"""Researcher agent -- tool-calling ReAct loop that produces Finding objects."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from research_swarm.agents._utils import schema_output_instruction
from research_swarm.schemas import Finding, Source
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


# Tool-turn limits per depth level.  Each turn is one LLM call, so this is
# the dominant factor in per-sub-question latency with slow cloud models.
_TOOL_TURNS_BY_DEPTH: dict[str, int] = {
    "shallow":  1,   # single search call, no follow-up fetches
    "standard": 3,   # balanced
    "deep":     6,   # thorough: original behaviour
}
MAX_TOOL_TURNS = _TOOL_TURNS_BY_DEPTH["standard"]  # module-level fallback


def _norm(s: str) -> str:
    """Normalise a sub-question string for comparison.

    Strips surrounding whitespace and lowercases so that trivial formatting
    differences between the plan's sub-questions and the strings stored on
    Finding objects don't prevent ID reuse on re-research passes (which would
    cause duplicate findings and an unbounded critic loop).
    """
    return s.strip().lower()


class FindingSynthesis(BaseModel):
    claim: str = Field(..., description="Concise claim answering the sub-question")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_SYSTEM_PROMPT = """\
You are an expert Research Agent. Your goal is to answer a research sub-question
by calling the available tools to gather evidence, then synthesising your findings.

Available tools:
  web_search        -- search the web (Tavily)
  arxiv_search      -- search arXiv preprints
  fetch_url         -- fetch and extract text from a URL
  retrieve_from_rag -- query documents already ingested into the session RAG index

Strategy ({depth} mode):
{strategy}

Session ID (required for retrieve_from_rag): {session_id}
Audience: {audience}
"""

_STRATEGY_SHALLOW = """\
1. Call web_search FIRST for a direct, fast answer.
2. Do NOT call retrieve_from_rag or fetch_url (budget is 1 tool call).
3. Synthesise immediately from the search results."""

_STRATEGY_DEFAULT = """\
1. Start with retrieve_from_rag to check if the answer is already in the session corpus.
2. Use web_search and arxiv_search for fresh information.
3. Fetch promising URLs for detail.
4. Stop calling tools once you have enough evidence (3-5 good sources)."""

_SYNTHESIS_JSON_SUFFIX = schema_output_instruction(FindingSynthesis)


def _synthesis_prompt(sub_question: str) -> str:
    """Build the synthesis prompt for a specific sub-question."""
    return (
        f"Based on the evidence gathered above, write a concise, factual claim that "
        f"directly answers the sub-question: \"{sub_question}\""
        + _SYNTHESIS_JSON_SUFFIX
    )


def _serialize_tool_result(result: Any, max_chars: int = 8000) -> str:
    """Serialize a tool result to a JSON string safe for inclusion in a ToolMessage.

    A naive ``json.dumps(result)[:max_chars]`` truncation can produce invalid JSON
    when the result is a list of source dicts whose snippets push the total length
    past the limit — ``json.loads`` then silently fails and all source metadata is
    lost.  This function truncates per-item snippets first so the outer JSON array
    always remains valid and ``_extract_sources_from_messages`` can parse it.
    """
    if isinstance(result, list):
        trimmed = []
        for item in result:
            if isinstance(item, dict) and "snippet" in item and isinstance(item["snippet"], str):
                item = {**item, "snippet": item["snippet"][:600]}
            trimmed.append(item)
        result = trimmed
    serialized = json.dumps(result, default=str)
    return serialized[:max_chars]


async def _run_tool_loop(
    sub_question: str,
    session_id: str,
    llm_with_tools: BaseChatModel,
    tool_map: dict[str, BaseTool],
    system_msg: SystemMessage,
    max_turns: int = MAX_TOOL_TURNS,
) -> list[Any]:
    """Run tool-calling loop for a single sub-question. Returns full message history."""
    messages: list[Any] = [
        system_msg,
        HumanMessage(content=f"Sub-question to research: {sub_question}"),
    ]

    for _ in range(max_turns):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break  # LLM decided it has enough info

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                result = f"[Unknown tool: {tool_name}]"
            else:
                try:
                    # Tools are sync (httpx/Tavily) — run in a thread so parallel
                    # worker nodes don't serialize on the event loop.
                    result = await asyncio.to_thread(tool_fn.invoke, tool_args)
                except Exception as exc:
                    result = f"[Tool error: {exc}]"

            messages.append(
                ToolMessage(
                    content=_serialize_tool_result(result),
                    tool_call_id=tool_id,
                )
            )

    return messages


def _extract_sources_from_messages(messages: list[Any]) -> list[Source]:
    """Parse Source dicts out of ToolMessage content."""
    sources: list[Source] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and "url" in item:
                try:
                    sources.append(Source(**item))
                except Exception:
                    pass
    return sources[:10]  # cap evidence list


async def run_researcher(
    state: AgentState,
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> list[Finding]:
    """Research sub-questions and return new / updated Finding objects.

    Targets sub-questions that either:
      - have no finding yet, OR
      - were marked weak/refuted by the critic (re-research pass).
    """
    plan = state.get("plan")
    if not plan:
        logger.warning("Researcher called with no plan -- skipping.")
        return []

    session_id = state.get("session_id", "default")
    query = state.get("query")
    audience = query.audience if query else "general"

    # Resolve depth → tool-turn budget (str or enum both work)
    _d = query.depth if query else "standard"
    raw_depth = str(_d.value if hasattr(_d, "value") else _d)
    max_turns = _TOOL_TURNS_BY_DEPTH.get(raw_depth, MAX_TOOL_TURNS)

    # Determine which sub-questions need research
    existing_findings = state.get("findings") or []
    critiques = state.get("critiques") or []

    # Build a set of sub-questions that are already well-supported.
    # Normalise strings so whitespace/casing differences don't cause a
    # supported sub-question to be researched again.
    ok_sub_questions: set[str] = set()
    for c in critiques:
        verdict = c.verdict if hasattr(c, "verdict") else c.get("verdict")
        fid = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id")
        if str(verdict) == CritiqueVerdict.supported or verdict == CritiqueVerdict.supported:
            # Find the sub_question for this finding
            for f in existing_findings:
                fid_key = f.id if hasattr(f, "id") else f.get("id")
                sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                if fid_key == fid:
                    ok_sub_questions.add(_norm(sub_q))

    targets = [q for q in plan.sub_questions if _norm(q) not in ok_sub_questions]

    if not targets:
        logger.info("All sub-questions already well-supported -- nothing to research.")
        return []

    # Set up tools
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    synthesis_llm = llm.with_structured_output(FindingSynthesis)

    strategy = _STRATEGY_SHALLOW if raw_depth == "shallow" else _STRATEGY_DEFAULT
    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(
            depth=raw_depth,
            strategy=strategy,
            session_id=session_id,
            audience=audience,
        )
    )

    new_findings: list[Finding] = []

    for sub_q in targets:
        logger.info("Researching: %s", sub_q)
        messages = await _run_tool_loop(
            sub_question=sub_q,
            session_id=session_id,
            llm_with_tools=llm_with_tools,
            tool_map=tool_map,
            system_msg=system_msg,
            max_turns=max_turns,
        )

        sources = _extract_sources_from_messages(messages)

        # Synthesise a finding from the gathered evidence
        synthesis_prompt = _synthesis_prompt(sub_q)
        synthesis_messages = messages + [HumanMessage(content=synthesis_prompt)]

        try:
            synthesis: FindingSynthesis = await synthesis_llm.ainvoke(synthesis_messages)
        except Exception as exc:
            logger.warning("Synthesis failed for sub-question %r: %s", sub_q, exc)
            synthesis = FindingSynthesis(
                claim=f"[Research incomplete for: {sub_q}]",
                confidence=0.2,
            )

        # Check if this sub-question already has a finding (re-research -> overwrite by id).
        # Normalise the stored sub_question string before comparing so that trivial
        # whitespace/casing differences don't cause a fresh UUID to be generated,
        # which would bypass the merge-by-id reducer and duplicate the finding.
        existing_id: str | None = None
        for f in existing_findings:
            sub_key = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
            if _norm(sub_key) == _norm(sub_q):
                existing_id = f.id if hasattr(f, "id") else f.get("id")
                break

        if existing_id is None and bool(existing_findings):
            logger.warning(
                "Re-research: no existing finding matched sub-question %r -- "
                "a new finding will be created (check for sub-question drift).",
                sub_q,
            )

        finding = Finding(
            id=existing_id or str(uuid.uuid4()),
            claim=synthesis.claim,
            evidence=sources,
            confidence=synthesis.confidence,
            sub_question=sub_q,
        )

        new_findings.append(finding)

    return new_findings
