"""Researcher agent — tool-calling ReAct loop that produces Finding objects."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from research_swarm.schemas import Finding, Source
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 6  # max tool-call rounds per sub-question

_SYSTEM_PROMPT = """\
You are an expert Research Agent. Your goal is to answer a research sub-question
by calling the available tools to gather evidence, then synthesising your findings.

Available tools:
  web_search      — search the web (Tavily)
  arxiv_search    — search arXiv preprints
  fetch_url       — fetch and extract text from a URL
  retrieve_from_rag — query documents already ingested into the session RAG index

Strategy:
1. Start with retrieve_from_rag to check if the answer is already in the session corpus.
2. Use web_search and arxiv_search for fresh information.
3. Fetch promising URLs for detail.
4. Stop calling tools once you have enough evidence (3–5 good sources).

Session ID (required for retrieve_from_rag): {session_id}
Audience: {audience}
"""

_SYNTHESIS_PROMPT = """\
Based on the evidence gathered above, write a concise, factual claim that directly
answers the sub-question: "{sub_question}"

Return ONLY a JSON object with these fields:
  claim       — one or two sentences summarising the answer
  confidence  — float 0.0–1.0 reflecting how well the evidence supports the claim
"""


class FindingSynthesis(BaseModel):
    claim: str = Field(..., description="Concise claim answering the sub-question")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


async def _run_tool_loop(
    sub_question: str,
    session_id: str,
    llm_with_tools: BaseChatModel,
    tool_map: dict[str, BaseTool],
    system_msg: SystemMessage,
) -> list[Any]:
    """Run tool-calling loop for a single sub-question. Returns full message history."""
    messages: list[Any] = [
        system_msg,
        HumanMessage(content=f"Sub-question to research: {sub_question}"),
    ]

    for _ in range(MAX_TOOL_TURNS):
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
                    result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    result = f"[Tool error: {exc}]"

            messages.append(
                ToolMessage(
                    content=json.dumps(result, default=str)[:4000],
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
    state: "AgentState",
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
        logger.warning("Researcher called with no plan — skipping.")
        return []

    session_id = state.get("session_id", "default")
    query = state.get("query")
    audience = query.audience if query else "general"

    # Determine which sub-questions need research
    existing_findings = state.get("findings") or []
    critiques = state.get("critiques") or []

    # Build a set of sub-questions that are already well-supported
    ok_sub_questions: set[str] = set()
    for c in critiques:
        verdict = c.verdict if hasattr(c, "verdict") else c.get("verdict")
        fid = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id")
        if verdict == CritiqueVerdict.supported:
            # Find the sub_question for this finding
            for f in existing_findings:
                fid_key = f.id if hasattr(f, "id") else f.get("id")
                sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                if fid_key == fid:
                    ok_sub_questions.add(sub_q)

    targets = [q for q in plan.sub_questions if q not in ok_sub_questions]

    if not targets:
        logger.info("All sub-questions already well-supported — nothing to research.")
        return []

    # Set up tools
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    synthesis_llm = llm.with_structured_output(FindingSynthesis)

    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(session_id=session_id, audience=audience)
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
        )

        sources = _extract_sources_from_messages(messages)

        # Synthesise a finding from the gathered evidence
        synthesis_prompt = _SYNTHESIS_PROMPT.format(sub_question=sub_q)
        synthesis_messages = messages + [HumanMessage(content=synthesis_prompt)]

        try:
            synthesis: FindingSynthesis = await synthesis_llm.ainvoke(synthesis_messages)
        except Exception as exc:
            logger.warning("Synthesis failed for sub-question %r: %s", sub_q, exc)
            synthesis = FindingSynthesis(
                claim=f"[Research incomplete for: {sub_q}]",
                confidence=0.2,
            )

        # Check if this sub-question already has a finding (re-research → overwrite by id)
        existing_id: str | None = None
        for f in existing_findings:
            sub_key = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
            if sub_key == sub_q:
                existing_id = f.id if hasattr(f, "id") else f.get("id")
                break

        finding = Finding(
            id=existing_id or Finding(claim="").id,  # reuse id to trigger merge
            claim=synthesis.claim,
            evidence=sources,
            confidence=synthesis.confidence,
            sub_question=sub_q,
        )
        # Ensure id is set correctly when reusing
        if existing_id:
            object.__setattr__(finding, "id", existing_id) if hasattr(finding, "__dict__") else None
            finding = finding.model_copy(update={"id": existing_id})

        new_findings.append(finding)

    return new_findings
