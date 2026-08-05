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

from research_swarm.agents._utils import recover_from_parse_failure, schema_output_instruction
from research_swarm.schemas import Finding, Source
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


# Tool-turn limits per depth level.  Each turn is one LLM call, so this is
# the dominant factor in per-sub-question latency with slow cloud models.
# standard reverted 4 -> 3: raising it (alongside more sub-questions) pushed
# a live run's LLM-call usage to 49 against a 40 budget, exhausting the
# shared budget before critic/fact-checker/writer ran. Combined with the
# separate research/review budget pools (runtime/budget.py), this keeps
# per-worker cost predictable instead of relying on the budget cap to save it.
_TOOL_TURNS_BY_DEPTH: dict[str, int] = {
    "shallow":  1,   # single search call, no follow-up fetches
    "standard": 3,   # balanced
    "deep":     6,   # thorough: original behaviour
}
MAX_TOOL_TURNS = _TOOL_TURNS_BY_DEPTH["standard"]  # module-level fallback

# Max sources kept per finding after relevance ranking. Same as the legacy
# flat cap (10) -- measured empirically (RAGAs faithfulness/context_precision
# on a live swarm run) that ranking by relevance is what fixes low context
# precision (~0.51 -> ~0.76-0.80), while *also* shrinking this cap to 6 traded
# away faithfulness (~0.64 -> ~0.45): claims now carry several distinct
# numeric/quantitative sub-statements (see the synthesis-prompt fix), each
# needing its own grounding, and a smaller pool starves that. Ranking alone,
# at the same cap, gets nearly all the precision gain without that cost. See
# _rank_sources_by_relevance.
MAX_EVIDENCE_SOURCES = 10


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


class _SnippetSummaries(BaseModel):
    summaries: list[str] = Field(
        default_factory=list,
        description=(
            "One condensed summary per excerpt, in the same order. Each summary is "
            "2-4 sentences that preserve every fact, number, date, and named entity "
            "relevant to the research question -- do not add opinions or omit specifics "
            "to save space."
        ),
    )


_SYSTEM_PROMPT = """\
You are an expert Research Agent. Your goal is to answer a research sub-question
by calling the available tools to gather evidence, then synthesising your findings.

Available tools:
  web_search        -- search the web (Tavily)
  arxiv_search      -- search arXiv preprints (physics/CS/math -- NOT biomedical)
  pubmed_search     -- search PubMed (biomedical/clinical/life-sciences literature)
  fetch_url         -- fetch and extract text from a URL
  retrieve_from_rag -- query evidence already discovered by earlier rounds of THIS
                       research session (NOT user-uploaded documents -- those were
                       already extracted into findings before research began, so
                       check the existing findings for that, not this tool)

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
1. Start with retrieve_from_rag to check if an earlier round of this session already
   found evidence for this (it will NOT contain user-uploaded documents -- those are
   already reflected in the existing findings, not in this tool).
2. Use web_search for fresh information. For medical/clinical/biological
   questions use pubmed_search; for CS/physics/math questions use
   arxiv_search -- arxiv_search barely indexes biomedical literature, and
   pubmed_search has no CS/physics/math coverage, so pick the one that
   matches the sub-question's actual domain.
3. Fetch promising URLs for detail.
4. Stop calling tools once you have enough evidence (3-5 good sources)."""

_SYNTHESIS_JSON_SUFFIX = schema_output_instruction(FindingSynthesis)

_SUMMARIZE_SYSTEM_PROMPT = (
    "You condense raw source excerpts into compact evidence for a research question. "
    "For each excerpt below, write a summary that preserves every specific fact, number, "
    "date, and named entity relevant to the question. Do not add opinions, and do not drop "
    "information to save space beyond removing filler prose."
    + schema_output_instruction(_SnippetSummaries)
)

# Excerpts shorter than this rarely lose anything to truncation, so skip the
# summarization call for them entirely -- it would cost more tokens than it saves.
_SUMMARIZE_THRESHOLD_CHARS = 400


def _synthesis_prompt(sub_question: str) -> str:
    """Build the synthesis prompt for a specific sub-question."""
    return (
        f"Based on the evidence gathered above, write a concise, factual claim that "
        f"directly answers the sub-question: \"{sub_question}\"\n\n"
        "If the evidence contains specific numbers -- effect sizes, percentages, "
        "sample sizes, p-values, confidence intervals, dosages, durations -- state "
        "them exactly as given. Do not compress a quantitative result down to a "
        "qualitative gist (e.g. \"showed improvement\"): the numbers ARE the finding, "
        "and a claim without them is far less useful even if it points the right "
        "direction.\n\n"
        "If the evidence gives you numbers from more than one source to compare "
        "(e.g. two techniques, a before/after, different model sizes), check they "
        "were actually measured under comparable conditions -- same model size/scale, "
        "same benchmark or task, same setup -- before presenting them as a direct "
        "comparison. Sources you gathered may cover different scales or setups even "
        "when they're both relevant to this sub-question. If the conditions don't "
        "match, say so in the claim itself (e.g. \"not directly comparable -- X was "
        "measured on a much smaller model\") instead of juxtaposing the numbers as if "
        "they were equivalent."
        + _SYNTHESIS_JSON_SUFFIX
    )


async def _summarize_tool_items(
    items: list[Any],
    sub_question: str,
    summarizer_llm: BaseChatModel,
) -> list[Any]:
    """Replace long snippets with LLM-condensed summaries instead of hard-truncating.

    Character truncation cuts mid-sentence and can drop the exact fact a finding
    needs. This runs the excerpts through the cheapest tier once (batched, one
    call per tool invocation) so the compressed version -- not a chopped one --
    is what gets re-sent on every subsequent turn of the ReAct loop.
    """
    idx_to_summarize = [
        i for i, item in enumerate(items)
        if isinstance(item, dict)
        and isinstance(item.get("snippet"), str)
        and len(item["snippet"]) > _SUMMARIZE_THRESHOLD_CHARS
        and not item["snippet"].startswith("[")  # skip error placeholders
    ]
    if not idx_to_summarize:
        return items

    excerpts_block = "\n\n".join(
        f"[{n + 1}] {items[i]['snippet'][:4000]}" for n, i in enumerate(idx_to_summarize)
    )
    user_msg = HumanMessage(
        content=(
            f"Research question: {sub_question}\n\n"
            f"Excerpts ({len(idx_to_summarize)}):\n{excerpts_block}\n\n"
            "Return one summary per excerpt above, in order."
        )
    )

    try:
        structured = summarizer_llm.with_structured_output(_SnippetSummaries)
        result: _SnippetSummaries = await structured.ainvoke(
            [SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT), user_msg]
        )
    except Exception as exc:
        logger.warning("Snippet summarization failed (%d excerpts): %s", len(idx_to_summarize), exc)
        return items

    if len(result.summaries) != len(idx_to_summarize):
        logger.warning(
            "Snippet summarizer returned %d summaries for %d excerpts; keeping originals.",
            len(result.summaries), len(idx_to_summarize),
        )
        return items

    out = list(items)
    for i, summary in zip(idx_to_summarize, result.summaries):
        out[i] = {**out[i], "snippet": summary}
    return out


def _serialize_tool_result(result: Any, max_chars: int = 8000) -> str:
    """Serialize a tool result to a JSON string safe for inclusion in a ToolMessage.

    A naive ``json.dumps(result)[:max_chars]`` truncation can produce invalid JSON
    when the result is a list of source dicts whose snippets push the total length
    past the limit — ``json.loads`` then silently fails and all source metadata is
    lost. This function caps per-item snippets first so the outer JSON array always
    remains valid and ``_extract_sources_from_messages`` can parse it. This cap is a
    safety net only: normal-sized snippets are expected to already have been condensed
    by ``_summarize_tool_items`` upstream, not hard-truncated here.
    """
    if isinstance(result, list):
        trimmed = []
        for item in result:
            if isinstance(item, dict) and "snippet" in item and isinstance(item["snippet"], str):
                item = {**item, "snippet": item["snippet"][:1200]}
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
    summarizer_llm: BaseChatModel | None = None,
) -> list[Any]:
    """Run tool-calling loop for a single sub-question. Returns full message history.

    ``summarizer_llm``, when given, condenses each tool result's snippets via a
    cheap-tier LLM call instead of hard-truncating them -- see
    ``_summarize_tool_items``. Pass ``None`` to fall back to truncation only.
    """
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

            logger.info("Tool call: %s(%s)", tool_name, tool_args)

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

            if summarizer_llm is not None and isinstance(result, list):
                result = await _summarize_tool_items(result, sub_question, summarizer_llm)

            messages.append(
                ToolMessage(
                    content=_serialize_tool_result(result),
                    tool_call_id=tool_id,
                )
            )

    return messages


async def _rank_sources_by_relevance(
    sub_question: str, sources: list[Source], top_k: int = MAX_EVIDENCE_SOURCES,
) -> list[Source]:
    """Return the top_k sources most relevant to sub_question.

    Sources previously reached the Finding's evidence list in whatever order
    the tool loop happened to accumulate them across turns -- unrelated to
    relevance -- then got flatly truncated. That's why RAGAs context_precision
    measured ~0.3-0.5: half the attached "evidence" was noise the LLM never
    actually used, just the first N results across however many tool calls
    fired. This ranks by embedding similarity to the actual sub-question
    instead of accumulation order.

    Uses BGE embeddings (get_embed_model), not the ms-marco cross-encoder in
    rag/reranker.py -- that reranker explicitly skips long sentence-style
    queries (>8 words) as out-of-distribution, and worker sub-questions are
    almost always long natural-language questions, so it would silently no-op
    on exactly the case this needs to handle. BGE is already used elsewhere
    in this project for this exact kind of long-question semantic matching
    (graph/stop.py, eval/faithfulness.py).

    The embedding work is CPU-bound (HuggingFace inference), so it's offloaded
    to a thread via asyncio.to_thread -- this runs once per worker's synthesis
    step, and workers execute concurrently as LangGraph Send-fanned tasks
    sharing one event loop; without to_thread this call would block that
    shared loop and silently serialize what's supposed to be parallel
    dispatch. All sources are embedded in a single batched call rather than
    one embedding call per source, for the same reason _ingest_round_evidence
    batches its Chroma writes -- N sequential model calls where one suffices.
    """
    if len(sources) <= top_k:
        return sources

    def _compute() -> list[Source]:
        from research_swarm.rag.indexes import cosine_similarity, get_embed_model
        model = get_embed_model()
        q_emb = model.get_text_embedding(sub_question)
        texts = [f"{s.title}\n{s.snippet}".strip() or s.url for s in sources]
        embeddings = model.get_text_embedding_batch(texts)
        scored = sorted(
            zip((cosine_similarity(q_emb, e) for e in embeddings), sources),
            key=lambda x: x[0],
            reverse=True,
        )
        return [s for _, s in scored[:top_k]]

    try:
        return await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.warning(
            "Source relevance ranking failed (%s) -- keeping first %d in original order.",
            exc, top_k,
        )
        return sources[:top_k]


async def _extract_sources_from_messages(
    messages: list[Any], sub_question: str | None = None,
) -> list[Source]:
    """Parse Source dicts out of ToolMessage content.

    When sub_question is given, the returned list is ranked by relevance to
    it (see _rank_sources_by_relevance) and capped at MAX_EVIDENCE_SOURCES.
    Without it, falls back to the original flat first-10 behaviour -- kept
    for callers that don't have a sub-question in scope.
    """
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
    if sub_question:
        return await _rank_sources_by_relevance(sub_question, sources)
    return sources[:10]  # legacy cap for callers without a sub_question


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

        sources = await _extract_sources_from_messages(messages, sub_question=sub_q)

        # Synthesise a finding from the gathered evidence
        synthesis_prompt = _synthesis_prompt(sub_q)
        synthesis_messages = messages + [HumanMessage(content=synthesis_prompt)]

        try:
            synthesis: FindingSynthesis = await synthesis_llm.ainvoke(synthesis_messages)
        except Exception as exc:
            logger.warning("Synthesis failed for sub-question %r: %s", sub_q, exc)
            recovered = recover_from_parse_failure(exc, FindingSynthesis)
            if recovered is not None:
                logger.info("Recovered synthesis from malformed completion for %r", sub_q)
                synthesis = recovered
            else:
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
