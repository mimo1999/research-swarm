"""Heterogeneous worker agents for parallel research dispatch.

Each ``WorkerRole`` gets a distinct research perspective and tool priority:

  general   -- balanced default (web + arXiv + RAG)
  academic  -- prioritises peer-reviewed literature and arXiv pre-prints
  industry  -- prioritises web, company blogs, case studies, official docs
  skeptic   -- actively seeks counter-evidence, limitations, failure modes
  benchmark -- seeks quantitative comparisons, benchmarks, ablation studies

All roles ultimately call ``run_researcher`` from ``agents.researcher``; they
differ only in their system-prompt strategy section and, where applicable, the
tool priority hints they embed in that strategy.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from research_swarm.agents.researcher import (
    _TOOL_TURNS_BY_DEPTH,
    MAX_TOOL_TURNS,
    FindingSynthesis,
    _extract_sources_from_messages,
    _norm,
    _run_tool_loop,
    _synthesis_prompt,
)
from research_swarm.schemas.finding import Finding
from research_swarm.schemas.worker import WorkerRole

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role-specific strategy sections (embedded in the shared system prompt)
# ---------------------------------------------------------------------------

_STRATEGIES: dict[WorkerRole, str] = {
    WorkerRole.general: """\
1. Start with retrieve_from_rag to check the session corpus.
2. Use web_search and arxiv_search for fresh information.
3. Fetch 1-2 promising URLs for detail.
4. Stop when you have 3-5 good sources.""",

    WorkerRole.academic: """\
You are the ACADEMIC worker. Prioritise peer-reviewed and pre-print sources.
1. Use arxiv_search FIRST — this is an academic question.
2. Check retrieve_from_rag for any ingested papers.
3. Use web_search only for supplementary context or DOI resolution.
4. Cite specific paper titles, authors, or DOIs wherever possible.
5. Flag if findings are preliminary (pre-print) vs peer-reviewed.""",

    WorkerRole.industry: """\
You are the INDUSTRY worker. Prioritise practical, real-world deployment evidence.
1. Use web_search FIRST with terms like "production", "deployed", "case study", "enterprise".
2. Look for official documentation, engineering blogs, and vendor announcements.
3. Avoid purely theoretical sources; prefer concrete adoption metrics where available.
4. Note the organisation behind each source to surface any potential bias.""",

    WorkerRole.skeptic: """\
You are the SKEPTIC worker. Your job is to find weaknesses, counter-evidence, and
known failure modes — NOT to confirm the mainstream view.
1. Use web_search with terms like "criticism", "limitation", "failure", "critique", "drawback".
2. Use arxiv_search with terms like "negative result", "limitations", "challenges".
3. Specifically look for published replications that failed, or papers challenging
   key assumptions of the mainstream position.
4. A finding that says "evidence against X is scarce" is valuable — report it explicitly.""",

    WorkerRole.benchmark: """\
You are the BENCHMARK worker. Seek quantitative comparisons and empirical evidence.
1. Use web_search with terms like "benchmark", "comparison", "performance", "evaluation".
2. Use arxiv_search for papers with ablation studies, leaderboard results, or metric tables.
3. Extract specific numbers: accuracy, latency, throughput, F1, BLEU, cost, etc.
4. Report the benchmark name, dataset, and date so the reader can assess recency.""",
}

_SYSTEM_TEMPLATE = """\
You are an expert Research Agent ({role} perspective). Answer the research sub-question
by calling the available tools, then synthesise your findings.

Available tools:
  web_search        -- search the web (Tavily)
  arxiv_search      -- search arXiv preprints
  fetch_url         -- fetch and extract text from a URL
  retrieve_from_rag -- query session RAG index

Strategy ({depth} mode, {role} perspective):
{strategy}

Audience: {audience}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_worker(
    sub_question: str,
    role: WorkerRole,
    state: AgentState,
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> Finding | None:
    """Research a single *sub_question* from the given *role* perspective.

    Returns a ``Finding`` or ``None`` if the run produced no usable claim.
    """
    query       = state.get("query")
    session_id  = state.get("session_id", "default")
    audience    = query.audience if query else "general"
    raw_depth   = str(
        query.depth.value if hasattr(query.depth, "value") else query.depth
    ) if query else "standard"
    max_turns   = _TOOL_TURNS_BY_DEPTH.get(raw_depth, MAX_TOOL_TURNS)

    # For shallow mode the skeptic/benchmark roles would be noisy with just 1
    # tool turn; fall back silently to the general strategy.
    strategy = _STRATEGIES.get(role, _STRATEGIES[WorkerRole.general])
    if raw_depth == "shallow":
        strategy = _STRATEGIES[WorkerRole.general]

    system_msg = SystemMessage(
        content=_SYSTEM_TEMPLATE.format(
            role=role.value,
            depth=raw_depth,
            strategy=strategy,
                audience=audience,
        )
    )

    tool_map       = {t.name: t for t in tools}
    # Shallow mode has only one tool turn, so require that turn to gather
    # evidence instead of allowing the model to answer from memory.
    if raw_depth == "shallow":
        llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    else:
        llm_with_tools = llm.bind_tools(tools)
    synthesis_llm  = llm.with_structured_output(FindingSynthesis)

    logger.info("Worker[%s] researching: %s", role.value, sub_question[:80])
    messages = await _run_tool_loop(
        sub_question=sub_question,
        session_id=session_id,
        llm_with_tools=llm_with_tools,
        tool_map=tool_map,
        system_msg=system_msg,
        max_turns=max_turns,
    )

    sources = _extract_sources_from_messages(messages)
    logger.info("Worker[%s] extracted %d sources", role.value, len(sources))

    synthesis_messages = messages + [HumanMessage(content=_synthesis_prompt(sub_question))]
    try:
        synthesis: FindingSynthesis = await synthesis_llm.ainvoke(synthesis_messages)
    except Exception as exc:
        logger.warning("Worker[%s] synthesis failed for %r: %s", role.value, sub_question, exc)
        synthesis = FindingSynthesis(
            claim=f"[Research incomplete for: {sub_question}]",
            confidence=0.2,
        )

    # Reuse existing finding ID on re-research passes (merge-by-id reducer).
    existing_findings = state.get("findings") or []
    existing_id: str | None = None
    for f in existing_findings:
        sub_key = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        if _norm(sub_key) == _norm(sub_question):
            existing_id = f.id if hasattr(f, "id") else f.get("id")
            break

    import uuid
    finding = Finding(
        id=existing_id or str(uuid.uuid4()),
        claim=synthesis.claim,
        evidence=sources,
        confidence=synthesis.confidence,
        sub_question=sub_question,
    )
    logger.info(
        "Worker[%s] produced finding (conf=%.2f, evidence=%d): %s",
        role.value, finding.confidence, len(finding.evidence), finding.claim[:80],
    )
    return finding
