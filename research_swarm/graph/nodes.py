"""LangGraph node functions — one async function per stage.

Topology (see CLAUDE.md for the diagram)
========================================
START → supervisor_node  (LLM: plan creation only)
          ↓ next_agent = "dispatch"
        document_pass_node  (deterministic: one-time fan-out)
          ├─ Send × B  document_worker_node  (uploaded documents packed into batches, one
          │            extraction call each)
          └─ Send × 1  paper_scout_node      (search → light-LLM relevance filter)
          ↓ both converge
        paper_worker_node  (extraction over the kept abstracts, one call per sub-question)
          ↓
        dispatch_node  (deterministic: record pre-round IDs, coverage gate, fan out via Send)
          ↓ Send × N (one per sub-question that still lacks min_grounded_facts grounded facts)
        worker_node  (gap fill: search → fetch → one extraction call)
          ↓ findings merged by the _merge_findings reducer
        collect_node  (deterministic: stop-signal check)
          ├─ loop  → dispatch_node  (novelty signal still open)
          └─ stop  → verifier_node → writer_node → END
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Send

from research_swarm.agents._utils import _field, _latest_verdicts
from research_swarm.agents.base import get_agent_llm, get_tiered_llm, without_thinking
from research_swarm.agents.question import research_topic
from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
from research_swarm.config import settings
from research_swarm.eval.llm_judge import judge_report
from research_swarm.runtime.budget import BudgetExceeded, get_budget
from research_swarm.runtime.limits import set_llm_context
from research_swarm.runtime.trace import TraceCallback, timed, trace_event, traced_node
from research_swarm.schemas.critique import CritiqueVerdict
from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------





def _stage_of_agent(agent: str) -> str:
    """The stage a ``_get_tiered_state_llm`` ``agent=`` label belongs to.

    "worker/general[sub-question]/summarizer" -> "summarizer"; "worker/general[...]" -> "worker";
    "paper_worker[sub-question]" -> "paper_worker"; "verifier" -> "verifier".
    """
    if agent.endswith("/summarizer"):
        return "summarizer"
    return agent.split("[", 1)[0].split("/", 1)[0]


def _get_tiered_state_llm(state: AgentState, tier: str, pool: str = "research", agent: str = ""):
    """Create a tiered LLM with budget callback attached.

    ``pool`` selects which of the session's two independent budget counters
    this call draws from -- "research" (supervisor, document workers,
    dispatch/worker loop -- the part that can genuinely iterate) or "review"
    (verifier/writer/judge -- a few batched calls). Kept separate
    so a research-loop overrun can't exhaust the budget the verifier/
    writer need to turn already-gathered findings into a real report. See
    runtime/budget.py.

    For the 'standard' (worker) tier, the session's user-selected provider
    overrides the static tier config -- so picking Anthropic/OpenAI in the UI
    actually routes workers to that provider's lowest-grade model instead of
    silently staying on the configured default (e.g. Ollama).

    Callback is set via the model's own ``callbacks`` field (model_copy),
    NOT ``.with_config({"callbacks": [...]})``. Every call site immediately
    chains ``.with_structured_output(...)`` on the returned model, and
    ``with_structured_output``/``bind_tools`` have no ``config`` parameter --
    so RunnableBinding.__getattr__'s config-merging only kicks in for methods
    that accept one, and for these it doesn't, silently returning
    ``self.bound.with_structured_output(...)`` on the *unwrapped* model and
    dropping the callback (and with it, all budget call/token counting).
    Setting the field directly on the model instance survives that because
    with_structured_output/bind_tools operate on `self` itself, not a wrapper.

    Stages listed in ``settings.no_thinking_stages`` (structured-JSON producers) run with thinking
    off and a capped output; see the setting for why.

    A stage listed in ``settings.large_model_stages`` (by its agent label, e.g. "supervisor",
    "writer", "paper_scout", "paper_worker", "verifier", "gap_fill") gets ``settings.large_model``
    on its own endpoint instead (by default Ollama Cloud directly, while the rest stays on the
    local daemon) -- so moving a stage is a config change, not a code change.
    """
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id, pool=pool)
    stage = _stage_of_agent(agent or tier)
    if settings.large_model and stage in settings.large_model_stages:
        provider = settings.large_model_provider
        base_url = settings.large_model_ollama_base_url if provider == "ollama" else ""
        llm = get_agent_llm(provider=provider, model=settings.large_model, temperature=0.0,
                            base_url=base_url or None)
        # A direct Ollama Cloud endpoint gets its own concurrency pool (limits.llm_slot).
        set_llm_context("ollama_cloud" if base_url else provider, session_id)
    else:
        provider_override = state.get("model_provider") if tier == "standard" else None
        llm = get_tiered_llm(tier=tier, provider_override=provider_override)
        # Which provider's concurrency cap (runtime/limits.py::llm_slot) this task's calls use.
        provider = provider_override or getattr(
            settings, f"tier_{tier}_provider", settings.default_model_provider,
        )
        set_llm_context(provider, session_id)
    if stage in settings.no_thinking_stages:
        llm = without_thinking(llm, settings.no_thinking_max_tokens)
    callbacks = [budget.callback, TraceCallback(session_id, agent or tier, tier)]
    return llm.model_copy(update={"callbacks": callbacks})


def _check_budget(
    state: AgentState, node_name: str, pool: str = "research",
) -> dict[str, Any] | None:
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id, pool=pool)
    try:
        budget.check()
        return None
    except BudgetExceeded as exc:
        logger.warning("%s: %s — forcing writer.", node_name, exc)
        unit = f"{exc.pool} calls" if exc.kind == "calls" else "session tokens"
        return {
            "next_agent": "writer",
            "messages": [
                AIMessage(
                    content=f"[{node_name}] Budget exceeded ({exc.used}/{exc.limit} "
                            f"{unit}); forcing report."
                )
            ],
        }


def _scope(plan: Any) -> str:
    """The plan's question-frame key constraint ("" without one)."""
    frame = getattr(plan, "frame", None) if plan else None
    return frame.key_constraint if frame is not None and frame.has_constraint else ""


def _counts_as_coverage(finding: Any, frame: Any) -> tuple[bool, str]:
    """(counts?, reason) for the round-0 coverage gate.

    A finding covers its sub-question only when its evidence was located, the extractor did not
    label it background, and -- when the question has a key constraint -- its claim or evidence
    actually talks about that constraint. The lexical check is a second opinion on purpose: a
    lenient small model labels everything "direct" (the KV-cache run had 14 compression facts
    "covering" a cross-model question, so gap fill never ran). A false miss costs one gap-fill
    call.
    """
    from research_swarm.agents.expansion import scope_hit

    if _field(finding, "grounding", "unknown") == "none":
        return False, "ungrounded"
    if _field(finding, "relevance", "unknown") in ("background", "off_topic"):
        return False, "background"
    evidence = _field(finding, "evidence", []) or []
    text = _field(finding, "claim", "") + " " + (
        _field(evidence[0], "snippet", "") if evidence else ""
    )
    if not scope_hit(text, frame):
        return False, "scope_miss"
    return True, "direct"


def _research_targets(state: AgentState, trace: bool = False) -> list[str]:
    """Return the sub-questions that still need research this round, at most the run depth's
    ``gap_fill_workers`` (least covered first; 0 = no cap).

    Shared by dispatch_node and route_from_dispatch so both agree.
    """
    targets = _uncapped_research_targets(state, trace=trace)
    workers = settings.for_depth("gap_fill_workers", getattr(state.get("query"), "depth", None))
    if workers <= 0 or len(targets) <= workers:
        return targets
    if trace:
        trace_event(state.get("session_id"), "dispatch.capped", "note",
                    workers=workers, dropped=[sq[:80] for sq in targets[workers:]])
    return targets[:workers]


def _uncapped_research_targets(state: AgentState, trace: bool = False) -> list[str]:
    """Every sub-question that still needs research this round, least covered first.

    Round 0: every sub-question with fewer than ``settings.min_grounded_facts`` findings that
    count as coverage (``_counts_as_coverage``: grounded, not background, within the question
    frame's scope). Later rounds: sub-questions with no finding at all (a worker that failed).
    """
    plan = state.get("plan")
    if not plan:
        return []

    findings = state.get("findings") or []

    if state.get("rework_instructions") is not None:
        return _rework_targets(state, trace=trace)

    if state.get("research_rounds", 0) == 0:
        frame = getattr(plan, "frame", None)
        covered: dict[str, int] = {}
        reasons: dict[str, dict[str, int]] = {}
        for f in findings:
            key = _field(f, "sub_question", "").strip().lower()
            ok, why = _counts_as_coverage(f, frame)
            bucket = reasons.setdefault(key, {})
            bucket[why] = bucket.get(why, 0) + 1
            if ok:
                covered[key] = covered.get(key, 0) + 1
        need = max(1, settings.min_grounded_facts)
        targets = [sq for sq in plan.sub_questions if covered.get(sq.strip().lower(), 0) < need]
        # Least covered first, so a worker cap drops the sub-questions that already have some.
        targets.sort(key=lambda sq: covered.get(sq.strip().lower(), 0))
        for sq in targets if trace else []:
            got = reasons.get(sq.strip().lower(), {})
            if got.get("background") or got.get("scope_miss"):
                trace_event(state.get("session_id"), "coverage.scope_miss", "note",
                            sub_question=sq[:80], **got)
        return targets

    answered = {_field(f, "sub_question", "").strip().lower() for f in findings}
    return [sq for sq in plan.sub_questions if sq.strip().lower() not in answered]


def _rework_targets(state: AgentState, trace: bool = False) -> list[str]:
    """Sub-questions to research again when a reviewer asks for more (graph/rework.py): those
    without a finding the verifier supported and judged on-topic. If every sub-question has one,
    all of them -- the reviewer explicitly asked for more research, so doing nothing is wrong."""
    plan = state.get("plan")
    if not plan:
        return []
    verdicts = _latest_verdicts(state.get("critiques") or [])
    answered: set[str] = set()
    for f in state.get("findings") or []:
        if (verdicts.get(_field(f, "id", "")) == CritiqueVerdict.supported.value
                and _field(f, "relevance", "unknown") not in ("background", "off_topic")):
            answered.add(_field(f, "sub_question", "").strip().lower())
    weak = [sq for sq in plan.sub_questions if sq.strip().lower() not in answered]
    targets = weak or list(plan.sub_questions)
    if trace:
        trace_event(state.get("session_id"), "rework.targets", "note", targets=targets,
                    all_answered=not weak,
                    instructions=(state.get("rework_instructions") or "")[:200])
    return targets


def _depth_str(state: AgentState) -> str:
    query = state.get("query")
    if not query:
        return "standard"
    d = query.depth
    return d.value if hasattr(d, "value") else str(d)


# ---------------------------------------------------------------------------
# document_pass_node  (deterministic bookkeeping, mirrors dispatch_node)
# ---------------------------------------------------------------------------

async def document_pass_node(state: AgentState) -> dict[str, Any]:
    """Bookkeeping node before the one-time document-extraction fan-out.

    The actual fan-out (one Send per document, or per size-bounded slice of
    an oversized one) is handled by the conditional edge
    route_from_document_pass. This node just logs -- same shape as
    dispatch_node, which does the equivalent bookkeeping for the
    sub-question fan-out.
    """
    docs = state.get("ingested_documents") or []
    return {
        "messages": [
            AIMessage(content=(
                f"[DocumentPass] {len(docs)} ingested document(s) to process."
                if docs else "[DocumentPass] No ingested documents."
            ))
        ],
    }


def _dispatch_bounce_payload(state: AgentState) -> dict[str, Any]:
    """Build the Send payload for a no-op bounce straight to dispatch_node.

    Send() gives the receiving node ONLY this payload, not the full graph
    state -- dispatch_node (and _research_targets, which it calls) needs
    plan, findings, critiques and research_rounds to make its routing decision. Mirrors
    _collect_bounce_payload, which exists for the identical reason on the dispatch->collect
    side.
    """
    return {
        "session_id":      state.get("session_id", "default"),
        "query":           state.get("query"),
        "plan":            state.get("plan"),
        "findings":        state.get("findings") or [],
        "critiques":       state.get("critiques") or [],
        "research_rounds": state.get("research_rounds", 0),
        "human_feedback":  state.get("human_feedback"),
        "model_provider":  state.get("model_provider"),
        "model_name":      state.get("model_name"),
    }


def route_from_document_pass(state: AgentState):
    """Return a list of Send objects for the one-time pre-dispatch fan-out.

    Two independent kinds of Sends, mixed in one list (LangGraph supports a
    single conditional edge fanning out to different target nodes):
      - document_worker_node — one per packed batch of documents.
      - paper_scout_node — builds the relevance-filtered paper corpus for the
        plan's sub-questions BEFORE round-0 dispatch.

    "Has docs" and "has plan" are independent gates: a plan with no uploaded
    documents still gets the paper scout. Only a missing plan bounces
    straight to dispatch_node, same as always.
    """
    docs = state.get("ingested_documents") or []
    plan = state.get("plan")

    if not plan or not plan.sub_questions:
        return [Send("dispatch_node", _dispatch_bounce_payload(state))]

    session_id     = state.get("session_id", "default")
    model_provider = state.get("model_provider")
    model_name     = state.get("model_name")
    sub_questions  = list(plan.sub_questions)

    sends = []
    query = state.get("query")
    topic = research_topic(query)

    if docs:
        # Short documents share extraction calls: one call per ~extract_batch_chars of text
        # rather than one per document.
        from research_swarm.agents.extractor import pack_sources

        sources = [
            {"url": d.get("url", ""), "title": d.get("title", ""), "text": d.get("text", ""),
             "source_type": d.get("source_type", "pdf"),
             "credibility_score": d.get("credibility_score", 0.8)}
            for d in docs
        ]
        for batch in pack_sources(sources, settings.extract_batch_chars):
            sends.append(Send("document_worker_node", {
                "active_batch":           batch,
                "topic":                  topic,
                "scope":                  _scope(plan),
                "sub_questions_snapshot": sub_questions,
                "session_id":             session_id,
                "model_provider":         model_provider,
                "model_name":             model_name,
            }))

    if settings.enable_fetch_pass:
        from research_swarm.agents.papers import keyword_query

        tasks = []
        for sq in sub_questions:
            assignment = plan.assignment_for(sq)
            planned = assignment.search_query.strip() if assignment else ""
            search_query = planned or keyword_query(sq)
            tasks.append({
                "sub_question": sq,
                "search_query": search_query,
                "domain":       assignment.domain if assignment else "other",
            })
        sends.append(Send("paper_scout_node", {
            "scout_tasks":    tasks,
            "frame":          plan.frame,
            "session_id":     session_id,
            "query":          state.get("query"),
            "model_provider": model_provider,
            "model_name":     model_name,
        }))

    return sends or [Send("dispatch_node", _dispatch_bounce_payload(state))]


# ---------------------------------------------------------------------------
# document_worker_node  (single-shot full-document extraction)
# ---------------------------------------------------------------------------

@traced_node("document_worker")
async def document_worker_node(state: AgentState) -> dict[str, Any]:
    """Extract facts from one packed batch of documents (agents/extractor.py)."""
    if (early := _check_budget(state, "DocumentWorker")):
        return early

    batch = state.get("active_batch")
    if not batch:
        return {"messages": [AIMessage(content="[DocumentWorker] No documents; skipping.")]}

    from research_swarm.agents.extractor import extract_facts
    from research_swarm.runtime.limits import limiter

    sub_questions = state.get("sub_questions_snapshot") or []
    session_id = state.get("session_id", "default")
    llm = _get_tiered_state_llm(state, "standard", agent="document_worker")
    # LangGraph starts every Send-fanned worker at once; cap how many call the LLM together.
    async with limiter("document_worker", session_id, settings.document_worker_concurrency):
        findings = await extract_facts(
            state.get("topic", ""), sub_questions, batch, llm,
            session_id=session_id, agent="document_worker", scope=state.get("scope", ""),
        )
    return {
        "findings": findings,
        "messages": [AIMessage(content=(
            f"[DocumentWorker] {len(batch)} source(s): {len(findings)} finding(s)."
        ))],
    }


# ---------------------------------------------------------------------------
# paper_scout_node / paper_worker_node  (relevance-filtered abstract corpus)
# ---------------------------------------------------------------------------

@traced_node("paper_scout")
async def paper_scout_node(state: AgentState) -> dict[str, Any]:
    """Build the relevance-filtered paper corpus for ALL sub-questions at once.

    Per sub-question: one supervisor-written keyword query per domain-routed
    tool, searched concurrently across every sub-question; each sub-question's
    deduplicated candidate pool is then scored against that sub-question in its
    own light-LLM call (calls run concurrently) and kept where
    >= settings.relevance_threshold. A sub-question that keeps nothing gets one
    retry on the tools its domain routing skipped. No PDF downloads, no
    embeddings; survivors are appended to ``paper_corpus`` for
    paper_worker_node.
    """
    tasks = state.get("scout_tasks") or []
    if not tasks:
        return {"messages": [AIMessage(content="[PaperScout] No sub-questions; skipping.")]}
    if (early := _check_budget(state, "PaperScout")):
        return early

    from research_swarm.agents.papers import (
        choose_papers,
        interleave,
        prefilter_candidates,
        routed_tools_union,
        score_pool,
        search_task,
        tool_registry,
    )

    session_id = state.get("session_id", "default")
    query = state.get("query")
    topic = research_topic(query) or tasks[0]["sub_question"]
    llm = _get_tiered_state_llm(state, "fast", agent="paper_scout")

    available = tool_registry()
    per_tool = settings.fetch_pass_results_per_tool
    depth = getattr(query, "depth", None)
    cap = settings.for_depth("paper_max_candidates", depth)
    pool_size = settings.for_depth("paper_prefilter_pool", depth)
    threshold = settings.relevance_threshold
    limit = settings.for_depth("paper_max_per_sub_question", depth)
    # The LLM's domain label picks the tools first; keywords in the sub-question/query widen the
    # set so a wrong label cannot keep a health question off PubMed.
    routed = [
        routed_tools_union(t["domain"], f"{t['sub_question']} {t['search_query']}", available)
        for t in tasks
    ]

    # The question frame's whole-question queries (which carry its key constraint) are searched
    # once and, with the probe's hits for the literal question, shared by every sub-question's
    # pool as extra round-robin lists -- so constraint-bearing papers get a fair share of the
    # candidate budget even when a sub-question's own query is generic.
    frame = state.get("frame")
    frame_queries = list(frame.search_queries) if frame else []
    frame_tools = routed_tools_union(
        tasks[0]["domain"], f"{topic} {' '.join(frame_queries)}", available,
    )

    with timed(session_id, "paper_scout", "step", name="search", n_tasks=len(tasks)) as info:
        per_task, per_frame_query = await asyncio.gather(
            asyncio.gather(*(
                search_task(t["sub_question"], t["search_query"], routed[j], available,
                            per_tool, session_id)
                for j, t in enumerate(tasks)
            )),
            asyncio.gather(*(
                search_task(topic, q, frame_tools, available, per_tool, session_id)
                for q in frame_queries
            )),
        )
        shared: dict[str, list[dict]] = {}
        for i, ranked in enumerate(per_frame_query):
            for tool, items in ranked.items():
                shared[f"frame{i}:{tool}"] = items
        if frame and frame.probe_hits:
            shared["probe"] = list(frame.probe_hits)
        # A wide net from the (cheap) searches, narrowed in code to the `cap` the LLM scorer
        # reads -- more candidates considered at no extra LLM cost.
        wide = [interleave({**ranked, **shared}, max(cap, pool_size))
                for ranked in per_task]
        pools = [
            prefilter_candidates(w, t["sub_question"], t["search_query"], frame, cap)
            for w, t in zip(wide, tasks)
        ]
        info["n_searched"] = sum(len(w) for w in wide)
        info["n_candidates"] = sum(len(p) for p in pools)
        info["n_frame_queries"] = len(frame_queries)

    async def _score_and_keep(j: int, pool: list[dict]) -> tuple[dict[int, float], list[dict]]:
        scores = await score_pool(topic, tasks[j]["sub_question"], pool, llm, frame=frame)
        return scores, choose_papers(pool, scores, limit)

    results = await asyncio.gather(*(_score_and_keep(j, p) for j, p in enumerate(pools)))
    all_scores = [r[0] for r in results]
    kept_by_task = [r[1] for r in results]

    # Safety net for a misclassified domain: retry once on the skipped tools when NO non-web
    # paper survived (web results alone are not evidence for a scholarly question). Not "fewer
    # than a few": that fires for nearly every industry/policy sub-question and doubles the
    # scout's scoring calls.
    retry = [
        (j, [n for n in available if n not in routed[j]])
        for j, kept in enumerate(kept_by_task)
        if not any(p.get("source_type") != "web" for p in kept)
    ]
    retry = [(j, names) for j, names in retry if names]
    if retry:
        with timed(session_id, "paper_scout", "step", name="retry_skipped_tools",
                   n_tasks=len(retry)):
            retried = await asyncio.gather(*(
                search_task(tasks[j]["sub_question"], tasks[j]["search_query"], names,
                            available, per_tool, session_id)
                for j, names in retry
            ))
            for (j, _), ranked in zip(retry, retried):
                have = {p["url"].strip().lower() for p in pools[j]}
                fresh = [p for p in interleave(ranked, cap) if p["url"].strip().lower() not in have]
                if not fresh:
                    continue
                fresh_scores = await score_pool(topic, tasks[j]["sub_question"], fresh, llm,
                                                frame=frame)
                merged = kept_by_task[j] + choose_papers(fresh, fresh_scores, limit)
                kept_by_task[j] = sorted(merged, key=lambda p: -p["score"])[:limit]
                pools[j] = pools[j] + fresh

    corpus: list[dict] = []
    for j, task in enumerate(tasks):
        by_type: dict[str, int] = {}
        for paper in kept_by_task[j]:
            kind = str(paper.get("source_type"))
            by_type[kind] = by_type.get(kind, 0) + 1
            corpus.append({
                "sub_question": task["sub_question"], "url": paper["url"],
                "title": paper.get("title", ""), "snippet": paper.get("snippet", ""),
                "source_type": paper.get("source_type", "web"),
                "credibility_score": paper.get("credibility_score", 0.6), "score": paper["score"],
            })
        hist: dict[int, int] = {}
        for v in all_scores[j].values():
            hist[int(round(v * 10))] = hist.get(int(round(v * 10)), 0) + 1
        # The final state keeps only the kept papers; the full candidate pool is recorded here so
        # a benchmark can measure candidate-pool recall separately from the relevance filter.
        trace_event(
            session_id, "paper_scout.candidates", "note",
            sub_question=task["sub_question"][:60],
            candidates=[[p["url"], p.get("title", ""), p.get("source_type")] for p in pools[j]],
        )
        trace_event(
            session_id, "paper_scout.kept", "note", sub_question=task["sub_question"][:60],
            domain=task["domain"], query=task["search_query"], tools=routed[j],
            n_candidates=len(pools[j]), n_scored=len(all_scores[j]),
            n_kept=len(kept_by_task[j]), kept_by_source=by_type, score_histogram=hist,
            n_topped_up=sum(1 for p in kept_by_task[j] if p.get("topped_up")),
            threshold=threshold,
        )
    return {
        "paper_corpus": corpus,
        "messages": [AIMessage(content=(
            f"[PaperScout] {len(corpus)} paper(s) kept across {len(tasks)} sub-question(s) "
            f"(threshold {threshold:.2f})."
        ))],
    }


@traced_node("paper_worker")
async def paper_worker_node(state: AgentState) -> dict[str, Any]:
    """Turn the relevance-filtered corpus into findings: 1-2 per paper.

    Runs once after every scout/document/fetch worker has finished (they all
    feed this node, which feeds dispatch_node). One extraction call per
    sub-question, run concurrently. Sub-questions with no paper above the
    threshold produce nothing here and fall through to dispatch's normal
    web-research worker.
    """
    corpus = state.get("paper_corpus") or []
    plan = state.get("plan")
    if not corpus or not plan:
        return {"messages": [AIMessage(content="[PaperWorker] No papers in corpus; skipping.")]}
    if (early := _check_budget(state, "PaperWorker")):
        return early

    session_id = state.get("session_id", "default")
    query = state.get("query")
    topic = research_topic(query)

    # Full text of the best primary papers, cut to the relevant passages (agents/deep_read.py):
    # extraction then quotes the original paper instead of leaving the details to blogs.
    from research_swarm.agents.deep_read import deep_read

    with timed(session_id, "paper_worker", "step", name="deep_read"):
        corpus = await deep_read(
            corpus, topic, getattr(plan, "frame", None), session_id,
            papers=settings.for_depth("deep_read_papers",
                                      getattr(state.get("query"), "depth", None)))

    from research_swarm.agents.papers import extract_findings

    by_sq: dict[str, list[dict]] = {}
    for paper in corpus:
        by_sq.setdefault(paper["sub_question"], []).append(paper)

    sem = asyncio.Semaphore(2)

    async def _one(sq: str, papers: list[dict]) -> list:
        async with sem:
            llm = _get_tiered_state_llm(state, "standard", agent=f"paper_worker[{sq[:32]}]")
            return await extract_findings(topic, sq, papers, llm, session_id=session_id,
                                          scope=_scope(plan))

    results = await asyncio.gather(*(_one(sq, ps) for sq, ps in by_sq.items()))
    findings = [f for r in results for f in r]
    for f in findings:
        trace_event(
            session_id, "paper_worker.finding", "note", sub_question=f.sub_question,
            confidence=f.confidence, text=f.claim, evidence_urls=[e.url for e in f.evidence],
        )
    return {
        "findings": findings,
        "messages": [AIMessage(content=(
            f"[PaperWorker] {len(findings)} finding(s) from {len(corpus)} paper(s) "
            f"across {len(by_sq)} sub-question(s)."
        ))],
    }


# ---------------------------------------------------------------------------
# supervisor_node  (called ONCE — plan creation only)
# ---------------------------------------------------------------------------

@traced_node("supervisor")
async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Create the initial research plan via LLM, then route to dispatch."""
    # Fast-path: if a plan already exists we should never be here again.
    # Return a no-op routing decision so the graph doesn't stall.
    if state.get("plan") is not None:
        return {
            "next_agent": "dispatch",
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": [AIMessage(content="[Supervisor] Plan exists; routing to dispatch.")],
        }

    if (early := _check_budget(state, "Supervisor")):
        return early

    # Supervisor is the orchestrator — called once per session (query expansion + plan), so the
    # larger model's cost doesn't compound the way it would for the per-sub-question worker
    # calls. The large model when "supervisor" is in settings.large_model_stages (the default).
    llm = _get_tiered_state_llm(state, "thorough", agent="supervisor")
    decision = await run_supervisor(state, llm)

    # Enforce dispatch routing regardless of LLM output
    if decision.plan is not None:
        decision = SupervisorDecision(
            reasoning=decision.reasoning,
            next_agent="dispatch",
            plan=decision.plan,
        )

    trace_event(
        state.get("session_id", "default"), "supervisor.plan", "note",
        reasoning=decision.reasoning,
        text=decision.plan.model_dump_json() if decision.plan else "(no plan)",
    )
    logger.info("Supervisor created plan with %d sub-question(s).",
                len(decision.plan.sub_questions) if decision.plan else 0)

    update: dict[str, Any] = {
        "next_agent": "dispatch",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [AIMessage(content=f"[Supervisor] {decision.reasoning}")],
    }
    if decision.plan is not None:
        update["plan"] = decision.plan
    return update


# ---------------------------------------------------------------------------
# dispatch_node  (deterministic fan-out)
# ---------------------------------------------------------------------------

@traced_node("dispatch")
async def dispatch_node(state: AgentState) -> dict[str, Any]:
    """Record pre-round finding IDs and set up the next research round.

    The actual fan-out is handled by the conditional edge ``route_from_dispatch``
    which returns a list of ``Send`` objects — one per target sub-question.
    This node just updates bookkeeping fields.
    """
    findings = state.get("findings") or []
    plan     = state.get("plan")

    if not plan:
        logger.error("dispatch_node called with no plan — skipping.")
        return {
            "next_agent": "writer",
            "messages": [AIMessage(content="[Dispatch] No plan found; forcing writer.")],
        }

    finding_ids = {f.id if hasattr(f, "id") else f.get("id", "") for f in findings}
    research_rounds = state.get("research_rounds", 0)
    targets = _research_targets(state, trace=True)
    if research_rounds > 0 and not targets:
        # Nothing left to re-research — let collect handle the transition
        logger.info("Dispatch: all sub-questions answered; signalling collect.")

    logger.info(
        "Dispatch round %d: %d target(s) from %d sub-question(s).",
        research_rounds, len(targets), len(plan.sub_questions),
    )

    return {
        "pre_dispatch_finding_ids": list(finding_ids),
        "messages": [
            AIMessage(
                content=(
                    f"[Dispatch] Round {research_rounds + 1}: "
                    f"dispatching {len(targets)} worker(s)."
                )
            )
        ],
    }


def _collect_bounce_payload(state: AgentState) -> dict[str, Any]:
    """Build the Send payload for a no-op bounce straight to collect_node.

    Send() gives the receiving node ONLY the payload dict, not the full graph
    state -- collect_node needs research_rounds, pre_dispatch_finding_ids,
    findings and critiques to make its stop decision.
    Omitting any of these makes every field silently reset to its default
    (0 / [] / {}) on that invocation, which defeats should_stop's hard round
    cap (it keeps re-reading research_rounds=0) and produces an infinite
    dispatch<->collect loop until LangGraph's recursion limit kills the run.
    """
    return {
        "active_sub_question": None,
        "session_id": state.get("session_id", "default"),
        "query": state.get("query"),
        "research_rounds": state.get("research_rounds", 0),
        "pre_dispatch_finding_ids": state.get("pre_dispatch_finding_ids") or [],
        "findings": state.get("findings") or [],
        "critiques": state.get("critiques") or [],
        "human_feedback": state.get("human_feedback"),
    }


def route_from_dispatch(state: AgentState):
    """Return a list of Send objects — one worker per target sub-question.

    If there are no targets (all sub-questions answered), send a single
    no-op worker that immediately routes to collect (which will stop the loop).
    """
    plan     = state.get("plan")
    findings = state.get("findings") or []

    if not plan:
        return [Send("collect_node", _collect_bounce_payload(state))]

    targets = _research_targets(state)
    if not targets:
        # Nothing to research — bounce through a no-op worker to collect
        return [Send("collect_node", _collect_bounce_payload(state))]

    # Pass all state fields worker_node needs — Send gives it ONLY the payload dict,
    # not the full graph state, so we must explicitly forward session context.
    session_id     = state.get("session_id", "default")
    query          = state.get("query")
    model_provider = state.get("model_provider")
    model_name     = state.get("model_name")

    from research_swarm.agents.papers import keyword_query

    # A reviewer's re-research request steers the new searches with its own keywords, so the
    # round does not just refetch the pages the first pass already read.
    steer = keyword_query(state.get("rework_instructions") or "", 5) \
        if state.get("rework_instructions") else ""

    sends = []
    for sq in targets:
        assignment = plan.assignment_for(sq)
        planned = assignment.search_query.strip() if assignment else ""
        query_text = planned or keyword_query(sq)
        sends.append(Send("worker_node", {
            "active_sub_question": sq,
            "search_query":        f"{query_text} {steer}".strip(),
            "scope":               _scope(plan),
            "session_id":          session_id,
            "query":               query,
            "model_provider":      model_provider,
            "model_name":          model_name,
            "findings":            findings,
        }))
    return sends


# ---------------------------------------------------------------------------
# worker_node  (gap fill for one under-covered sub-question)
# ---------------------------------------------------------------------------

async def _get_gap_fill_sources(sub_question: str, query: str, session_id: str) -> list[dict]:
    """Where gap fill gets its sources. A module-level hook so a closed-corpus benchmark can
    replace live web search with the task's own documents."""
    from research_swarm.agents.gap_fill import web_sources

    return await web_sources(sub_question, query, session_id)


@traced_node("worker")
async def worker_node(state: AgentState) -> dict[str, Any]:
    """Gap fill for one sub-question: search -> fetch -> one extraction call."""
    if (early := _check_budget(state, "Worker")):
        return early

    sub_question = state.get("active_sub_question")
    if not sub_question:
        # No-op worker (sent when nothing needed researching)
        return {"messages": [AIMessage(content="[Worker] No sub-question assigned; skipping.")]}

    from research_swarm.agents.gap_fill import run_gap_fill
    from research_swarm.agents.papers import keyword_query

    session_id = state.get("session_id", "default")
    llm = _get_tiered_state_llm(state, "standard", agent=f"gap_fill[{sub_question[:32]}]")
    findings = await run_gap_fill(
        research_topic(state.get("query")), sub_question,
        state.get("search_query") or keyword_query(sub_question), llm, session_id,
        source_fn=_get_gap_fill_sources, scope=state.get("scope", ""),
    )
    return {
        "findings": findings,
        "messages": [AIMessage(content=(
            f"[GapFill] {len(findings)} finding(s) for: {sub_question[:60]}"
        ))],
    }


# ---------------------------------------------------------------------------
# collect_node  (stop-signal check + routing)
# ---------------------------------------------------------------------------

@traced_node("collect")
async def collect_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the stop signal after a dispatch round; route to the verifier or re-dispatch."""
    from research_swarm.graph.stop import should_stop

    findings             = state.get("findings") or []
    pre_ids              = state.get("pre_dispatch_finding_ids") or []
    research_rounds      = state.get("research_rounds", 0)
    depth                = _depth_str(state)
    max_rounds           = settings.max_research_rounds(depth)
    human_feedback       = state.get("human_feedback")

    new_rounds = research_rounds + 1

    # Human feedback always overrides stop signal — more research requested.
    if human_feedback:
        logger.info("Collect: human_feedback present — forcing another dispatch round.")
        return {
            "research_rounds": new_rounds,
            "next_agent": "dispatch",
            "human_feedback": None,   # consume so it doesn't re-trigger
            "rework_instructions": None,
            "messages": [AIMessage(
                content=f"[Collect] Round {new_rounds}: re-dispatching (human feedback).",
            )],
        }

    stop, reason = should_stop(
        pre_dispatch_finding_ids=pre_ids,
        all_findings=findings,
        research_rounds=new_rounds,
        max_rounds=max_rounds,
        novelty_threshold=settings.stop_novelty_threshold,
    )

    logger.info("Collect round %d: stop=%s reason=%s", new_rounds, stop, reason)

    next_agent = "verifier" if stop else "dispatch"
    return {
        "research_rounds": new_rounds,
        "next_agent": next_agent,
        # A reviewer-requested round (graph/rework.py) is exactly one round: clear the request.
        "rework_instructions": None,
        "messages": [
            AIMessage(
                content=(
                    f"[Collect] Round {new_rounds}: {'→ verifier' if stop else '→ re-dispatch'}. "
                    f"Reason: {reason}"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# verifier_node  (one pass over every finding)
# ---------------------------------------------------------------------------

@traced_node("verifier")
async def verifier_node(state: AgentState) -> dict[str, Any]:
    """One pass that checks every finding against its evidence window (agents/verifier.py)."""
    if (early := _check_budget(state, "Verifier", pool="review")):
        return early
    from research_swarm.agents.verifier import run_verifier

    llm = _get_tiered_state_llm(state, "fast", pool="review", agent="verifier")
    findings, critiques, conflicts = await run_verifier(state, llm)
    return {
        "findings": findings,
        "critiques": critiques,
        "fact_conflicts": conflicts,
        "messages": [AIMessage(content=(
            f"[Verifier] Checked {len(findings)} finding(s); {len(conflicts)} conflict(s)."
        ))],
    }






# ---------------------------------------------------------------------------
# writer_node
# ---------------------------------------------------------------------------

@traced_node("writer")
async def writer_node(state: AgentState) -> dict[str, Any]:
    """Synthesise the final report.

    Deliberately NOT gated by a budget check, unlike every other node --
    this is the one call that turns whatever findings the research loop
    managed to gather into the user-facing report. Skipping it in favour of
    an empty "budget exceeded" placeholder would throw away real, already-
    paid-for research the moment the (separate, smaller) review pool ran
    dry, which is a worse outcome than just letting this one call through.
    Only the *optional* LLM judge pass below stays budget-gated -- it's
    supplementary, not the report itself.
    """
    # The large model when "writer" is in settings.large_model_stages (the default) -- synthesis
    # quality matters most here
    llm = _get_tiered_state_llm(state, "thorough", pool="review", agent="writer")
    from research_swarm.agents.writer import run_attributed_writer

    report = await run_attributed_writer(state, llm)

    if settings.llm_judge_enabled:
        session_id = state.get("session_id", "default")
        budget = get_budget(session_id, pool="review")
        try:
            budget.check()
        except BudgetExceeded:
            logger.info("Writer: skipping LLM judge — budget exhausted.")
        else:
            judge_llm = _get_tiered_state_llm(
                state, settings.llm_judge_tier, pool="review", agent="judge",
            )
            query = state.get("query")
            plan = state.get("plan")
            judge_result = await judge_report(
                report, plan, judge_llm, topic=query.topic if query else ""
            )
            report = report.model_copy(update={"llm_judge": judge_result})

    trace_event(
        state.get("session_id", "default"), "writer.report", "note",
        title=report.title, n_sections=len(report.sections or []),
        n_references=len(report.references or []),
        faithfulness=getattr(report.quality_score, "faithfulness", None),
        text=report.model_dump_json(exclude={"references"}),
    )
    logger.info("Writer produced report: %r", report.title)
    return {
        "final_report": report,
        "draft_report": report,
        "writer_instructions": None,
        "messages": [AIMessage(content=f"[Writer] Report complete: {report.title}")],
    }




# ---------------------------------------------------------------------------
# Packet path (settings.pipeline_mode == "packet"; CONTEXT.md):
#   supervisor (packet_plan_node) -> packet_node -> writer (synthesis_node)
# Milestone 1 covers supplied sources: no search, and no planning call when the sources are given.
# ---------------------------------------------------------------------------

@traced_node("supervisor")
async def _supplied_sources_plan(state: AgentState) -> dict[str, Any]:
    from research_swarm.schemas.plan import ResearchPlan
    from research_swarm.schemas.worker import SubQuestionAssignment

    question = research_topic(state.get("query")) or "the question"
    plan = ResearchPlan(
        sub_questions=[question],
        strategy="Supplied sources: one evidence packet, one synthesis call.",
        assignments=[SubQuestionAssignment(sub_question=question)],
    )
    trace_event(state.get("session_id", "default"), "supervisor.plan", "note",
                source="supplied", text=plan.model_dump_json())
    return {
        "plan": plan,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [AIMessage(content="[Supervisor] Sources supplied: no planning call.")],
    }


async def packet_plan_node(state: AgentState) -> dict[str, Any]:
    """Planning for the packet path: only when there is something to search. With supplied
    sources the whole question is the single sub-question and no LLM is called; otherwise the
    usual planner runs (searching for the packet path is milestone 2)."""
    if state.get("plan") is None and state.get("ingested_documents"):
        return await _supplied_sources_plan(state)
    return await supervisor_node(state)


@traced_node("packet")
async def packet_node(state: AgentState) -> dict[str, Any]:
    """Build the evidence packet from the run's sources (agents/packet.py)."""
    from research_swarm.agents.packet import build_packet, llm_screener

    query = state.get("query")
    plan = state.get("plan")
    question = research_topic(query) or ""
    frame = getattr(plan, "frame", None) if plan else None
    sources = state.get("ingested_documents") or []
    budget = settings.for_depth("packet_budget", getattr(query, "depth", None))
    # The local small model screens passages only when the sources overflow the budget.
    screener = llm_screener(_get_tiered_state_llm(state, "fast", agent="packet_screen"),
                            state.get("session_id")) if sources else None
    packet = await build_packet(
        question, list(plan.sub_questions) if plan else [question], sources, budget,
        screener=screener,
        scope_phrases=frame.scope_phrases() if frame is not None and frame.has_constraint
        else (),
    )
    trace_event(state.get("session_id", "default"), "packet.built", "note",
                budget=budget, **packet.stats)
    return {
        "evidence_packet": packet.to_dict(),
        "messages": [AIMessage(content=(
            f"[Packet] {packet.stats['kept_sentences']} sentence(s) from "
            f"{packet.stats['sources']} source(s), ~{packet.stats['kept_tokens']} tokens "
            f"(budget {budget}, fit: {packet.stats['fit']})."))],
    }


@traced_node("writer")
async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """One large-model call over the evidence packet, audited by the code render
    (agents/synthesis.py). Takes the writer's place in the graph, so the review pause and the
    UI's diagram treat it as the writer."""
    from research_swarm.agents.synthesis import run_synthesis

    llm = _get_tiered_state_llm(state, "thorough", pool="review", agent="synthesis")
    report, cited = await run_synthesis(state, llm)
    trace_event(
        state.get("session_id", "default"), "writer.report", "note",
        title=report.title, n_sections=len(report.sections or []),
        n_references=len(report.references or []), n_cited_sentences=len(cited),
        text=report.model_dump_json(exclude={"references"}),
    )
    return {
        "final_report": report,
        "draft_report": report,
        "findings": cited,
        "writer_instructions": None,
        "messages": [AIMessage(content=f"[Synthesis] Report complete: {report.title}")],
    }
