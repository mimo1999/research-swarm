"""LangGraph node functions — one async function per agent.

Phase-4 topology
================
START → supervisor_node  (LLM: plan creation only)
          ↓ next_agent = "dispatch"
        document_pass_node  (deterministic: one-time fan-out over ingested_documents)
          ↓ Send × N (one per document, or per size-bounded slice of an oversized one)
          │    — bounces straight to dispatch_node when there are no documents
        document_worker_node  (single-shot full-document claim extraction, no tool loop)
          ↓ findings merged by _merge_findings reducer
        dispatch_node  (deterministic: record pre-round IDs, fan out via Send)
          ↓ Send × N (one per target sub-question — skips ones the document
          │    pass already answered, via _research_targets' round-0 check)
        worker_node  (role-aware researcher for a single sub-question)
          ↓ findings merged by _merge_findings reducer
        collect_node  (deterministic: stop-signal check + rework bookkeeping)
          ├─ stop  → critic_node
          └─ loop  → dispatch_node  (novelty/similarity signal still open)
        critic_node  (LLM: verdicts, then decides whether to loop back)
          ├─ weak/refuted findings under the rework cap → dispatch_node
          │    (re-research just those; see settings.max_rework_attempts)
          └─ else → fact_checker_node  → writer_node  → END
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Send

from research_swarm.agents.base import get_tiered_llm
from research_swarm.agents.critic import run_critic
from research_swarm.agents.fact_checker import run_fact_checker
from research_swarm.agents.researcher import run_researcher  # kept for legacy node + test patching
from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
from research_swarm.agents.workers import run_worker
from research_swarm.agents.writer import run_writer
from research_swarm.config import settings
from research_swarm.eval.llm_judge import judge_report
from research_swarm.runtime.budget import BudgetExceeded, get_budget
from research_swarm.schemas.state import AgentState
from research_swarm.schemas.worker import WorkerRole
from research_swarm.tools import (
    arxiv_search,
    europe_pmc_search,
    fetch_url,
    pubmed_search,
    web_search,
)
from research_swarm.tools.retriever_tool import build_retriever_tool
from research_swarm.tools.web_search import is_configured as _tavily_configured

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_researcher_tools(max_sources: int | None = None, session_id: str | None = None):
    tools = [arxiv_search, pubmed_search, europe_pmc_search, fetch_url]
    if _tavily_configured():
        tools.insert(0, web_search)
    try:
        tools.append(build_retriever_tool(max_sources=max_sources, session_id=session_id))
    except Exception as exc:
        logger.warning("RAG retriever unavailable: %s", exc)
    return tools


def _finding_evidence_dicts(finding: Any) -> list[dict]:
    """Return a Finding's evidence as plain dicts, regardless of Source vs dict shape."""
    evidence = finding.evidence if hasattr(finding, "evidence") else finding.get("evidence", [])
    out: list[dict] = []
    for s in evidence:
        if hasattr(s, "model_dump"):
            out.append(s.model_dump(mode="json"))
        elif isinstance(s, dict):
            out.append(s)
    return out


async def _ingest_round_evidence(state: AgentState, findings: list) -> None:
    """Persist this round's evidence into the session's RAG index.

    Runs once per round from collect_node -- after all of this round's
    Send-dispatched worker_node tasks have already merged -- so writes to the
    session's Chroma collection are always sequential, never concurrent
    across parallel workers. IngestionPipeline.ingest_new_source_dicts skips
    URLs already present in the collection, so calling this on every round
    (not just once) is a cheap no-op for evidence ingested in an earlier
    round -- it only does real work for genuinely new sources.

    This lets a rework pass (or a later dispatch round) check the
    accumulated corpus via retrieve_from_rag before re-hitting the web for
    the same ground a sibling worker already covered in an earlier round.

    Best-effort: failure here (embedding model unavailable, disk issue) must
    never block the graph's stop/rework routing decision.
    """
    all_evidence = [s for f in findings for s in _finding_evidence_dicts(f)]
    if not all_evidence:
        return
    try:
        from research_swarm.rag.indexes import get_embed_model
        from research_swarm.rag.ingestion import IngestionPipeline

        session_id = state.get("session_id", "default")
        pipeline = IngestionPipeline(session_id)
        embed_model = get_embed_model()
        added = await asyncio.to_thread(
            pipeline.ingest_new_source_dicts, all_evidence, embed_model
        )
        if added:
            logger.info(
                "Collect: persisted %d new evidence chunk(s) into session RAG index.",
                added,
            )
    except Exception as exc:
        logger.warning("Collect: evidence ingestion failed (%s) -- continuing.", exc)


def _get_tiered_state_llm(state: AgentState, tier: str, pool: str = "research"):
    """Create a tiered LLM with budget callback attached.

    ``pool`` selects which of the session's two independent budget counters
    this call draws from -- "research" (supervisor, document workers,
    dispatch/worker loop -- the part that can genuinely iterate) or "review"
    (critic/fact-checker/writer/judge -- a few batched calls). Kept separate
    so a research-loop overrun can't exhaust the budget critic/fact-checker/
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
    """
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id, pool=pool)
    provider_override = state.get("model_provider") if tier == "standard" else None
    llm = get_tiered_llm(tier=tier, provider_override=provider_override)
    return llm.model_copy(update={"callbacks": [budget.callback]})


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


def _research_targets(state: AgentState) -> list[str]:
    """Return the sub-questions needing (re-)research this round.

    Round 0: every sub-question in the plan.  Later rounds: only sub-questions
    whose latest finding is weak or refuted, AND haven't already hit the
    per-finding rework cap (``settings.max_rework_attempts`` — see
    ``rework_counts`` in AgentState). Shared by dispatch_node, route_from_dispatch,
    critic_node, and collect_node so all four views of a round agree exactly.
    """
    plan = state.get("plan")
    if not plan:
        return []

    findings = state.get("findings") or []

    if state.get("research_rounds", 0) == 0:
        # Round 0 used to unconditionally return every sub-question, which
        # was safe only because nothing ran before dispatch. document_pass_node
        # can now produce findings before round 0 (one-time full-document
        # extraction from ingested documents) -- skip any sub-question that
        # already has one of those, so round-0 web dispatch doesn't duplicate
        # work a document already did. No weak/refuted distinction here:
        # critiques is empty at genuine round 0, so any existing finding can
        # only have come from the document pass, not a rejected re-research.
        already_has_finding = {
            (f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", ""))
            .strip().lower()
            for f in findings
        }
        return [sq for sq in plan.sub_questions if sq.strip().lower() not in already_has_finding]

    weak_or_refuted_sqs = _weak_or_refuted_sub_questions(state)
    rework_counts = state.get("rework_counts") or {}
    max_rework = settings.max_rework_attempts

    answered_sqs: set[str] = set()
    capped_sqs: set[str] = set()
    for f in findings:
        sq = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        norm_sq = sq.strip().lower()
        if norm_sq not in weak_or_refuted_sqs:
            answered_sqs.add(norm_sq)
        elif rework_counts.get(norm_sq, 0) >= max_rework:
            capped_sqs.add(norm_sq)

    return [
        sq for sq in plan.sub_questions
        if sq.strip().lower() not in answered_sqs
        and sq.strip().lower() not in capped_sqs
    ]


def _weak_or_refuted_sub_questions(state: AgentState) -> set[str]:
    """Return normalized sub-questions whose latest finding was critiqued as weak or refuted.

    Distinct from "has no finding at all": a sub-question that never got a
    finding (a worker failure, or the mandatory round-0->round-1 loop that
    always fires before critic ever runs -- see should_stop's "first round"
    fallback) hasn't been rejected by anything, it just hasn't succeeded yet.
    collect_node uses this set (not _research_targets' full return value,
    which also includes findingless sub-questions) to increment
    rework_counts, so a sub-question's max_rework_attempts budget isn't
    partially spent by rounds that happened before any critique existed.
    """
    from research_swarm.agents._utils import _latest_verdicts

    findings = state.get("findings") or []
    critiques = state.get("critiques") or []
    latest_verdicts = _latest_verdicts(critiques)
    weak_or_refuted_ids = {fid for fid, v in latest_verdicts.items() if v in {"weak", "refuted"}}

    result: set[str] = set()
    for f in findings:
        fid = f.id if hasattr(f, "id") else f.get("id", "")
        if fid in weak_or_refuted_ids:
            sq = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
            result.add(sq.strip().lower())
    return result


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
    plan, findings, critiques, research_rounds, and rework_counts to make
    its routing decision. Mirrors _collect_bounce_payload, which exists for
    the identical reason on the dispatch->collect side.
    """
    return {
        "session_id":      state.get("session_id", "default"),
        "query":           state.get("query"),
        "plan":            state.get("plan"),
        "findings":        state.get("findings") or [],
        "critiques":       state.get("critiques") or [],
        "research_rounds": state.get("research_rounds", 0),
        "rework_counts":   state.get("rework_counts") or {},
        "human_feedback":  state.get("human_feedback"),
        "model_provider":  state.get("model_provider"),
        "model_name":      state.get("model_name"),
    }


def route_from_document_pass(state: AgentState):
    """Return a list of Send objects for the one-time pre-dispatch fan-out.

    Two independent kinds of Sends, mixed in one list (LangGraph supports a
    single conditional edge fanning out to different target nodes):
      - document_worker_node — one per (document, part), same as before.
      - fetch_worker_node — one per plan sub-question, deep-fetching and
        embedding search results into the session's RAG index BEFORE round-0
        dispatch, so retrieve_from_rag has real substance from round 1
        instead of only what workers' own live searches turn up mid-round.

    "Has docs" and "has plan" are independent gates: a plan with no uploaded
    documents still gets the fetch pass (nothing to extract, but still
    something to search-and-embed). Only a missing plan bounces straight to
    dispatch_node, same as always.
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

    if docs:
        from research_swarm.agents.document_worker import _split_into_parts

        for doc in docs:
            parts = _split_into_parts(doc.get("text", ""))
            for i, part_text in enumerate(parts):
                sends.append(Send("document_worker_node", {
                    "active_document":        doc,
                    "active_doc_part_text":   part_text,
                    "active_doc_part_index":  i,
                    "active_doc_part_total":  len(parts),
                    "sub_questions_snapshot": sub_questions,
                    "session_id":             session_id,
                    "model_provider":         model_provider,
                    "model_name":             model_name,
                }))

    if settings.enable_fetch_pass:
        for sq in sub_questions:
            sends.append(Send("fetch_worker_node", {
                "active_fetch_query": sq,
                "session_id":         session_id,
            }))

    return sends or [Send("dispatch_node", _dispatch_bounce_payload(state))]


# ---------------------------------------------------------------------------
# document_worker_node  (single-shot full-document extraction)
# ---------------------------------------------------------------------------

async def document_worker_node(state: AgentState) -> dict[str, Any]:
    """Extract claims from a single document (or one slice of an oversized one)."""
    if (early := _check_budget(state, "DocumentWorker")):
        return early

    document  = state.get("active_document")
    part_text = state.get("active_doc_part_text")
    if not document or not part_text:
        return {"messages": [AIMessage(content="[DocumentWorker] No document assigned; skipping.")]}

    sub_questions = state.get("sub_questions_snapshot") or []
    part_index    = state.get("active_doc_part_index", 0)
    part_total    = state.get("active_doc_part_total", 1)

    llm = _get_tiered_state_llm(state, "standard")

    from research_swarm.agents.document_worker import run_document_worker
    findings = await run_document_worker(
        document, part_index, part_total, part_text, sub_questions, llm,
    )

    label = document.get("title") or document.get("url", "doc")
    part_note = f" (part {part_index + 1}/{part_total})" if part_total > 1 else ""
    return {
        "findings": findings,
        "messages": [
            AIMessage(content=f"[DocumentWorker] {label}{part_note}: {len(findings)} finding(s).")
        ],
    }


# ---------------------------------------------------------------------------
# fetch_worker_node  (one-time deep-fetch-and-embed pass, no LLM call)
# ---------------------------------------------------------------------------

async def fetch_worker_node(state: AgentState) -> dict[str, Any]:
    """Search once per sub-question and deep-embed results into the session's
    RAG index, before any research round runs.

    Deliberately makes NO LLM call -- pure deterministic tool calls plus
    embedding work, so it never touches the LLM call/token budget. Reuses
    the exact search tools (pubmed_search/arxiv_search/europe_pmc_search/
    web_search) and the exact multi-chunk embedding path
    (IngestionPipeline.ingest_pdf/ingest_url/ingest_text) already proven for
    the uploaded-document path -- see rag/ingestion.py. Each source's
    fetch+embed is independently try/excepted so one bad download (a dead
    link, a malformed PDF) can't drop the others fanned out from the same
    Send.
    """
    query = state.get("active_fetch_query")
    if not query:
        return {"messages": [AIMessage(content="[FetchPass] No query assigned; skipping.")]}

    session_id = state.get("session_id", "default")
    max_results = settings.fetch_pass_results_per_tool

    import os
    import tempfile
    import xml.etree.ElementTree as ET

    import httpx

    from research_swarm.rag.indexes import get_embed_model
    from research_swarm.rag.ingestion import IngestionPipeline

    pipeline = IngestionPipeline(session_id)
    embed_model = get_embed_model()
    embedded = 0

    # PubMed: ingest the abstract as-is -- already close to that source
    # type's practical ceiling. Real PubMed full text needs a separate
    # PMID -> PMCID -> PMC-efetch chain this pass doesn't build.
    try:
        pubmed_results = await asyncio.to_thread(
            pubmed_search.invoke, {"query": query, "max_results": max_results}
        )
        for source in pubmed_results:
            try:
                embedded += await asyncio.to_thread(
                    pipeline.ingest_source_dict, source, embed_model
                )
            except Exception as exc:
                logger.warning("FetchPass: PubMed source ingest failed (%s) -- skipping.", exc)
    except Exception as exc:
        logger.warning("FetchPass: pubmed_search failed for %r (%s) -- skipping.", query[:60], exc)

    # Europe PMC: overlaps with PubMed (both draw on MEDLINE) but additionally
    # exposes open-access full text -- europe_pmc_tool.py encodes that as a
    # "/article/PMC/{pmcid}" URL. Same two-tier pattern as arXiv below: try
    # the deep fetch, fall back to the abstract (already carried in the
    # Source dict) if it's not open access or the fetch fails.
    try:
        epmc_results = await asyncio.to_thread(
            europe_pmc_search.invoke, {"query": query, "max_results": max_results}
        )
        for source in epmc_results:
            url = source.get("url", "")
            pmcid = url.rsplit("/", 1)[-1] if "/article/PMC/" in url else None
            if not pmcid:
                # Not open access (or no PMCID) -- abstract is the ceiling.
                try:
                    embedded += await asyncio.to_thread(
                        pipeline.ingest_source_dict, source, embed_model
                    )
                except Exception as exc:
                    logger.warning(
                        "FetchPass: Europe PMC source ingest failed (%s) -- skipping.", exc
                    )
                continue
            try:
                resp = await asyncio.to_thread(
                    httpx.get,
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                    timeout=20,
                )
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                body = root.find(".//body")
                if body is None:
                    raise ValueError("no <body> in full-text XML")
                text = " ".join(
                    "".join(p.itertext()).strip() for p in body.iter("p")
                ).strip()
                if not text:
                    raise ValueError("full-text XML body had no paragraph text")
                metadata = {
                    "url": url,
                    "title": source.get("title", ""),
                    "source_type": "europe_pmc",
                    "credibility_score": source.get("credibility_score", 0.9),
                }
                embedded += await asyncio.to_thread(
                    pipeline.ingest_text, text, metadata, embed_model
                )
            except Exception as exc:
                # Full-text fetch/parse failed -- fall back to the abstract
                # rather than losing this source entirely, same as arXiv.
                logger.warning(
                    "FetchPass: Europe PMC full text failed for %r (%s) -- "
                    "falling back to abstract.", url, exc
                )
                try:
                    embedded += await asyncio.to_thread(
                        pipeline.ingest_source_dict, source, embed_model
                    )
                except Exception as fallback_exc:
                    logger.warning(
                        "FetchPass: Europe PMC abstract fallback also failed for %r (%s) "
                        "-- skipping.", url, fallback_exc,
                    )
    except Exception as exc:
        logger.warning(
            "FetchPass: europe_pmc_search failed for %r (%s) -- skipping.", query[:60], exc
        )

    # arXiv: download the actual PDF and embed every page via
    # IngestionPipeline.ingest_pdf's existing multi-page chunker -- the same
    # one already proven for uploaded PDFs -- instead of just the abstract.
    # Falls back to the abstract if the PDF isn't fetchable (404, withdrawn,
    # network error) so a bad download never means zero content for that paper.
    try:
        arxiv_results = await asyncio.to_thread(
            arxiv_search.invoke, {"query": query, "max_results": max_results}
        )
        for source in arxiv_results:
            url = source.get("url", "")
            if not url.startswith("http"):
                continue  # error sentinel, e.g. "arxiv://search/..."
            pdf_url = url.replace("/abs/", "/pdf/")
            tmp_path = None
            try:
                resp = await asyncio.to_thread(
                    httpx.get, pdf_url, timeout=20, follow_redirects=True
                )
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                embedded += await asyncio.to_thread(pipeline.ingest_pdf, tmp_path, embed_model)
            except Exception as exc:
                # PDF unavailable (404, withdrawn, network error, etc.) -- fall
                # back to the abstract rather than losing this source entirely.
                # Same graceful-degradation level PubMed already gets.
                logger.warning(
                    "FetchPass: arXiv PDF ingest failed for %r (%s) -- "
                    "falling back to abstract.", url, exc
                )
                try:
                    embedded += await asyncio.to_thread(
                        pipeline.ingest_source_dict, source, embed_model
                    )
                except Exception as fallback_exc:
                    logger.warning(
                        "FetchPass: arXiv abstract fallback also failed for %r (%s) -- skipping.",
                        url, fallback_exc,
                    )
            finally:
                if tmp_path:
                    os.unlink(tmp_path)
    except Exception as exc:
        logger.warning("FetchPass: arxiv_search failed for %r (%s) -- skipping.", query[:60], exc)

    # Web: full page text (up to fetch_url's own ceiling), not just Tavily's
    # search-engine extract. Skipped entirely when no Tavily key is
    # configured -- calling it anyway would just embed a useless
    # "[Search error: ...]" placeholder for every sub-question.
    if not _tavily_configured():
        logger.info("FetchPass: Tavily not configured -- skipping web search.")
    else:
        try:
            web_results = await asyncio.to_thread(
                web_search.invoke, {"query": query, "max_results": max_results}
            )
            for source in web_results:
                url = source.get("url", "")
                if not url:
                    continue
                try:
                    embedded += await asyncio.to_thread(
                        pipeline.ingest_url, url, embed_model, 20000
                    )
                except Exception as exc:
                    logger.warning(
                        "FetchPass: web page ingest failed for %r (%s) -- skipping.", url, exc
                    )
        except Exception as exc:
            logger.warning("FetchPass: web_search failed for %r (%s) -- skipping.", query[:60], exc)

    return {
        "messages": [
            AIMessage(content=f"[FetchPass] {embedded} chunk(s) embedded for: {query[:60]}")
        ],
    }


# ---------------------------------------------------------------------------
# supervisor_node  (called ONCE — plan creation only)
# ---------------------------------------------------------------------------

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

    # Supervisor is the orchestrator — called once per session, so the larger
    # model's cost doesn't compound the way it would for the per-sub-question
    # worker calls. Use the thorough tier for plan-quality reasoning.
    llm = _get_tiered_state_llm(state, "thorough")
    decision = await run_supervisor(state, llm)

    # Enforce dispatch routing regardless of LLM output
    if decision.plan is not None:
        decision = SupervisorDecision(
            reasoning=decision.reasoning,
            next_agent="dispatch",
            plan=decision.plan,
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
    targets = _research_targets(state)
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
    findings, critiques, and rework_counts to make its stop/rework decision.
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
        "rework_counts": state.get("rework_counts") or {},
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

    sends = []
    for sq in targets:
        role = plan.role_for(sq)
        sends.append(Send("worker_node", {
            "active_sub_question": sq,
            "active_worker_role":  role.value,
            "session_id":          session_id,
            "query":               query,
            "model_provider":      model_provider,
            "model_name":          model_name,
            "findings":            findings,
        }))
    return sends


# ---------------------------------------------------------------------------
# worker_node  (role-aware researcher for one sub-question)
# ---------------------------------------------------------------------------

async def worker_node(state: AgentState) -> dict[str, Any]:
    """Research a single sub-question using the assigned worker role."""
    if (early := _check_budget(state, "Worker")):
        return early

    sub_question = state.get("active_sub_question")
    if not sub_question:
        # No-op worker (sent when nothing needed re-researching)
        return {"messages": [AIMessage(content="[Worker] No sub-question assigned; skipping.")]}

    role_str = state.get("active_worker_role") or WorkerRole.general.value
    try:
        role = WorkerRole(role_str)
    except ValueError:
        role = WorkerRole.general

    query = state.get("query")
    max_sources = query.max_sources if query else None

    # Workers use the standard tier — the smallest model that can still do
    # reliable tool-calling + synthesis, since this is called once per
    # sub-question per tool turn (the highest call-volume node in the graph).
    llm   = _get_tiered_state_llm(state, "standard")
    # Snippet condensation uses the fast tier — cheaper than the truncation
    # it replaces would cost across the loop's repeated re-sends.
    summarizer_llm = _get_tiered_state_llm(state, "fast")
    session_id = state.get("session_id", "default")
    tools = _get_researcher_tools(max_sources=max_sources, session_id=session_id)

    finding = await run_worker(sub_question, role, state, llm, tools, summarizer_llm=summarizer_llm)

    if finding is None:
        return {
            "messages": [
                AIMessage(content=f"[Worker/{role.value}] No finding for: {sub_question[:60]}")
            ]
        }

    return {
        "findings": [finding],
        "messages": [
            AIMessage(
                content=(
                    f"[Worker/{role.value}] Finding (conf={finding.confidence:.2f}): "
                    f"{finding.claim[:80]}"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# collect_node  (stop-signal check + routing)
# ---------------------------------------------------------------------------

async def collect_node(state: AgentState) -> dict[str, Any]:
    """Evaluate stop signal after a dispatch round; route to critic or re-dispatch.

    Also records rework attempts: for any round beyond the first, every
    sub-question the critic actually flagged weak/refuted (see
    ``_weak_or_refuted_sub_questions``) gets its rework_counts bumped by one
    -- not every ``_research_targets(state)`` entry, which also includes
    sub-questions with no finding at all (worker failure, or the mandatory
    round-0->round-1 loop that always fires before critic ever runs). This
    runs before dispatch_node/route_from_dispatch see the new counts for the
    *next* round, and after they saw the (unchanged) counts for *this*
    round — so nothing drifts mid-round.
    """
    from research_swarm.graph.stop import should_stop

    findings             = state.get("findings") or []
    pre_ids              = state.get("pre_dispatch_finding_ids") or []
    research_rounds      = state.get("research_rounds", 0)
    depth                = _depth_str(state)
    max_rounds           = settings.max_research_rounds(depth)
    human_feedback       = state.get("human_feedback")

    await _ingest_round_evidence(state, findings)

    new_rounds = research_rounds + 1

    # Only sub-questions the critic actually flagged weak/refuted count
    # against the rework budget -- not every _research_targets() entry, which
    # also includes sub-questions with no finding at all (a worker failure,
    # or the mandatory round-0->round-1 loop that fires before critic ever
    # runs). See _weak_or_refuted_sub_questions.
    rework_counts = dict(state.get("rework_counts") or {})
    if research_rounds > 0:
        for key in _weak_or_refuted_sub_questions(state):
            rework_counts[key] = rework_counts.get(key, 0) + 1

    # Human feedback always overrides stop signal — more research requested.
    if human_feedback:
        logger.info("Collect: human_feedback present — forcing another dispatch round.")
        return {
            "research_rounds": new_rounds,
            "next_agent": "dispatch",
            "human_feedback": None,   # consume so it doesn't re-trigger
            "rework_counts": rework_counts,
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
        similarity_threshold=settings.stop_similarity_threshold,
    )

    logger.info("Collect round %d: stop=%s reason=%s", new_rounds, stop, reason)

    next_agent = "critic" if stop else "dispatch"
    return {
        "research_rounds": new_rounds,
        "next_agent": next_agent,
        "rework_counts": rework_counts,
        "messages": [
            AIMessage(
                content=(
                    f"[Collect] Round {new_rounds}: {'→ critic' if stop else '→ re-dispatch'}. "
                    f"Reason: {reason}"
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# critic_node
# ---------------------------------------------------------------------------

async def critic_node(state: AgentState) -> dict[str, Any]:
    """Review findings, then decide whether to loop back for rework.

    Weak/refuted findings that haven't hit the per-finding rework cap
    (settings.max_rework_attempts) get routed back through dispatch_node for
    another attempt; everything else proceeds to fact_checker. This is the
    only place next_agent="dispatch" gets set after critic runs, so
    route_from_critic just reads it.
    """
    if (early := _check_budget(state, "Critic", pool="review")):
        return early
    # Critic uses fast tier — structured extraction, not synthesis
    llm = _get_tiered_state_llm(state, "fast", pool="review")
    new_critiques = await run_critic(state, llm)
    logger.info("Critic produced %d critique(s).", len(new_critiques))

    # _research_targets needs this round's critiques to compute weak/refuted
    # targets, but they aren't merged into state until this node returns --
    # build a temp view so the same shared function sees them now.
    temp_state: AgentState = {  # type: ignore[typeddict-item]
        **state,
        "critiques": (state.get("critiques") or []) + new_critiques,
    }
    rework_targets = _research_targets(temp_state)

    depth = _depth_str(state)
    max_rounds = settings.max_research_rounds(depth)
    research_rounds = state.get("research_rounds", 0)

    if rework_targets and research_rounds < max_rounds:
        next_agent = "dispatch"
        logger.info(
            "Critic: %d finding(s) weak/refuted and under the rework cap (%d) — re-dispatching.",
            len(rework_targets), settings.max_rework_attempts,
        )
    else:
        next_agent = "fact_checker"
        if rework_targets:
            logger.info(
                "Critic: %d finding(s) remain weak/refuted but hit the round cap "
                "(%d/%d) — proceeding to fact-checker.",
                len(rework_targets), research_rounds, max_rounds,
            )

    return {
        "critiques": new_critiques,
        "next_agent": next_agent,
        "messages": [
            AIMessage(content=f"[Critic] Reviewed {len(new_critiques)} finding(s).")
        ],
    }


# ---------------------------------------------------------------------------
# fact_checker_node
# ---------------------------------------------------------------------------

async def fact_checker_node(state: AgentState) -> dict[str, Any]:
    if (early := _check_budget(state, "FactChecker", pool="review")):
        return early
    llm = _get_tiered_state_llm(state, "fast", pool="review")
    updated_findings = await run_fact_checker(state, llm)
    logger.info("FactChecker updated %d finding(s).", len(updated_findings))
    return {
        "findings": updated_findings,
        "messages": [
            AIMessage(
                content=f"[FactChecker] Updated confidence on {len(updated_findings)} finding(s)."
            )
        ],
    }


# ---------------------------------------------------------------------------
# writer_node
# ---------------------------------------------------------------------------

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
    # Writer uses the thorough tier — synthesis quality matters most here
    llm = _get_tiered_state_llm(state, "thorough", pool="review")
    report = await run_writer(state, llm)

    if settings.llm_judge_enabled:
        session_id = state.get("session_id", "default")
        budget = get_budget(session_id, pool="review")
        try:
            budget.check()
        except BudgetExceeded:
            logger.info("Writer: skipping LLM judge — budget exhausted.")
        else:
            judge_llm = _get_tiered_state_llm(state, settings.llm_judge_tier, pool="review")
            query = state.get("query")
            plan = state.get("plan")
            judge_result = await judge_report(
                report, plan, judge_llm, topic=query.topic if query else ""
            )
            report = report.model_copy(update={"llm_judge": judge_result})

    logger.info("Writer produced report: %r", report.title)
    return {
        "final_report": report,
        "draft_report": report,
        "writer_instructions": None,
        "messages": [AIMessage(content=f"[Writer] Report complete: {report.title}")],
    }


# ---------------------------------------------------------------------------
# researcher_node  (legacy — kept for backward-compat with old tests/checkpoints)
# ---------------------------------------------------------------------------

async def researcher_node(state: AgentState) -> dict[str, Any]:
    """Legacy researcher node — routes through dispatch in Phase 4.

    Retained so existing tests and old checkpoints that reference 'researcher'
    as a next_agent value continue to work.  New sessions use dispatch_node.
    """
    if (early := _check_budget(state, "Researcher")):
        return early

    query = state.get("query")
    if query is None:
        logger.error("researcher_node called with no query — skipping.")
        return {"messages": [AIMessage(content="[Researcher] No query; skipping.")]}

    llm   = _get_tiered_state_llm(state, "standard")
    tools = _get_researcher_tools(max_sources=query.max_sources if query else None)
    new_findings = await run_researcher(state, llm, tools)
    logger.info("Researcher (legacy) produced %d finding(s).", len(new_findings))
    return {
        "findings": new_findings,
        "human_feedback": None,
        "messages": [AIMessage(content=f"[Researcher] Produced {len(new_findings)} finding(s).")],
    }
