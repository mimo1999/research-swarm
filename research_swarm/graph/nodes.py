"""LangGraph node functions — one async function per agent.

Phase-4 topology
================
START → supervisor_node  (LLM: plan creation only)
          ↓ next_agent = "dispatch"
        dispatch_node  (deterministic: record pre-round IDs, fan out via Send)
          ↓ Send × N (one per target sub-question)
        worker_node  (role-aware researcher for a single sub-question)
          ↓ findings merged by _merge_findings reducer
        collect_node  (deterministic: stop-signal check)
          ├─ stop  → critic_node
          └─ loop  → dispatch_node  (re-research weak sub-questions)
        critic_node  → fact_checker_node  → writer_node  → END
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Send

from research_swarm.agents.base import get_agent_llm, get_tiered_llm
from research_swarm.agents.critic import run_critic
from research_swarm.agents.fact_checker import run_fact_checker
from research_swarm.agents.researcher import run_researcher  # kept for legacy node + test patching
from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
from research_swarm.agents.workers import run_worker
from research_swarm.agents.writer import run_writer
from research_swarm.runtime.budget import BudgetExceeded, get_budget
from research_swarm.schemas.state import AgentState
from research_swarm.schemas.worker import WorkerRole
from research_swarm.tools import arxiv_search, fetch_url, web_search
from research_swarm.tools.retriever_tool import build_retriever_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_researcher_tools(max_sources: int | None = None, session_id: str | None = None):
    tools = [web_search, arxiv_search, fetch_url]
    try:
        tools.append(build_retriever_tool(max_sources=max_sources, session_id=session_id))
    except Exception as exc:
        logger.warning("RAG retriever unavailable: %s", exc)
    return tools


def _get_state_llm(state: AgentState):
    """Create the chat model requested by this run's state, with budget callback."""
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id)
    return get_agent_llm(
        provider=state.get("model_provider"),
        model=state.get("model_name"),
    ).with_config({"callbacks": [budget.callback]})


def _get_tiered_state_llm(state: AgentState, tier: str):
    """Create a tiered LLM with budget callback attached."""
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id)
    return get_tiered_llm(tier=tier).with_config({"callbacks": [budget.callback]})


def _check_budget(state: AgentState, node_name: str) -> dict[str, Any] | None:
    session_id = state.get("session_id", "default")
    budget = get_budget(session_id)
    try:
        budget.check()
        return None
    except BudgetExceeded as exc:
        logger.warning("%s: %s — forcing writer.", node_name, exc)
        return {
            "next_agent": "writer",
            "messages": [
                AIMessage(
                    content=f"[{node_name}] Budget exceeded ({exc.used}/{exc.limit} calls); "
                            "forcing report."
                )
            ],
        }


def _research_targets(state: AgentState) -> list[str]:
    """Return the sub-questions needing (re-)research this round.

    Round 0: every sub-question in the plan.  Later rounds: only sub-questions
    whose latest finding is missing, weak, or refuted.  Shared by dispatch_node
    and route_from_dispatch so their views of the round can never drift apart.
    """
    from research_swarm.agents._utils import _latest_verdicts

    plan = state.get("plan")
    if not plan:
        return []

    findings  = state.get("findings") or []
    critiques = state.get("critiques") or []

    if state.get("research_rounds", 0) == 0:
        return list(plan.sub_questions)

    latest_verdicts = _latest_verdicts(critiques)
    weak_or_refuted = {
        fid for fid, v in latest_verdicts.items() if v in {"weak", "refuted"}
    }
    answered_sqs: set[str] = set()
    for f in findings:
        sq  = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        fid = f.id if hasattr(f, "id") else f.get("id", "")
        if fid not in weak_or_refuted:
            answered_sqs.add(sq.strip().lower())

    return [sq for sq in plan.sub_questions if sq.strip().lower() not in answered_sqs]


def _depth_str(state: AgentState) -> str:
    query = state.get("query")
    if not query:
        return "standard"
    d = query.depth
    return d.value if hasattr(d, "value") else str(d)


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

    # supervisor uses the fast tier — it's just generating a plan structure
    llm = _get_tiered_state_llm(state, "fast")
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


def route_from_dispatch(state: AgentState):
    """Return a list of Send objects — one worker per target sub-question.

    If there are no targets (all sub-questions answered), send a single
    no-op worker that immediately routes to collect (which will stop the loop).
    """
    plan     = state.get("plan")
    findings = state.get("findings") or []

    if not plan:
        return [Send("collect_node", {"active_sub_question": None})]

    targets = _research_targets(state)
    if not targets:
        # Nothing to research — bounce through a no-op worker to collect
        return [Send("collect_node", {"active_sub_question": None,
                                      "session_id": state.get("session_id", "default")})]

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

    # Workers use the standard tier (quality matters here)
    llm   = _get_tiered_state_llm(state, "standard")
    session_id = state.get("session_id", "default")
    tools = _get_researcher_tools(max_sources=max_sources, session_id=session_id)

    finding = await run_worker(sub_question, role, state, llm, tools)

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
    """Evaluate stop signal after a dispatch round; route to critic or re-dispatch."""
    from research_swarm.config import settings
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
        "messages": [
            AIMessage(
                content=(
                    f"[Collect] Round {new_rounds}: {'→ critic' if stop else '→ re-dispatch'}. "
                    f"Reason: {reason}"
                )
            )
        ],
    }


def route_from_collect(state: AgentState) -> str:
    """Deterministic edge: read next_agent set by collect_node."""
    return state.get("next_agent") or "critic"


# ---------------------------------------------------------------------------
# critic_node
# ---------------------------------------------------------------------------

async def critic_node(state: AgentState) -> dict[str, Any]:
    if (early := _check_budget(state, "Critic")):
        return early
    # Critic uses fast tier — structured extraction, not synthesis
    llm = _get_tiered_state_llm(state, "fast")
    new_critiques = await run_critic(state, llm)
    logger.info("Critic produced %d critique(s).", len(new_critiques))
    return {
        "critiques": new_critiques,
        "messages": [
            AIMessage(content=f"[Critic] Reviewed {len(new_critiques)} finding(s).")
        ],
    }


# ---------------------------------------------------------------------------
# fact_checker_node
# ---------------------------------------------------------------------------

async def fact_checker_node(state: AgentState) -> dict[str, Any]:
    if (early := _check_budget(state, "FactChecker")):
        return early
    llm = _get_tiered_state_llm(state, "fast")
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
    if _check_budget(state, "Writer"):
        query = state.get("query")
        from research_swarm.schemas import FinalReport
        report = FinalReport(
            title=f"Research Report: {query.topic if query else 'Topic'}",
            exec_summary="Budget exceeded; report could not be completed.",
        )
        return {
            "final_report": report,
            "draft_report": report,
            "writer_instructions": None,
            "messages": [AIMessage(
                content=f"[Writer] Budget exceeded — partial report: {report.title}",
            )],
        }

    # Writer uses the thorough tier — synthesis quality matters most here
    llm = _get_tiered_state_llm(state, "thorough")
    report = await run_writer(state, llm)

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
