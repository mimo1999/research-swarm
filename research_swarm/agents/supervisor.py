"""Supervisor agent -- decides which agent to invoke next."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.config import settings
from research_swarm.schemas import ResearchPlan
from research_swarm.schemas.state import AgentName

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState


_SYSTEM_PROMPT = """\
You are the Supervisor of a multi-agent research system.
Analyse the current research state and decide which agent should act next.

Available agents:
  researcher   -- searches the web, arXiv, and the RAG index; produces Finding objects
  critic       -- reviews each Finding and assigns supported / weak / refuted
  fact_checker -- cross-checks claims against source snippets; updates confidence scores
  writer       -- drafts the final report (will pause for human review before running)
  end          -- terminate the session

Decision guidelines:
1. No plan yet            -> create a ResearchPlan, set next_agent = "researcher"
2. Findings present, no critiques yet
                          -> next_agent = "critic"
3. Critiques present with weak/refuted verdicts AND iteration < max_iterations
                          -> next_agent = "researcher"  (re-research weak points)
4. All findings supported OR iteration >= max_iterations
                          -> next_agent = "fact_checker"
5. fact_checker has run (findings have updated confidence)
                          -> next_agent = "writer"
6. Report is complete     -> next_agent = "end"

Always set `plan` when creating it for the first time; otherwise leave it null.
"""


class SupervisorDecision(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of your decision")
    next_agent: AgentName = Field(..., description="Which agent to run next")
    plan: ResearchPlan | None = Field(
        default=None,
        description="Provide ONLY on the very first call to create the research plan",
    )


async def run_supervisor(state: AgentState, llm: BaseChatModel) -> SupervisorDecision:
    """Return the next graph step.

    Plan creation still uses the LLM, but lifecycle routing is deterministic so
    the graph does not depend on prompt compliance for ordinary control flow.
    """
    deterministic_decision = _route_from_state(state)
    if deterministic_decision is not None:
        return deterministic_decision

    structured_llm = llm.with_structured_output(SupervisorDecision)

    # Build a compact state summary for the prompt
    query = state.get("query")
    plan = state.get("plan")
    findings = state.get("findings") or []
    critiques = state.get("critiques") or []
    iteration = state.get("iteration_count", 0)
    draft = state.get("draft_report")

    # Verdict tally for the critic summary
    verdict_counts: dict[str, int] = {}
    for c in critiques:
        v = c.verdict if hasattr(c, "verdict") else c.get("verdict", "?")
        verdict_counts[str(v)] = verdict_counts.get(str(v), 0) + 1

    state_summary = {
        "query": query.topic if query else "N/A",
        "depth": query.depth if query else "N/A",
        "audience": query.audience if query else "N/A",
        "plan_exists": plan is not None,
        "sub_questions": plan.sub_questions if plan else [],
        "n_findings": len(findings),
        "n_critiques": len(critiques),
        "critique_verdicts": verdict_counts,
        "draft_report_exists": draft is not None,
        "iteration_count": iteration,
        "max_iterations": settings.max_iterations,
        "human_feedback": state.get("human_feedback"),
    }

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Current state:\n{json.dumps(state_summary, indent=2)}"
                "\n\nWhat should happen next?"
            )
        ),
    ]
    return await structured_llm.ainvoke(messages)


def _route_from_state(state: AgentState) -> SupervisorDecision | None:
    """Return a deterministic routing decision, or None when a plan is needed."""
    plan = state.get("plan")
    findings = state.get("findings") or []
    critiques = state.get("critiques") or []
    iteration = state.get("iteration_count", 0)

    if state.get("final_report") is not None:
        return SupervisorDecision(
            reasoning="Final report exists; ending session.",
            next_agent="end",
        )

    if state.get("draft_report") is not None:
        return SupervisorDecision(
            reasoning="Draft report exists; ending session.",
            next_agent="end",
        )

    if state.get("next_agent") == "researcher" and state.get("human_feedback"):
        return SupervisorDecision(
            reasoning="Human feedback requested another research pass.",
            next_agent="researcher",
        )

    if plan is None:
        return None

    if not findings:
        return SupervisorDecision(
            reasoning="Research plan exists but no findings have been gathered.",
            next_agent="researcher",
        )

    last_agent = _last_worker_agent(state)
    if last_agent == "researcher":
        return SupervisorDecision(
            reasoning="Researcher produced findings; sending them to the critic.",
            next_agent="critic",
        )

    latest_verdicts = _latest_verdicts(critiques)
    finding_ids = {f.id if hasattr(f, "id") else f.get("id", "") for f in findings}
    unreviewed = finding_ids - set(latest_verdicts)
    if unreviewed:
        return SupervisorDecision(
            reasoning=f"{len(unreviewed)} finding(s) still need critique.",
            next_agent="critic",
        )

    if last_agent == "fact_checker":
        return SupervisorDecision(
            reasoning="Fact-checking is complete; writing the final report.",
            next_agent="writer",
        )

    weak_or_refuted = {
        fid
        for fid, verdict in latest_verdicts.items()
        if verdict in {"weak", "refuted"}
    }
    if weak_or_refuted and iteration < settings.max_iterations:
        return SupervisorDecision(
            reasoning=f"{len(weak_or_refuted)} finding(s) need stronger evidence.",
            next_agent="researcher",
        )

    return SupervisorDecision(
        reasoning="All reviewed findings are ready for fact-checking.",
        next_agent="fact_checker",
    )


def _latest_verdicts(critiques: list) -> dict[str, str]:
    latest: dict[str, str] = {}
    for critique in critiques:
        fid = (
            critique.finding_id
            if hasattr(critique, "finding_id")
            else critique.get("finding_id", "")
        )
        verdict = critique.verdict if hasattr(critique, "verdict") else critique.get("verdict", "")
        latest[fid] = verdict.value if hasattr(verdict, "value") else str(verdict)
    return latest


def _last_worker_agent(state: AgentState) -> str | None:
    for message in reversed(state.get("messages") or []):
        content = getattr(message, "content", "")
        if content.startswith("[Researcher]"):
            return "researcher"
        if content.startswith("[Critic]"):
            return "critic"
        if content.startswith("[FactChecker]"):
            return "fact_checker"
        if content.startswith("[Writer]"):
            return "writer"
    return None
