"""Supervisor agent -- decides which agent to invoke next."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import _latest_verdicts, json_output_instruction
from research_swarm.config import settings
from research_swarm.schemas import ResearchPlan
from research_swarm.schemas.state import AgentName

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState


_SYSTEM_PROMPT = (
    "You are the Supervisor of a multi-agent research system.\n"
    "Analyse the current research state and decide which agent should act next.\n\n"
    "Available agents:\n"
    "  researcher   -- searches the web, arXiv, and the RAG index; produces Finding objects\n"
    "  critic       -- reviews each Finding and assigns supported / weak / refuted\n"
    "  fact_checker -- cross-checks claims against source snippets; updates confidence scores\n"
    "  writer       -- drafts the final report (will pause for human review before running)\n"
    "  end          -- terminate the session\n\n"
    "Decision guidelines:\n"
    "1. No plan yet            -> create a ResearchPlan, set next_agent = \"researcher\"\n"
    "2. Findings present, no critiques yet\n"
    "                          -> next_agent = \"critic\"\n"
    "3. Critiques present with weak/refuted verdicts AND iteration < max_iterations\n"
    "                          -> next_agent = \"researcher\"  (re-research weak points)\n"
    "4. All findings supported OR iteration >= max_iterations\n"
    "                          -> next_agent = \"fact_checker\"\n"
    "5. fact_checker has run (findings have updated confidence)\n"
    "                          -> next_agent = \"writer\"\n"
    "6. Report is complete     -> next_agent = \"end\"\n\n"
    "Always set `plan` when creating it for the first time; otherwise leave it null."
    + json_output_instruction({
        "reasoning": "<brief explanation of your decision>",
        "next_agent": "researcher | critic | fact_checker | writer | end",
        "plan": {
            "sub_questions": ["<question 1>", "<question 2>"],
            "strategy": "<research strategy>",
            "required_tools": ["web_search", "arxiv_search"],
        },
    })
)


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
    try:
        return await structured_llm.ainvoke(messages)
    except Exception as exc:
        # LLM unavailable or returned unparseable output -- build a minimal fallback
        # plan so the graph can still proceed rather than crashing.
        topic = query.topic if query else "research topic"
        logger.warning(
            "Supervisor LLM failed during plan generation (%s: %s) -- using fallback plan.",
            type(exc).__name__,
            exc,
        )
        return SupervisorDecision(
            reasoning=f"LLM unavailable ({type(exc).__name__}); using minimal fallback plan.",
            next_agent="researcher",
            plan=ResearchPlan(
                sub_questions=[topic],
                strategy="Direct research of the main topic",
                required_tools=["web_search"],
            ),
        )


def _route_from_state(state: AgentState) -> SupervisorDecision | None:
    """Return a deterministic routing decision, or None when a plan is needed."""
    plan = state.get("plan")
    findings = state.get("findings") or []
    critiques = state.get("critiques") or []
    iteration = state.get("iteration_count", 0)

    # Hard ceiling: if the session has consumed more than 4× max_iterations
    # supervisor calls (counting critic/fact_checker rounds too), force it to
    # end rather than loop indefinitely.  This guards against edge cases not
    # covered by the researcher-specific iteration limit.
    if iteration >= settings.max_iterations * 4:
        logger.warning(
            "Hard step ceiling reached (%d steps, limit=%d). Forcing fact_checker.",
            iteration,
            settings.max_iterations * 4,
        )
        return SupervisorDecision(
            reasoning=(
                f"Hard step ceiling reached ({iteration} steps). "
                "Forcing fact-checking to terminate the session."
            ),
            next_agent="fact_checker",
        )

    # A completed report ends the session — UNLESS there are pending writer
    # instructions (HITL revision requested), in which case we let the normal
    # routing continue so the writer can be re-invoked with the new feedback.
    if state.get("final_report") is not None and not state.get("writer_instructions"):
        return SupervisorDecision(
            reasoning="Final report exists; ending session.",
            next_agent="end",
        )

    # Human provided feedback: route to researcher for another pass, but only
    # when the researcher has NOT just finished (next_agent would still be
    # "researcher" in that case, causing an immediate re-entry loop).
    # The researcher_node also clears human_feedback after consuming it, so
    # this guard is a belt-and-suspenders check.
    if state.get("human_feedback") and state.get("next_agent") != "researcher":
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

    # next_agent reflects what the previous supervisor call scheduled, so it
    # tells us which worker just completed without scanning the messages list.
    last_agent = state.get("next_agent")
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
    # Use <= max_iterations (not <) to avoid an off-by-one: iteration_count is
    # read *before* the supervisor_node increments it, so the effective limit
    # would otherwise be max_iterations + 1 researcher cycles.
    if weak_or_refuted and iteration <= settings.max_iterations:
        return SupervisorDecision(
            reasoning=f"{len(weak_or_refuted)} finding(s) need stronger evidence.",
            next_agent="researcher",
        )

    return SupervisorDecision(
        reasoning="All reviewed findings are ready for fact-checking.",
        next_agent="fact_checker",
    )


