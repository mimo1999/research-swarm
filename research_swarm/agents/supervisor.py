"""Supervisor agent — decides which agent to invoke next."""
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
  researcher   — searches the web, arXiv, and the RAG index; produces Finding objects
  critic       — reviews each Finding and assigns supported / weak / refuted
  fact_checker — cross-checks claims against source snippets; updates confidence scores
  writer       — drafts the final report (will pause for human review before running)
  end          — terminate the session

Decision guidelines:
1. No plan yet            → create a ResearchPlan, set next_agent = "researcher"
2. Findings present, no critiques yet
                          → next_agent = "critic"
3. Critiques present with weak/refuted verdicts AND iteration < max_iterations
                          → next_agent = "researcher"  (re-research weak points)
4. All findings supported OR iteration >= max_iterations
                          → next_agent = "fact_checker"
5. fact_checker has run (findings have updated confidence)
                          → next_agent = "writer"
6. Report is complete     → next_agent = "end"

Always set `plan` when creating it for the first time; otherwise leave it null.
"""


class SupervisorDecision(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of your decision")
    next_agent: AgentName = Field(..., description="Which agent to run next")
    plan: ResearchPlan | None = Field(
        default=None,
        description="Provide ONLY on the very first call to create the research plan",
    )


async def run_supervisor(state: "AgentState", llm: BaseChatModel) -> SupervisorDecision:
    """Invoke the supervisor LLM and return a structured decision."""
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
        HumanMessage(content=f"Current state:\n{json.dumps(state_summary, indent=2)}\n\nWhat should happen next?"),
    ]
    return await structured_llm.ainvoke(messages)
