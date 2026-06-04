"""Supervisor agent -- decides which agent to invoke next."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import json_output_instruction
from research_swarm.schemas import ResearchPlan
from research_swarm.schemas.state import AgentName

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState


# Sub-question limits by depth — controls how many sub-questions the LLM is
# allowed to generate in the initial research plan.
_SUB_QUESTIONS_BY_DEPTH: dict[str, int] = {
    "shallow":  1,
    "standard": 4,
    "deep":     6,
}


def _build_system_prompt(depth: str = "standard") -> str:
    """Return a supervisor system prompt focused on plan creation only."""
    max_sq = _SUB_QUESTIONS_BY_DEPTH.get(depth, 4)
    return (
        "You are the Supervisor of a multi-agent research system.\n"
        "Your ONLY job in this call is to create the initial research plan.\n\n"
        "Worker roles available for assignment:\n"
        "  general   -- balanced web + arXiv + RAG research\n"
        "  academic  -- prioritises peer-reviewed papers and arXiv pre-prints\n"
        "  industry  -- prioritises real-world deployment and case studies\n"
        "  skeptic   -- actively seeks counter-evidence and known failure modes\n"
        "  benchmark -- seeks quantitative comparisons and empirical metrics\n\n"
        f"Generate AT MOST {max_sq} sub-questions (depth = {depth!r}).\n"
        "Assign a worker role to each sub-question based on the angle that best answers it.\n"
        "For shallow depth, always use role 'general'.\n"
        "Set complexity_score 0.0–1.0 (0 = single-fact, 1 = deep multi-faceted).\n\n"
        "Always set `plan`. Leave `next_agent` as \"dispatch\"."
        + json_output_instruction({
            "reasoning": "<why this decomposition covers the topic>",
            "next_agent": "dispatch",
            "plan": {
                "sub_questions": ["<question 1>", "<question 2>"],
                "strategy": "<overall research strategy>",
                "required_tools": ["web_search", "arxiv_search"],
                "complexity_score": 0.5,
                "assignments": [
                    {"sub_question": "<question 1>", "worker_role": "academic"},
                    {"sub_question": "<question 2>", "worker_role": "skeptic"},
                ],
            },
        })
    )


class SupervisorDecision(BaseModel):
    reasoning:  str          = Field(..., description="Brief explanation")
    next_agent: AgentName    = Field(..., description="Which agent to run next")
    plan:       ResearchPlan | None = Field(
        default=None,
        description="Provide on the first call only — omit on all subsequent calls",
    )


async def run_supervisor(state: AgentState, llm: BaseChatModel) -> SupervisorDecision:
    """Create the initial research plan (LLM call) or route deterministically.

    After Phase 4 the supervisor is invoked only once — at session start — to
    produce the plan with sub-question assignments.  All subsequent routing
    decisions are handled deterministically inside collect_node and the graph
    edge functions, so the LLM is never called again for routing purposes.
    """
    # If a plan already exists, return a deterministic decision without any LLM call.
    if state.get("plan") is not None:
        return SupervisorDecision(
            reasoning="Plan already exists; routing to dispatch.",
            next_agent="dispatch",
        )

    # --- Plan creation via LLM ---
    query     = state.get("query")
    depth_str = str(
        query.depth.value if hasattr(query.depth, "value") else query.depth
    ) if query else "standard"

    structured_llm = llm.with_structured_output(SupervisorDecision)
    messages = [
        SystemMessage(content=_build_system_prompt(depth_str)),
        HumanMessage(
            content=(
                f"Research topic: {query.topic if query else 'unknown'}\n"
                f"Depth: {depth_str}\n"
                f"Audience: {query.audience if query else 'general'}\n\n"
                "Create the research plan now."
            )
        ),
    ]
    try:
        decision = await structured_llm.ainvoke(messages)
        # Enforce next_agent = "dispatch" regardless of what the LLM returned.
        return SupervisorDecision(
            reasoning=decision.reasoning,
            next_agent="dispatch",
            plan=decision.plan,
        )
    except Exception as exc:
        topic = query.topic if query else "research topic"
        logger.warning(
            "Supervisor LLM failed (%s: %s) — using fallback plan.",
            type(exc).__name__, exc,
        )
        from research_swarm.schemas.worker import SubQuestionAssignment, WorkerRole
        return SupervisorDecision(
            reasoning="LLM unavailable; using minimal fallback plan.",
            next_agent="dispatch",
            plan=ResearchPlan(
                sub_questions=[topic],
                strategy="Direct research of the main topic",
                required_tools=["web_search"],
                complexity_score=0.3,
                assignments=[SubQuestionAssignment(
                    sub_question=topic, worker_role=WorkerRole.general,
                )],
            ),
        )



