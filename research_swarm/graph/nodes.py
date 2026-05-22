"""LangGraph node functions -- one async function per agent."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from research_swarm.agents.base import get_agent_llm
from research_swarm.agents.critic import run_critic
from research_swarm.agents.fact_checker import run_fact_checker
from research_swarm.agents.researcher import run_researcher
from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
from research_swarm.agents.writer import run_writer
from research_swarm.schemas.state import AgentState
from research_swarm.tools import arxiv_search, fetch_url, web_search
from research_swarm.tools.retriever_tool import build_retriever_tool

logger = logging.getLogger(__name__)


def _get_researcher_tools(max_sources: int | None = None):
    """Return the tool list for the researcher, including the RAG retriever.

    max_sources is forwarded to the retriever so the vector store uses the
    per-session limit rather than the process-wide default.
    """
    tools = [web_search, arxiv_search, fetch_url]
    try:
        tools.append(build_retriever_tool(max_sources=max_sources))
    except Exception as exc:
        logger.warning("RAG retriever unavailable: %s", exc)
    return tools


def _get_state_llm(state: AgentState):
    """Create the chat model requested by this run's state."""
    return get_agent_llm(
        provider=state.get("model_provider"),
        model=state.get("model_name"),
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Central router -- decides which agent to invoke next."""
    from research_swarm.config import settings

    iteration = state.get("iteration_count", 0)

    # Fast-path: check the hard ceiling BEFORE constructing the LLM.  When the
    # ceiling fires we know exactly where to route (fact_checker) without asking
    # the model, so there is no reason to pay the object-construction cost or
    # risk authentication noise in logs / tests.
    ceiling = settings.max_iterations * 4
    if iteration >= ceiling:
        from research_swarm.agents.supervisor import SupervisorDecision as _SD
        logger.warning(
            "Hard step ceiling reached (%d steps, limit=%d). Forcing fact_checker.",
            iteration,
            ceiling,
        )
        decision = _SD(
            reasoning=(
                f"Hard step ceiling reached ({iteration} steps). "
                "Forcing fact-checking to terminate the session."
            ),
            next_agent="fact_checker",
        )
    else:
        # _get_state_llm is cheap (constructs a LangChain model object, no
        # network call), but run_supervisor's deterministic path may not use it
        # at all.  We still create it here so run_supervisor has it available.
        llm = _get_state_llm(state)
        decision = await run_supervisor(state, llm)

    # When the LLM just created a plan, always route to researcher regardless
    # of what the LLM returned for next_agent.  This makes the post-plan
    # transition deterministic and removes a source of fragility.
    if decision.plan is not None:
        decision = SupervisorDecision(
            reasoning=decision.reasoning,
            next_agent="researcher",
            plan=decision.plan,
        )

    logger.info("Supervisor decision: %s (iter %d)", decision.next_agent, iteration)

    update: dict[str, Any] = {
        "next_agent": decision.next_agent,
        "iteration_count": iteration + 1,
        "messages": [AIMessage(content=f"[Supervisor] {decision.reasoning}")],
    }

    # Attach plan on first call
    if decision.plan is not None:
        update["plan"] = decision.plan

    return update


async def researcher_node(state: AgentState) -> dict[str, Any]:
    """Research sub-questions using web/arXiv/RAG tools; emit Findings."""
    query = state.get("query")
    if query is None:
        logger.error("researcher_node called with no query in state -- skipping.")
        return {
            "messages": [AIMessage(content="[Researcher] No query found; skipping.")],
        }

    llm = _get_state_llm(state)
    max_sources = query.max_sources if query else None
    tools = _get_researcher_tools(max_sources=max_sources)
    new_findings = await run_researcher(state, llm, tools)

    logger.info("Researcher produced %d finding(s).", len(new_findings))
    return {
        "findings": new_findings,
        # Consume human_feedback so the supervisor's human-feedback guard
        # does not re-trigger on the very next call (infinite-loop fix).
        "human_feedback": None,
        "messages": [
            AIMessage(content=f"[Researcher] Produced {len(new_findings)} finding(s).")
        ],
    }


async def critic_node(state: AgentState) -> dict[str, Any]:
    """Review findings and emit Critiques."""
    llm = _get_state_llm(state)
    new_critiques = await run_critic(state, llm)

    logger.info("Critic produced %d critique(s).", len(new_critiques))
    return {
        "critiques": new_critiques,
        "messages": [
            AIMessage(content=f"[Critic] Reviewed {len(new_critiques)} finding(s).")
        ],
    }


async def fact_checker_node(state: AgentState) -> dict[str, Any]:
    """Cross-check claims and update Finding confidence scores."""
    llm = _get_state_llm(state)
    updated_findings = await run_fact_checker(state, llm)

    logger.info("FactChecker updated %d finding(s).", len(updated_findings))
    return {
        "findings": updated_findings,  # merge-by-id reducer will overwrite
        "messages": [
            AIMessage(
                content=(
                    f"[FactChecker] Updated confidence on "
                    f"{len(updated_findings)} finding(s)."
                )
            )
        ],
    }


async def writer_node(state: AgentState) -> dict[str, Any]:
    """Generate the FinalReport from validated findings."""
    llm = _get_state_llm(state)
    report = await run_writer(state, llm)

    logger.info("Writer produced report: %r", report.title)
    return {
        "final_report": report,
        "draft_report": report,
        # Clear writer_instructions so they don't persist into the next run or
        # cause the supervisor to think another revision is pending.
        "writer_instructions": None,
        "messages": [
            AIMessage(content=f"[Writer] Report complete: {report.title}")
        ],
    }
