"""LangGraph node functions -- one async function per agent."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from research_swarm.agents.base import get_agent_llm
from research_swarm.agents.critic import run_critic
from research_swarm.agents.fact_checker import run_fact_checker
from research_swarm.agents.researcher import run_researcher
from research_swarm.agents.supervisor import run_supervisor
from research_swarm.agents.writer import run_writer
from research_swarm.config import settings
from research_swarm.schemas.state import AgentState
from research_swarm.tools import arxiv_search, fetch_url, web_search
from research_swarm.tools.retriever_tool import build_retriever_tool

logger = logging.getLogger(__name__)


def _get_researcher_tools():
    """Return the tool list for the researcher, including the RAG retriever."""
    tools = [web_search, arxiv_search, fetch_url]
    try:
        tools.append(build_retriever_tool())
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
    iteration = state.get("iteration_count", 0)

    # Hard iteration cap (deterministic -- no LLM call needed)
    if iteration >= settings.max_iterations:
        logger.warning("Iteration cap reached (%d) -- forcing END.", settings.max_iterations)
        return {
            "next_agent": "end",
            "iteration_count": iteration + 1,
            "messages": [
                AIMessage(
                    content=(
                        f"[Supervisor] Iteration cap reached "
                        f"({settings.max_iterations}). Ending session."
                    )
                )
            ],
        }

    llm = _get_state_llm(state)
    decision = await run_supervisor(state, llm)
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
    llm = _get_state_llm(state)
    tools = _get_researcher_tools()
    new_findings = await run_researcher(state, llm, tools)

    logger.info("Researcher produced %d finding(s).", len(new_findings))
    return {
        "findings": new_findings,
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
        "messages": [
            AIMessage(content=f"[Writer] Report complete: {report.title}")
        ],
    }
