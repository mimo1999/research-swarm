"""Conditional edge routing for the research swarm graph."""
from __future__ import annotations

from langgraph.graph import END

from research_swarm.schemas.state import AgentName, AgentState


def route_from_supervisor(state: AgentState) -> str:
    """Map the supervisor's `next_agent` decision to a graph node name.

    Returns a string matching one of the graph node names or END.
    """
    next_agent: AgentName | None = state.get("next_agent")

    routing: dict[str, str] = {
        "researcher": "researcher",
        "critic": "critic",
        "fact_checker": "fact_checker",
        "writer": "writer",
        "human": "writer",   # human HITL is handled via interrupt_before writer
        "end": END,
    }

    destination = routing.get(str(next_agent), END)
    return destination
