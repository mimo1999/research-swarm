"""Conditional edge routing for the research swarm graph."""
from __future__ import annotations

import logging

from langgraph.graph import END

from research_swarm.schemas.state import AgentName, AgentState

logger = logging.getLogger(__name__)

_ROUTING: dict[str, str] = {
    "researcher": "researcher",
    "critic": "critic",
    "fact_checker": "fact_checker",
    "writer": "writer",
    "human": "writer",   # HITL: interrupt_before writer handles the pause
    "end": END,
}


def route_from_supervisor(state: AgentState) -> str:
    """Map the supervisor's ``next_agent`` decision to a graph node name.

    Returns a string matching one of the graph node names or ``END``.
    Logs a WARNING when an unrecognised value is encountered so routing
    failures surface in logs rather than silently terminating the session.
    """
    next_agent: AgentName | None = state.get("next_agent")
    key = str(next_agent)

    if key not in _ROUTING:
        logger.warning(
            "route_from_supervisor: unrecognised next_agent=%r -- routing to END. "
            "Check supervisor output for unexpected values.",
            next_agent,
        )

    return _ROUTING.get(key, END)
