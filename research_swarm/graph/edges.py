"""Conditional edge routing for the research swarm graph.

Phase-4 topology — all routing after plan creation is deterministic.

  route_from_supervisor  → always "dispatch_node" (plan just created)
  route_from_dispatch    → list[Send], one per target sub-question (in nodes.py)
  route_from_collect     → "dispatch_node" (loop) or "critic" (stop)
"""
from __future__ import annotations

import logging

from langgraph.graph import END

from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


def route_from_supervisor(state: AgentState) -> str:
    """After supervisor_node: always routes to dispatch_node."""
    next_agent = state.get("next_agent", "dispatch")
    if str(next_agent) in ("end", END):
        return END
    # Everything else (incl. "dispatch", unexpected values) → dispatch_node
    if str(next_agent) not in ("dispatch",):
        logger.warning(
            "route_from_supervisor: unexpected next_agent=%r — routing to dispatch_node.",
            next_agent,
        )
    return "dispatch_node"


def route_from_collect(state: AgentState) -> str:
    """After collect_node: route to dispatch_node (loop) or critic (stop)."""
    next_agent = state.get("next_agent", "critic")
    if str(next_agent) == "dispatch":
        return "dispatch_node"
    return "critic"
