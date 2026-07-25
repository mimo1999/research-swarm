"""Conditional edge routing for the research swarm graph.

Phase-4 topology — all routing after plan creation is deterministic.

  route_from_supervisor      → always "document_pass_node" (plan just created)
  route_from_document_pass   → list[Send], one per ingested document/part,
                                or a bounce straight to "dispatch_node" when
                                there are none (in nodes.py)
  route_from_dispatch        → list[Send], one per target sub-question (in nodes.py)
  route_from_collect         → "dispatch_node" (loop) or "critic" (stop)
  route_from_critic          → "dispatch_node" (rework weak/refuted findings,
                                capped by settings.max_rework_attempts) or
                                "fact_checker" (done)
"""
from __future__ import annotations

import logging

from langgraph.graph import END

from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


def route_from_supervisor(state: AgentState) -> str:
    """After supervisor_node: always routes to document_pass_node.

    document_pass_node runs the one-time ingested-document extraction fan-out
    (or bounces straight through to dispatch_node when there are no ingested
    documents), then joins into dispatch_node for the normal sub-question
    round-0 dispatch.
    """
    next_agent = state.get("next_agent", "dispatch")
    if str(next_agent) in ("end", END):
        return END
    # Everything else (incl. "dispatch", unexpected values) → document_pass_node
    if str(next_agent) not in ("dispatch",):
        logger.warning(
            "route_from_supervisor: unexpected next_agent=%r — routing to document_pass_node.",
            next_agent,
        )
    return "document_pass_node"


def route_from_collect(state: AgentState) -> str:
    """After collect_node: route to dispatch_node (loop) or critic (stop)."""
    next_agent = state.get("next_agent", "critic")
    if str(next_agent) == "dispatch":
        return "dispatch_node"
    return "critic"


def route_from_critic(state: AgentState) -> str:
    """After critic_node: loop back to dispatch_node for rework, or proceed to fact_checker."""
    next_agent = state.get("next_agent", "fact_checker")
    if str(next_agent) == "dispatch":
        return "dispatch_node"
    return "fact_checker"
