"""Reviewer-requested re-research from the HITL pause.

The graph pauses *before the writer* (``interrupt_before=["writer"]``), so simply resuming it
runs the writer: setting ``human_feedback`` on the paused state (what the UI and API used to do)
could never lead back to research. Instead the reviewer's request is written *as the output of
collect_node*: collect's routing then sends the run to dispatch_node, which targets the weakly
answered sub-questions for one more gap-fill round (``nodes._rework_targets``), and the normal
round cap takes it back through the verifier to the same pause before the writer.
"""
from __future__ import annotations

from typing import Any


async def request_rework(graph: Any, config: dict, instructions: str | None) -> None:
    """Point a run paused before the writer back at research. Resume it afterwards with
    ``graph.astream(None, config)``. *instructions* (may be empty) steer the new searches."""
    await graph.aupdate_state(
        config,
        {"next_agent": "dispatch", "rework_instructions": (instructions or "").strip()},
        as_node="collect_node",
    )
