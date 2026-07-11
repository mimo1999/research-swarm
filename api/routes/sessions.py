from __future__ import annotations

import re
from collections import Counter

from fastapi import APIRouter, HTTPException, Request

from research_swarm.graph.builder import get_thread_config
from research_swarm.persistence.sessions import delete_session, list_sessions
from research_swarm.runtime.budget import get_budget
from research_swarm.runtime.migrations import migrate_state

from api.serialize import to_jsonable

router = APIRouter(prefix="/api/sessions")

_NODE_TAG = re.compile(r"^\[([^\]]+)\]")


def _node_visit_counts(messages: list) -> dict[str, int]:
    """Count agent-node invocations from the '[NodeName] ...' message prefix
    every node writes to state['messages'] (see graph/nodes.py). This is the
    "iterations per agent" figure -- more reliable than reading LangGraph's
    internal checkpoint metadata, whose shape isn't a stable per-node map in
    this LangGraph version.
    """
    counts: Counter[str] = Counter()
    for m in messages:
        content = m.content if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else None)
        if not isinstance(content, str):
            continue
        if match := _NODE_TAG.match(content):
            counts[match.group(1)] += 1
    return dict(counts)


@router.get("")
async def get_sessions():
    sessions = list_sessions()
    return [
        {
            "thread_id": s.thread_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "step_count": s.step_count,
            "has_report": s.has_report,
        }
        for s in sessions
    ]


@router.get("/{thread_id}")
async def get_session(thread_id: str, request: Request):
    graph = request.app.state.graphs[True]
    config = get_thread_config(thread_id)
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    state = migrate_state(dict(snapshot.values))

    return {
        "final_report": to_jsonable(state.get("final_report")),
        "findings": to_jsonable(state.get("findings", [])),
        "critiques": to_jsonable(state.get("critiques", [])),
        "plan": to_jsonable(state.get("plan")),
        "node_visits": _node_visit_counts(state.get("messages", [])),
        # bool(snapshot.next) is true for ANY pending step -- including a
        # normal rework loop back to dispatch, or a mid-run crash whose last
        # good checkpoint just happens to have a queued next node. The only
        # real HITL pause point is interrupt_before=["writer"], so check for
        # that specifically rather than "any next step exists".
        "is_interrupted": "writer" in (snapshot.next or ()),
    }


@router.get("/{thread_id}/usage")
async def get_session_usage(thread_id: str):
    budget = get_budget(thread_id)
    return {
        "llm_calls": budget.used,
        "llm_call_limit": budget.limit,
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "total_tokens": budget.total_tokens,
    }


@router.delete("/{thread_id}")
async def remove_session(thread_id: str):
    n = delete_session(thread_id)
    return {"deleted_count": n}
