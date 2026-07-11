from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from research_swarm.graph.builder import get_thread_config
from research_swarm.runtime.budget import clear_budget
from research_swarm.runtime.session_ctx import (
    SessionCredentials,
    bind_session,
    session_scope,
)
from research_swarm.schemas import ResearchQuery

from api.runs import create_run, forget_run, get_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research")


class ResearchSubmit(BaseModel):
    topic: str
    audience: str = "technical"
    depth: str = "shallow"
    max_sources: int = 10
    provider: str = "ollama"
    model: str
    ollama_url: str | None = None
    ollama_deployment: str | None = None
    hitl_enabled: bool = True


@router.post("")
async def start_research(
    body: ResearchSubmit,
    # Credentials arrive as headers, never in the request body -- a body is
    # what gets echoed into validation errors and access logs. They are bound
    # to the session in memory and never enter AgentState or the checkpoint.
    x_anthropic_api_key: str | None = Header(default=None),
    x_openai_api_key: str | None = Header(default=None),
    x_ollama_api_key: str | None = Header(default=None),
):
    if not body.topic.strip():
        raise HTTPException(status_code=422, detail="topic is required")

    session_id = str(uuid.uuid4())
    clear_budget(session_id)

    # Per-session, NOT on the global settings singleton: mutating settings here
    # would let a concurrent session read this user's key and connection.
    bind_session(
        session_id,
        SessionCredentials(
            anthropic_api_key=x_anthropic_api_key or "",
            openai_api_key=x_openai_api_key or "",
            ollama_api_key=x_ollama_api_key or "",
            ollama_base_url=body.ollama_url if body.provider == "ollama" else None,
            ollama_deployment=(
                (body.ollama_deployment or "local") if body.provider == "ollama" else None
            ),
        ),
    )

    query = ResearchQuery(
        topic=body.topic.strip(),
        depth=body.depth,
        max_sources=body.max_sources,
        audience=body.audience,
    )
    initial_state = {
        "messages": [],
        "query": query,
        "plan": None,
        "findings": [],
        "critiques": [],
        "draft_report": None,
        "final_report": None,
        "human_feedback": None,
        "writer_instructions": None,
        "iteration_count": 0,
        "next_agent": None,
        "session_id": session_id,
        "model_provider": body.provider,
        "model_name": body.model,
        "schema_version": 1,
    }
    await create_run(session_id, initial_state, body.hitl_enabled)
    return {"session_id": session_id}


@router.get("/{session_id}/stream")
async def stream_research(
    session_id: str,
    request: Request,
    # A reconnecting EventSource sends this automatically; the explicit
    # ?from= param covers a client rejoining after a full page reload.
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    from_: int | None = Query(default=None, alias="from"),
):
    run = await get_run(session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")

    # The run starts on first connect, not at POST time: the client uploads
    # documents between the two, and the graph must not start before they are
    # ingested. A reconnect finds the run already running and just subscribes.
    if run.state == "pending":
        run.start(request.app.state.graphs[run.hitl], get_thread_config(session_id))

    start = _replay_cursor(last_event_id, from_)

    async def event_gen():
        # Only forwards events; the graph runs in the run's own task, so a
        # client disconnect cancels this generator and nothing else.
        async for event in run.subscribe(start):
            yield event

    return EventSourceResponse(event_gen())


@router.get("/{session_id}/status")
async def run_status(session_id: str):
    """Whether a run is still alive, so a reloaded client knows to rejoin."""
    run = await get_run(session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    return {"state": run.state, "next_event_id": run.next_event_id, "hitl": run.hitl}


def _replay_cursor(last_event_id: str | None, from_: int | None) -> int:
    """Resolve where to resume the event log, tolerating a junk header."""
    if from_ is not None:
        return max(from_, 0)
    if last_event_id is not None:
        try:
            # Last-Event-ID is the last event *seen*; resume after it.
            return max(int(last_event_id) + 1, 0)
        except ValueError:
            logger.warning("Ignoring malformed Last-Event-ID %r", last_event_id)
    return 0


class ResumeBody(BaseModel):
    action: Literal["approve", "edit", "discard"]
    feedback: str | None = None


@router.post("/{session_id}/resume")
async def resume_research(session_id: str, body: ResumeBody, request: Request):
    run = await get_run(session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session")

    graph = request.app.state.graphs[run.hitl]
    config = get_thread_config(session_id)

    if body.action == "discard":
        clear_budget(session_id)
        await forget_run(session_id)
        return {"status": "discarded"}

    with session_scope(session_id):
        if body.action == "approve":
            await graph.aupdate_state(config, {"writer_instructions": body.feedback or "Approved."})
        else:  # edit
            await graph.aupdate_state(config, {
                "human_feedback": (
                    body.feedback or "Please re-research weak findings more thoroughly."
                ),
                "next_agent": None,
            })

    # Back to 'pending' with the event log intact -- the client's reconnect
    # starts the next segment, and its Last-Event-ID stays meaningful because
    # event ids keep counting across the pause.
    run.reopen()
    return {"status": "resumed", "resume_from": run.next_event_id}
