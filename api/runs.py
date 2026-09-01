"""Background research runs, decoupled from the HTTP connection.

Why this exists
---------------
The graph used to be driven from inside the SSE response generator, and the
generator polled ``request.is_disconnected()`` to decide whether to keep going.
That tied a multi-minute research run to one fragile TCP connection: a refresh,
a flaky network, or a proxy timeout killed the run and threw away every token
already spent on it.

Here the run is the durable thing and the connection is disposable:

* A :class:`ResearchRun` owns an ``asyncio.Task`` driving ``graph.astream`` and
  an append-only buffer of the SSE events it has emitted.
* ``GET /stream`` merely *subscribes* to that buffer. Dropping the connection
  cancels the subscriber, never the task.
* Every event carries a monotonic ``id``. A reconnecting ``EventSource``
  automatically sends ``Last-Event-ID``, so the client replays what it missed
  and picks up live -- no duplicate events, no lost final report.

The run is started lazily by the first ``/stream`` connection rather than by
``POST /api/research``, because the client uploads documents in between and the
graph must not start before they are ingested. It also means a client that
posts and never connects never burns tokens.

Single-process only, like the checkpoint hand-off it replaces: the buffer and
task live in this worker's memory. Scaling past one worker means moving events
to Redis (or similar) and the task to a real queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from research_swarm.runtime.session_ctx import session_scope, unbind_session

from api.serialize import to_jsonable, update_jsonable

logger = logging.getLogger(__name__)

# Hard cap so a pathological run cannot grow the buffer without bound. A real
# run emits on the order of tens of events, so this is a backstop, not a limit
# anyone should hit.
MAX_BUFFERED_EVENTS = 2000

# How long a finished run stays replayable. Long enough that a client which
# reconnects after a laptop sleep still gets its report back.
FINISHED_RUN_TTL_SECONDS = 900.0

RunState = Literal["pending", "running", "interrupted", "finished", "failed"]

# States in which no further events will arrive on the *current* stream
# segment. "interrupted" is closed but not terminal -- a resume opens a new
# segment appending to the same buffer.
_CLOSED_STATES: frozenset[str] = frozenset({"interrupted", "finished", "failed"})
_TERMINAL_STATES: frozenset[str] = frozenset({"finished", "failed"})


class ResearchRun:
    """One research session: a background task plus a replayable event log."""

    def __init__(self, session_id: str, initial_state: dict, hitl: bool) -> None:
        self.session_id = session_id
        self.hitl = hitl
        self.state: RunState = "pending"
        # Consumed by the first stream connection, then None -- a resumed run
        # continues from the checkpoint rather than from an initial state.
        self.initial_state: dict | None = initial_state
        self.task: asyncio.Task | None = None
        self.finished_at: float | None = None

        self._events: list[dict[str, str]] = []
        # Absolute index of _events[0]; nonzero once the cap starts evicting.
        self._base = 0
        self._waiters: list[asyncio.Future[None]] = []

    # -- event log ---------------------------------------------------------

    @property
    def next_event_id(self) -> int:
        return self._base + len(self._events)

    def emit(self, event: str, payload: Any) -> None:
        """Append an SSE event to the log and wake every subscriber."""
        self._events.append({
            "id": str(self.next_event_id),
            "event": event,
            "data": json.dumps(payload),
        })
        if len(self._events) > MAX_BUFFERED_EVENTS:
            overflow = len(self._events) - MAX_BUFFERED_EVENTS
            del self._events[:overflow]
            self._base += overflow
            logger.warning(
                "Run[%s] event buffer overflowed; dropped %d oldest events",
                self.session_id, overflow,
            )
        self._notify()

    def _notify(self) -> None:
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    def _set_state(self, state: RunState) -> None:
        self.state = state
        if state in _TERMINAL_STATES:
            self.finished_at = time.monotonic()
            # The run is over: the user's API key has no further use here.
            unbind_session(self.session_id)
        self._notify()

    @property
    def stream_closed(self) -> bool:
        return self.state in _CLOSED_STATES

    async def subscribe(self, start: int = 0) -> AsyncIterator[dict[str, str]]:
        """Yield buffered events from *start*, then live ones until the segment ends.

        Cancelling this iterator (client disconnect) does not touch the task.
        """
        cursor = max(start, self._base)
        if start < self._base:
            logger.warning(
                "Run[%s] replay requested from %d but buffer starts at %d",
                self.session_id, start, self._base,
            )
        while True:
            # Register before draining: an event appended mid-drain resolves
            # this future, so the await below returns instead of hanging.
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            try:
                while cursor < self.next_event_id:
                    yield self._events[cursor - self._base]
                    cursor += 1
                if self.stream_closed:
                    return
                await waiter
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

    # -- execution ---------------------------------------------------------

    def start(self, graph: Any, config: dict) -> None:
        """Launch the background task, unless one is already running."""
        if self.state == "running" and self.task and not self.task.done():
            return
        initial_state, self.initial_state = self.initial_state, None
        self._set_state("running")
        self.task = asyncio.create_task(
            self._drive(graph, initial_state, config),
            name=f"research-run-{self.session_id}",
        )

    async def _drive(self, graph: Any, initial_state: dict | None, config: dict) -> None:
        """Drive the graph to completion, emitting events as it goes.

        Runs inside ``session_scope`` so every node, tool, and LLM factory
        beneath it resolves this session's credentials -- the scope belongs to
        the task, not to whichever connection happens to be attached.
        """
        with session_scope(self.session_id):
            try:
                async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        # When multiple tasks for the same node resolve within
                        # one superstep -- e.g. dispatch_node's Send() fan-out
                        # to N parallel worker_node calls -- LangGraph reports
                        # them as a list/tuple of per-task update dicts under
                        # one node key instead of a single dict (see
                        # map_output_updates in langgraph.pregel._io). Emit one
                        # event per task so every consumer downstream (the SSE
                        # client, update_jsonable) only ever sees a plain dict.
                        is_multi = isinstance(node_update, (list, tuple))
                        updates = node_update if is_multi else (node_update,)
                        for update in updates:
                            self.emit("node_update", {
                                "node": node_name,
                                "update": update_jsonable(update),
                            })
                            if node_name == "writer" and update and update.get("final_report"):
                                self.emit("final_report", to_jsonable(update["final_report"]))

                snapshot = await graph.aget_state(config)
                if snapshot.next:
                    # Paused for HITL. The segment ends, but the run stays
                    # alive (and credentials stay bound) for the resume.
                    values = snapshot.values
                    self.emit("interrupted", {
                        "findings": to_jsonable(values.get("findings", [])),
                        "critiques": to_jsonable(values.get("critiques", [])),
                    })
                    self.emit("done", {})
                    self._set_state("interrupted")
                else:
                    self.emit("done", {})
                    self._set_state("finished")

            except asyncio.CancelledError:
                # Shutdown, not client disconnect -- disconnects cancel the
                # subscriber, never this task.
                self._set_state("failed")
                raise
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                logger.exception("Run[%s] failed", self.session_id)
                self.emit("error", {"message": str(exc)})
                self._set_state("failed")

    def reopen(self) -> None:
        """Put an interrupted run back in 'pending' so a resume can restart it.

        The event buffer carries over, so ids stay monotonic across the pause
        and a reconnecting client's ``Last-Event-ID`` remains meaningful.
        """
        if self.state == "interrupted":
            self.state = "pending"
            self.task = None

    async def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_runs: dict[str, ResearchRun] = {}
_lock = asyncio.Lock()


def _sweep_finished(now: float) -> None:
    """Drop finished runs past their replay window. Caller holds ``_lock``."""
    stale = [
        sid
        for sid, run in _runs.items()
        if run.finished_at is not None and now - run.finished_at > FINISHED_RUN_TTL_SECONDS
    ]
    for sid in stale:
        _runs.pop(sid, None)


async def create_run(session_id: str, initial_state: dict, hitl: bool) -> ResearchRun:
    """Register a run in 'pending'. The first /stream connection starts it."""
    async with _lock:
        _sweep_finished(time.monotonic())
        run = ResearchRun(session_id, initial_state, hitl)
        _runs[session_id] = run
        return run


async def get_run(session_id: str) -> ResearchRun | None:
    async with _lock:
        return _runs.get(session_id)


async def forget_run(session_id: str) -> None:
    """Cancel and drop a run. Safe to call more than once."""
    async with _lock:
        run = _runs.pop(session_id, None)
    if run is not None:
        await run.cancel()
        unbind_session(session_id)


async def shutdown_all_runs() -> None:
    """Cancel every in-flight run. Called from the app's lifespan teardown."""
    async with _lock:
        runs = list(_runs.values())
        _runs.clear()
    for run in runs:
        await run.cancel()
        unbind_session(run.session_id)
