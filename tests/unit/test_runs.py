"""Background research runs: the run outlives the HTTP connection.

The property under test is that the graph task and the SSE subscriber have
independent lifetimes -- dropping a client must not stop research, and
reconnecting must replay exactly what was missed.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from api import runs as runs_mod
from api.routes.research import _replay_cursor
from api.runs import ResearchRun, create_run, get_run
from research_swarm.runtime import session_ctx


@pytest.fixture(autouse=True)
def _clean_registries():
    runs_mod._runs.clear()
    session_ctx._registry.clear()
    yield
    runs_mod._runs.clear()
    session_ctx._registry.clear()


class FakeGraph:
    """Minimal stand-in for a compiled LangGraph.

    ``gate`` lets a test hold the graph mid-run so it can assert on a live
    subscriber before the run completes.
    """

    def __init__(self, chunks, *, next_nodes=(), values=None, raises=None, gate=None):
        self.chunks = chunks
        self.next_nodes = next_nodes
        self.values = values or {"findings": [], "critiques": []}
        self.raises = raises
        self.gate = gate
        self.astream_calls = 0

    async def astream(self, initial_state, config, stream_mode=None):  # noqa: ARG002
        self.astream_calls += 1
        for chunk in self.chunks:
            if self.gate is not None:
                await self.gate.wait()
            yield chunk
        if self.raises is not None:
            raise self.raises

    async def aget_state(self, config):  # noqa: ARG002
        return SimpleNamespace(next=self.next_nodes, values=self.values)


def _node_chunk(node="researcher", **update):
    return {node: update or {"plan": "x"}}


async def _drain(run: ResearchRun, start: int = 0) -> list[dict]:
    return [event async for event in run.subscribe(start)]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_disconnect_does_not_cancel_the_run():
    """The regression this whole design exists for: a dropped client used to
    kill the research run and throw away every token already spent."""
    gate = asyncio.Event()
    graph = FakeGraph([_node_chunk(), _node_chunk(), _node_chunk()], gate=gate)
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})

    # Attach a subscriber, take one event, then hang up mid-run.
    subscriber = run.subscribe(0)
    gate.set()
    first = await subscriber.__anext__()
    await subscriber.aclose()

    await run.task

    assert first["event"] == "node_update"
    assert run.state == "finished"
    assert not run._waiters  # the abandoned subscriber deregistered itself
    # Everything the disconnected client missed is still on the log.
    replayed = await _drain(run)
    assert [e["event"] for e in replayed] == ["node_update"] * 3 + ["done"]


# ---------------------------------------------------------------------------
# Event log and replay
# ---------------------------------------------------------------------------


async def test_replay_returns_only_missed_events():
    graph = FakeGraph([_node_chunk(), _node_chunk(), _node_chunk()])
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})
    await run.task

    # Client saw ids 0 and 1, then dropped. Last-Event-ID: 1 -> resume at 2.
    replayed = await _drain(run, start=_replay_cursor("1", None))
    assert [e["id"] for e in replayed] == ["2", "3"]


# ---------------------------------------------------------------------------
# HITL pause / resume
# ---------------------------------------------------------------------------

async def test_interrupt_closes_the_stream_but_keeps_the_run():
    graph = FakeGraph([_node_chunk()], next_nodes=("writer",))
    run = await create_run("s1", {"topic": "x"}, hitl=True)
    run.start(graph, {})
    await run.task

    events = await _drain(run)
    assert [e["event"] for e in events] == ["node_update", "interrupted", "done"]
    assert run.state == "interrupted"
    assert await get_run("s1") is run  # still resumable


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

async def test_failure_emits_error_and_closes():
    graph = FakeGraph([_node_chunk()], raises=RuntimeError("llm exploded"))
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})
    await run.task

    events = await _drain(run)
    assert [e["event"] for e in events] == ["node_update", "error"]
    assert json.loads(events[-1]["data"])["message"] == "llm exploded"
    assert run.state == "failed"
