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
from api.runs import ResearchRun, create_run, forget_run, get_run, shutdown_all_runs
from research_swarm.runtime import session_ctx
from research_swarm.runtime.session_ctx import SessionCredentials, bind_session


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

async def test_create_does_not_start_the_graph():
    """Documents are uploaded between POST and connect -- don't run before them."""
    graph = FakeGraph([_node_chunk()])
    run = await create_run("s1", {"topic": "x"}, hitl=False)

    assert run.state == "pending"
    assert graph.astream_calls == 0


async def test_first_connect_starts_the_run():
    graph = FakeGraph([_node_chunk()])
    run = await create_run("s1", {"topic": "x"}, hitl=False)

    run.start(graph, {})
    await run.task

    assert graph.astream_calls == 1
    assert run.state == "finished"


async def test_second_connect_does_not_restart_the_run():
    """A reconnect subscribes to the existing task instead of re-running it."""
    gate = asyncio.Event()
    graph = FakeGraph([_node_chunk(), _node_chunk()], gate=gate)
    run = await create_run("s1", {"topic": "x"}, hitl=False)

    run.start(graph, {})
    first_task = run.task
    run.start(graph, {})  # simulate a reconnect while still running

    assert run.task is first_task
    gate.set()
    await run.task
    assert graph.astream_calls == 1


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


async def test_shutdown_cancels_in_flight_runs():
    graph = FakeGraph([_node_chunk()], gate=asyncio.Event())  # never opened
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})

    await shutdown_all_runs()

    assert run.task.cancelled() or run.task.done()
    assert await get_run("s1") is None


# ---------------------------------------------------------------------------
# Node update shaping
# ---------------------------------------------------------------------------

async def test_parallel_fanout_emits_one_event_per_task():
    """dispatch_node's Send() fan-out can land several worker_node results in
    the same superstep. LangGraph's `updates` stream mode then reports them
    as a list of per-task update dicts under one node key instead of a
    single dict (see map_output_updates in langgraph.pregel._io) -- this
    used to crash update_jsonable with "'list' object has no attribute
    'items'" the first time two workers actually finished together."""
    graph = FakeGraph([
        {"worker_node": [{"findings": ["a"]}, {"findings": ["b"]}]},
    ])
    run = await create_run("s1", {"topic": "x"}, hitl=False)

    run.start(graph, {})
    await run.task

    assert run.state == "finished"
    events = await _drain(run)
    node_updates = [json.loads(e["data"]) for e in events if e["event"] == "node_update"]
    assert node_updates == [
        {"node": "worker_node", "update": {"findings": ["a"]}},
        {"node": "worker_node", "update": {"findings": ["b"]}},
    ]


async def test_node_with_no_output_channel_writes_does_not_crash():
    """A node that wrote to no observed output channel is reported as a bare
    `None` for that step, not `{}` -- update_jsonable must tolerate it."""
    graph = FakeGraph([{"dispatch_node": None}])
    run = await create_run("s1", {"topic": "x"}, hitl=False)

    run.start(graph, {})
    await run.task

    assert run.state == "finished"
    events = await _drain(run)
    node_updates = [json.loads(e["data"]) for e in events if e["event"] == "node_update"]
    assert node_updates == [{"node": "dispatch_node", "update": {}}]


# ---------------------------------------------------------------------------
# Event log and replay
# ---------------------------------------------------------------------------

async def test_events_carry_monotonic_ids():
    graph = FakeGraph([_node_chunk(), _node_chunk()])
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})
    await run.task

    events = await _drain(run)
    assert [e["id"] for e in events] == ["0", "1", "2"]


async def test_replay_returns_only_missed_events():
    graph = FakeGraph([_node_chunk(), _node_chunk(), _node_chunk()])
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})
    await run.task

    # Client saw ids 0 and 1, then dropped. Last-Event-ID: 1 -> resume at 2.
    replayed = await _drain(run, start=_replay_cursor("1", None))
    assert [e["id"] for e in replayed] == ["2", "3"]


async def test_late_subscriber_gets_the_whole_log():
    graph = FakeGraph([_node_chunk(node="writer", final_report={"title": "T"})])
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})
    await run.task

    events = await _drain(run)
    names = [e["event"] for e in events]
    assert names == ["node_update", "final_report", "done"]
    assert json.loads(events[1]["data"]) == {"title": "T"}


async def test_subscriber_wakes_on_live_events():
    """A subscriber attached mid-run receives events as they are emitted."""
    gate = asyncio.Event()
    graph = FakeGraph([_node_chunk(), _node_chunk()], gate=gate)
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(graph, {})

    collected = []

    async def consume():
        async for event in run.subscribe(0):
            collected.append(event["event"])

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    assert collected == []  # nothing emitted while the graph is gated

    gate.set()
    await run.task
    await consumer

    assert collected == ["node_update", "node_update", "done"]


async def test_buffer_overflow_drops_oldest_and_advances_base(monkeypatch):
    monkeypatch.setattr(runs_mod, "MAX_BUFFERED_EVENTS", 3)
    run = ResearchRun("s1", {}, hitl=False)

    for i in range(5):
        run.emit("node_update", {"i": i})

    assert run._base == 2
    assert run.next_event_id == 5
    # Ids stay absolute, so a reconnecting client's cursor is still valid.
    assert [e["id"] for e in run._events] == ["2", "3", "4"]


async def test_replay_below_base_is_clamped_not_crashed(monkeypatch):
    monkeypatch.setattr(runs_mod, "MAX_BUFFERED_EVENTS", 2)
    run = ResearchRun("s1", {}, hitl=False)
    for i in range(4):
        run.emit("node_update", {"i": i})
    run._set_state("finished")

    replayed = await _drain(run, start=0)
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


async def test_resume_continues_event_ids_across_the_pause():
    paused = FakeGraph([_node_chunk()], next_nodes=("writer",))
    run = await create_run("s1", {"topic": "x"}, hitl=True)
    run.start(paused, {})
    await run.task
    assert run.next_event_id == 3

    run.reopen()
    assert run.state == "pending"

    resumed = FakeGraph([_node_chunk(node="writer", final_report={"title": "T"})])
    run.start(resumed, {})
    await run.task

    # The second segment appends rather than restarting the numbering, so a
    # client reconnecting with Last-Event-ID: 2 gets exactly the new events.
    replayed = await _drain(run, start=_replay_cursor("2", None))
    assert [e["id"] for e in replayed] == ["3", "4", "5"]
    assert [e["event"] for e in replayed] == ["node_update", "final_report", "done"]


async def test_resume_passes_no_initial_state():
    """A resumed run must continue from the checkpoint, not re-seed the graph."""
    captured = {}

    class RecordingGraph(FakeGraph):
        async def astream(self, initial_state, config, stream_mode=None):
            captured["initial_state"] = initial_state
            async for chunk in super().astream(initial_state, config, stream_mode):
                yield chunk

    run = await create_run("s1", {"topic": "x"}, hitl=True)
    run.start(RecordingGraph([_node_chunk()], next_nodes=("writer",)), {})
    await run.task
    assert captured["initial_state"] == {"topic": "x"}

    run.reopen()
    run.start(RecordingGraph([_node_chunk()]), {})
    await run.task
    assert captured["initial_state"] is None


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


# ---------------------------------------------------------------------------
# Credential lifetime
# ---------------------------------------------------------------------------

async def test_credentials_released_when_the_run_finishes():
    bind_session("s1", SessionCredentials(anthropic_api_key="k"))
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(FakeGraph([_node_chunk()]), {})
    await run.task

    assert "s1" not in session_ctx._registry


async def test_credentials_survive_an_hitl_pause():
    """The resume still needs to call the LLM, so the key must stay bound."""
    bind_session("s1", SessionCredentials(anthropic_api_key="k"))
    run = await create_run("s1", {"topic": "x"}, hitl=True)
    run.start(FakeGraph([_node_chunk()], next_nodes=("writer",)), {})
    await run.task

    assert session_ctx._registry["s1"][0].anthropic_api_key == "k"


async def test_credentials_released_on_failure():
    bind_session("s1", SessionCredentials(anthropic_api_key="k"))
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(FakeGraph([], raises=RuntimeError("boom")), {})
    await run.task

    assert "s1" not in session_ctx._registry


async def test_forget_run_cancels_and_releases():
    bind_session("s1", SessionCredentials(anthropic_api_key="k"))
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(FakeGraph([_node_chunk()], gate=asyncio.Event()), {})

    await forget_run("s1")

    assert await get_run("s1") is None
    assert "s1" not in session_ctx._registry
    assert run.task.done()


async def test_run_task_resolves_its_own_session_credentials():
    """The scope belongs to the task, not to whichever connection is attached."""
    from research_swarm.runtime.session_ctx import resolve_api_key

    seen = {}

    class ProbingGraph(FakeGraph):
        async def astream(self, initial_state, config, stream_mode=None):
            seen["key"] = resolve_api_key("anthropic")
            for chunk in self.chunks:
                yield chunk

    bind_session("s1", SessionCredentials(anthropic_api_key="run-scoped-key"))
    run = await create_run("s1", {"topic": "x"}, hitl=False)
    run.start(ProbingGraph([_node_chunk()]), {})
    await run.task

    assert seen["key"] == "run-scoped-key"


# ---------------------------------------------------------------------------
# Cursor parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("last_event_id", "from_", "expected"),
    [
        (None, None, 0),      # fresh connection
        ("4", None, 5),       # resume after the last event seen
        (None, 7, 7),         # explicit ?from= wins for a post-reload rejoin
        ("4", 7, 7),
        ("garbage", None, 0), # malformed header -> full replay, not a crash
        ("-3", None, 0),      # never negative
    ],
)
def test_replay_cursor(last_event_id, from_, expected):
    assert _replay_cursor(last_event_id, from_) == expected
