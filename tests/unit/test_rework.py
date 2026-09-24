"""Reviewer-requested re-research from the HITL pause (graph/rework.py).

The graph pauses before the writer, so the old "Edit & Re-research" (set human_feedback, resume)
could only ever run the writer. These tests drive a real compiled graph -- real dispatch_node,
route_from_dispatch and collect_node; only the LLM-backed nodes are stubbed -- through
pause -> request_rework -> resume and check it researches the weak sub-question again, steered by
the reviewer's text, and comes back to the same pause without writing.
"""
from __future__ import annotations

from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

import research_swarm.graph.nodes as _nodes
from research_swarm.graph.builder import _serde, build_graph, get_thread_config
from research_swarm.graph.rework import request_rework
from research_swarm.schemas import Critique, CritiqueVerdict, Finding, Source
from research_swarm.schemas.query import ResearchDepth, ResearchQuery
from tests.unit.test_graph import _make_plan, _make_state


async def _run(both_supported: bool = False, feedback: str = "focus on cost"):
    plan = _make_plan(2)
    sq1, sq2 = plan.sub_questions
    worker_calls: list[tuple[str, str]] = []
    rounds = {"n": 0}

    async def fake_supervisor(state):
        return {"next_agent": "dispatch", "iteration_count": 1, "plan": plan, "messages": []}

    async def fake_scout(state):
        return {"messages": []}

    async def fake_worker(state):
        sq = state.get("active_sub_question")
        if not sq:
            return {"messages": []}
        worker_calls.append((sq, state.get("search_query", "")))
        return {"findings": [Finding(
            id=f"{sq}-{len(worker_calls)}", claim=f"fact about {sq}", sub_question=sq,
            evidence=[Source(url=f"http://x/{len(worker_calls)}", title="t", snippet="s")],
        )], "messages": []}

    async def fake_verifier(state):
        rounds["n"] += 1
        crits = [
            Critique(finding_id=f.id, reasoning="r", verdict=(
                CritiqueVerdict.supported if both_supported or f.sub_question == sq1
                else CritiqueVerdict.weak))
            for f in state.get("findings") or []
        ]
        return {"critiques": crits, "messages": []}

    async def fake_writer(state):  # pragma: no cover
        raise AssertionError("re-research must return to the review pause, not write")

    with patch.object(_nodes, "supervisor_node", fake_supervisor), \
         patch.object(_nodes, "paper_scout_node", fake_scout), \
         patch.object(_nodes, "worker_node", fake_worker), \
         patch.object(_nodes, "verifier_node", fake_verifier), \
         patch.object(_nodes, "writer_node", fake_writer):
        graph = build_graph(checkpointer=MemorySaver(serde=_serde), interrupt_before_writer=True)
        config = get_thread_config(f"rework-{both_supported}")
        initial = _make_state(query=ResearchQuery(topic="t", depth=ResearchDepth.shallow))
        await graph.ainvoke(initial, config)
        first_pass = list(worker_calls)
        assert (await graph.aget_state(config)).next == ("writer",)

        await request_rework(graph, config, feedback)
        assert (await graph.aget_state(config)).next == ("dispatch_node",)
        await graph.ainvoke(None, config)
        snap = await graph.aget_state(config)

    return sq1, sq2, first_pass, worker_calls[len(first_pass):], snap, rounds["n"]


async def test_rework_researches_the_weak_sub_question_and_pauses_again():
    sq1, sq2, first_pass, rework, snap, verifier_runs = await _run()
    assert {sq for sq, _ in first_pass} == {sq1, sq2}
    assert [sq for sq, _ in rework] == [sq2]                  # only the weak one
    assert rework[0][1].endswith("focus cost")               # steered by the reviewer
    assert snap.next == ("writer",)                          # back at the review pause
    assert verifier_runs == 2                                # new facts were verified
    assert snap.values.get("rework_instructions") is None    # one round only


async def test_rework_with_everything_supported_researches_all_sub_questions():
    sq1, sq2, _first, rework, snap, _ = await _run(both_supported=True, feedback="")
    assert sorted(sq for sq, _ in rework) == sorted([sq1, sq2])
    assert all(not q.endswith(" ") for _, q in rework)       # no steer text appended
    assert snap.next == ("writer",)
