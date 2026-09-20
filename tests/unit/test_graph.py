"""Unit tests for Phase 4 graph — nodes, edges, routing, and stop signal.

All LLM calls are replaced with AsyncMock / MagicMock.
No API keys or network access required.

Topology:
  START → supervisor  (plan creation only)
          ↓ always "dispatch_node"
        dispatch_node  (records pre-round IDs, fans out via Send)
          ↓ Send × N
        worker_node  (gap fill for one sub-question)
          ↓ all join
        collect_node  (stop-signal check)
          ├─ stop  → verifier → writer → END
          └─ loop  → dispatch_node
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.schemas import (
    Critique,
    CritiqueVerdict,
    FinalReport,
    Finding,
    ResearchPlan,
    ResearchQuery,
)
from research_swarm.schemas.state import AgentState
from research_swarm.schemas.worker import SubQuestionAssignment, WorkerRole

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "query": ResearchQuery(topic="AI safety", audience="technical"),
        "plan": None,
        "findings": [],
        "critiques": [],
        "draft_report": None,
        "final_report": None,
        "human_feedback": None,
        "writer_instructions": None,
        "iteration_count": 0,
        "next_agent": None,
        "session_id": "test-session",
        "model_provider": "ollama",
        "model_name": "test-model",
        "schema_version": 2,
        "research_rounds": 0,
        "pre_dispatch_finding_ids": [],
        "active_sub_question": None,
        "active_worker_role": None,
    }
    base.update(overrides)
    return base


def _make_plan(n_questions: int = 2) -> ResearchPlan:
    sqs = [f"Sub-question {i+1}" for i in range(n_questions)]
    return ResearchPlan(
        sub_questions=sqs,
        strategy="Search web and arXiv",
        required_tools=["web_search"],
        complexity_score=0.5,
        assignments=[
            SubQuestionAssignment(sub_question=sq, worker_role=WorkerRole.general)
            for sq in sqs
        ],
    )


def _make_finding(sub_q: str = "test", confidence: float = 0.7) -> Finding:
    return Finding(
        claim=f"Claim for: {sub_q}",
        evidence=[],
        confidence=confidence,
        sub_question=sub_q,
    )


def _make_critique(finding_id: str, verdict: CritiqueVerdict) -> Critique:
    return Critique(
        finding_id=finding_id,
        verdict=verdict,
        reasoning="Test reasoning",
    )


def _mock_llm():
    """Return a MagicMock that looks enough like a ChatModel for node tests."""
    llm = MagicMock()
    llm.with_config = MagicMock(return_value=llm)
    return llm


# ---------------------------------------------------------------------------
# schemas/state — merge reducer
# ---------------------------------------------------------------------------

class TestFindingsMergeReducer:

    def test_overwrite_by_id(self):
        from research_swarm.schemas.state import _merge_findings
        f1 = _make_finding("q1", confidence=0.4)
        f2 = f1.model_copy(update={"confidence": 0.9})
        result = _merge_findings([f1], [f2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# graph/edges.py — Phase 4 routing functions
# ---------------------------------------------------------------------------

class TestRoutingEdges:
    def test_route_from_supervisor_returns_document_pass(self):
        from research_swarm.graph.edges import route_from_supervisor
        # After plan creation, supervisor always routes to document_pass_node
        # (the one-time ingested-document extraction fan-out, which itself
        # bounces straight through to dispatch_node when there are no
        # ingested documents -- see route_from_document_pass).
        for na in ("dispatch", "verifier", "writer", None):
            result = route_from_supervisor(_make_state(next_agent=na))
            assert result == "document_pass_node", (
                f"Expected document_pass_node, got {result!r} for next_agent={na!r}"
            )


# ---------------------------------------------------------------------------
# graph/stop.py — stop signal
# ---------------------------------------------------------------------------

class TestStopSignal:

    def test_hard_cap_always_stops(self):
        from research_swarm.graph.stop import should_stop
        f = _make_finding("q1")
        stop, reason = should_stop(
            pre_dispatch_finding_ids=[f.id],
            all_findings=[f],
            research_rounds=3,
            max_rounds=3,
        )
        assert stop
        assert "Hard cap" in reason


    def test_low_novelty_stops(self):
        from research_swarm.graph.stop import should_stop
        # 10 existing, 1 new → 0.1 novelty rate, below threshold 0.15
        existing = [_make_finding(f"q{i}") for i in range(10)]
        new_f = _make_finding("q_new")
        stop, reason = should_stop(
            pre_dispatch_finding_ids=[f.id for f in existing],
            all_findings=existing + [new_f],
            research_rounds=2,
            max_rounds=5,
            novelty_threshold=0.15,
        )
        assert stop
        assert "novelty" in reason.lower()


# ---------------------------------------------------------------------------
# nodes.py — dispatch_node + route_from_dispatch
# ---------------------------------------------------------------------------

class TestDispatchNode:


    def test_route_from_dispatch_returns_sends_for_all_sqs_on_round_0(self):
        from langgraph.types import Send

        from research_swarm.graph.nodes import route_from_dispatch

        plan = _make_plan(3)
        state = _make_state(plan=plan, research_rounds=0)
        sends = route_from_dispatch(state)

        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)
        sub_questions = [s.arg.get("active_sub_question") for s in sends]
        assert set(sub_questions) == set(plan.sub_questions)


# ---------------------------------------------------------------------------
# nodes.py — collect_node
# ---------------------------------------------------------------------------

class TestCollectNode:
    @pytest.mark.asyncio
    async def test_stops_at_max_rounds(self):
        from research_swarm.config import settings
        from research_swarm.graph.nodes import collect_node

        f = _make_finding("q1")
        state = _make_state(
            findings=[f],
            pre_dispatch_finding_ids=[f.id],
            research_rounds=settings.max_research_rounds_shallow,
            query=ResearchQuery(topic="test", depth="shallow"),
        )
        result = await collect_node(state)
        assert result["next_agent"] == "verifier"
        assert result["research_rounds"] == settings.max_research_rounds_shallow + 1


# ---------------------------------------------------------------------------
# graph/builder.py — graph compilation
# ---------------------------------------------------------------------------

class TestGraphBuilder:


    def test_all_phase4_nodes_present(self):
        from research_swarm.graph.builder import build_graph
        g = build_graph(interrupt_before_writer=False)
        nodes = set(g.nodes.keys())
        for required in ("supervisor", "dispatch_node", "worker_node", "collect_node",
                         "verifier", "writer",
                         "document_pass_node", "document_worker_node", "paper_scout_node", "paper_worker_node"):
            assert required in nodes, f"Missing node: {required}"

    def test_get_graph_includes_the_send_fanned_edges_for_the_ui_diagram(self):
        """document_pass_node and dispatch_node route via Send-returning functions with no
        static path map; without an explicit path_map (see builder.py) LangGraph can't infer
        their targets and get_graph() silently drops the edge -- which would make the UI's live
        topology diagram (ui/graph_view.py) miss exactly the two most important fan-outs."""
        from research_swarm.graph.builder import build_graph

        g = build_graph(interrupt_before_writer=False)
        mermaid = g.get_graph().draw_mermaid()
        for edge in ("document_pass_node -.-> document_worker_node",
                     "document_pass_node -.-> paper_scout_node",
                     "dispatch_node -.-> worker_node"):
            assert edge in mermaid, mermaid


# ---------------------------------------------------------------------------
# Async checkpointer guards
# ---------------------------------------------------------------------------

class TestAsyncCheckpointer:

    @pytest.mark.asyncio
    async def test_astream_works_with_async_sqlite_saver(self):
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            conn = await aiosqlite.connect(":memory:")
            saver = AsyncSqliteSaver(conn)
            graph = build_graph(checkpointer=saver, interrupt_before_writer=False)
            chunks = [c async for c in graph.astream(_make_state(), get_thread_config("t2"), stream_mode="updates")]
        finally:
            _nodes.supervisor_node = orig
            await conn.close()
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# agents/supervisor.py
# ---------------------------------------------------------------------------

class TestRunSupervisor:


    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback_plan(self):
        from research_swarm.agents.supervisor import run_supervisor
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )
        result = await run_supervisor(_make_state(), mock_llm)
        assert result.plan is not None
        assert result.next_agent == "dispatch"


# ---------------------------------------------------------------------------
# HITL interrupt / resume
# ---------------------------------------------------------------------------

class TestHITLInterruptResume:
    @pytest.mark.asyncio
    async def test_graph_pauses_before_writer(self):
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import _serde, build_graph, get_thread_config

        plan = _make_plan(1)
        finding = _make_finding(plan.sub_questions[0], confidence=0.8)
        critique = _make_critique(finding.id, CritiqueVerdict.supported)
        fc_finding = finding.model_copy(update={"confidence": 0.9})

        async def fake_supervisor(state):
            return {"next_agent": "dispatch", "iteration_count": 1, "plan": plan, "messages": []}

        async def fake_worker(state):
            sq = state.get("active_sub_question")
            return {"findings": [finding] if sq else [], "messages": []}

        async def fake_collect(state):
            return {"next_agent": "verifier", "research_rounds": 1, "messages": []}

        async def fake_verifier(state):
            return {"critiques": [critique], "findings": [fc_finding], "messages": []}

        async def fake_writer(state):  # pragma: no cover
            raise AssertionError("Writer must not fire before HITL approval")

        async def fake_paper_scout_node(state):
            return {"messages": []}

        with patch.object(_nodes, "supervisor_node",    fake_supervisor), \
             patch.object(_nodes, "worker_node",        fake_worker), \
             patch.object(_nodes, "paper_scout_node",  fake_paper_scout_node), \
             patch.object(_nodes, "collect_node",       fake_collect), \
             patch.object(_nodes, "verifier_node",      fake_verifier), \
             patch.object(_nodes, "writer_node",        fake_writer):

            graph  = build_graph(checkpointer=MemorySaver(serde=_serde), interrupt_before_writer=True)
            config = get_thread_config("hitl-pause")
            initial = {**_make_state(), "query": ResearchQuery(topic="HITL test")}
            await graph.ainvoke(initial, config)
            snap = await graph.aget_state(config)

        assert snap.next, "Graph should be paused before writer"
        assert "writer" in snap.next

    @pytest.mark.asyncio
    async def test_graph_resumes_after_approval(self):
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import _serde, build_graph, get_thread_config

        plan = _make_plan(1)
        finding = _make_finding(plan.sub_questions[0], confidence=0.8)
        critique = _make_critique(finding.id, CritiqueVerdict.supported)
        fc_finding = finding.model_copy(update={"confidence": 0.9})
        report = FinalReport(title="HITL Report", exec_summary="Approved.")
        writer_states: list[dict] = []

        async def fake_supervisor(state):
            return {"next_agent": "dispatch", "iteration_count": 1, "plan": plan, "messages": []}

        async def fake_worker(state):
            sq = state.get("active_sub_question")
            return {"findings": [finding] if sq else [], "messages": []}

        async def fake_collect(state):
            return {"next_agent": "verifier", "research_rounds": 1, "messages": []}

        async def fake_verifier(state):
            return {"critiques": [critique], "findings": [fc_finding], "messages": []}

        async def fake_writer(state):
            writer_states.append(dict(state))
            return {"final_report": report, "draft_report": report,
                    "writer_instructions": None, "messages": []}

        async def fake_paper_scout_node(state):
            return {"messages": []}

        with patch.object(_nodes, "supervisor_node",    fake_supervisor), \
             patch.object(_nodes, "worker_node",        fake_worker), \
             patch.object(_nodes, "paper_scout_node",  fake_paper_scout_node), \
             patch.object(_nodes, "collect_node",       fake_collect), \
             patch.object(_nodes, "verifier_node",      fake_verifier), \
             patch.object(_nodes, "writer_node",        fake_writer):

            graph  = build_graph(checkpointer=MemorySaver(serde=_serde), interrupt_before_writer=True)
            config = get_thread_config("hitl-resume")
            initial = {**_make_state(), "query": ResearchQuery(topic="HITL resume")}

            # Phase 1: run until interrupt
            await graph.ainvoke(initial, config)
            snap = await graph.aget_state(config)
            assert snap.next, "Expected pause before writer"

            # Phase 2: inject approval and resume
            await graph.aupdate_state(config, {"writer_instructions": "Approve"})
            final = await graph.ainvoke(None, config)

        assert len(writer_states) == 1, "Writer should fire exactly once"
        assert final["final_report"].title == "HITL Report"
