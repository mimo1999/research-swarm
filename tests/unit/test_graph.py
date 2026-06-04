"""Unit tests for Phase 4 graph — nodes, edges, routing, and stop signal.

All LLM calls are replaced with AsyncMock / MagicMock.
No API keys or network access required.

Phase 4 topology:
  START → supervisor  (plan creation only)
          ↓ always "dispatch_node"
        dispatch_node  (records pre-round IDs, fans out via Send)
          ↓ Send × N
        worker_node  (per-sub-question role-aware researcher)
          ↓ all join
        collect_node  (stop-signal check)
          ├─ stop  → critic → fact_checker → writer → END
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
    def test_append_new_finding(self):
        from research_swarm.schemas.state import _merge_findings
        f1 = _make_finding("q1")
        f2 = _make_finding("q2")
        assert len(_merge_findings([f1], [f2])) == 2

    def test_overwrite_by_id(self):
        from research_swarm.schemas.state import _merge_findings
        f1 = _make_finding("q1", confidence=0.4)
        f2 = f1.model_copy(update={"confidence": 0.9})
        result = _merge_findings([f1], [f2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_empty_existing(self):
        from research_swarm.schemas.state import _merge_findings
        assert len(_merge_findings([], [_make_finding()])) == 1

    def test_empty_new(self):
        from research_swarm.schemas.state import _merge_findings
        f = _make_finding()
        assert len(_merge_findings([f], [])) == 1


# ---------------------------------------------------------------------------
# graph/edges.py — Phase 4 routing functions
# ---------------------------------------------------------------------------

class TestRoutingEdges:
    def test_route_from_supervisor_returns_dispatch(self):
        from research_swarm.graph.edges import route_from_supervisor
        # After plan creation, supervisor always routes to dispatch_node
        for na in ("dispatch", "researcher", "critic", "writer", None):
            result = route_from_supervisor(_make_state(next_agent=na))
            assert result == "dispatch_node", f"Expected dispatch_node, got {result!r} for next_agent={na!r}"

    def test_route_from_supervisor_respects_end(self):
        from langgraph.graph import END

        from research_swarm.graph.edges import route_from_supervisor
        assert route_from_supervisor(_make_state(next_agent="end")) == END

    def test_route_from_collect_dispatch(self):
        from research_swarm.graph.edges import route_from_collect
        assert route_from_collect(_make_state(next_agent="dispatch")) == "dispatch_node"

    def test_route_from_collect_critic(self):
        from research_swarm.graph.edges import route_from_collect
        assert route_from_collect(_make_state(next_agent="critic")) == "critic"

    def test_route_from_collect_default_critic(self):
        from research_swarm.graph.edges import route_from_collect
        # Unset next_agent defaults to critic (stop)
        assert route_from_collect(_make_state(next_agent=None)) == "critic"


# ---------------------------------------------------------------------------
# graph/stop.py — stop signal
# ---------------------------------------------------------------------------

class TestStopSignal:
    def test_first_round_never_stops(self):
        from research_swarm.graph.stop import should_stop
        f = _make_finding("q1")
        stop, reason = should_stop(
            pre_dispatch_finding_ids=[],
            all_findings=[f],
            research_rounds=1,
            max_rounds=3,
        )
        assert not stop
        assert "first round" in reason

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

    def test_no_new_findings_stops(self):
        from research_swarm.graph.stop import should_stop
        f = _make_finding("q1")
        # All findings are pre-existing — nothing new produced
        stop, reason = should_stop(
            pre_dispatch_finding_ids=[f.id],
            all_findings=[f],
            research_rounds=1,
            max_rounds=5,
        )
        assert stop
        assert "no new findings" in reason

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

    def test_sufficient_novelty_continues(self):
        from research_swarm.graph.stop import should_stop
        existing = [_make_finding("q1")]
        new_findings = [_make_finding("q_new1"), _make_finding("q_new2")]
        stop, _ = should_stop(
            pre_dispatch_finding_ids=[existing[0].id],
            all_findings=existing + new_findings,
            research_rounds=1,
            max_rounds=5,
            novelty_threshold=0.15,
            similarity_threshold=0.99,  # very high so it doesn't fire
        )
        assert not stop


# ---------------------------------------------------------------------------
# nodes.py — supervisor_node
# ---------------------------------------------------------------------------

class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_creates_plan_on_first_call(self):
        from research_swarm.agents.supervisor import SupervisorDecision
        mock_decision = SupervisorDecision(
            reasoning="Creating plan",
            next_agent="dispatch",
            plan=_make_plan(),
        )
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes.run_supervisor", new=AsyncMock(return_value=mock_decision)):
            from research_swarm.graph.nodes import supervisor_node
            result = await supervisor_node(_make_state())

        assert result["next_agent"] == "dispatch"
        assert result["plan"] is not None
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_skips_llm_when_plan_exists(self):
        """If a plan already exists, supervisor returns 'dispatch' without any LLM call."""
        with patch("research_swarm.graph.nodes._get_tiered_state_llm") as mock_llm_fn, \
             patch("research_swarm.graph.nodes.run_supervisor") as mock_run_sup:
            from research_swarm.graph.nodes import supervisor_node
            state = _make_state(plan=_make_plan(), iteration_count=1)
            result = await supervisor_node(state)

        # LLM factory and run_supervisor must NOT be called
        mock_llm_fn.assert_not_called()
        mock_run_sup.assert_not_called()
        assert result["next_agent"] == "dispatch"
        assert "plan" not in result  # plan not re-written when it already exists


# ---------------------------------------------------------------------------
# nodes.py — dispatch_node + route_from_dispatch
# ---------------------------------------------------------------------------

class TestDispatchNode:
    @pytest.mark.asyncio
    async def test_records_pre_dispatch_finding_ids(self):
        f1 = _make_finding("q1")
        f2 = _make_finding("q2")
        state = _make_state(plan=_make_plan(2), findings=[f1, f2], research_rounds=0)

        from research_swarm.graph.nodes import dispatch_node
        result = await dispatch_node(state)

        assert set(result["pre_dispatch_finding_ids"]) == {f1.id, f2.id}

    @pytest.mark.asyncio
    async def test_no_plan_forces_writer(self):
        from research_swarm.graph.nodes import dispatch_node
        result = await dispatch_node(_make_state(plan=None))
        assert result["next_agent"] == "writer"

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

    def test_route_from_dispatch_only_targets_weak_on_round_1(self):
        """Second dispatch should only re-research weak/refuted findings."""
        from research_swarm.graph.nodes import route_from_dispatch

        plan = _make_plan(2)
        f_good = _make_finding(plan.sub_questions[0], confidence=0.9)
        f_weak = _make_finding(plan.sub_questions[1], confidence=0.3)
        c_good = _make_critique(f_good.id, CritiqueVerdict.supported)
        c_weak = _make_critique(f_weak.id, CritiqueVerdict.weak)

        state = _make_state(
            plan=plan,
            findings=[f_good, f_weak],
            critiques=[c_good, c_weak],
            research_rounds=1,
        )
        sends = route_from_dispatch(state)

        # Only the weak sub-question should be re-dispatched
        sub_questions = [s.arg.get("active_sub_question") for s in sends]
        assert plan.sub_questions[1] in sub_questions
        assert plan.sub_questions[0] not in sub_questions

    def test_route_from_dispatch_no_targets_returns_collect_send(self):
        """When all sub-questions are answered, route straight to collect_node."""
        from research_swarm.graph.nodes import route_from_dispatch

        plan = _make_plan(1)
        f = _make_finding(plan.sub_questions[0])
        c = _make_critique(f.id, CritiqueVerdict.supported)
        state = _make_state(plan=plan, findings=[f], critiques=[c], research_rounds=1)
        sends = route_from_dispatch(state)

        assert len(sends) == 1
        assert sends[0].node == "collect_node"


# ---------------------------------------------------------------------------
# nodes.py — worker_node
# ---------------------------------------------------------------------------

class TestWorkerNode:
    @pytest.mark.asyncio
    async def test_worker_calls_run_worker_with_role(self):
        from research_swarm.graph.nodes import worker_node

        f = _make_finding("Sub-question 1")
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]), \
             patch("research_swarm.graph.nodes.run_worker", new=AsyncMock(return_value=f)):
            state = _make_state(
                plan=_make_plan(),
                active_sub_question="Sub-question 1",
                active_worker_role="academic",
            )
            result = await worker_node(state)

        assert result["findings"] == [f]

    @pytest.mark.asyncio
    async def test_worker_noop_when_no_sub_question(self):
        from research_swarm.graph.nodes import worker_node
        result = await worker_node(_make_state(active_sub_question=None))
        assert result.get("findings") is None
        assert "[Worker] No sub-question" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_worker_handles_unknown_role_gracefully(self):
        from research_swarm.graph.nodes import worker_node
        f = _make_finding("q1")
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]), \
             patch("research_swarm.graph.nodes.run_worker", new=AsyncMock(return_value=f)):
            state = _make_state(active_sub_question="q1", active_worker_role="nonexistent_role")
            result = await worker_node(state)
        # Should not crash — falls back to general
        assert result["findings"] == [f]


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
        assert result["next_agent"] == "critic"
        assert result["research_rounds"] == settings.max_research_rounds_shallow + 1

    @pytest.mark.asyncio
    async def test_continues_when_first_round(self):
        from research_swarm.graph.nodes import collect_node
        f = _make_finding("q1")
        state = _make_state(
            findings=[f],
            pre_dispatch_finding_ids=[],   # first round: no pre-existing IDs
            research_rounds=0,
            query=ResearchQuery(topic="test", depth="standard"),
        )
        result = await collect_node(state)
        assert result["next_agent"] == "dispatch"

    @pytest.mark.asyncio
    async def test_human_feedback_overrides_stop(self):
        """human_feedback should force another dispatch round even when stop signal fires."""
        from research_swarm.graph.nodes import collect_node
        f = _make_finding("q1")
        state = _make_state(
            findings=[f],
            pre_dispatch_finding_ids=[f.id],
            research_rounds=99,  # well past any limit
            human_feedback="Please research topic X more thoroughly.",
            query=ResearchQuery(topic="test", depth="standard"),
        )
        result = await collect_node(state)
        assert result["next_agent"] == "dispatch"
        assert result.get("human_feedback") is None  # consumed

    @pytest.mark.asyncio
    async def test_increments_research_rounds(self):
        from research_swarm.graph.nodes import collect_node
        state = _make_state(research_rounds=1, pre_dispatch_finding_ids=[])
        result = await collect_node(state)
        assert result["research_rounds"] == 2


# ---------------------------------------------------------------------------
# nodes.py — critic_node / fact_checker_node / writer_node
# ---------------------------------------------------------------------------

class TestCriticNode:
    @pytest.mark.asyncio
    async def test_returns_critiques(self):
        f = _make_finding("q1")
        c = _make_critique(f.id, CritiqueVerdict.supported)
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes.run_critic", new=AsyncMock(return_value=[c])):
            from research_swarm.graph.nodes import critic_node
            result = await critic_node(_make_state(findings=[f]))
        assert result["critiques"] == [c]

class TestFactCheckerNode:
    @pytest.mark.asyncio
    async def test_returns_updated_findings(self):
        f = _make_finding("q1", confidence=0.5)
        updated = f.model_copy(update={"confidence": 0.9})
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes.run_fact_checker", new=AsyncMock(return_value=[updated])):
            from research_swarm.graph.nodes import fact_checker_node
            result = await fact_checker_node(_make_state(findings=[f]))
        assert result["findings"][0].confidence == pytest.approx(0.9)

class TestWriterNode:
    @pytest.mark.asyncio
    async def test_returns_final_report(self):
        report = FinalReport(title="Test", exec_summary="Summary.")
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=_mock_llm()), \
             patch("research_swarm.graph.nodes.run_writer", new=AsyncMock(return_value=report)):
            from research_swarm.graph.nodes import writer_node
            result = await writer_node(_make_state(findings=[_make_finding()]))
        assert result["final_report"] == report
        assert result["writer_instructions"] is None


# ---------------------------------------------------------------------------
# graph/builder.py — graph compilation
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_builds_without_error(self):
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        assert build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False) is not None

    def test_interrupt_before_writer_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        assert build_graph(checkpointer=MemorySaver(), interrupt_before_writer=True) is not None

    def test_all_phase4_nodes_present(self):
        from research_swarm.graph.builder import build_graph
        g = build_graph(interrupt_before_writer=False)
        nodes = set(g.nodes.keys())
        for required in ("supervisor", "dispatch_node", "worker_node", "collect_node",
                         "critic", "fact_checker", "writer"):
            assert required in nodes, f"Missing node: {required}"

    def test_get_thread_config_format(self):
        from research_swarm.graph.builder import get_thread_config
        assert get_thread_config("sess") == {"configurable": {"thread_id": "sess"}}

    def test_default_build_graph_uses_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        assert isinstance(build_graph().checkpointer, MemorySaver)


# ---------------------------------------------------------------------------
# Async checkpointer guards
# ---------------------------------------------------------------------------

class TestAsyncCheckpointer:
    @pytest.mark.asyncio
    async def test_astream_works_with_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            graph = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False)
            chunks = [c async for c in graph.astream(_make_state(), get_thread_config("t1"), stream_mode="updates")]
        finally:
            _nodes.supervisor_node = orig
        assert len(chunks) >= 1

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

    @pytest.mark.asyncio
    async def test_sync_sqlite_saver_raises_on_astream(self):
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            saver = SqliteSaver(conn)
            graph = build_graph(checkpointer=saver, interrupt_before_writer=False)
            with pytest.raises(Exception, match="(?i)(SqliteSaver|async|synchronous)"):
                async for _ in graph.astream(_make_state(), get_thread_config("t3"), stream_mode="updates"):
                    pass
        finally:
            _nodes.supervisor_node = orig
            conn.close()

    @pytest.mark.asyncio
    async def test_make_async_checkpointer_returns_async_sqlite_saver(self):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from research_swarm.graph.builder import make_async_checkpointer
        saver = await make_async_checkpointer()
        assert isinstance(saver, AsyncSqliteSaver)
        await saver.conn.close()


# ---------------------------------------------------------------------------
# agents/supervisor.py
# ---------------------------------------------------------------------------

class TestRunSupervisor:
    @pytest.mark.asyncio
    async def test_creates_plan_with_llm(self):
        from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
        mock_decision = SupervisorDecision(
            reasoning="Start", next_agent="dispatch", plan=_make_plan(),
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_decision)

        result = await run_supervisor(_make_state(), mock_llm)

        assert result.plan is not None
        assert result.next_agent == "dispatch"

    @pytest.mark.asyncio
    async def test_no_llm_call_when_plan_exists(self):
        """If plan already exists, run_supervisor must not call the LLM."""
        from research_swarm.agents.supervisor import run_supervisor
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock()

        state = _make_state(plan=_make_plan())
        result = await run_supervisor(state, mock_llm)

        mock_llm.with_structured_output.return_value.ainvoke.assert_not_called()
        assert result.next_agent == "dispatch"

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

    @pytest.mark.asyncio
    async def test_prompt_contains_topic(self):
        from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
        captured = []

        async def fake_ainvoke(messages):
            captured.extend(messages)
            return SupervisorDecision(reasoning="ok", next_agent="dispatch")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=fake_ainvoke)

        await run_supervisor(_make_state(), mock_llm)
        assert any("AI safety" in str(m.content) for m in captured)


# ---------------------------------------------------------------------------
# agents/critic.py
# ---------------------------------------------------------------------------

class TestRunCritic:
    @pytest.mark.asyncio
    async def test_reviews_unreviewed_findings(self):
        from research_swarm.agents.critic import run_critic
        f = _make_finding("q1")
        c = _make_critique(f.id, CritiqueVerdict.supported)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=c)
        result = await run_critic(_make_state(findings=[f]), mock_llm)
        assert len(result) == 1
        assert result[0].finding_id == f.id

    @pytest.mark.asyncio
    async def test_skips_already_supported_findings(self):
        from research_swarm.agents.critic import run_critic
        f = _make_finding("q1")
        c = _make_critique(f.id, CritiqueVerdict.supported)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock()
        result = await run_critic(_make_state(findings=[f], critiques=[c]), mock_llm)
        assert result == []


# ---------------------------------------------------------------------------
# agents/fact_checker.py
# ---------------------------------------------------------------------------

class TestRunFactChecker:
    @pytest.mark.asyncio
    async def test_updates_confidence(self):
        from research_swarm.agents.fact_checker import FactCheckResult, run_fact_checker
        from research_swarm.schemas.source import Source, SourceType
        f = _make_finding("q1", confidence=0.4)
        src = Source(url="https://x.com", snippet="Evidence", source_type=SourceType.web)
        f = f.model_copy(update={"evidence": [src]})
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=FactCheckResult(confidence_score=0.85, notes="ok")
        )
        updated = await run_fact_checker(_make_state(findings=[f]), mock_llm)
        assert updated[0].confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_skips_refuted_findings(self):
        from research_swarm.agents.fact_checker import run_fact_checker
        f = _make_finding("q1")
        c = _make_critique(f.id, CritiqueVerdict.refuted)
        mock_llm = MagicMock()
        result = await run_fact_checker(_make_state(findings=[f], critiques=[c]), mock_llm)
        assert result == []

    @pytest.mark.asyncio
    async def test_penalises_findings_without_evidence(self):
        from research_swarm.agents.fact_checker import run_fact_checker
        f = Finding(claim="Unsupported", evidence=[], confidence=0.7, sub_question="q1")
        updated = await run_fact_checker(_make_state(findings=[f]), MagicMock())
        assert updated[0].confidence < 0.4


# ---------------------------------------------------------------------------
# agents/writer.py
# ---------------------------------------------------------------------------

class TestRunWriter:
    @pytest.mark.asyncio
    async def test_produces_final_report(self):
        from research_swarm.agents.writer import run_writer
        f = _make_finding("q1", confidence=0.8)
        report = FinalReport(title="AI Safety Report", exec_summary="Summary.")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=report)
        result = await run_writer(_make_state(findings=[f], plan=_make_plan()), mock_llm)
        assert result.title == "AI Safety Report"

    @pytest.mark.asyncio
    async def test_empty_findings_returns_insufficient_report(self):
        from research_swarm.agents.writer import run_writer
        result = await run_writer(_make_state(findings=[]), MagicMock())
        assert "Insufficient" in result.exec_summary


# ---------------------------------------------------------------------------
# Phase 4 end-to-end pipeline
# ---------------------------------------------------------------------------

class TestResearchPipelinePhase4:
    """Full pipeline: supervisor → dispatch → worker(s) → collect → critic → fact_checker → writer."""

    @pytest.mark.asyncio
    async def test_full_phase4_pipeline_produces_report(self):
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import _serde, build_graph, get_thread_config
        from research_swarm.schemas.report import ReportSection

        plan     = _make_plan(2)
        finding1 = _make_finding(plan.sub_questions[0], confidence=0.8)
        finding2 = _make_finding(plan.sub_questions[1], confidence=0.75)
        critique1 = _make_critique(finding1.id, CritiqueVerdict.supported)
        critique2 = _make_critique(finding2.id, CritiqueVerdict.supported)
        fc1 = finding1.model_copy(update={"confidence": 0.9})
        fc2 = finding2.model_copy(update={"confidence": 0.88})
        report = FinalReport(
            title="AI Safety Research",
            exec_summary="Comprehensive findings on AI safety.",
            sections=[ReportSection(heading="Overview", body_md="...", citations=[])],
            references=[],
        )

        async def fake_supervisor_node(state):
            return {
                "next_agent": "dispatch",
                "iteration_count": 1,
                "plan": plan,
                "messages": [],
            }

        # worker_node gets called once per sub-question (in parallel)
        worker_calls = []
        async def fake_worker_node(state):
            sq = state.get("active_sub_question", "")
            worker_calls.append(sq)
            f = finding1 if sq == plan.sub_questions[0] else finding2
            return {"findings": [f], "messages": []}

        async def fake_collect_node(state):
            return {"next_agent": "critic", "research_rounds": 1, "messages": []}

        async def fake_critic_node(state):
            return {"critiques": [critique1, critique2], "messages": []}

        async def fake_fact_checker_node(state):
            return {"findings": [fc1, fc2], "messages": []}

        async def fake_writer_node(state):
            return {"final_report": report, "draft_report": report,
                    "writer_instructions": None, "messages": []}

        with patch.object(_nodes, "supervisor_node",    fake_supervisor_node), \
             patch.object(_nodes, "worker_node",        fake_worker_node), \
             patch.object(_nodes, "collect_node",       fake_collect_node), \
             patch.object(_nodes, "critic_node",        fake_critic_node), \
             patch.object(_nodes, "fact_checker_node",  fake_fact_checker_node), \
             patch.object(_nodes, "writer_node",        fake_writer_node):

            graph  = build_graph(checkpointer=MemorySaver(serde=_serde), interrupt_before_writer=False)
            config = get_thread_config("phase4-pipeline")
            initial = {**_make_state(), "query": ResearchQuery(topic="AI safety", audience="technical")}
            final   = await graph.ainvoke(initial, config)

        assert final["final_report"] is not None
        assert final["final_report"].title == "AI Safety Research"
        # Both workers should have been called (parallel dispatch)
        assert set(worker_calls) == set(plan.sub_questions)
        # Findings and critiques should be present
        assert len(final["findings"]) == 2
        assert len(final["critiques"]) == 2


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
            return {"next_agent": "critic", "research_rounds": 1, "messages": []}

        async def fake_critic(state):
            return {"critiques": [critique], "messages": []}

        async def fake_fc(state):
            return {"findings": [fc_finding], "messages": []}

        async def fake_writer(state):  # pragma: no cover
            raise AssertionError("Writer must not fire before HITL approval")

        with patch.object(_nodes, "supervisor_node",    fake_supervisor), \
             patch.object(_nodes, "worker_node",        fake_worker), \
             patch.object(_nodes, "collect_node",       fake_collect), \
             patch.object(_nodes, "critic_node",        fake_critic), \
             patch.object(_nodes, "fact_checker_node",  fake_fc), \
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
            return {"next_agent": "critic", "research_rounds": 1, "messages": []}

        async def fake_critic(state):
            return {"critiques": [critique], "messages": []}

        async def fake_fc(state):
            return {"findings": [fc_finding], "messages": []}

        async def fake_writer(state):
            writer_states.append(dict(state))
            return {"final_report": report, "draft_report": report,
                    "writer_instructions": None, "messages": []}

        with patch.object(_nodes, "supervisor_node",    fake_supervisor), \
             patch.object(_nodes, "worker_node",        fake_worker), \
             patch.object(_nodes, "collect_node",       fake_collect), \
             patch.object(_nodes, "critic_node",        fake_critic), \
             patch.object(_nodes, "fact_checker_node",  fake_fc), \
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
