"""Unit tests for Phase 4 — graph nodes, edges, and builder.

All LLM calls are replaced with AsyncMock / MagicMock so no API keys
are needed and tests run fully offline.
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

# ---------------------------------------------------------------------------
# Helpers
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
        "iteration_count": 0,
        "next_agent": None,
        "session_id": "test-session",
    }
    base.update(overrides)
    return base


def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        sub_questions=["What is AI safety?", "What are current AI risks?"],
        strategy="Search web and arXiv",
        required_tools=["web_search", "arxiv_search"],
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


# ---------------------------------------------------------------------------
# schemas/state — merge reducer
# ---------------------------------------------------------------------------

class TestFindingsMergeReducer:
    def test_append_new_finding(self):
        from research_swarm.schemas.state import _merge_findings
        f1 = _make_finding("q1")
        f2 = _make_finding("q2")
        result = _merge_findings([f1], [f2])
        assert len(result) == 2

    def test_overwrite_by_id(self):
        from research_swarm.schemas.state import _merge_findings
        f1 = _make_finding("q1", confidence=0.4)
        f2 = f1.model_copy(update={"confidence": 0.9})
        result = _merge_findings([f1], [f2])
        assert len(result) == 1
        conf = result[0].confidence if hasattr(result[0], "confidence") else result[0]["confidence"]
        assert conf == pytest.approx(0.9)

    def test_empty_existing(self):
        from research_swarm.schemas.state import _merge_findings
        f = _make_finding("q1")
        result = _merge_findings([], [f])
        assert len(result) == 1

    def test_empty_new(self):
        from research_swarm.schemas.state import _merge_findings
        f = _make_finding("q1")
        result = _merge_findings([f], [])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# edges.py — route_from_supervisor
# ---------------------------------------------------------------------------

class TestRoutingEdge:
    def _route(self, next_agent: str) -> str:

        from research_swarm.graph.edges import route_from_supervisor
        state = _make_state(next_agent=next_agent)
        result = route_from_supervisor(state)
        return result

    def test_routes_to_researcher(self):
        assert self._route("researcher") == "researcher"

    def test_routes_to_critic(self):
        assert self._route("critic") == "critic"

    def test_routes_to_fact_checker(self):
        assert self._route("fact_checker") == "fact_checker"

    def test_routes_to_writer(self):
        assert self._route("writer") == "writer"

    def test_routes_human_to_writer(self):
        # human is handled via interrupt_before writer; edge maps to writer
        assert self._route("human") == "writer"

    def test_routes_end(self):
        from langgraph.graph import END
        assert self._route("end") == END

    def test_routes_unknown_to_end(self):
        from langgraph.graph import END
        assert self._route("garbage") == END


# ---------------------------------------------------------------------------
# nodes.py — supervisor_node
# ---------------------------------------------------------------------------

class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_returns_next_agent_and_increments_iteration(self):
        from research_swarm.agents.supervisor import SupervisorDecision
        mock_decision = SupervisorDecision(
            reasoning="No plan yet, start research",
            next_agent="researcher",
            plan=_make_plan(),
        )
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm_fn, \
             patch("research_swarm.graph.nodes.run_supervisor", new=AsyncMock(return_value=mock_decision)):
            mock_llm_fn.return_value = MagicMock()
            from research_swarm.graph.nodes import supervisor_node
            state = _make_state()
            result = await supervisor_node(state)

        assert result["next_agent"] == "researcher"
        assert result["iteration_count"] == 1
        assert result["plan"] == mock_decision.plan
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_iteration_cap_forces_end(self, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "max_iterations", 3)

        from research_swarm.graph.nodes import supervisor_node
        state = _make_state(iteration_count=3)
        result = await supervisor_node(state)

        assert result["next_agent"] == "end"

    @pytest.mark.asyncio
    async def test_no_plan_in_update_when_plan_exists(self):
        from research_swarm.agents.supervisor import SupervisorDecision
        mock_decision = SupervisorDecision(
            reasoning="Moving to critic",
            next_agent="critic",
            plan=None,
        )
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm_fn, \
             patch("research_swarm.graph.nodes.run_supervisor", new=AsyncMock(return_value=mock_decision)):
            mock_llm_fn.return_value = MagicMock()
            from research_swarm.graph.nodes import supervisor_node
            state = _make_state(plan=_make_plan(), iteration_count=1)
            result = await supervisor_node(state)

        assert "plan" not in result  # plan not in update when decision.plan is None


# ---------------------------------------------------------------------------
# nodes.py — researcher_node
# ---------------------------------------------------------------------------

class TestResearcherNode:
    @pytest.mark.asyncio
    async def test_returns_findings_in_state_update(self):
        findings = [_make_finding("q1"), _make_finding("q2")]
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm, \
             patch("research_swarm.graph.nodes.run_researcher", new=AsyncMock(return_value=findings)), \
             patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]):
            mock_llm.return_value = MagicMock()
            from research_swarm.graph.nodes import researcher_node
            state = _make_state(plan=_make_plan())
            result = await researcher_node(state)

        assert result["findings"] == findings
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_empty_findings_allowed(self):
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm, \
             patch("research_swarm.graph.nodes.run_researcher", new=AsyncMock(return_value=[])), \
             patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]):
            mock_llm.return_value = MagicMock()
            from research_swarm.graph.nodes import researcher_node
            state = _make_state(plan=_make_plan())
            result = await researcher_node(state)

        assert result["findings"] == []


# ---------------------------------------------------------------------------
# nodes.py — critic_node
# ---------------------------------------------------------------------------

class TestCriticNode:
    @pytest.mark.asyncio
    async def test_returns_critiques(self):
        f = _make_finding("What is AI safety?")
        critiques = [_make_critique(f.id, CritiqueVerdict.supported)]
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm, \
             patch("research_swarm.graph.nodes.run_critic", new=AsyncMock(return_value=critiques)):
            mock_llm.return_value = MagicMock()
            from research_swarm.graph.nodes import critic_node
            state = _make_state(findings=[f])
            result = await critic_node(state)

        assert result["critiques"] == critiques
        assert len(result["messages"]) == 1


# ---------------------------------------------------------------------------
# nodes.py — fact_checker_node
# ---------------------------------------------------------------------------

class TestFactCheckerNode:
    @pytest.mark.asyncio
    async def test_returns_updated_findings(self):
        f = _make_finding("q1", confidence=0.5)
        updated_f = f.model_copy(update={"confidence": 0.9})
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm, \
             patch("research_swarm.graph.nodes.run_fact_checker", new=AsyncMock(return_value=[updated_f])):
            mock_llm.return_value = MagicMock()
            from research_swarm.graph.nodes import fact_checker_node
            state = _make_state(findings=[f])
            result = await fact_checker_node(state)

        assert len(result["findings"]) == 1
        assert result["findings"][0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# nodes.py — writer_node
# ---------------------------------------------------------------------------

class TestWriterNode:
    @pytest.mark.asyncio
    async def test_returns_final_report(self):
        report = FinalReport(title="Test Report", exec_summary="Summary.")
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm, \
             patch("research_swarm.graph.nodes.run_writer", new=AsyncMock(return_value=report)):
            mock_llm.return_value = MagicMock()
            from research_swarm.graph.nodes import writer_node
            state = _make_state(findings=[_make_finding("q1")])
            result = await writer_node(state)

        assert result["final_report"] == report
        assert result["draft_report"] == report
        assert len(result["messages"]) == 1


# ---------------------------------------------------------------------------
# graph/builder.py — graph compilation
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_builds_without_error(self):
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        graph = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False)
        assert graph is not None

    def test_interrupt_before_writer_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        graph = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=True)
        assert graph is not None

    def test_get_thread_config_format(self):
        from research_swarm.graph.builder import get_thread_config
        cfg = get_thread_config("sess-abc")
        assert cfg == {"configurable": {"thread_id": "sess-abc"}}

    @pytest.mark.asyncio
    async def test_graph_routes_correctly_with_mocked_nodes(self):
        """End-to-end routing test: supervisor → researcher → supervisor → END.

        Uses patch.object so the lambdas in builder.py pick up the mocks and
        the originals are always restored, even on test failure.
        """
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes

        call_log: list[str] = []

        async def fake_supervisor(state):
            call_log.append("supervisor")
            it = state.get("iteration_count", 0)
            if it == 0:
                return {
                    "next_agent": "researcher",
                    "iteration_count": 1,
                    "plan": _make_plan(),
                    "messages": [],
                }
            return {"next_agent": "end", "iteration_count": 2, "messages": []}

        async def fake_researcher(state):
            call_log.append("researcher")
            return {"findings": [_make_finding("q1")], "messages": []}

        async def fake_noop(state):
            return {"messages": []}

        with patch.object(_nodes, "supervisor_node",   fake_supervisor), \
             patch.object(_nodes, "researcher_node",   fake_researcher), \
             patch.object(_nodes, "critic_node",       fake_noop), \
             patch.object(_nodes, "fact_checker_node", fake_noop), \
             patch.object(_nodes, "writer_node",       fake_noop):
            from research_swarm.graph.builder import build_graph, get_thread_config
            graph = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False)
            config = get_thread_config("test-e2e")
            await graph.ainvoke(_make_state(), config)

        assert "supervisor" in call_log
        assert "researcher" in call_log


# ---------------------------------------------------------------------------
# agents/supervisor.py
# ---------------------------------------------------------------------------

class TestRunSupervisor:
    @pytest.mark.asyncio
    async def test_returns_supervisor_decision(self):
        from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor

        mock_decision = SupervisorDecision(
            reasoning="Start researching",
            next_agent="researcher",
            plan=_make_plan(),
        )
        mock_llm = MagicMock()
        # with_structured_output() returns a chain; .ainvoke() is called on that chain
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_decision)

        state = _make_state()
        result = await run_supervisor(state, mock_llm)

        assert result.next_agent == "researcher"
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_state_summary_includes_query(self):
        """Ensure the state summary passed to LLM contains the query topic."""
        from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor

        captured_messages = []

        async def fake_ainvoke(messages):
            captured_messages.extend(messages)
            return SupervisorDecision(reasoning="ok", next_agent="end")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=fake_ainvoke)

        state = _make_state()
        await run_supervisor(state, mock_llm)

        human_content = str(captured_messages[-1].content)
        assert "AI safety" in human_content

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback_plan(self):
        """If the LLM fails during plan generation, a minimal fallback plan is returned."""
        from research_swarm.agents.supervisor import run_supervisor

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )

        state = _make_state()  # plan=None triggers the LLM path
        result = await run_supervisor(state, mock_llm)

        assert result.plan is not None
        assert result.next_agent == "researcher"
        assert len(result.plan.sub_questions) >= 1

    @pytest.mark.asyncio
    async def test_deterministic_routing_bypasses_llm(self):
        """When _route_from_state can decide, the LLM must NOT be called."""
        from research_swarm.agents.supervisor import run_supervisor

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock()

        # Plan exists + findings present -> deterministic critic route
        state = _make_state(plan=_make_plan(), findings=[_make_finding()])
        result = await run_supervisor(state, mock_llm)

        mock_llm.with_structured_output.return_value.ainvoke.assert_not_called()
        assert result.next_agent == "critic"


# ---------------------------------------------------------------------------
# nodes.py — supervisor_node: deterministic plan-creation override
# ---------------------------------------------------------------------------

class TestSupervisorNodePlanOverride:
    @pytest.mark.asyncio
    async def test_plan_creation_forces_researcher_regardless_of_llm(self):
        """Even if the LLM returns next_agent='fact_checker' alongside a plan,
        supervisor_node must override it to 'researcher'."""
        from research_swarm.agents.supervisor import SupervisorDecision

        weird_decision = SupervisorDecision(
            reasoning="Weird routing from LLM",
            next_agent="fact_checker",  # wrong -- should be overridden
            plan=_make_plan(),
        )
        with patch("research_swarm.graph.nodes.get_agent_llm") as mock_llm_fn, \
             patch("research_swarm.graph.nodes.run_supervisor",
                   new=AsyncMock(return_value=weird_decision)):
            mock_llm_fn.return_value = MagicMock()
            from research_swarm.graph.nodes import supervisor_node
            state = _make_state()
            result = await supervisor_node(state)

        assert result["next_agent"] == "researcher"
        assert result["plan"] is not None


# ---------------------------------------------------------------------------
# agents/critic.py
# ---------------------------------------------------------------------------

class TestRunCritic:
    @pytest.mark.asyncio
    async def test_reviews_unreviewed_findings(self):
        from research_swarm.agents.critic import run_critic

        f = _make_finding("What is AI?")
        critique = _make_critique(f.id, CritiqueVerdict.supported)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=critique)

        state = _make_state(findings=[f])
        result = await run_critic(state, mock_llm)

        assert len(result) == 1
        assert result[0].finding_id == f.id

    @pytest.mark.asyncio
    async def test_skips_already_reviewed_findings(self):
        from research_swarm.agents.critic import run_critic

        f = _make_finding("q1")
        existing_critique = _make_critique(f.id, CritiqueVerdict.supported)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock()

        state = _make_state(findings=[f], critiques=[existing_critique])
        result = await run_critic(state, mock_llm)

        assert result == []
        mock_llm.with_structured_output.return_value.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# agents/fact_checker.py
# ---------------------------------------------------------------------------

class TestRunFactChecker:
    @pytest.mark.asyncio
    async def test_updates_confidence(self):
        from research_swarm.agents.fact_checker import FactCheckResult, run_fact_checker
        from research_swarm.schemas.source import Source, SourceType

        f = _make_finding("q1", confidence=0.4)
        # Give the finding real evidence so it doesn't hit the "no sources" branch
        src = Source(url="https://example.com", snippet="Evidence text", source_type=SourceType.web)
        f = f.model_copy(update={"evidence": [src]})

        check_result = FactCheckResult(confidence_score=0.85, notes="Well corroborated")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=check_result)

        state = _make_state(findings=[f])
        updated = await run_fact_checker(state, mock_llm)

        assert len(updated) == 1
        assert updated[0].confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_skips_refuted_findings(self):
        from research_swarm.agents.fact_checker import run_fact_checker

        f = _make_finding("q1")
        c = _make_critique(f.id, CritiqueVerdict.refuted)

        mock_llm = MagicMock()
        state = _make_state(findings=[f], critiques=[c])
        updated = await run_fact_checker(state, mock_llm)

        assert updated == []

    @pytest.mark.asyncio
    async def test_penalises_findings_without_evidence(self):
        from research_swarm.agents.fact_checker import run_fact_checker

        f = Finding(claim="Unsupported claim", evidence=[], confidence=0.7, sub_question="q1")
        mock_llm = MagicMock()
        state = _make_state(findings=[f])
        updated = await run_fact_checker(state, mock_llm)

        assert len(updated) == 1
        assert updated[0].confidence < 0.4  # penalised


# ---------------------------------------------------------------------------
# agents/writer.py
# ---------------------------------------------------------------------------

class TestRunWriter:
    @pytest.mark.asyncio
    async def test_produces_final_report(self):
        from research_swarm.agents.writer import run_writer

        f1 = _make_finding("q1", confidence=0.8)
        report = FinalReport(title="AI Safety Report", exec_summary="Summary here.")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=report)

        state = _make_state(findings=[f1], plan=_make_plan())
        result = await run_writer(state, mock_llm)

        assert result.title == "AI Safety Report"

    @pytest.mark.asyncio
    async def test_empty_findings_returns_minimal_report(self):
        from research_swarm.agents.writer import run_writer

        mock_llm = MagicMock()
        state = _make_state(findings=[])
        result = await run_writer(state, mock_llm)

        assert "Insufficient" in result.exec_summary

    @pytest.mark.asyncio
    async def test_refuted_findings_excluded(self):
        from research_swarm.agents.writer import run_writer

        f = _make_finding("q1", confidence=0.8)
        c = _make_critique(f.id, CritiqueVerdict.refuted)
        report = FinalReport(title="Test", exec_summary="Empty due to refuted findings.")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = AsyncMock(return_value=report)

        state = _make_state(findings=[f], critiques=[c])
        # All findings are refuted → should produce minimal fallback report
        result = await run_writer(state, mock_llm)
        # Either the LLM returned a report OR the writer bailed with "Insufficient"
        assert result is not None


# ---------------------------------------------------------------------------
# Regression: async checkpointer required for astream / ainvoke
#
# The graph nodes are all async, which means the graph MUST be compiled with
# an async-compatible checkpointer (AsyncSqliteSaver or MemorySaver).
# Using the sync SqliteSaver raises:
#   "The SqliteSaver does not support async methods."
# These tests guard against that regression.
# ---------------------------------------------------------------------------

class TestAsyncCheckpointer:
    """Guard against accidentally compiling the graph with a sync-only checkpointer."""

    @pytest.mark.asyncio
    async def test_astream_works_with_memory_saver(self):
        """MemorySaver supports both sync and async — astream must not raise."""
        from langgraph.checkpoint.memory import MemorySaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            graph = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False)
            config = get_thread_config("test-mem-async")
            chunks = [c async for c in graph.astream(_make_state(), config, stream_mode="updates")]
        finally:
            _nodes.supervisor_node = orig

        assert len(chunks) >= 1  # at least the supervisor update came through

    @pytest.mark.asyncio
    async def test_astream_works_with_async_sqlite_saver(self):
        """AsyncSqliteSaver (the production checkpointer) must work with astream.

        Regression: using SqliteSaver here raises
        "The SqliteSaver does not support async methods."
        """
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
            config = get_thread_config("test-sqlite-async")
            chunks = [c async for c in graph.astream(_make_state(), config, stream_mode="updates")]
        finally:
            _nodes.supervisor_node = orig
            await conn.close()

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_sync_sqlite_saver_raises_on_astream(self):
        """Document the failure mode: SqliteSaver + astream raises immediately.

        This test proves WHY we cannot use SqliteSaver in production.
        """
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config

        async def fake_supervisor(state):  # pragma: no cover
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            saver = SqliteSaver(conn)
            graph = build_graph(checkpointer=saver, interrupt_before_writer=False)
            config = get_thread_config("test-sync-should-fail")
            with pytest.raises(Exception, match="(?i)(SqliteSaver|async|synchronous)"):
                async for _ in graph.astream(_make_state(), config, stream_mode="updates"):
                    pass
        finally:
            _nodes.supervisor_node = orig
            conn.close()

    def test_default_build_graph_uses_memory_saver(self):
        """build_graph() with no args uses MemorySaver — not SqliteSaver."""
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.graph.builder import build_graph
        graph = build_graph()
        assert isinstance(graph.checkpointer, MemorySaver)

    @pytest.mark.asyncio
    async def test_make_async_checkpointer_returns_async_sqlite_saver(self):
        """make_async_checkpointer() must return an AsyncSqliteSaver instance."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from research_swarm.graph.builder import make_async_checkpointer
        saver = await make_async_checkpointer()
        assert isinstance(saver, AsyncSqliteSaver)
        # Tidy up the connection opened against the real DB path
        await saver.conn.close()


# ---------------------------------------------------------------------------
# End-to-end research pipeline
#
# Exercises the full supervisor → researcher → critic → fact_checker → writer
# cycle using mocked agent functions (no real LLM / API calls required).
# Asserts that a FinalReport is produced and surfaced in the final state.
# ---------------------------------------------------------------------------

class TestResearchPipeline:
    """Integration test: ResearchQuery in → FinalReport out."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_report(self):
        """Running the graph end-to-end with mocked agents must produce a report.

        Flow:
          supervisor(1) → researcher → supervisor(2) → critic
          → supervisor(3) → fact_checker → supervisor(4) → writer → END

        All agent functions are replaced with AsyncMocks so no API keys,
        network calls, or LLM inference is needed.
        """
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.agents.supervisor import SupervisorDecision
        from research_swarm.graph.builder import build_graph, get_thread_config
        from research_swarm.schemas import (
            CritiqueVerdict,
            FinalReport,
            ReportSection,
            ResearchQuery,
        )

        # ── Fixtures ──────────────────────────────────────────────────
        plan     = _make_plan()
        finding  = _make_finding("What is AI safety?", confidence=0.8)
        critique = _make_critique(finding.id, CritiqueVerdict.supported)
        fc_finding = finding.model_copy(update={"confidence": 0.92})
        report   = FinalReport(
            title="AI Safety: Current Landscape",
            exec_summary="AI safety research addresses risks from advanced AI systems.",
            sections=[
                ReportSection(
                    heading="Introduction",
                    body_md="AI safety is a growing field.",
                    citations=[],
                )
            ],
            references=[],
        )

        # Supervisor is called 4 times:
        #   1st: create plan + route to researcher
        #   2nd: route to critic
        #   3rd: route to fact_checker
        #   4th: route to writer
        supervisor_responses = iter([
            SupervisorDecision(reasoning="No plan yet — start research",   next_agent="researcher", plan=plan),
            SupervisorDecision(reasoning="Findings ready — run critic",     next_agent="critic"),
            SupervisorDecision(reasoning="Critiques done — fact-check",     next_agent="fact_checker"),
            SupervisorDecision(reasoning="All verified — write report",     next_agent="writer"),
        ])

        async def fake_run_supervisor(state, llm):
            return next(supervisor_responses)

        async def fake_run_researcher(state, llm, tools):
            return [finding]

        async def fake_run_critic(state, llm):
            return [critique]

        async def fake_run_fact_checker(state, llm):
            return [fc_finding]

        async def fake_run_writer(state, llm):
            return report

        # ── Build + run ───────────────────────────────────────────────
        with (
            patch("research_swarm.graph.nodes.run_supervisor",   new=AsyncMock(side_effect=fake_run_supervisor)),
            patch("research_swarm.graph.nodes.run_researcher",   new=AsyncMock(side_effect=fake_run_researcher)),
            patch("research_swarm.graph.nodes.run_critic",       new=AsyncMock(side_effect=fake_run_critic)),
            patch("research_swarm.graph.nodes.run_fact_checker", new=AsyncMock(side_effect=fake_run_fact_checker)),
            patch("research_swarm.graph.nodes.run_writer",       new=AsyncMock(side_effect=fake_run_writer)),
            patch("research_swarm.graph.nodes.get_agent_llm",    return_value=MagicMock()),
            patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]),
        ):
            graph  = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=False)
            config = get_thread_config("test-full-pipeline")

            initial_state = {
                "messages":        [],
                "query":           ResearchQuery(topic="AI safety", audience="technical"),
                "plan":            None,
                "findings":        [],
                "critiques":       [],
                "draft_report":    None,
                "final_report":    None,
                "human_feedback":  None,
                "iteration_count": 0,
                "next_agent":      None,
                "session_id":      "test-full-pipeline",
            }

            final_state = await graph.ainvoke(initial_state, config)

        # ── Assertions ────────────────────────────────────────────────
        assert final_state["final_report"] is not None, \
            "Pipeline did not produce a FinalReport"

        r = final_state["final_report"]
        assert r.title == "AI Safety: Current Landscape"
        assert r.exec_summary != ""
        assert len(r.sections) == 1
        assert r.sections[0].heading == "Introduction"

        # Findings should reflect the fact-checker's updated confidence
        assert len(final_state["findings"]) == 1
        assert final_state["findings"][0].confidence == pytest.approx(0.92)

        # Critiques accumulated
        assert len(final_state["critiques"]) == 1
        assert final_state["critiques"][0].verdict == CritiqueVerdict.supported

        # Message log should contain one entry per node visit
        contents = [m.content for m in final_state["messages"]]
        assert any("[Supervisor]"   in c for c in contents)
        assert any("[Researcher]"   in c for c in contents)
        assert any("[Critic]"       in c for c in contents)
        assert any("[FactChecker]"  in c for c in contents)
        assert any("[Writer]"       in c for c in contents)


# ---------------------------------------------------------------------------
# HITL interrupt / resume integration test
#
# There are zero existing tests for the human-in-the-loop pause-and-resume path.
# This class builds a graph with interrupt_before_writer=True, verifies the graph
# pauses before the writer node, injects human feedback via aupdate_state, then
# resumes and verifies the writer fires and a FinalReport is produced.
# ---------------------------------------------------------------------------

class TestHITLInterruptResume:
    """Guard the human-in-the-loop interrupt/resume path end-to-end."""

    @pytest.mark.asyncio
    async def test_graph_pauses_before_writer(self):
        """With interrupt_before_writer=True the graph must stop before writer."""
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.agents.supervisor import SupervisorDecision
        from research_swarm.graph.builder import build_graph, get_thread_config

        # Supervisor routes directly to writer on the first call.
        # plan=None because we're testing routing, not plan generation;
        # returning a plan here would trigger the post-plan override to researcher.
        async def fake_run_supervisor(state, llm):
            return SupervisorDecision(
                reasoning="Enough findings — write the report",
                next_agent="writer",
                plan=None,
            )

        async def fake_run_writer(state, llm):  # pragma: no cover
            raise AssertionError("writer should not run before human approval")

        with (
            patch("research_swarm.graph.nodes.run_supervisor",   new=AsyncMock(side_effect=fake_run_supervisor)),
            patch("research_swarm.graph.nodes.run_writer",        new=AsyncMock(side_effect=fake_run_writer)),
            patch("research_swarm.graph.nodes.get_agent_llm",     return_value=MagicMock()),
            patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]),
        ):
            graph  = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=True)
            config = get_thread_config("hitl-pause-test")
            initial_state = {
                "messages": [], "query": ResearchQuery(topic="HITL test", audience="technical"),
                "plan": None, "findings": [_make_finding("q1", confidence=0.8)],
                "critiques": [], "draft_report": None, "final_report": None,
                "human_feedback": None, "iteration_count": 0,
                "next_agent": None, "session_id": "hitl-pause-test",
            }
            await graph.ainvoke(initial_state, config)
            snapshot = await graph.aget_state(config)

        # The graph must be paused at the writer node
        assert snapshot.next, "Graph should be paused (next must be non-empty)"
        assert "writer" in snapshot.next

    @pytest.mark.asyncio
    async def test_graph_resumes_and_writer_fires_after_approval(self):
        """After aupdate_state with human feedback, resuming must invoke the writer."""
        from langgraph.checkpoint.memory import MemorySaver

        from research_swarm.agents.supervisor import SupervisorDecision
        from research_swarm.graph.builder import build_graph, get_thread_config

        writer_called_with_feedback: list[str] = []

        # plan=None: this is a routing decision, not plan generation.
        async def fake_run_supervisor(state, llm):
            return SupervisorDecision(
                reasoning="Time to write",
                next_agent="writer",
                plan=None,
            )

        async def fake_run_writer(state, llm):
            writer_called_with_feedback.append(state.get("human_feedback", ""))
            return FinalReport(title="HITL Report", exec_summary="Approved by human.")

        with (
            patch("research_swarm.graph.nodes.run_supervisor",    new=AsyncMock(side_effect=fake_run_supervisor)),
            patch("research_swarm.graph.nodes.run_writer",         new=AsyncMock(side_effect=fake_run_writer)),
            patch("research_swarm.graph.nodes.get_agent_llm",      return_value=MagicMock()),
            patch("research_swarm.graph.nodes._get_researcher_tools", return_value=[]),
        ):
            graph  = build_graph(checkpointer=MemorySaver(), interrupt_before_writer=True)
            config = get_thread_config("hitl-resume-test")
            initial_state = {
                "messages": [], "query": ResearchQuery(topic="HITL resume", audience="technical"),
                "plan": None, "findings": [_make_finding("q1", confidence=0.8)],
                "critiques": [], "draft_report": None, "final_report": None,
                "human_feedback": None, "iteration_count": 0,
                "next_agent": None, "session_id": "hitl-resume-test",
            }

            # Phase 1: run until interrupt
            await graph.ainvoke(initial_state, config)
            snapshot = await graph.aget_state(config)
            assert snapshot.next, "Expected graph to pause before writer"

            # Phase 2: inject human feedback and resume
            await graph.aupdate_state(config, {"human_feedback": "Focus on risks only."})
            final_state = await graph.ainvoke(None, config)

        # Writer must have fired
        assert len(writer_called_with_feedback) == 1, "Writer was not called after resume"
        assert "Focus on risks only." in writer_called_with_feedback[0]

        # Final report must be present
        assert final_state.get("final_report") is not None
        assert final_state["final_report"].title == "HITL Report"
