"""Integration test: shallow research flow through the Phase 4 graph topology.

Exercises the full graph pipeline with mocked nodes (no LLM/API calls):
  supervisor → dispatch_node → worker_node(s) → collect_node → critic →
  fact_checker → writer → END

Also verifies:
  - Schema migration v0 → v2
  - Budget guard (LLM call counting per session)
  - Serialization (JsonPlusSerializer with all registered types)
  - Worker role assignment via ResearchPlan.assignments
  - Stop signal fires after one round (shallow = max_rounds=1)
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from research_swarm.runtime.budget import clear_budget, get_budget


@pytest.mark.asyncio
async def test_shallow_research_flow():
    """End-to-end shallow research through Phase 4 topology."""
    from langgraph.checkpoint.memory import MemorySaver

    import research_swarm.graph.nodes as _nodes
    from research_swarm.graph.builder import _serde, build_graph, get_thread_config
    from research_swarm.runtime.migrations import migrate_state
    from research_swarm.schemas.critique import Critique, CritiqueVerdict
    from research_swarm.schemas.finding import Finding
    from research_swarm.schemas.plan import ResearchPlan
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery
    from research_swarm.schemas.report import FinalReport, ReportSection
    from research_swarm.schemas.source import Source, SourceType
    from research_swarm.schemas.worker import SubQuestionAssignment, WorkerRole

    session_id = f"shallow-test-{uuid.uuid4().hex[:8]}"
    topic      = "Python async/await"
    sub_q      = f"What are the core concepts of {topic}?"

    # ── Verify migration v0 → v2 ─────────────────────────────────────────
    v0_state = {
        "messages": [], "query": None, "plan": None, "findings": [],
        "critiques": [], "draft_report": None, "final_report": None,
        "human_feedback": None, "session_id": session_id,
    }
    migrated = migrate_state(v0_state)
    assert migrated["schema_version"] == 2
    assert migrated["model_provider"] in ("anthropic", "openai", "ollama")
    assert migrated["research_rounds"] == 0
    assert migrated["pre_dispatch_finding_ids"] == []

    # ── Domain objects ────────────────────────────────────────────────────
    plan = ResearchPlan(
        sub_questions=[sub_q],
        strategy="Shallow research",
        required_tools=["web_search"],
        complexity_score=0.2,
        assignments=[SubQuestionAssignment(sub_question=sub_q, worker_role=WorkerRole.general)],
    )
    finding = Finding(
        id="f1",
        claim=f"{topic} allows concurrent I/O without threads via async/await",
        evidence=[Source(
            url="https://python.org/async",
            title="Python async docs",
            snippet=f"{topic} provides async/await syntax",
            source_type=SourceType.web,
            credibility_score=0.9,
        )],
        confidence=0.8,
        sub_question=sub_q,
    )
    critique = Critique(
        finding_id="f1", verdict=CritiqueVerdict.supported,
        reasoning="Evidence supports the claim.", suggested_followup="",
    )
    fc_finding = finding.model_copy(update={"confidence": 0.85})
    report = FinalReport(
        title=f"Report: {topic}",
        exec_summary=finding.claim,
        sections=[ReportSection(heading="Overview", body_md=finding.claim, citations=[1])],
        references=list(finding.evidence),
        methodology="Shallow research",
        limitations="Single tool turn",
    )

    # ── Node mocks ────────────────────────────────────────────────────────
    worker_calls: list[str] = []

    async def fake_supervisor(state):
        return {
            "next_agent": "dispatch",
            "iteration_count": 1,
            "plan": plan,
            "messages": [],
        }

    async def fake_worker(state):
        sq = state.get("active_sub_question")
        if sq:
            worker_calls.append(sq)
            return {"findings": [finding], "messages": []}
        return {"messages": []}

    async def fake_collect(state):
        # Shallow: 1 round max — stop immediately
        return {"next_agent": "critic", "research_rounds": 1, "messages": []}

    async def fake_critic(state):
        return {"critiques": [critique], "messages": []}

    async def fake_fc(state):
        return {"findings": [fc_finding], "messages": []}

    async def fake_writer(state):
        return {
            "final_report": report, "draft_report": report,
            "writer_instructions": None, "messages": [],
        }

    async def fake_fetch_worker_node(state):
        return {"messages": []}

    checkpointer = MemorySaver(serde=_serde)
    budget = get_budget(session_id, limit=25)
    calls_before = budget.used

    with (
        patch.object(_nodes, "supervisor_node",    fake_supervisor),
        patch.object(_nodes, "worker_node",        fake_worker),
        patch.object(_nodes, "fetch_worker_node",  fake_fetch_worker_node),
        patch.object(_nodes, "collect_node",       fake_collect),
        patch.object(_nodes, "critic_node",        fake_critic),
        patch.object(_nodes, "fact_checker_node",  fake_fc),
        patch.object(_nodes, "writer_node",        fake_writer),
    ):
        graph  = build_graph(checkpointer=checkpointer, interrupt_before_writer=False)
        config = get_thread_config(session_id)
        initial = {
            **migrated,
            "query": ResearchQuery(
                topic=topic, depth=ResearchDepth.shallow, max_sources=3, audience="technical",
            ),
            "plan": None, "findings": [], "critiques": [],
            "draft_report": None, "final_report": None, "human_feedback": None,
            "writer_instructions": None, "iteration_count": 0, "next_agent": None,
            "model_provider": "anthropic", "model_name": "claude-3-5-sonnet",
            "schema_version": 2, "research_rounds": 0, "pre_dispatch_finding_ids": [],
        }

        async for _ in graph.astream(initial, config, stream_mode="updates"):
            pass

        snap = await graph.aget_state(config)
        result_report = snap.values.get("final_report")
        calls_after = budget.used

    clear_budget(session_id)

    # ── Assertions ────────────────────────────────────────────────────────
    assert result_report is not None, "No final report produced"
    assert topic.lower() in result_report.title.lower()
    assert "concurrent" in result_report.exec_summary.lower()
    assert len(result_report.sections) >= 1
    assert result_report.references

    # Worker was called for the sub-question
    assert sub_q in worker_calls, f"Expected worker called for {sub_q!r}, got {worker_calls}"

    # Budget guard: mocked nodes don't invoke real LLMs → 0 LLM calls tracked
    assert calls_after - calls_before == 0

    # Findings confidence was updated by fake fact-checker
    findings = snap.values.get("findings") or []
    assert len(findings) >= 1
    mean_conf = sum(
        f.confidence if hasattr(f, "confidence") else f.get("confidence", 0)
        for f in findings
    ) / len(findings)
    assert mean_conf >= 0.5

    print("\n✓ Phase 4 shallow flow test passed")
    print(f"  Topic:       {topic}")
    print(f"  Report:      {result_report.title}")
    print(f"  Workers:     {worker_calls}")
    print(f"  Findings:    {len(findings)}")
    print(f"  Mean conf:   {mean_conf:.2f}")
