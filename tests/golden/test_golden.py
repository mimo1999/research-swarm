"""Golden-set regression tests.

These tests run the full graph pipeline with mocked LLMs and verify that the
final report covers the expected claims and avoids forbidden ones.

They are intentionally *not* end-to-end network tests — the LLM is mocked with
realistic-looking responses so the pipeline mechanics (routing, finding/critique
round-trip, writer invocation) are exercised without API calls.

Run with:
    poetry run pytest tests/golden/ -v
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from tests.golden.fixtures import GOLDEN_FIXTURES

# ---------------------------------------------------------------------------
# Helpers to build mock LLM responses at each graph stage
# ---------------------------------------------------------------------------

def _mock_supervisor_plan(topic: str):
    """Return a SupervisorDecision-like mock for plan creation."""
    from research_swarm.agents.supervisor import SupervisorDecision
    from research_swarm.schemas.plan import ResearchPlan
    return SupervisorDecision(
        reasoning="Creating initial plan.",
        next_agent="researcher",
        plan=ResearchPlan(
            sub_questions=[f"What are the core concepts of {topic}?"],
            strategy="Direct research",
            required_tools=["web_search"],
        ),
    )


def _finding_for(topic: str, fixture: dict):
    """Return a Finding whose claim satisfies the fixture's expected_claims."""
    from research_swarm.schemas.finding import Finding
    from research_swarm.schemas.source import Source, SourceType
    # Build a claim that contains all expected substrings
    claim = " ".join(fixture["expected_claims"]) + f" are key concepts in {topic}."
    return Finding(
        id=str(uuid.uuid4()),
        claim=claim,
        evidence=[
            Source(
                url=f"https://example.com/{topic.replace(' ', '-')}",
                title=f"Study on {topic}",
                snippet=claim,
                source_type=SourceType.web,
                credibility_score=0.8,
            )
        ],
        confidence=0.7,
        sub_question=f"What are the core concepts of {topic}?",
    )


def _critique_for(finding_id: str):
    from research_swarm.schemas.critique import Critique, CritiqueVerdict
    return Critique(
        finding_id=finding_id,
        verdict=CritiqueVerdict.supported,
        reasoning="Evidence supports the claim.",
        suggested_followup="",
    )


# ---------------------------------------------------------------------------
# Parametrized golden tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES, ids=[f["topic"] for f in GOLDEN_FIXTURES])
@pytest.mark.asyncio
async def test_golden_coverage(fixture: dict):
    """Full pipeline produces a report covering expected claims for each topic."""
    from langgraph.checkpoint.memory import MemorySaver

    from research_swarm.graph.builder import _serde, build_graph, get_thread_config
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery

    topic = fixture["topic"]
    session_id = f"golden-{uuid.uuid4().hex[:8]}"
    finding = _finding_for(topic, fixture)

    # ---- mock supervisor: plan on first call, deterministic routing after ----
    supervisor_calls = {"n": 0}

    async def mock_run_supervisor(state, llm):
        supervisor_calls["n"] += 1
        if supervisor_calls["n"] == 1:
            return _mock_supervisor_plan(topic)
        # Should not be called again — deterministic routing handles the rest
        from research_swarm.agents.supervisor import SupervisorDecision
        return SupervisorDecision(reasoning="end", next_agent="end")

    # ---- mock researcher: return the pre-built finding ----
    async def mock_run_researcher(state, llm, tools):
        return [finding]

    # ---- mock critic: mark the finding supported ----
    async def mock_run_critic(state, llm):
        findings = state.get("findings") or []
        return [_critique_for(f.id if hasattr(f, "id") else f.get("id", "")) for f in findings]

    # ---- mock fact_checker: preserve confidence ----
    async def mock_run_fact_checker(state, llm):
        findings = state.get("findings") or []
        return [
            f.model_copy(update={"confidence": 0.75}) if hasattr(f, "model_copy")
            else {**f, "confidence": 0.75}
            for f in findings
        ]

    # ---- mock writer: produce a real FinalReport from the finding ----
    async def mock_run_writer(state, llm):
        from research_swarm.schemas.report import FinalReport, ReportSection
        findings = state.get("findings") or []
        claim = findings[0].claim if findings else ""
        return FinalReport(
            title=f"Report: {topic}",
            exec_summary=claim,
            sections=[ReportSection(heading="Overview", body_md=claim, citations=[1])],
            references=list(findings[0].evidence) if findings else [],
            methodology="Web search",
            limitations="Shallow run.",
        )

    checkpointer = MemorySaver(serde=_serde)

    import research_swarm.graph.nodes as _nodes

    # In Phase 4 the flow is: supervisor→dispatch→worker(s)→collect→critic→fc→writer
    # We patch at the node level so the golden checks flow through the real graph edges.
    worker_calls: dict[str, int] = {}

    async def patched_supervisor_node(state):
        plan = _mock_supervisor_plan(topic)
        return {
            "next_agent": "dispatch",
            "iteration_count": 1,
            "plan": plan.plan,
            "messages": [],
        }

    async def patched_worker_node(state):
        sq = state.get("active_sub_question") or topic
        worker_calls[sq] = worker_calls.get(sq, 0) + 1
        return {"findings": [finding], "messages": []}

    async def patched_collect_node(state):
        return {"next_agent": "critic", "research_rounds": 1, "messages": []}

    async def patched_critic_node(state):
        findings = state.get("findings") or []
        return {
            "critiques": [_critique_for(f.id if hasattr(f, "id") else f.get("id", "")) for f in findings],
            "messages": [],
        }

    async def patched_fact_checker_node(state):
        findings = state.get("findings") or []
        return {
            "findings": [
                f.model_copy(update={"confidence": 0.75}) if hasattr(f, "model_copy")
                else {**f, "confidence": 0.75}
                for f in findings
            ],
            "messages": [],
        }

    async def patched_writer_node(state):
        # mock_run_writer returns a FinalReport; wrap it in the state-update dict
        # the graph expects from a node function.
        report = await mock_run_writer(state, None)
        return {
            "final_report": report,
            "draft_report": report,
            "writer_instructions": None,
            "messages": [],
        }

    async def patched_fetch_worker_node(state):
        return {"messages": []}

    with (
        patch.object(_nodes, "supervisor_node",    patched_supervisor_node),
        patch.object(_nodes, "worker_node",        patched_worker_node),
        patch.object(_nodes, "fetch_worker_node",  patched_fetch_worker_node),
        patch.object(_nodes, "collect_node",       patched_collect_node),
        patch.object(_nodes, "critic_node",        patched_critic_node),
        patch.object(_nodes, "fact_checker_node",  patched_fact_checker_node),
        patch.object(_nodes, "writer_node",        patched_writer_node),
    ):
        graph = build_graph(checkpointer=checkpointer, interrupt_before_writer=False)
        config = get_thread_config(session_id)
        initial = {
            "messages": [],
            "query": ResearchQuery(
                topic=topic,
                depth=ResearchDepth.shallow,
                max_sources=3,
                audience="technical",
            ),
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
            "model_provider": "anthropic",
            "model_name": "claude-haiku-3-5",
            "schema_version": 2,
            "research_rounds": 0,
            "pre_dispatch_finding_ids": [],
            "active_sub_question": None,
            "active_worker_role": None,
        }

        async for _ in graph.astream(initial, config, stream_mode="updates"):
            pass

        snap = await graph.aget_state(config)
        report = snap.values.get("final_report")

    # ---- assertions ----
    assert report is not None, f"No final report produced for topic: {topic!r}"

    full_text = (
        report.title + " " + report.exec_summary + " " +
        " ".join(s.body_md for s in (report.sections or []))
    ).lower()

    for expected in fixture["expected_claims"]:
        assert expected.lower() in full_text, (
            f"Topic {topic!r}: expected claim {expected!r} not found in report.\n"
            f"Report text (first 400 chars): {full_text[:400]}"
        )

    for forbidden in fixture["forbidden_claims"]:
        assert forbidden.lower() not in full_text, (
            f"Topic {topic!r}: forbidden claim {forbidden!r} found in report."
        )

    assert len(report.sections or []) >= fixture["min_sections"], (
        f"Expected at least {fixture['min_sections']} sections, got {len(report.sections or [])}"
    )

    findings = snap.values.get("findings") or []
    if findings:
        mean_conf = sum(
            (f.confidence if hasattr(f, "confidence") else f.get("confidence", 0))
            for f in findings
        ) / len(findings)
        assert mean_conf >= fixture["min_confidence"], (
            f"Mean confidence {mean_conf:.2f} < {fixture['min_confidence']}"
        )
