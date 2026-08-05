"""Unit tests for Phase 1 Pydantic schemas."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from research_swarm.schemas import (
    Critique,
    CritiqueVerdict,
    FinalReport,
    Finding,
    ReportQualityScore,
    ResearchDepth,
    ResearchPlan,
    ResearchQuery,
    Source,
    SourceType,
)
from research_swarm.schemas.state import AgentState


def test_research_query_defaults():
    q = ResearchQuery(topic="AI safety")
    assert q.depth == ResearchDepth.standard
    assert q.max_sources == 15
    assert q.audience == "general"


def test_research_query_depth_validation():
    q = ResearchQuery(topic="test", depth="deep")
    assert q.depth == ResearchDepth.deep


def test_research_query_max_sources_bounds():
    with pytest.raises(ValidationError):
        ResearchQuery(topic="test", max_sources=0)
    with pytest.raises(ValidationError):
        ResearchQuery(topic="test", max_sources=51)


def test_source_defaults():
    s = Source(url="https://example.com")
    assert s.source_type == SourceType.web
    assert 0.0 <= s.credibility_score <= 1.0
    assert isinstance(s.retrieved_at, datetime)


def test_research_plan():
    plan = ResearchPlan(
        sub_questions=["What is X?", "How does Y work?"],
        strategy="Use web search then cross-check with arXiv",
        required_tools=["web_search", "arxiv"],
    )
    assert len(plan.sub_questions) == 2
    assert "web_search" in plan.required_tools


def test_finding_auto_id():
    f1 = Finding(claim="X causes Y")
    f2 = Finding(claim="A causes B")
    assert f1.id != f2.id


def test_critique_verdict_enum():
    c = Critique(
        finding_id="abc",
        verdict="weak",
        reasoning="Only one source",
    )
    assert c.verdict == CritiqueVerdict.weak


def test_report_quality_score_overall():
    score = ReportQualityScore(faithfulness=0.8, relevance=0.7, completeness=0.9)
    assert abs(score.overall - 0.8) < 0.01


def test_report_quality_score_overall_all_zero():
    """Edge case: all dimensions zero → overall is exactly 0.0."""
    score = ReportQualityScore(faithfulness=0.0, relevance=0.0, completeness=0.0)
    assert score.overall == 0.0


def test_report_quality_score_overall_all_one():
    """Edge case: all dimensions one → overall is exactly 1.0."""
    score = ReportQualityScore(faithfulness=1.0, relevance=1.0, completeness=1.0)
    assert score.overall == 1.0


def test_report_quality_score_relevance_and_completeness_default_to_none():
    """Nothing in the codebase computes relevance/completeness yet -- only
    faithfulness is. They must default to None ("not computed"), not 0.0,
    since a 0.0 default is indistinguishable from "computed and genuinely
    zero" and previously made every report show a misleading 0% badge."""
    score = ReportQualityScore(faithfulness=0.9)
    assert score.relevance is None
    assert score.completeness is None


def test_report_quality_score_overall_ignores_uncomputed_dimensions():
    """overall must average only the dimensions actually computed --
    treating an uncomputed None as 0.0 would deflate a good report's score
    (0.9 faithfulness alone used to report overall=0.3, not 0.9)."""
    score = ReportQualityScore(faithfulness=0.9)
    assert score.overall == 0.9


def test_report_quality_score_overall_none_when_nothing_computed():
    score = ReportQualityScore()
    assert score.overall == 0.0


class TestNextAgentReducer:
    """next_agent must tolerate >=1 concurrent writes within one LangGraph
    step without raising -- e.g. several Send-fanned worker_node/
    document_worker_node branches each independently hitting an exhausted
    budget and returning {"next_agent": "writer", ...} in the same step."""

    def test_last_value_returns_the_new_value(self):
        from research_swarm.schemas.state import _last_value

        assert _last_value("dispatch", "writer") == "writer"
        assert _last_value(None, "writer") == "writer"

    def test_concurrent_identical_writes_do_not_raise(self):
        """Reproduces the real failure shape via LangGraph's own channel
        machinery: BinaryOperatorAggregate.update() is what a Send fan-out's
        simultaneous writes actually go through. Before adding the reducer,
        next_agent was a plain LastValue channel, which raises
        InvalidUpdateError whenever len(values) != 1 in a single update()
        call -- regardless of whether the values are equal."""
        from langgraph.channels.binop import BinaryOperatorAggregate

        from research_swarm.schemas.state import AgentName, _last_value

        channel: BinaryOperatorAggregate = BinaryOperatorAggregate(
            AgentName | None, _last_value,
        )
        # Two (or more) concurrent branches writing the same value in one step.
        channel.update(["writer", "writer", "writer"])
        assert channel.get() == "writer"

    def test_concurrent_writes_take_the_last_value(self):
        from langgraph.channels.binop import BinaryOperatorAggregate

        from research_swarm.schemas.state import AgentName, _last_value

        channel: BinaryOperatorAggregate = BinaryOperatorAggregate(
            AgentName | None, _last_value,
        )
        channel.update(["dispatch", "critic"])
        assert channel.get() == "critic"


def test_final_report_quality_optional():
    report = FinalReport(
        title="Test Report",
        exec_summary="Summary here",
    )
    assert report.quality_score is None
    assert report.sections == []
    assert report.references == []


def test_agent_state_is_typeddict():
    # AgentState is a TypedDict — check key existence via annotations
    keys = AgentState.__annotations__.keys()
    assert "messages" in keys
    assert "findings" in keys
    assert "critiques" in keys
    assert "next_agent" in keys
    assert "iteration_count" in keys
