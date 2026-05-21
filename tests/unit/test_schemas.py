"""Unit tests for Phase 1 Pydantic schemas."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from research_swarm.schemas import (
    ResearchQuery,
    ResearchDepth,
    Source,
    SourceType,
    ResearchPlan,
    Finding,
    Critique,
    CritiqueVerdict,
    ReportSection,
    FinalReport,
    ReportQualityScore,
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
