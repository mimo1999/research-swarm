"""Unit tests for Phase 1 Pydantic schemas."""

import pytest
from pydantic import ValidationError

from research_swarm.schemas import (
    ReportQualityScore,
    ResearchQuery,
)


def test_research_query_max_sources_bounds():
    with pytest.raises(ValidationError):
        ResearchQuery(topic="test", max_sources=0)
    with pytest.raises(ValidationError):
        ResearchQuery(topic="test", max_sources=51)


def test_report_quality_score_overall_ignores_uncomputed_dimensions():
    """overall must average only the dimensions actually computed --
    treating an uncomputed None as 0.0 would deflate a good report's score
    (0.9 faithfulness alone used to report overall=0.3, not 0.9)."""
    score = ReportQualityScore(faithfulness=0.9)
    assert score.overall == 0.9


class TestNextAgentReducer:
    """next_agent must tolerate >=1 concurrent writes within one LangGraph
    step without raising -- e.g. several Send-fanned worker_node/
    document_worker_node branches each independently hitting an exhausted
    budget and returning {"next_agent": "writer", ...} in the same step."""


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
