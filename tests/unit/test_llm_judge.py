"""Unit tests for eval/llm_judge.py -- LLM-as-a-judge report review."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_swarm.schemas.judge import JudgeVerdict, LLMJudgeResult
from research_swarm.schemas.plan import ResearchPlan
from research_swarm.schemas.report import FinalReport, ReportSection


def _make_plan(sub_questions: list[str]) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=sub_questions,
        strategy="test strategy",
        complexity_score=0.5,
    )


def _mock_llm(structured_return):
    """Mock BaseChatModel whose with_structured_output(...).ainvoke(...) returns a fixed value."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=structured_return)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


class TestJudgeReport:
    @pytest.mark.asyncio
    async def test_returns_llm_result_on_success(self):
        from research_swarm.eval.llm_judge import judge_report

        expected = LLMJudgeResult(
            coherence=4, relevance=5, completeness=3, citation_quality=4,
            verdict=JudgeVerdict.revise, reasoning="Missing citation in section 2.",
        )
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="H1", body_md="body", citations=[1])],
        )
        llm = _mock_llm(expected)

        result = await judge_report(report, _make_plan(["q1"]), llm, topic="Test topic")

        assert result == expected
        llm.with_structured_output.assert_called_once_with(LLMJudgeResult)

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_neutral_fallback(self):
        from research_swarm.eval.llm_judge import judge_report

        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        report = FinalReport(title="T", exec_summary="S")
        result = await judge_report(report, None, llm)

        assert result.verdict == JudgeVerdict.revise
        assert result.coherence == 3
        assert "boom" in result.reasoning

    @pytest.mark.asyncio
    async def test_recovers_result_from_malformed_json_completion(self):
        """A JSON-escape parse failure (e.g. LaTeX in reasoning) must be
        recovered via recover_from_parse_failure instead of always
        discarding real model output for the neutral fallback."""
        from langchain_core.exceptions import OutputParserException

        from research_swarm.eval.llm_judge import judge_report

        raw = (
            '{"coherence":4,"relevance":5,"completeness":4,"citation_quality":3,'
            r'"verdict":"revise","reasoning":"Missing \theta derivation in section 2."}'
        )
        structured = MagicMock()
        structured.ainvoke = AsyncMock(
            side_effect=OutputParserException(f"Invalid json output: {raw}", llm_output=raw)
        )
        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        report = FinalReport(title="T", exec_summary="S")
        result = await judge_report(report, None, llm)

        assert result.verdict == JudgeVerdict.revise
        assert result.coherence == 4
        assert "theta" in result.reasoning or r"\theta" in result.reasoning

    @pytest.mark.asyncio
    async def test_no_plan_still_works(self):
        from research_swarm.eval.llm_judge import judge_report

        expected = LLMJudgeResult(
            coherence=5, relevance=5, completeness=5, citation_quality=5,
            verdict=JudgeVerdict.approve, reasoning="no significant issues",
        )
        report = FinalReport(title="T", exec_summary="S")
        llm = _mock_llm(expected)

        result = await judge_report(report, None, llm)

        assert result == expected


class TestLLMJudgeResultOverall:
    def test_overall_is_mean_of_criteria(self):
        result = LLMJudgeResult(
            coherence=4, relevance=4, completeness=4, citation_quality=4,
            verdict=JudgeVerdict.approve, reasoning="fine",
        )
        assert result.overall == pytest.approx(4.0)

    def test_overall_rounds_to_two_places(self):
        result = LLMJudgeResult(
            coherence=5, relevance=4, completeness=4, citation_quality=3,
            verdict=JudgeVerdict.revise, reasoning="mixed",
        )
        assert result.overall == pytest.approx(4.0)
