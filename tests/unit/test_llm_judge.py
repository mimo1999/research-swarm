"""Unit tests for eval/llm_judge.py -- LLM-as-a-judge report review."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_swarm.schemas.judge import JudgeVerdict
from research_swarm.schemas.plan import ResearchPlan
from research_swarm.schemas.report import FinalReport


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
