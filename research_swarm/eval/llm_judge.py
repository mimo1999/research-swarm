"""LLM-as-a-judge reviewer for generated reports.

Complements ``eval/faithfulness.py`` (embedding cosine similarity between
section bodies and cited snippets) with an independent LLM review pass --
embeddings can't catch a report that answers the wrong question, skips a
sub-question, cites a reference that doesn't exist, or just reads poorly.

Public API::

    from research_swarm.eval.llm_judge import judge_report, JUDGE_PASS_THRESHOLD

    result = await judge_report(report, plan, llm, topic=query.topic)
    if result.overall < JUDGE_PASS_THRESHOLD:
        ...
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.agents._utils import recover_from_parse_failure, schema_output_instruction
from research_swarm.schemas.judge import JudgeVerdict, LLMJudgeResult

if TYPE_CHECKING:
    from research_swarm.schemas.plan import ResearchPlan
    from research_swarm.schemas.report import FinalReport

logger = logging.getLogger(__name__)

# Mean of the four 1-5 criteria at/above this reads as a pass. Secondary to
# `result.verdict` (the model's own recommendation) -- use this for a
# numeric gate (dashboards, benchmarks), not to override the verdict.
JUDGE_PASS_THRESHOLD = 3.5

_SYSTEM_PROMPT = (
    "You are an independent Quality Reviewer grading a research report you did NOT write.\n"
    "Score the report on four criteria, each on a 1-5 scale:\n\n"
    "  coherence        -- Is the report clearly written, well-organized, and easy to follow?\n"
    "  relevance        -- Does the report directly address the research topic and stay on-topic?\n"
    "  completeness     -- Are all listed sub-questions substantively addressed?\n"
    "  citation_quality -- Are claims backed by [N]-style citations pointing at real, listed\n"
    "                      references? Penalize uncited factual claims and citations to\n"
    "                      references that don't exist in the list below.\n\n"
    "Give a `verdict`:\n"
    "  approve -- the report is ready to ship as-is\n"
    "  revise  -- the report has fixable gaps (missing citations, an unaddressed\n"
    "             sub-question, weak prose)\n"
    "  reject  -- the report is fundamentally inadequate (wrong topic, fabricated\n"
    "             claims, empty)\n\n"
    "Be a skeptical, specific reviewer. `reasoning` must name the concrete gaps you found,\n"
    "or say \"no significant issues\" if genuinely none."
    + schema_output_instruction(LLMJudgeResult)
)

_REVIEW_TEMPLATE = """\
Research topic: {topic}

Sub-questions the report was expected to cover:
{sub_questions}

--- REPORT UNDER REVIEW ---
Title: {title}

Executive summary:
{exec_summary}

Sections:
{sections}

References ({n_refs} total):
{references}
--- END REPORT ---

Review the report now."""


def _format_sections(report: FinalReport) -> str:
    lines = []
    for s in report.sections:
        heading = getattr(s, "heading", "?")
        body = getattr(s, "body_md", "")
        citations = getattr(s, "citations", []) or []
        cite_str = ", ".join(f"[{n}]" for n in citations) or "(no citations)"
        lines.append(f"## {heading} {cite_str}\n{body}")
    return "\n\n".join(lines) or "(no sections)"


def _format_references(report: FinalReport) -> str:
    lines = [f"[{i + 1}] {r.url} -- {r.title}" for i, r in enumerate(report.references)]
    return "\n".join(lines) or "(none)"


def _fallback_result(exc: Exception) -> LLMJudgeResult:
    # Neutral midpoint scores + 'revise' -- an LLM outage should surface as
    # "needs a look", not silently pass or silently fail the report.
    return LLMJudgeResult(
        coherence=3, relevance=3, completeness=3, citation_quality=3,
        verdict=JudgeVerdict.revise,
        reasoning=f"LLM judge call failed ({type(exc).__name__}: {exc}); scores default neutral.",
    )


async def judge_report(
    report: FinalReport,
    plan: ResearchPlan | None,
    llm: BaseChatModel,
    topic: str = "",
) -> LLMJudgeResult:
    """Run an independent LLM review of a generated report. Never raises."""
    structured_llm = llm.with_structured_output(LLMJudgeResult)
    sub_qs = plan.sub_questions if plan else []
    sub_questions = "\n".join(f"  - {q}" for q in sub_qs) or "  (none)"

    user_msg = HumanMessage(
        content=_REVIEW_TEMPLATE.format(
            topic=topic or report.title,
            sub_questions=sub_questions,
            title=report.title,
            exec_summary=report.exec_summary,
            sections=_format_sections(report),
            n_refs=len(report.references),
            references=_format_references(report),
        )
    )

    try:
        result: LLMJudgeResult = await structured_llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), user_msg]
        )
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        recovered = recover_from_parse_failure(exc, LLMJudgeResult)
        if recovered is None:
            return _fallback_result(exc)
        logger.info("LLM judge recovered a result from a malformed completion.")
        return recovered

    logger.info(
        "LLM judge verdict=%s overall=%.2f (coherence=%d relevance=%d "
        "completeness=%d citation_quality=%d)",
        result.verdict, result.overall, result.coherence, result.relevance,
        result.completeness, result.citation_quality,
    )
    return result
