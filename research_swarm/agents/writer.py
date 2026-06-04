"""Writer agent -- synthesises validated findings into a FinalReport."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.agents._utils import _field, _latest_verdicts, json_output_instruction
from research_swarm.schemas import FinalReport, ReportSection, Source
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Research Writer. Produce a comprehensive, well-structured report
from the provided research findings.

Guidelines:
  - Write in clear, professional prose appropriate for the specified audience.
  - Cite sources using [N] notation where N is the 1-based index in the references list.
  - Each section should directly address the corresponding sub-question.
  - Be accurate: only include claims supported by the evidence.
  - Acknowledge limitations honestly.
  - Incorporate any human feedback provided below.

Audience: {audience}
Human feedback: {human_feedback}
"""

_FINDINGS_TEMPLATE = (
    "Research findings ({n} total):\n\n"
    "{findings_text}\n\n"
    "Sub-questions to cover:\n"
    "{sub_questions}\n\n"
    "All sources referenced:\n"
    "{sources_text}\n\n"
    "Write the final report now."
)

# Appended after .format() so the JSON braces don't clash with str.format()
_FINDINGS_JSON_SUFFIX = json_output_instruction({
    "title": "<report title>",
    "exec_summary": "<executive summary in Markdown>",
    "sections": [
        {
            "heading": "<section heading>",
            "body_md": "<section body in Markdown>",
            "citations": [1],
        }
    ],
    "methodology": "<research methodology>",
    "limitations": "<known limitations>",
})


def _collect_references(findings: list) -> list[Source]:
    """Deduplicate sources across all findings; return ordered reference list."""
    seen_urls: set[str] = set()
    refs: list[Source] = []
    for f in findings:
        evidence = f.evidence if hasattr(f, "evidence") else f.get("evidence", [])
        for e in evidence:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                refs.append(e if hasattr(e, "url") else Source(**e))
    return refs


def _format_findings(findings: list, ref_index: dict[str, int]) -> str:
    lines = []
    for i, f in enumerate(findings, 1):
        claim = f.claim if hasattr(f, "claim") else f.get("claim", "")
        confidence = f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.5)
        sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        evidence = f.evidence if hasattr(f, "evidence") else f.get("evidence", [])
        citation_nums = [
            str(ref_index[e.url if hasattr(e, "url") else e.get("url", "")])
            for e in evidence
            if (e.url if hasattr(e, "url") else e.get("url", "")) in ref_index
        ]
        citations = ", ".join(f"[{n}]" for n in citation_nums[:5])
        lines.append(
            f"{i}. [{sub_q}] {claim} (confidence={confidence:.2f}) {citations}"
        )
    return "\n".join(lines)


async def run_writer(
    state: AgentState,
    llm: BaseChatModel,
) -> FinalReport:
    """Generate a FinalReport from validated findings."""
    findings: list = state.get("findings") or []
    critiques: list = state.get("critiques") or []
    query = state.get("query")
    plan = state.get("plan")
    # writer_instructions is the dedicated HITL channel for report revisions.
    # Fall back to human_feedback for backwards compatibility with checkpoints
    # that pre-date the writer_instructions field.
    human_feedback = (
        state.get("writer_instructions")
        or state.get("human_feedback")
        or "None provided."
    )

    refuted_ids = {
        fid
        for fid, verdict in _latest_verdicts(critiques).items()
        if verdict == CritiqueVerdict.refuted.value
    }

    valid_findings = [
        f for f in findings
        if _field(f, "id", "") not in refuted_ids
        and _field(f, "confidence", 0) >= 0.1  # low bar — writer acknowledges uncertainty
    ]

    if not valid_findings:
        logger.warning("Writer: no valid findings -- producing empty report.")
        return FinalReport(
            title=f"Research Report: {query.topic if query else 'Unknown'}",
            exec_summary="Insufficient evidence was gathered to produce a report.",
        )

    references = _collect_references(valid_findings)
    ref_index = {r.url: i + 1 for i, r in enumerate(references)}

    sources_text = "\n".join(
        f"[{i+1}] {r.url} -- {r.title}" for i, r in enumerate(references)
    )
    findings_text = _format_findings(valid_findings, ref_index)
    sub_questions = "\n".join(
        f"  - {q}" for q in (plan.sub_questions if plan else [])
    )

    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(
            audience=query.audience if query else "general",
            human_feedback=human_feedback,
        )
    )
    user_msg = HumanMessage(
        content=_FINDINGS_TEMPLATE.format(
            n=len(valid_findings),
            findings_text=findings_text,
            sub_questions=sub_questions or "  (none)",
            sources_text=sources_text or "  (none)",
        ) + _FINDINGS_JSON_SUFFIX
    )

    structured_llm = llm.with_structured_output(FinalReport)

    try:
        report: FinalReport = await structured_llm.ainvoke([system_msg, user_msg])
        # Ensure references list is populated
        if not report.references:
            report = report.model_copy(update={"references": references})
    except Exception as exc:
        logger.error("Writer structured output failed: %s", exc)
        # Fallback: build a minimal report manually
        report = FinalReport(
            title=f"Research Report: {query.topic if query else 'Topic'}",
            exec_summary="\n".join(
                f"- {f.claim if hasattr(f, 'claim') else f.get('claim','')}"
                for f in valid_findings[:5]
            ),
            sections=[
                ReportSection(
                    heading=(
                        f.sub_question
                        if hasattr(f, "sub_question")
                        else f.get("sub_question", "Finding")
                    ),
                    body_md=f.claim if hasattr(f, "claim") else f.get("claim", ""),
                    citations=[],
                )
                for f in valid_findings
            ],
            references=references,
            methodology=plan.strategy if plan else "",
            limitations="Report generated in fallback mode due to LLM error.",
        )

    # ------------------------------------------------------------------
    # Faithfulness check — one rewrite attempt if score is too low.
    # Uses embedding cosine similarity between section bodies and cited
    # source snippets; no extra LLM call for the check itself.
    # ------------------------------------------------------------------
    report = await _faithfulness_rewrite(report, references, structured_llm, system_msg, user_msg)

    return report


async def _faithfulness_rewrite(
    report: FinalReport,
    references: list[Source],
    structured_llm,
    system_msg: SystemMessage,
    original_user_msg: HumanMessage,
) -> FinalReport:
    """Return *report* unchanged if faithfulness is acceptable; else rewrite once."""
    try:
        from research_swarm.eval.faithfulness import FAITHFULNESS_THRESHOLD, score_report
    except ImportError:
        return report  # eval not available — skip

    try:
        faith_score = score_report(report, references)
    except Exception as exc:
        logger.warning("Faithfulness scoring failed (%s) — skipping rewrite.", exc)
        return report

    logger.info("Faithfulness score: %.3f (threshold=%.2f)", faith_score, FAITHFULNESS_THRESHOLD)
    if faith_score >= FAITHFULNESS_THRESHOLD:
        return report

    logger.warning(
        "Faithfulness %.3f < %.2f — requesting one rewrite pass.",
        faith_score, FAITHFULNESS_THRESHOLD,
    )
    rewrite_msg = HumanMessage(
        content=(
            f"The previous report scored {faith_score:.2f} on faithfulness "
            f"(threshold {FAITHFULNESS_THRESHOLD}).  "
            "Please rewrite it so that every claim is directly supported by the "
            "cited source snippets.  Remove or qualify any claim that lacks "
            "clear evidential support.  Return the full corrected report in the "
            "same JSON format."
        )
    )
    try:
        rewritten: FinalReport = await structured_llm.ainvoke(
            [system_msg, original_user_msg, rewrite_msg]
        )
        if not rewritten.references:
            rewritten = rewritten.model_copy(update={"references": references})
        logger.info("Rewrite faithfulness: %.3f", score_report(rewritten, references))
        return rewritten
    except Exception as exc:
        logger.warning("Faithfulness rewrite failed (%s) — keeping original.", exc)
        return report
