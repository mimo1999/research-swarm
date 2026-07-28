"""Writer agent -- synthesises validated findings into a FinalReport."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.agents._utils import _field, _latest_verdicts, schema_output_instruction
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
  - Preserve exact numbers from the findings below -- effect sizes, percentages,
    sample sizes, p-values, confidence intervals, dosages, durations. If a finding
    states "-3.5 points (95% CI -6.7 to -0.3; p=0.03)", write that, not "showed
    improvement". Collapsing a quantitative result into a qualitative gist during
    synthesis is a bigger loss than any single missing detail below -- the numbers
    ARE the finding.
  - Cite precisely, not in bulk. Each finding below lists its sources as
    "[N] Title" so you can tell which source plausibly covers which specific
    number or sub-claim. When a finding bundles several distinct facts from
    different sources, attach each sentence only the citation(s) that actually
    support IT -- do not tack the finding's entire source list onto every
    sentence derived from it. If you genuinely cannot tell which source backs a
    specific number, cite only the source(s) whose title plausibly covers that
    claim rather than citing all of them defensively.
  - Before presenting two quantitative results side by side as a comparison
    (e.g. "X vs Y", a before/after, a table row), verify they were actually
    measured under comparable conditions -- same model size/scale, same
    benchmark or task, same evaluation setup. Findings may bundle numbers
    gathered from sources covering different scales or setups; if the
    conditions don't match, either compare only the matched-scale figures or
    state plainly that the comparison isn't apples-to-apples (e.g. "not
    directly comparable -- measured on a much smaller model") instead of
    presenting mismatched figures as if they were equivalent.
  - Acknowledge limitations honestly.
  - Incorporate any human feedback provided below.
  - Leave the `references` array EMPTY ([]) — it is populated programmatically
    from the collected sources; do not re-list them.

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
_FINDINGS_JSON_SUFFIX = schema_output_instruction(FinalReport)


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


def _format_findings(findings: list, references: list[Source]) -> str:
    """Format findings for the writer prompt, one source-per-citation.

    Each evidence source is listed as "[N] Title" rather than a bare number
    cluster -- the writer's prompt instructs it to attach only the citation(s)
    that plausibly cover a given sentence/number, which requires being able to
    tell sources apart by more than an index. A bare "[2],[3],[4],[5],[6]"
    block gives it nothing to distinguish between them, and it just re-attaches
    the whole cluster to every sentence derived from the finding.
    """
    ref_lookup = {r.url: (i + 1, r.title) for i, r in enumerate(references)}
    lines = []
    for i, f in enumerate(findings, 1):
        claim = f.claim if hasattr(f, "claim") else f.get("claim", "")
        confidence = f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.5)
        sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        evidence = f.evidence if hasattr(f, "evidence") else f.get("evidence", [])
        cite_parts = []
        for e in evidence[:5]:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            if url in ref_lookup:
                num, title = ref_lookup[url]
                cite_parts.append(f"[{num}] {title}" if title else f"[{num}]")
        sources_line = "; ".join(cite_parts) if cite_parts else "(no sources)"
        lines.append(
            f"{i}. [{sub_q}] {claim} (confidence={confidence:.2f})\n   Sources: {sources_line}"
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

    sources_text = "\n".join(
        f"[{i+1}] {r.url} -- {r.title}" for i, r in enumerate(references)
    )
    findings_text = _format_findings(valid_findings, references)
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
        # References are always set programmatically — the LLM is instructed to
        # leave them empty (saves output tokens and avoids hallucinated URLs).
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
        rewritten = rewritten.model_copy(update={"references": references})
        logger.info("Rewrite faithfulness: %.3f", score_report(rewritten, references))
        return rewritten
    except Exception as exc:
        logger.warning("Faithfulness rewrite failed (%s) — keeping original.", exc)
        return report
