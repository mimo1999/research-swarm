"""Writer agent -- synthesises validated findings into a FinalReport."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

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

_FINDINGS_TEMPLATE = """\
Research findings ({n} total):

{findings_text}

Sub-questions to cover:
{sub_questions}

All sources referenced:
{sources_text}

Write the final report now.
"""


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
    human_feedback = state.get("human_feedback") or "None provided."

    # Filter: include supported findings + fact-checked weak ones (confidence >= 0.4)
    latest_verdict_by_id: dict[str, str] = {}
    for c in critiques:
        verdict = c.verdict if hasattr(c, "verdict") else c.get("verdict", "")
        fid = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
        latest_verdict_by_id[fid] = verdict.value if hasattr(verdict, "value") else str(verdict)
    refuted_ids = {
        fid
        for fid, verdict in latest_verdict_by_id.items()
        if verdict == CritiqueVerdict.refuted.value
    }

    valid_findings = [
        f for f in findings
        if (f.id if hasattr(f, "id") else f.get("id", "")) not in refuted_ids
        and (f.confidence if hasattr(f, "confidence") else f.get("confidence", 0)) >= 0.3
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
        )
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

    return report
