"""Fact-checker agent -- cross-checks claims against source snippets and updates confidence."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import _field, _latest_verdicts, json_output_instruction
from research_swarm.schemas import Finding
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous Fact-Checker. Given a research claim and its source evidence,"
    " score the factual accuracy of the claim on a scale of 0.0 (completely unsupported)"
    " to 1.0 (fully corroborated by multiple sources).\n\n"
    "Consider:\n"
    "  - Does each cited source actually contain text supporting the claim?\n"
    "  - Is the claim an overstatement or misrepresentation of the evidence?\n"
    "  - Are there contradictions between sources?"
    + json_output_instruction({
        "confidence_score": 0.0,
        "notes": "<one sentence explaining your assessment>",
    })
)

_CLAIM_TEMPLATE = """\
Claim: {claim}

Sources ({n}):
{sources_text}

Cross-check the claim against these sources and return a confidence score.
"""


class FactCheckResult(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    notes: str = Field(default="")


async def run_fact_checker(
    state: AgentState,
    llm: BaseChatModel,
) -> list[Finding]:
    """Cross-check every supported/weak finding and return updated Finding objects.

    Returns the same Finding objects with revised confidence scores.
    The merge-by-id reducer in AgentState ensures these overwrite the originals.
    """
    findings: list[Finding] = state.get("findings") or []
    critiques: list = state.get("critiques") or []

    refuted_ids = {
        fid
        for fid, verdict in _latest_verdicts(critiques).items()
        if verdict == CritiqueVerdict.refuted.value
    }

    # Skip findings the critic already marked as refuted
    to_check = [
        f for f in findings
        if _field(f, "id", "") not in refuted_ids
    ]

    if not to_check:
        logger.info("FactChecker: no findings to check.")
        return []

    structured_llm = llm.with_structured_output(FactCheckResult)
    system_msg = SystemMessage(content=_SYSTEM_PROMPT)
    updated_findings: list[Finding] = []

    for finding in to_check:
        fid = _field(finding, "id", "")
        claim = _field(finding, "claim", "")
        evidence = _field(finding, "evidence", [])
        if not evidence:
            # No sources -- penalise confidence
            updated = _update_confidence(finding, 0.1)
            updated_findings.append(updated)
            continue

        sources_text = "\n".join(
            f"[{i+1}] {(e.url if hasattr(e, 'url') else e.get('url',''))} -- "
            f"{(e.snippet if hasattr(e, 'snippet') else e.get('snippet',''))[:200]}"
            for i, e in enumerate(evidence[:5])
        )

        user_msg = HumanMessage(
            content=_CLAIM_TEMPLATE.format(
                claim=claim,
                n=len(evidence[:5]),
                sources_text=sources_text,
            )
        )

        try:
            result: FactCheckResult = await structured_llm.ainvoke([system_msg, user_msg])
            # Apply a floor of 0.15 when evidence is present: a finding backed by
            # real sources should never score below the no-evidence baseline (0.1).
            # This guards against models that systematically return 0.0.
            new_confidence = max(result.confidence_score, 0.15)
        except Exception as exc:
            logger.warning("FactChecker failed for finding %s: %s", fid[:8], exc)
            new_confidence = _field(finding, "confidence", 0.5)

        updated = _update_confidence(finding, new_confidence)
        updated_findings.append(updated)
        logger.info("FactCheck %s: confidence %.2f -> %.2f", fid[:8],
                    _field(finding, "confidence", 0.5),
                    new_confidence)

    return updated_findings


def _update_confidence(finding, new_confidence: float) -> Finding:
    """Return a copy of the finding with updated confidence."""
    if hasattr(finding, "model_copy"):
        return finding.model_copy(update={"confidence": new_confidence})
    # dict fallback
    f = dict(finding)
    f["confidence"] = new_confidence
    return Finding(**f)
