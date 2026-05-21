"""Fact-checker agent — cross-checks claims against source snippets and updates confidence."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.schemas import Finding
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a meticulous Fact-Checker. Given a research claim and its source evidence,
score the factual accuracy of the claim on a scale of 0.0 (completely unsupported)
to 1.0 (fully corroborated by multiple sources).

Consider:
  - Does each cited source actually contain text supporting the claim?
  - Is the claim an overstatement or misrepresentation of the evidence?
  - Are there contradictions between sources?

Return only a JSON object with:
  confidence_score  — float 0.0–1.0
  notes             — one sentence explaining your assessment
"""

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
    state: "AgentState",
    llm: BaseChatModel,
) -> list[Finding]:
    """Cross-check every supported/weak finding and return updated Finding objects.

    Returns the same Finding objects with revised confidence scores.
    The merge-by-id reducer in AgentState ensures these overwrite the originals.
    """
    findings: list[Finding] = state.get("findings") or []
    critiques: list = state.get("critiques") or []

    # Index critiques by finding_id — store the .value string for reliable comparison
    refuted_ids: set[str] = set()
    for c in critiques:
        fid = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
        v = c.verdict if hasattr(c, "verdict") else c.get("verdict", "")
        # StrEnum: `.value` gives "refuted"; plain str also works
        verdict_val = v.value if hasattr(v, "value") else str(v)
        if verdict_val == CritiqueVerdict.refuted.value:
            refuted_ids.add(fid)

    # Skip findings the critic already marked as refuted
    to_check = [
        f for f in findings
        if (f.id if hasattr(f, "id") else f.get("id", "")) not in refuted_ids
    ]

    if not to_check:
        logger.info("FactChecker: no findings to check.")
        return []

    structured_llm = llm.with_structured_output(FactCheckResult)
    system_msg = SystemMessage(content=_SYSTEM_PROMPT)
    updated_findings: list[Finding] = []

    for finding in to_check:
        fid = finding.id if hasattr(finding, "id") else finding.get("id", "")
        claim = finding.claim if hasattr(finding, "claim") else finding.get("claim", "")
        evidence = finding.evidence if hasattr(finding, "evidence") else finding.get("evidence", [])
        sub_q = finding.sub_question if hasattr(finding, "sub_question") else finding.get("sub_question", "")

        if not evidence:
            # No sources — penalise confidence
            updated = _update_confidence(finding, 0.1)
            updated_findings.append(updated)
            continue

        sources_text = "\n".join(
            f"[{i+1}] {(e.url if hasattr(e, 'url') else e.get('url',''))} — "
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
            new_confidence = result.confidence_score
        except Exception as exc:
            logger.warning("FactChecker failed for finding %s: %s", fid[:8], exc)
            new_confidence = finding.confidence if hasattr(finding, "confidence") else finding.get("confidence", 0.5)

        updated = _update_confidence(finding, new_confidence)
        updated_findings.append(updated)
        logger.info("FactCheck %s: confidence %.2f → %.2f", fid[:8],
                    finding.confidence if hasattr(finding, "confidence") else 0.5,
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
