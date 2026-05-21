"""Critic agent — reviews each Finding and assigns a verdict."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.schemas import Critique, CritiqueVerdict, Finding

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a rigorous Research Critic. Your job is to evaluate each research finding
and assign one of three verdicts:

  supported — the claim is well-supported by the cited evidence
  weak      — the claim needs more or stronger evidence
  refuted   — the evidence contradicts or does not support the claim

Be concise. If the verdict is weak or refuted, suggest a specific follow-up
research question in `suggested_followup`.
"""

_FINDING_TEMPLATE = """\
Finding to review:
  Sub-question : {sub_question}
  Claim        : {claim}
  Confidence   : {confidence}
  Sources      : {n_sources} source(s)
  Evidence     : {evidence_summary}

Assign a verdict.
"""


async def run_critic(
    state: "AgentState",
    llm: BaseChatModel,
) -> list[Critique]:
    """Review all findings that don't yet have a critique. Return new Critique objects."""
    findings: list[Finding] = state.get("findings") or []
    existing_critiques: list[Critique] = state.get("critiques") or []

    reviewed_ids: set[str] = {
        (c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", ""))
        for c in existing_critiques
    }

    to_review = [
        f for f in findings
        if (f.id if hasattr(f, "id") else f.get("id", "")) not in reviewed_ids
    ]

    if not to_review:
        logger.info("Critic: no new findings to review.")
        return []

    structured_llm = llm.with_structured_output(Critique)
    system_msg = SystemMessage(content=_SYSTEM_PROMPT)
    new_critiques: list[Critique] = []

    for finding in to_review:
        fid = finding.id if hasattr(finding, "id") else finding.get("id", "")
        claim = finding.claim if hasattr(finding, "claim") else finding.get("claim", "")
        confidence = finding.confidence if hasattr(finding, "confidence") else finding.get("confidence", 0.5)
        sub_q = finding.sub_question if hasattr(finding, "sub_question") else finding.get("sub_question", "")
        evidence = finding.evidence if hasattr(finding, "evidence") else finding.get("evidence", [])

        evidence_summary = "; ".join(
            (e.snippet if hasattr(e, "snippet") else e.get("snippet", ""))[:150]
            for e in evidence[:3]
        ) or "(no evidence provided)"

        user_msg = HumanMessage(
            content=_FINDING_TEMPLATE.format(
                sub_question=sub_q,
                claim=claim,
                confidence=confidence,
                n_sources=len(evidence),
                evidence_summary=evidence_summary,
            )
        )

        try:
            critique: Critique = await structured_llm.ainvoke([system_msg, user_msg])
            # Force the finding_id to match (structured output may hallucinate it)
            critique = critique.model_copy(update={"finding_id": fid})
        except Exception as exc:
            logger.warning("Critic failed for finding %s: %s", fid, exc)
            critique = Critique(
                finding_id=fid,
                verdict=CritiqueVerdict.weak,
                reasoning=f"Critique generation failed: {exc}",
                suggested_followup="",
            )

        new_critiques.append(critique)
        logger.info("Critique for %s: %s", fid[:8], critique.verdict)

    return new_critiques
