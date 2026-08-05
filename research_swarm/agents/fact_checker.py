"""Fact-checker agent -- cross-checks claims against source snippets and updates confidence."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    _field,
    _latest_verdicts,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.config import settings
from research_swarm.schemas import Finding
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


class FactCheckResult(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    notes: str = Field(default="")


class FactCheckBatch(BaseModel):
    results: list[FactCheckResult] = Field(
        default_factory=list, description="One result per claim reviewed, in the same order"
    )


_SYSTEM_PROMPT = (
    "You are a meticulous Fact-Checker. Given a list of research claims and their source"
    " evidence, score the factual accuracy of each claim on a scale of 0.0 (completely"
    " unsupported) to 1.0 (fully corroborated by multiple sources).\n\n"
    "Consider, for each claim independently:\n"
    "  - Does each cited source actually contain text supporting the claim?\n"
    "  - Is the claim an overstatement or misrepresentation of the evidence?\n"
    "  - Are there contradictions between sources?\n\n"
    "Return exactly one result per claim, in the same order as given."
    + schema_output_instruction(FactCheckBatch)
)

_BATCH_TEMPLATE = """\
Sources referenced below (shared across claims, referenced by [S#]):
{sources_block}

Claims to check ({n_claims}):
{claims_block}

Cross-check each claim against its sources and return one confidence score per
claim, in order.
"""

_CLAIM_ENTRY_TEMPLATE = """\
Claim   : {claim}
Sources : {source_refs}"""


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

    # Findings with no evidence never hit the LLM — penalise directly.
    no_evidence = [f for f in to_check if not _field(f, "evidence", [])]
    with_evidence = [f for f in to_check if _field(f, "evidence", [])]

    updated_findings: list[Finding] = [
        _update_confidence(f, 0.1) for f in no_evidence
    ]

    if with_evidence:
        structured_llm = llm.with_structured_output(FactCheckBatch)
        system_msg = SystemMessage(content=_SYSTEM_PROMPT)
        # Each batch is an independent LLM call -- gather concurrently rather
        # than awaiting one at a time (see critic.py's run_critic for the
        # same fix and rationale). Findings merge into state by id, so the
        # order updated_findings ends up in doesn't matter.
        batch_size = max(1, settings.judge_batch_size)
        batches = [
            with_evidence[i:i + batch_size] for i in range(0, len(with_evidence), batch_size)
        ]
        results = await asyncio.gather(
            *(_fact_check_batch(batch, structured_llm, system_msg) for batch in batches)
        )
        for batch_result in results:
            updated_findings.extend(batch_result)

    return updated_findings


async def _fact_check_batch(
    batch: list[Finding],
    structured_llm,
    system_msg: SystemMessage,
) -> list[Finding]:
    """Fact-check a batch of findings in a single LLM call, deduping shared sources."""
    source_index: dict[str, int] = {}
    sources_block_lines: list[str] = []
    claim_entries: list[str] = []

    for finding in batch:
        evidence = _field(finding, "evidence", [])
        refs: list[str] = []
        for e in evidence[:5]:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            snippet = (e.snippet if hasattr(e, "snippet") else e.get("snippet", ""))[:200]
            if url not in source_index:
                source_index[url] = len(source_index) + 1
                sources_block_lines.append(f"[S{source_index[url]}] {url} -- {snippet}")
            refs.append(f"S{source_index[url]}")

        claim_entries.append(
            _CLAIM_ENTRY_TEMPLATE.format(
                claim=_field(finding, "claim", ""),
                source_refs=", ".join(refs),
            )
        )

    user_msg = HumanMessage(
        content=_BATCH_TEMPLATE.format(
            sources_block="\n".join(sources_block_lines) or "(none)",
            n_claims=len(batch),
            claims_block="\n\n".join(claim_entries),
        )
    )

    try:
        result: FactCheckBatch = await structured_llm.ainvoke([system_msg, user_msg])
    except Exception as exc:
        logger.warning("FactChecker batch failed (%d findings): %s", len(batch), exc)
        recovered = recover_from_parse_failure(exc, FactCheckBatch)
        if recovered is None or len(recovered.results) != len(batch):
            return [
                _update_confidence(f, _field(f, "confidence", 0.5)) for f in batch
            ]
        logger.info(
            "FactChecker batch recovered %d result(s) from a malformed completion.", len(batch)
        )
        result = recovered

    if len(result.results) != len(batch):
        logger.warning(
            "FactChecker batch returned %d result(s) for %d finding(s); "
            "preserving original confidence for mismatched batch.",
            len(result.results), len(batch),
        )
        return [
            _update_confidence(f, _field(f, "confidence", 0.5)) for f in batch
        ]

    out: list[Finding] = []
    for finding, result_item in zip(batch, result.results):
        fid = _field(finding, "id", "")
        old_confidence = _field(finding, "confidence", 0.5)
        # Apply a floor of 0.15 when evidence is present: a finding backed by
        # real sources should never score below the no-evidence baseline (0.1).
        # This guards against models that systematically return 0.0.
        new_confidence = max(result_item.confidence_score, 0.15)
        out.append(_update_confidence(finding, new_confidence))
        logger.info(
            "FactCheck %s: confidence %.2f -> %.2f", fid[:8], old_confidence, new_confidence
        )

    return out


def _update_confidence(finding, new_confidence: float) -> Finding:
    """Return a copy of the finding with updated confidence."""
    if hasattr(finding, "model_copy"):
        return finding.model_copy(update={"confidence": new_confidence})
    # dict fallback
    f = dict(finding)
    f["confidence"] = new_confidence
    return Finding(**f)
