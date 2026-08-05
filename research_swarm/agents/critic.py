"""Critic agent -- reviews each Finding and assigns a verdict."""
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
from research_swarm.schemas import Critique, CritiqueVerdict, Finding

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)


class CritiqueBatch(BaseModel):
    critiques: list[Critique] = Field(
        default_factory=list, description="One critique per finding reviewed, in the same order"
    )


_SYSTEM_PROMPT = (
    "You are a rigorous Research Critic. Your job is to evaluate each research finding"
    " below and assign one of three verdicts:\n\n"
    "  supported -- the claim is well-supported by the cited evidence\n"
    "  weak      -- the claim needs more or stronger evidence\n"
    "  refuted   -- the evidence contradicts or does not support the claim\n\n"
    "Be concise. If the verdict is weak or refuted, suggest a specific follow-up"
    " research question in `suggested_followup`. Return exactly one critique per"
    " finding, in the same order, with `finding_id` set to the finding's ID."
    + schema_output_instruction(CritiqueBatch)
)

_BATCH_TEMPLATE = """\
Sources referenced below (shared across findings, referenced by [S#]):
{sources_block}

Findings to review ({n_findings}):
{findings_block}

Return one critique per finding above, in order, with `finding_id` set exactly
as shown.
"""

_FINDING_ENTRY_TEMPLATE = """\
Finding ID   : {finding_id}
Sub-question : {sub_question}
Claim        : {claim}
Confidence   : {confidence}
Sources      : {source_refs}"""


async def run_critic(
    state: AgentState,
    llm: BaseChatModel,
) -> list[Critique]:
    """Review all findings that don't yet have a critique. Return new Critique objects."""
    findings: list[Finding] = state.get("findings") or []
    existing_critiques: list[Critique] = state.get("critiques") or []

    latest_verdict_by_id = _latest_verdicts(existing_critiques)

    to_review = [
        f for f in findings
        if latest_verdict_by_id.get(_field(f, "id", "")) != CritiqueVerdict.supported.value
    ]

    if not to_review:
        logger.info("Critic: no new findings to review.")
        return []

    structured_llm = llm.with_structured_output(CritiqueBatch)
    system_msg = SystemMessage(content=_SYSTEM_PROMPT)

    # Each batch is an independent LLM call (different findings, no shared
    # mutable state) -- gather them concurrently instead of one at a time.
    # Sequential awaiting turned N/judge_batch_size batches into N/batch_size
    # times the latency of a single call for no reason; concurrent batches
    # take roughly the latency of the slowest one.
    batch_size = max(1, settings.judge_batch_size)
    batches = [to_review[i:i + batch_size] for i in range(0, len(to_review), batch_size)]
    results = await asyncio.gather(
        *(_critique_batch(batch, structured_llm, system_msg) for batch in batches)
    )
    return [critique for batch_result in results for critique in batch_result]


def _fallback_critique(fid: str, exc: Exception) -> Critique:
    # Fallback to 'supported' rather than 'weak': an LLM outage should
    # not trigger expensive re-research loops.  The fact-checker will
    # still validate confidence scores independently.
    return Critique(
        finding_id=fid,
        verdict=CritiqueVerdict.supported,
        reasoning=f"Critique generation failed ({type(exc).__name__}); defaulting to supported.",
        suggested_followup="",
    )


async def _critique_batch(
    batch: list[Finding],
    structured_llm,
    system_msg: SystemMessage,
) -> list[Critique]:
    """Review a batch of findings in a single LLM call, deduping shared sources."""
    fids = [_field(f, "id", "") for f in batch]

    # Dedup sources shared across findings in this batch by URL so a snippet
    # cited by multiple findings is only transmitted once.
    source_index: dict[str, int] = {}
    sources_block_lines: list[str] = []
    finding_entries: list[str] = []

    for finding in batch:
        fid = _field(finding, "id", "")
        evidence = _field(finding, "evidence", [])
        refs: list[str] = []
        for e in evidence[:3]:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            snippet = (e.snippet if hasattr(e, "snippet") else e.get("snippet", ""))[:150]
            if url not in source_index:
                source_index[url] = len(source_index) + 1
                sources_block_lines.append(f"[S{source_index[url]}] {url} -- {snippet}")
            refs.append(f"S{source_index[url]}")

        finding_entries.append(
            _FINDING_ENTRY_TEMPLATE.format(
                finding_id=fid,
                sub_question=_field(finding, "sub_question", ""),
                claim=_field(finding, "claim", ""),
                confidence=_field(finding, "confidence", 0.5),
                source_refs=", ".join(refs) or "(no evidence provided)",
            )
        )

    user_msg = HumanMessage(
        content=_BATCH_TEMPLATE.format(
            sources_block="\n".join(sources_block_lines) or "(none)",
            n_findings=len(batch),
            findings_block="\n\n".join(finding_entries),
        )
    )

    try:
        result: CritiqueBatch = await structured_llm.ainvoke([system_msg, user_msg])
    except Exception as exc:
        logger.warning("Critic batch failed (%d findings): %s", len(batch), exc)
        recovered = recover_from_parse_failure(exc, CritiqueBatch)
        if recovered is None or len(recovered.critiques) != len(fids):
            return [_fallback_critique(fid, exc) for fid in fids]
        logger.info("Critic batch recovered %d critique(s) from a malformed completion.", len(fids))
        result = recovered

    # Match by position (we asked for "one critique per finding, in order") — more
    # robust than matching on finding_id, which structured output may hallucinate
    # or omit. The real finding_id is always force-stamped below regardless.
    if len(result.critiques) != len(fids):
        logger.warning(
            "Critic batch returned %d critique(s) for %d finding(s); defaulting mismatched batch.",
            len(result.critiques), len(fids),
        )
        return [_fallback_critique(fid, ValueError("batch count mismatch")) for fid in fids]

    out: list[Critique] = []
    for fid, critique in zip(fids, result.critiques):
        # Force the finding_id to match (structured output may hallucinate it)
        critique = critique.model_copy(update={"finding_id": fid})
        out.append(critique)
        logger.info("Critique for %s: %s", fid[:8], critique.verdict)

    return out
