"""Verifier: one pass over every finding.

The verifier makes a single enum decision per fact against the evidence window the grounding
step located (no free-text reasoning), then applies a fixed policy in code:

    supported (quote-grounded)   -> critique supported, confidence 0.90
    supported (other)            -> critique supported, confidence 0.75
    partial                      -> critique weak,      confidence 0.50   (kept, writer hedges)
    unsupported                  -> critique refuted,   confidence 0.10   (writer never sees it)
    unsupported, but its quote was found verbatim in the source
                                 -> critique weak,      confidence 0.30   (the verifier disagrees
                                    with the source's own words: hedge it, do not hide it)

Facts with no supporting passage at all (``grounding == "none"``) are refuted without an LLM
call. Verdicts are written as ordinary ``Critique`` objects so the writer's refuted-filter and
the benchmark's false-refute metric work unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    _field,
    _latest_verdicts,
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.question import research_topic
from research_swarm.config import settings
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas import Critique, Finding
from research_swarm.schemas.critique import CritiqueVerdict

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 1000
BATCH_SIZE = 10
_GROUNDING_RANK = {"quote": 0, "passage": 1, "unknown": 1, "none": 2}


class FactVerdict(BaseModel):
    fact: int = Field(..., description="Fact number F#")
    verdict: Literal["supported", "partial", "unsupported"]
    conflicts_with: list[int] = Field(
        default_factory=list,
        description="Numbers of other facts in this list that state the opposite",
    )
    relevance: Literal["direct", "background", "off_topic"] | None = Field(
        default=None,
        description="direct = answers the research question as asked; background = context on "
                    "the general subject; off_topic = unrelated to the question",
    )


class VerifyBatch(BaseModel):
    verdicts: list[FactVerdict] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You check research facts against the evidence text quoted with each fact.\n"
    "For each fact answer ONLY from its own evidence text:\n"
    "  supported   - the evidence states this fact (paraphrase is fine; numbers must match)\n"
    "  partial     - the evidence supports part of it, or it overstates/generalises the evidence\n"
    "  unsupported - the evidence does not state it, or says something different\n"
    "A fact that says a source does NOT mention / report / describe something is supported\n"
    "when the evidence text really does not mention it (absence claims are valid facts).\n"
    "Also list in conflicts_with any other fact number in this list that directly\n"
    "contradicts it (same subject, incompatible statement).\n"
    "Return exactly one verdict per fact. No explanations."
    + schema_output_instruction(VerifyBatch)
)

# Appended when the question has a key constraint (the question frame, agents/expansion.py).
_RELEVANCE_RULE = (
    "\nAlso set `relevance` for each fact, judged against the research question and its "
    "specific scope ({scope}):\n"
    "  direct     - the fact itself answers the question within that scope\n"
    "  background - true context about the general subject, but outside that scope\n"
    "  off_topic  - unrelated to the question"
)

# verdict -> (critique verdict, confidence); "supported" is split by grounding in _apply
_POLICY = {
    "partial": (CritiqueVerdict.weak, 0.5),
    "unsupported": (CritiqueVerdict.refuted, 0.1),
}


def cap_findings(findings: list, cap: int) -> tuple[list, list]:
    """(kept, dropped): at most *cap* findings, best grounded first then highest confidence;
    kept ones stay in their original order. ``cap <= 0`` keeps everything."""
    if cap <= 0 or len(findings) <= cap:
        return list(findings), []
    ranked = sorted(
        range(len(findings)),
        key=lambda i: (
            _GROUNDING_RANK.get(_field(findings[i], "grounding", "unknown"), 1),
            -float(_field(findings[i], "confidence", 0.5)),
            i,
        ),
    )
    keep = set(ranked[:cap])
    return ([f for i, f in enumerate(findings) if i in keep],
            [f for i, f in enumerate(findings) if i not in keep])


def _apply(finding: Finding, verdict: str) -> tuple[Finding, Critique]:
    """The policy table: (updated finding, its critique)."""
    grounding = _field(finding, "grounding", "unknown")
    if verdict == "supported":
        crit = CritiqueVerdict.supported
        conf = 0.9 if grounding == "quote" else 0.75
    elif verdict == "unsupported" and grounding == "quote":
        crit, conf = CritiqueVerdict.weak, 0.3
    else:
        crit, conf = _POLICY[verdict]
    updated = finding.model_copy(update={"confidence": conf})
    return updated, Critique(
        finding_id=finding.id, verdict=crit, reasoning=f"verifier:{verdict}",
    )


def _checked_relevance(label: str, finding: Finding, frame: object) -> str:
    """The verifier's relevance label, except that a fact whose own claim states the question's
    scope is never downgraded to background / off_topic: a 2B verifier called 5 of 6 facts about
    "cross-model KV cache transfer" background for a cross-model question. Code is the second
    opinion here, as in the coverage gate."""
    from research_swarm.agents.expansion import scope_hit

    if label != "direct" and scope_hit(finding.claim, frame):  # type: ignore[arg-type]
        return "direct"
    return label


def _format(number: int, finding: Finding) -> str:
    ev = finding.evidence[0] if finding.evidence else None
    snippet = (ev.snippet if ev else "")[:SNIPPET_CHARS].replace("\n", " ")
    return (
        f"F{number} [Q: {finding.sub_question[:80]}] {finding.claim}\n"
        f"   Evidence ({ev.title if ev else ''}): «{snippet}»\n"
    )


async def _verify_batch(
    batch: list[tuple[int, Finding]], topic: str, structured: object, session_id: str | None,
    scope: str = "",
) -> dict[int, FactVerdict] | None:
    """Verdicts by fact number for one batch; None when the call failed for good."""
    user = HumanMessage(content=(
        f"Research question: {topic}\n\nFacts:\n" + "\n".join(_format(n, f) for n, f in batch)
    ))
    system = _SYSTEM_PROMPT + (_RELEVANCE_RULE.format(scope=scope) if scope else "")
    try:
        result: VerifyBatch = await ainvoke_with_retry(
            structured, [SystemMessage(content=system), user],
            session_id=session_id, agent="verifier",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Verifier batch failed (%s)", exc)
        recovered = recover_from_parse_failure(exc, VerifyBatch)
        if recovered is None:
            return None
        result = recovered
    return {v.fact: v for v in result.verdicts}


async def run_verifier(
    state, llm: BaseChatModel,
) -> tuple[list[Finding], list[Critique], list[list[str]]]:
    """Verify every finding: (updated findings, critiques, conflicting id pairs)."""
    findings: list[Finding] = list(state.get("findings") or [])
    session_id = state.get("session_id")
    query = state.get("query")
    topic = research_topic(query)
    plan = state.get("plan")
    frame = getattr(plan, "frame", None) if plan else None
    scope = frame.key_constraint if frame is not None and frame.has_constraint else ""
    if not findings:
        return [], [], []

    kept, capped = cap_findings(
        findings, settings.for_depth("max_facts_for_writer", getattr(query, "depth", None)))
    updated: list[Finding] = []
    critiques: list[Critique] = []
    # Over the cap: not refuted (nothing is wrong with them), just kept out of the writer's
    # input by confidence < 0.1 (its existing rule).
    for f in capped:
        updated.append(f.model_copy(update={"confidence": 0.05}))
    if capped:
        trace_event(session_id, "verifier.capped", "note", n_capped=len(capped), n_kept=len(kept))

    ungrounded = [f for f in kept if _field(f, "grounding", "unknown") == "none"]
    todo = [f for f in kept if _field(f, "grounding", "unknown") != "none"]
    for f in ungrounded:
        new, crit = _apply(f, "unsupported")
        updated.append(new)
        critiques.append(crit)

    numbered = list(enumerate(todo, 1))
    by_number = {n: f for n, f in numbered}
    verdicts: dict[int, str] = {}
    relevance: dict[int, str] = {}
    conflicts: set[tuple[str, str]] = set()
    if numbered:
        structured = llm.with_structured_output(VerifyBatch)
        batches = [numbered[i:i + BATCH_SIZE] for i in range(0, len(numbered), BATCH_SIZE)]
        results = await asyncio.gather(
            *(_verify_batch(b, topic, structured, session_id, scope) for b in batches)
        )
        for batch, got in zip(batches, results, strict=True):
            if got is None:
                trace_event(session_id, "verifier.fallback", "note", n_facts=len(batch))
            missing = 0
            for n, _f in batch:
                fv = None if got is None else got.get(n)
                if fv is None:
                    missing += got is not None
                    verdicts[n] = "partial"        # never refute what the model did not judge
                    continue
                verdicts[n] = fv.verdict
                if scope and fv.relevance:
                    relevance[n] = _checked_relevance(fv.relevance, by_number[n], frame)
                for other in fv.conflicts_with:
                    if other in by_number and other != n:
                        pair = sorted((by_number[n].id, by_number[other].id))
                        conflicts.add((pair[0], pair[1]))
            if missing:
                trace_event(session_id, "verifier.missing", "note", n=missing)

    for n, f in numbered:
        new, crit = _apply(f, verdicts[n])
        if n in relevance:                     # a missing label keeps the extractor's
            new = new.model_copy(update={"relevance": relevance[n]})
        updated.append(new)
        critiques.append(crit)
        trace_event(
            session_id, "verifier.verdict", "note", finding_id=str(f.id)[:8],
            verdict=verdicts[n], grounding=_field(f, "grounding", "unknown"),
            relevance=_field(new, "relevance", "unknown"),
        )

    counts: dict[str, int] = {}
    for c in critiques:
        counts[c.verdict.value] = counts.get(c.verdict.value, 0) + 1
    rel_counts: dict[str, int] = {}
    for f in updated:
        key = f"relevance_{_field(f, 'relevance', 'unknown')}"
        rel_counts[key] = rel_counts.get(key, 0) + 1
    trace_event(
        session_id, "verifier.summary", "note", n_findings=len(findings),
        n_conflicts=len(conflicts), n_ungrounded=len(ungrounded), **counts, **rel_counts,
    )
    return updated, critiques, [list(p) for p in sorted(conflicts)]


def latest_verdict_map(critiques: list) -> dict[str, str]:
    """finding_id -> latest verdict string (thin alias so the writer imports one module)."""
    return _latest_verdicts(critiques)
