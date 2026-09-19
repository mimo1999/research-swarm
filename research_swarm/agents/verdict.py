"""Claim verdict: SUPPORT / CONTRADICT / NOT_ENOUGH_INFO decided per fact, aggregated in code.

Asked for a verdict in one go, the small model answered SUPPORT whenever it found a related
abstract: on the 300-task benchmark, 17/34 CONTRADICT and 22/33 NOT_ENOUGH_INFO claims came
back SUPPORT, and 9/100 reports named two different verdicts. Here the model does the narrower
job it is better at -- for each verified fact, does it *directly test* the claim, and if so in
which direction -- and the verdict follows from those relations by a fixed rule:

    no fact directly tests the claim        -> the insufficient label
    tests only support / only contradict    -> that label
    both directions                         -> the side with more deciding facts; a tie is
                                               insufficient (the evidence is mixed)

One extra LLM call, made only when the question enumerates claim-check labels
(``QuestionSpec.is_claim_check``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    _field,
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.question import QuestionSpec
from research_swarm.runtime.trace import trace_event

logger = logging.getLogger(__name__)

EVIDENCE_CHARS = 600


class FactRelation(BaseModel):
    fact: int = Field(..., description="Fact number F#")
    relation: Literal["supports", "contradicts", "unrelated"]


class ClaimRelations(BaseModel):
    relations: list[FactRelation] = Field(default_factory=list)


_SYSTEM = (
    "You decide how each fact bears on ONE claim.\n"
    "  supports    - the fact directly tests the claim's exact assertion (same subject,\n"
    "                population, intervention/exposure, outcome) and agrees with it\n"
    "  contradicts - the fact directly tests the same assertion and finds the opposite,\n"
    "                no effect, or a different direction/size\n"
    "  unrelated   - anything else: a different outcome, population or question, background,\n"
    "                or a fact that is merely on the same topic\n"
    "Being on the same topic is NOT support. When unsure, answer unrelated.\n"
    "Return one relation per fact."
    + schema_output_instruction(ClaimRelations)
)


@dataclass
class Verdict:
    label: str                                  # "" = insufficient, and the question has no
    role: str                                   # label for it (e.g. YES / NO)
    # role: support | contradict | insufficient
    deciding_facts: list[int] = field(default_factory=list)   # 1-based fact numbers
    counts: dict[str, int] = field(default_factory=dict)


def aggregate(relations: dict[int, str], spec: QuestionSpec) -> Verdict:
    """The fixed rule (module docstring) from per-fact relations."""
    sup = sorted(n for n, r in relations.items() if r == "supports")
    con = sorted(n for n, r in relations.items() if r == "contradicts")
    counts = {"supports": len(sup), "contradicts": len(con),
              "unrelated": len(relations) - len(sup) - len(con)}
    if sup and len(sup) > len(con):
        role, deciding = "support", sup
    elif con and len(con) > len(sup):
        role, deciding = "contradict", con
    else:
        role, deciding = "insufficient", []
    # No label for "insufficient" (YES / NO): do not borrow another label, that would assert an
    # answer on no evidence; the render states the insufficiency in words instead.
    label = spec.label_for(role) or ""
    return Verdict(label=label, role=role, deciding_facts=deciding, counts=counts)


async def decide_verdict(
    spec: QuestionSpec, facts: list, llm: BaseChatModel, session_id: str | None = None,
) -> Verdict | None:
    """Per-fact relations -> verdict; None when the call failed (the caller keeps the draft)."""
    if not spec.is_claim_check:
        return None
    if not facts:
        return aggregate({}, spec)
    lines = []
    for n, f in enumerate(facts, 1):
        ev = (_field(f, "evidence", []) or [None])[0]
        snippet = (_field(ev, "snippet", "") if ev else "")[:EVIDENCE_CHARS].replace("\n", " ")
        lines.append(f"F{n} {_field(f, 'claim', '')}\n   Evidence: «{snippet}»")
    user = HumanMessage(content=f"Claim: {spec.content}\n\nFacts:\n" + "\n".join(lines))
    structured = llm.with_structured_output(ClaimRelations)
    try:
        result: Any = await ainvoke_with_retry(
            structured, [SystemMessage(content=_SYSTEM), user], session_id=session_id,
            agent="verdict",
        )
    except Exception as exc:  # noqa: BLE001
        result = recover_from_parse_failure(exc, ClaimRelations)
        if result is None:
            logger.warning("Claim verdict call failed (%s); keeping the writer's answer.", exc)
            trace_event(session_id, "verdict.fallback", "note",
                        error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return None
    relations = {r.fact: r.relation for r in result.relations if 1 <= r.fact <= len(facts)}
    verdict = aggregate(relations, spec)
    trace_event(session_id, "verdict.result", "note", label=verdict.label, role=verdict.role,
                deciding=verdict.deciding_facts, **verdict.counts)
    return verdict
