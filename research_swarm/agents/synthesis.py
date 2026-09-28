"""Synthesis: one large-model call that reads the evidence packet and judges.

It returns the direct answer, the claim verdict (claim-check questions), the stance and the report
sentences, each citing sentence IDs (``S3.4``). Code then audits it (see CONTEXT.md): the output
reaches the existing ``writer_render.render_report`` through an adapter in which every packet
sentence is one fact whose evidence is exactly that sentence, so the render's checks (numbers,
scope, strict terms, duplicates, audience shape, references) run against the precise text. Unknown
sentence IDs are dropped, and a SUPPORT / CONTRADICT verdict that cites no packet sentence becomes
the insufficient-evidence label. The large model is the judge; code is the auditor.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.packet import EvidencePacket
from research_swarm.agents.question import QuestionSpec, parse_question
from research_swarm.agents.verdict import Verdict
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas.finding import Finding
from research_swarm.schemas.report import FinalReport
from research_swarm.schemas.source import Source, SourceType

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"S(\d+)\.(\d+)", re.IGNORECASE)


class CitedSentence(BaseModel):
    text: str = Field(..., description="One sentence, no citation markers in the text")
    ids: list[str] = Field(default_factory=list,
                           description="Sentence IDs (e.g. S3.4) this sentence rests on")


class SynthesisSection(BaseModel):
    heading: str
    sentences: list[CitedSentence] = Field(default_factory=list)


class Synthesis(BaseModel):
    title: str
    direct_answer: str = Field(..., description="1-2 sentences answering the question")
    answer_ids: list[str] = Field(default_factory=list,
                                  description="Sentence IDs the direct answer rests on")
    verdict: str = Field(default="", description="Only when allowed answers are listed: exactly "
                                                  "one of them; otherwise empty")
    verdict_ids: list[str] = Field(default_factory=list,
                                   description="Sentence IDs that decide the verdict")
    stance: Literal["answered", "partial", "insufficient"]
    summary: list[CitedSentence] = Field(default_factory=list, description="1-3 sentences")
    sections: list[SynthesisSection] = Field(default_factory=list)
    limitations: str = ""


_SYSTEM = """\
You answer a research question from an evidence packet: numbered sentences copied from the
sources, each with an ID like S3.4 (source 3, sentence 4). You are the judge: read the sentences
themselves and decide what they show.

Rules:
- Cite the IDs of the sentences each of your sentences rests on in `ids`. Use only IDs that
  appear in the packet. Do not put IDs or [n] markers in the text.
- direct_answer answers the question in 1-2 sentences; answer_ids are the sentences it rests on.
  If allowed answers are listed, `verdict` is exactly one of them, chosen by what the sentences
  directly test: the supporting answer only if a sentence directly tests the claim and agrees;
  the contradicting answer if a sentence directly tests it and finds the opposite, no effect,
  or a different direction or size; otherwise the insufficient-evidence answer. Being on the
  same topic is NOT support.
  verdict_ids are the deciding sentences.
- stance = insufficient only if no sentence addresses the question.
- Use only what the sentences say. Copy numbers exactly; never compute new ones. Hedge findings
  the sentences themselves hedge. If sentences conflict, give both sides with their IDs.
- A sentence with no ID may only be a short transition with no facts or numbers.
- Never repeat a sentence: summary is a 1-3 sentence overview, sections hold the detail.
{scope_rules}
Report structure for this audience:
{structure_guidance}

Audience: {audience}. Human feedback: {human_feedback}
"""

_USER = """\
Research question: {question}
{answer_format}
Sub-questions:
{sub_questions}

Evidence packet:
{packet}
"""


def normalize_ids(ids: list[str] | None) -> list[str]:
    """``["[s3.4]", "S1.2, S1.3"]`` -> ``["S3.4", "S1.2", "S1.3"]`` (order kept, no duplicates)."""
    out: list[str] = []
    for raw in ids or []:
        for m in _ID_RE.finditer(str(raw)):
            sid = f"S{int(m.group(1))}.{int(m.group(2))}"
            if sid not in out:
                out.append(sid)
    return out


def packet_facts(packet: EvidencePacket, frame: Any = None) -> list[Finding]:
    """Every packet sentence as one fact whose evidence is exactly that sentence (F1 = the first
    packet sentence). With a question frame, a sentence that does not reach its scope is
    background, as the fact chain labels it."""
    from research_swarm.agents.expansion import scope_hit

    facts = []
    for s in packet.sentences:
        src = packet.source(s.source)
        url = src.url if src else ""
        try:
            stype = SourceType(src.source_type) if src else SourceType.web
        except ValueError:
            stype = SourceType.web
        relevance = "direct" if frame is None or scope_hit(s.text, frame) else "background"
        facts.append(Finding(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}|{s.id}|{s.text[:80]}")),
            claim=s.text, sub_question=s.sub_question, grounding="quote", quote=s.text,
            relevance=relevance, confidence=0.9,
            evidence=[Source(url=url, title=src.title if src else "", snippet=s.text,
                             source_type=stype,
                             credibility_score=src.credibility_score if src else 0.6)],
        ))
    return facts


def to_draft(synthesis: Synthesis, packet: EvidencePacket) -> tuple[Any, dict[str, Any]]:
    """The synthesis as a ``WriterDraft`` citing fact numbers, and stats on unknown IDs."""
    from research_swarm.agents.writer_render import DraftSection, DraftSentence, WriterDraft

    number = {s.id: n for n, s in enumerate(packet.sentences, 1)}
    stats = {"cited_ids": 0, "unknown_ids": 0}

    def nums(ids: list[str]) -> list[int]:
        out = []
        for sid in normalize_ids(ids):
            if sid in number:
                out.append(number[sid])
                stats["cited_ids"] += 1
            else:
                stats["unknown_ids"] += 1
        return out

    def sentences(items: list[CitedSentence]) -> list[Any]:
        return [DraftSentence(text=c.text, facts=nums(c.ids)) for c in items if c.text.strip()]

    draft = WriterDraft(
        title=synthesis.title or packet.question[:80],
        direct_answer=synthesis.direct_answer,
        answer_facts=nums(synthesis.answer_ids),
        stance=synthesis.stance,
        summary=sentences(synthesis.summary),
        sections=[DraftSection(heading=sec.heading, sentences=sentences(sec.sentences))
                  for sec in synthesis.sections],
        limitations=synthesis.limitations,
    )
    return draft, stats


def audit_verdict(synthesis: Synthesis, spec: QuestionSpec,
                  packet: EvidencePacket) -> Verdict | None:
    """The synthesis verdict, checked: an unlisted label, or a SUPPORT / CONTRADICT verdict that
    cites no packet sentence, becomes the insufficient-evidence label. None for questions that
    are not claim checks."""
    if not spec.is_claim_check:
        return None
    number = {s.id: n for n, s in enumerate(packet.sentences, 1)}
    deciding = [number[sid] for sid in normalize_ids(synthesis.verdict_ids) if sid in number]
    raw = re.sub(r"[\s-]+", "_", synthesis.verdict.strip().upper())
    label = next((lab for lab in spec.labels if lab.upper() == raw), "")
    role = spec.label_roles.get(label, "insufficient")
    if role in ("support", "contradict") and not deciding:
        role, deciding = "insufficient", []
    if role == "insufficient":
        label = spec.label_for("insufficient") or ""
        deciding = []
    return Verdict(label=label, role=role, deciding_facts=deciding,
                   counts={"cited": len(deciding)})


def _parsed(result: Any) -> Synthesis:
    """The Synthesis from an ``include_raw`` result ({"raw", "parsed", "parsing_error"}), parsing
    the raw text (code fence and surrounding prose stripped) when the parser returned nothing.
    A plain Synthesis (fakes, providers without include_raw) passes through."""
    from langchain_core.exceptions import OutputParserException

    from research_swarm.agents._utils import (
        _extract_json_object,
        _strip_code_fence,
        repair_json_brackets,
    )

    if not isinstance(result, dict):
        return result
    if result.get("parsed") is not None:
        return result["parsed"]
    raw = getattr(result.get("raw"), "content", "") or ""
    text = raw if isinstance(raw, str) else str(raw)
    body = _extract_json_object(_strip_code_fence(text))
    try:
        return Synthesis.model_validate_json(body)
    except Exception:  # noqa: BLE001 -- one repair attempt: a forgotten closing bracket
        try:
            return Synthesis.model_validate_json(repair_json_brackets(body))
        except Exception as exc:  # noqa: BLE001 -- re-raised in the shape recovery understands
            raise OutputParserException(f"Failed to parse Synthesis: {exc}",
                                        llm_output=text) from exc


def _answer_only_ok(synthesis: Synthesis, verdict: Verdict | None, draft: Any) -> bool:
    """An empty body is still a complete report when the direct answer stands on its own: the
    model found no evidence (insufficient stance or verdict, which cites nothing), or the answer
    or verdict cites real packet sentences."""
    if synthesis.stance == "insufficient":
        return True
    if verdict is not None and (verdict.role == "insufficient" or verdict.deciding_facts):
        return True
    return bool(draft.answer_facts)


def _no_evidence_report(question: str) -> FinalReport:
    """The report for an empty packet: no source sentence to answer from, so no LLM call."""
    return FinalReport(
        title=f"Research Report: {question}"[:200],
        exec_summary=("**Answer:** No source text was available to answer this question, so it "
                      "could not be answered."),
        sections=[], references=[],
        methodology="Evidence packet path: the packet held no source sentences.",
        limitations="No sources were supplied, or none contained usable sentences. In the "
                    "packet path, sources must currently be supplied (search comes in "
                    "milestone 2).",
    )


def _scope_rules(frame: Any) -> str:
    from research_swarm.agents.writer import _PROOF_RULES, _SCOPE_RULES

    if frame is None or not frame.has_constraint:
        return ""
    rules = _SCOPE_RULES.format(scope=frame.key_constraint)
    if frame.define_terms:
        rules += _PROOF_RULES.format(
            terms="; ".join(frame.define_terms),
            criterion=frame.proof_criterion or "a source stating it is met, for the whole method",
        )
    return rules


async def run_synthesis(state: dict[str, Any], llm: BaseChatModel
                        ) -> tuple[FinalReport, list[Finding]]:
    """The report for the run's evidence packet, and the packet sentences it cited (as facts)."""
    from research_swarm.agents.writer import structure_guidance
    from research_swarm.agents.writer_render import deterministic_report, render_report

    session_id = state.get("session_id")
    query = state.get("query")
    plan = state.get("plan")
    topic = query.topic if query else ""
    audience = query.audience if query else "general"
    spec = parse_question(topic)
    question = spec.content or topic
    frame = getattr(plan, "frame", None) if plan else None
    sub_qs = list(plan.sub_questions) if plan else [question]
    packet = EvidencePacket.from_dict(state.get("evidence_packet") or {})
    facts = packet_facts(packet, frame)
    if not facts:
        trace_event(session_id, "synthesis.empty_packet", "note")
        return _no_evidence_report(question), []

    answer_format = ""
    if spec.instruction:
        answer_format += f"Answer format: {spec.instruction}\n"
    if spec.labels:
        answer_format += f"Allowed answers: {', '.join(spec.labels)}\n"
    human_feedback = state.get("writer_instructions") or "None provided."
    system = SystemMessage(content=_SYSTEM.format(
        scope_rules=_scope_rules(frame), structure_guidance=structure_guidance(audience),
        audience=audience, human_feedback=human_feedback,
    ) + schema_output_instruction(Synthesis))
    user = HumanMessage(content=_USER.format(
        question=question, answer_format=answer_format,
        sub_questions="\n".join(f"- {sq}" for sq in sub_qs) or "(none)",
        packet=packet.render(),
    ))
    # include_raw: a reply the parser rejects (e.g. valid JSON wrapped in a ```json fence came
    # back as "completion null") is parsed here from the raw text instead of being lost.
    structured = llm.with_structured_output(Synthesis, include_raw=True)
    try:
        synthesis: Any = _parsed(await ainvoke_with_retry(
            structured, [system, user], session_id=session_id, agent="synthesis"))
    except Exception as exc:  # noqa: BLE001
        synthesis = recover_from_parse_failure(exc, Synthesis)
        if synthesis is None:
            logger.error("Synthesis failed (%s) -- using the no-LLM report.", exc)
            trace_event(session_id, "synthesis.fallback", "note",
                        error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return deterministic_report(facts, question, plan, audience=audience), []

    draft, id_stats = to_draft(synthesis, packet)
    verdict = audit_verdict(synthesis, spec, packet)
    report, stats = render_report(
        draft, facts, question, plan, verdict=verdict, labels=spec.labels, audience=audience,
        frame=frame, sub_questions=sub_qs,
    )
    trace_event(session_id, "synthesis.render", "note", stance=synthesis.stance,
                model_verdict=synthesis.verdict, verdict=verdict.label if verdict else None,
                verdict_role=verdict.role if verdict else None, **id_stats, **stats)
    if stats["empty"] and not _answer_only_ok(synthesis, verdict, draft):
        trace_event(session_id, "synthesis.fallback", "note", reason="empty_render")
        return deterministic_report(facts, question, plan, audience=audience), []
    if stats["empty"]:
        # The direct answer (and verdict) is the whole report, not a failed render: "no source
        # addresses this" legitimately cites nothing, and a cited verdict needs no body.
        trace_event(session_id, "synthesis.answer_only", "note", stance=synthesis.stance)

    cited = set(draft.answer_facts)
    for s in draft.summary:
        cited.update(s.facts)
    for sec in draft.sections:
        for s in sec.sentences:
            cited.update(s.facts)
    if verdict is not None:
        cited.update(verdict.deciding_facts)
    return report, [f for n, f in enumerate(facts, 1) if n in cited]
