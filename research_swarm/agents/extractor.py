"""Shared fact extractor: one call turns a batch of source texts into grounded findings.

Replaces the per-document worker and the per-sub-question paper worker with one prompt:
  * several short sources are *packed* into one call (``pack_sources``), so a 10-paragraph
    HotpotQA task needs one call instead of ten;
  * every distinct fact becomes its own item (up to ``extract_max_facts_per_pair`` per source and
    sub-question) instead of one gist per document;
  * sources and sub-questions are referred to by NUMBER, so a paraphrased sub-question can no
    longer be silently dropped by an exact-string match;
  * each fact's evidence is located in the source by code (``grounding.ground``).
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
    is_transient,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.grounding import ground
from research_swarm.agents.text import split_into_parts
from research_swarm.config import settings
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas import Finding, Source
from research_swarm.schemas.source import SourceType

logger = logging.getLogger(__name__)

# Confidence a fact starts with, by how well its evidence was located (the verifier revises it).
_GROUNDING_CONFIDENCE = {"quote": 0.6, "passage": 0.5, "none": 0.3}


class ExtractedFact(BaseModel):
    source: int = Field(..., description="Number of the source [S#] this fact comes from")
    sub_question: int = Field(..., description="Number of the sub-question [Q#] it answers")
    claim: str = Field(
        ..., description="One specific fact, self-contained, exact numbers and names kept",
    )
    quote: str = Field(
        default="", description="1-2 sentences copied EXACTLY from that source (max 40 words)",
    )
    relevance: Literal["direct", "background"] = Field(
        default="direct",
        description="direct = answers the sub-question as asked; background = context only",
    )


class Extraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You extract facts from source texts to answer research sub-questions.\n"
    "Rules:\n"
    "- Use ONLY the source texts below. No outside knowledge.\n"
    "- For every sub-question, list EACH distinct fact in the sources that helps answer it,\n"
    "  as its own item (up to {per_pair} per source per sub-question). A fact is one specific,\n"
    "  checkable statement: a name, date, number, result, relationship or conclusion.\n"
    "- Keep exact numbers, units, names and dates. Never turn a number into a vague word.\n"
    "- Each claim must be understandable on its own (no \"it\", \"this study\").\n"
    "- `quote` must be copied EXACTLY, character for character, from that source.\n"
    "- Say what kind of evidence it is when the source makes it clear (RCT, meta-analysis,\n"
    "  animal study, review, preprint, news).\n"
    "- If a source does not help with a sub-question, output nothing for that pair."
    "{scope_rule}"
    "{schema}"
)

_SCOPE_RULE = (
    "\n- `relevance`: \"direct\" when the fact itself answers the sub-question as asked, within\n"
    "  this specific scope: {scope}. \"background\" when it is context about the general\n"
    "  subject that does not address that scope."
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _source_type(value: Any) -> SourceType:
    try:
        return SourceType(value)
    except ValueError:
        return SourceType.web


def pack_sources(docs: list[dict], budget_chars: int) -> list[list[dict]]:
    """Group sources into batches of at most *budget_chars* of text, keeping their order.

    A source longer than the budget is first split at sentence boundaries into parts (same url,
    title suffixed ``(part i/n)``); a single part or source is never split further, so a batch
    only exceeds the budget when it holds exactly one item.
    """
    items: list[dict] = []
    for doc in docs:
        text = doc.get("text", "") or ""
        parts = split_into_parts(text, budget_chars) if len(text) > budget_chars else [text]
        for i, part in enumerate(parts):
            title = doc.get("title", "")
            if len(parts) > 1:
                title = f"{title} (part {i + 1}/{len(parts)})".strip()
            items.append({**doc, "title": title, "text": part})

    batches: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for item in items:
        n = len(item["text"])
        if current and size + n > budget_chars:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += n
    if current:
        batches.append(current)
    return batches


async def extract_facts(
    topic: str,
    sub_questions: list[str],
    sources: list[dict],
    llm: BaseChatModel,
    *,
    session_id: str | None = None,
    agent: str = "document_worker",
    scope: str = "",
) -> list[Finding]:
    """One structured call over *sources* -> grounded findings for *sub_questions*.

    Transient provider errors (after ``ainvoke_with_retry``'s retries) are logged at ERROR and
    traced as ``extractor.failed`` and yield ``[]``; a parse failure is recovered when possible.
    With a *scope* (the question frame's key constraint) each fact is labelled direct /
    background against it; without one every fact's relevance stays "unknown".
    """
    if not sub_questions or not sources:
        return []
    per_pair = settings.extract_max_facts_per_pair

    sq_block = "\n".join(f"[Q{i}] {sq}" for i, sq in enumerate(sub_questions, 1))
    src_block = "\n\n".join(
        f"[S{i}] {s.get('title', '') or s.get('url', '')}\n{s.get('text', '')}"
        for i, s in enumerate(sources, 1)
    )
    user = HumanMessage(content=(
        f"Overall research question: {topic}\n\nSub-questions:\n{sq_block}\n\nSources:\n{src_block}"
    ))
    system = SystemMessage(content=_SYSTEM_PROMPT.format(
        per_pair=per_pair, schema=schema_output_instruction(Extraction),
        scope_rule=_SCOPE_RULE.format(scope=scope) if scope else "",
    ))

    structured = llm.with_structured_output(Extraction)
    try:
        result: Extraction = await ainvoke_with_retry(
            structured, [system, user], session_id=session_id, agent=agent,
        )
    except Exception as exc:  # noqa: BLE001
        if is_transient(exc):
            logger.error("Extractor gave up after retries (%d source(s)): %s", len(sources), exc)
            trace_event(
                session_id, "extractor.failed", "note", n_sources=len(sources),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            return []
        logger.warning("Extractor failed (%d source(s)): %s", len(sources), exc)
        recovered = recover_from_parse_failure(exc, Extraction)
        if recovered is None:
            return []
        result = recovered

    dropped: dict[str, int] = {}
    per_key: dict[tuple[int, int], int] = {}
    counts = {"quote": 0, "passage": 0, "none": 0}
    findings: list[Finding] = []
    for fact in result.facts:
        if not (1 <= fact.source <= len(sources)):
            dropped["bad_source"] = dropped.get("bad_source", 0) + 1
            continue
        if not (1 <= fact.sub_question <= len(sub_questions)):
            dropped["bad_sub_question"] = dropped.get("bad_sub_question", 0) + 1
            continue
        claim = fact.claim.strip()
        if not claim:
            dropped["empty_claim"] = dropped.get("empty_claim", 0) + 1
            continue
        key = (fact.source, fact.sub_question)
        if per_key.get(key, 0) >= per_pair:
            dropped["over_cap"] = dropped.get("over_cap", 0) + 1
            continue
        per_key[key] = per_key.get(key, 0) + 1

        src = sources[fact.source - 1]
        sq = sub_questions[fact.sub_question - 1]
        snippet, how = ground(claim, fact.quote, src.get("text", ""))
        counts[how] = counts.get(how, 0) + 1
        url = src.get("url", "")
        fid = (
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}|{sq.strip().lower()}|{_norm(claim)[:80]}"))
            if url else str(uuid.uuid4())
        )
        findings.append(Finding(
            id=fid, claim=claim, sub_question=sq, grounding=how,
            relevance=fact.relevance if scope else "unknown",
            confidence=_GROUNDING_CONFIDENCE.get(how, 0.5),
            evidence=[Source(
                url=url, title=src.get("title", ""), snippet=snippet,
                source_type=_source_type(src.get("source_type", "pdf")),
                credibility_score=float(src.get("credibility_score", 0.6) or 0.6),
            )],
        ))

    if dropped:
        trace_event(session_id, "extractor.dropped", "note", n=sum(dropped.values()), **dropped)
    trace_event(
        session_id, "ground.result", "note", stage=agent, n_facts=len(findings),
        quote=counts["quote"], passage=counts["passage"], none=counts["none"],
    )
    return findings
