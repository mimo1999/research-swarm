"""Evidence packet: the budgeted set of source sentences the synthesis call reads.

Sources are segmented once into sentences, each with a stable sentence ID (``S<source>.<n>``,
e.g. ``S3.4``); every citation in the packet path is one of these IDs, and code resolves it to the
exact source text (see CONTEXT.md). Building the packet is code first:

  1. segment every source (abbreviation- and decimal-safe) and drop exact duplicates;
  2. if everything fits the packet budget, keep it all, in source order;
  3. otherwise group sentences into passages of up to ``PASSAGE_SENTENCES`` consecutive sentences,
     score each passage by term overlap with the question and sub-questions (plus the question
     frame's scope phrases), and -- only then -- let the optional *screener* (the local small
     model, passed in) mark the best-scored candidates relevant or not; keep relevant passages
     first, best score first, until the budget is full.

The large model never sees anything outside the packet, so a citation can only point at text the
run actually read. Token counts are estimated at four characters per token.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from research_swarm.agents.text import terms

CHARS_PER_TOKEN = 4
PASSAGE_SENTENCES = 4
# Candidates handed to the screener, as a multiple of the budget: enough to fill the budget with
# relevant passages even when the screener rejects most of the lexical top.
SCREEN_OVERSAMPLE = 2.0
MIN_SENTENCE_CHARS = 12

# A sentence ends at . ! or ? followed by whitespace and then a capital, digit, quote or bracket
# -- but not after common abbreviations ("et al.", "e.g.", "Fig.") or a single initial ("J.").
_ABBREVIATIONS = {
    "al", "e.g", "i.e", "fig", "figs", "vs", "etc", "approx", "ca", "cf", "dr", "mr", "mrs", "ms",
    "no", "vol", "eq", "eqs", "ref", "refs", "resp", "sec", "st", "inc", "ltd", "co", "jr", "sr",
}
_BOUNDARY_RE = re.compile(r"([.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """*text* as sentences (whitespace-normalised, empty pieces dropped)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences, start = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.start(1) + 1
        head = text[start:end]
        last_word = re.findall(r"([A-Za-z][A-Za-z.]*)\.$", head)
        if last_word:
            word = last_word[-1].lower()
            if word in _ABBREVIATIONS or len(word) == 1:
                continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class PacketSource:
    number: int
    url: str
    title: str = ""
    source_type: str = "web"
    credibility_score: float = 0.6


@dataclass(frozen=True)
class PacketSentence:
    id: str                 # "S3.4"
    source: int             # 1-based source number
    index: int              # 1-based sentence number within its source
    text: str
    sub_question: str = ""  # the sub-question it matches best
    score: float = 0.0


@dataclass(frozen=True)
class Passage:
    """Up to PASSAGE_SENTENCES consecutive sentences of one source: the screening unit."""
    source: int
    sentences: tuple[PacketSentence, ...]
    score: float

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


# (question, sub_questions, passages) -> one bool per passage (relevant or not).
Screener = Callable[[str, list[str], list[Passage]], Awaitable[list[bool]]]


@dataclass
class EvidencePacket:
    question: str
    sub_questions: list[str]
    sources: list[PacketSource]
    sentences: list[PacketSentence]        # kept sentences, in source order
    budget_tokens: int
    stats: dict[str, Any] = field(default_factory=dict)

    def by_id(self) -> dict[str, PacketSentence]:
        return {s.id: s for s in self.sentences}

    def source(self, number: int) -> PacketSource | None:
        return next((s for s in self.sources if s.number == number), None)

    def tokens(self) -> int:
        return sum(estimate_tokens(s.text) for s in self.sentences)

    def render(self) -> str:
        """The packet as the synthesis call reads it: grouped by source, one sentence per line."""
        blocks = []
        for src in self.sources:
            lines = [f"{s.id} {s.text}" for s in self.sentences if s.source == src.number]
            if lines:
                title = src.title or src.url
                blocks.append(f"[S{src.number}] {title}\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for graph state and checkpoints."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidencePacket:
        return cls(
            question=data.get("question", ""),
            sub_questions=list(data.get("sub_questions") or []),
            sources=[PacketSource(**s) for s in data.get("sources") or []],
            sentences=[PacketSentence(**s) for s in data.get("sentences") or []],
            budget_tokens=int(data.get("budget_tokens") or 0),
            stats=dict(data.get("stats") or {}),
        )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _score(text_terms: set[str], query_terms: set[str], scope_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    base = len(text_terms & query_terms) / len(query_terms)
    return base + (0.5 if scope_terms and text_terms & scope_terms else 0.0)


async def build_packet(
    question: str,
    sub_questions: list[str],
    sources: list[dict[str, Any]],
    budget_tokens: int,
    screener: Screener | None = None,
    scope_phrases: list[str] | tuple[str, ...] = (),
) -> EvidencePacket:
    """The evidence packet for *sources* (dicts with ``url``, ``title``, ``text`` and optionally
    ``source_type`` / ``credibility_score``). See the module docstring for the rules."""
    sub_questions = [sq for sq in sub_questions if sq.strip()] or [question]
    sq_terms = [(sq, terms(sq)) for sq in sub_questions]
    query_terms = terms(question).union(*(t for _, t in sq_terms))
    scope_terms = set().union(*(terms(p) for p in scope_phrases)) if scope_phrases else set()

    packet_sources: list[PacketSource] = []
    candidates: list[PacketSentence] = []
    seen: set[str] = set()
    duplicates = 0
    for number, src in enumerate(sources, 1):
        packet_sources.append(PacketSource(
            number=number, url=str(src.get("url", "")), title=str(src.get("title", "")),
            source_type=str(src.get("source_type", "web") or "web"),
            credibility_score=float(src.get("credibility_score", 0.6) or 0.6),
        ))
        for index, text in enumerate(split_sentences(str(src.get("text", ""))), 1):
            key = _norm(text)
            if len(key) < MIN_SENTENCE_CHARS:
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            t = terms(text)
            best_sq = max(sq_terms, key=lambda item, t=t: len(t & item[1]))[0]
            candidates.append(PacketSentence(
                id=f"S{number}.{index}", source=number, index=index, text=text,
                sub_question=best_sq, score=round(_score(t, query_terms, scope_terms), 4),
            ))

    total = sum(estimate_tokens(s.text) for s in candidates)
    stats: dict[str, Any] = {
        "sources": len(packet_sources), "candidate_sentences": len(candidates),
        "candidate_tokens": total, "duplicates": duplicates, "screened_passages": 0,
        "rejected_passages": 0, "fit": "whole",
    }
    if total <= budget_tokens:
        kept = candidates
    else:
        kept = await _fit_budget(candidates, question, sub_questions, budget_tokens, screener,
                                 stats)
    packet = EvidencePacket(question=question, sub_questions=sub_questions,
                            sources=packet_sources, sentences=kept, budget_tokens=budget_tokens,
                            stats=stats)
    stats["kept_sentences"] = len(kept)
    stats["kept_tokens"] = packet.tokens()
    return packet


SCREEN_BATCH = 8


def llm_screener(llm: Any, session_id: str | None = None,
                 batch_size: int = SCREEN_BATCH) -> Screener:
    """A screener backed by the local small model: batched yes/no calls, one per *batch_size*
    passages ("which of these passages help answer the question?"). A failed batch keeps its
    passages (screening only ever narrows the lexical ranking, it never loses a batch)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from research_swarm.agents._utils import (
        ainvoke_with_retry,
        recover_from_parse_failure,
        schema_output_instruction,
    )
    from research_swarm.runtime.trace import trace_event

    class RelevantPassages(BaseModel):
        relevant: list[int] = Field(default_factory=list,
                                    description="Numbers P# of the passages that help answer")

    system = SystemMessage(content=(
        "You screen source passages for a research question. List the numbers of the passages "
        "that contain evidence for answering the question or one of its sub-questions: a "
        "finding, number, method or conclusion about the question's subject. Omit passages that "
        "are only background, methods boilerplate or about something else."
        + schema_output_instruction(RelevantPassages)
    ))
    async def screen(question: str, sub_questions: list[str],
                     passages: list[Passage]) -> list[bool]:
        # Built on first use: a packet that fits its budget never touches the local model.
        structured = llm.with_structured_output(RelevantPassages)
        verdicts: list[bool] = []
        sqs = "\n".join(f"- {sq}" for sq in sub_questions if sq != question)
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            listing = "\n\n".join(f"P{n} {p.text}" for n, p in enumerate(batch, 1))
            user = HumanMessage(content=f"Question: {question}\n"
                                        + (f"Sub-questions:\n{sqs}\n" if sqs else "")
                                        + f"\nPassages:\n{listing}")
            try:
                result = await ainvoke_with_retry(structured, [system, user],
                                                  session_id=session_id, agent="packet_screen")
            except Exception as exc:  # noqa: BLE001
                result = recover_from_parse_failure(exc, RelevantPassages)
                if result is None:
                    trace_event(session_id, "packet.screen_failed", "note",
                                error=f"{type(exc).__name__}: {str(exc)[:200]}")
                    verdicts.extend([True] * len(batch))
                    continue
            keep = {n for n in result.relevant if 1 <= n <= len(batch)}
            verdicts.extend(n in keep for n in range(1, len(batch) + 1))
        return verdicts

    return screen


def _passages(candidates: list[PacketSentence]) -> list[Passage]:
    passages: list[Passage] = []
    by_source: dict[int, list[PacketSentence]] = {}
    for s in candidates:
        by_source.setdefault(s.source, []).append(s)
    for source, sents in by_source.items():
        for i in range(0, len(sents), PASSAGE_SENTENCES):
            chunk = tuple(sents[i:i + PASSAGE_SENTENCES])
            passages.append(Passage(source=source, sentences=chunk,
                                    score=max(s.score for s in chunk)))
    return passages


async def _fit_budget(candidates: list[PacketSentence], question: str, sub_questions: list[str],
                      budget: int, screener: Screener | None,
                      stats: dict[str, Any]) -> list[PacketSentence]:
    """Best passages until the budget is full; the screener (if any) vets the lexical top first."""
    passages = sorted(_passages(candidates), key=lambda p: (-p.score, p.source,
                                                            p.sentences[0].index))
    stats["fit"] = "scored"
    relevant = passages
    if screener is not None:
        pool, pool_tokens = [], 0
        for p in passages:
            if pool_tokens >= budget * SCREEN_OVERSAMPLE:
                break
            pool.append(p)
            pool_tokens += sum(estimate_tokens(s.text) for s in p.sentences)
        verdicts = await screener(question, sub_questions, pool)
        if len(verdicts) == len(pool):
            stats["fit"] = "screened"
            stats["screened_passages"] = len(pool)
            stats["rejected_passages"] = sum(1 for v in verdicts if not v)
            keep = [p for p, v in zip(pool, verdicts, strict=True) if v]
            rest = [p for p in passages if p not in pool]
            # Relevant passages first; unscreened leftovers only if relevant ones don't fill it.
            relevant = keep + rest
    chosen: list[PacketSentence] = []
    used = 0
    for p in relevant:
        size = sum(estimate_tokens(s.text) for s in p.sentences)
        if used + size > budget:
            continue
        chosen.extend(p.sentences)
        used += size
    order = {s.id: n for n, s in enumerate(candidates)}
    return sorted(chosen, key=lambda s: order[s.id])
