"""Locate a model-written quote in its source text, so evidence is real text, not a claim.

The extractors ask the model for a verbatim ``quote`` per fact. Small models often paraphrase,
truncate or re-space it, so the verifier used to judge claims against text that was not in the
source. Here the quote is *located* in the source (exact, then fuzzy) and the surrounding
sentences become the evidence window; if the quote cannot be found the best lexical passage for
the claim is used instead, and a claim with no supporting passage at all is marked ``none``.

Standard library only; everything is deterministic and costs no LLM call.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from research_swarm.agents.text import terms as _terms

MIN_QUOTE_CHARS = 12
FUZZY_SEARCH_CHARS = 60_000          # performance guard: only the head of very long texts
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_TRANSLATE = (
    {ord(c): "'" for c in "‘’´`"}
    | {ord(c): '"' for c in "“”"}
    | {ord(c): "-" for c in "‐‑‒–—−"}
)


@dataclass(frozen=True)
class QuoteMatch:
    start: int          # offsets into the ORIGINAL text
    end: int
    score: float        # 1.0 for exact, the SequenceMatcher ratio for fuzzy
    method: str         # "exact" | "fuzzy"


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes, collapse whitespace runs to one space.

    Returns ``(normalized, index_map)``; ``index_map[i]`` is the offset in *text* of the
    character that produced ``normalized[i]``.
    """
    out: list[str] = []
    index: list[int] = []
    prev_space = True                       # also strips leading whitespace
    for pos, ch in enumerate(text):
        ch = unicodedata.normalize("NFKC", ch.translate(_TRANSLATE)).lower()
        for c in ch:                        # NFKC can expand one char into several
            if c.isspace():
                if prev_space:
                    continue
                c = " "
                prev_space = True
            else:
                prev_space = False
            out.append(c)
            index.append(pos)
    return "".join(out), index


def locate_quote(quote: str, text: str, min_ratio: float = 0.85) -> QuoteMatch | None:
    """Where *quote* sits in *text*: exact (after normalisation) first, then fuzzy over windows
    of 1-3 consecutive sentences; None if it is too short or nothing reaches *min_ratio*."""
    nq, _ = _normalize_with_map(quote.strip(" \"'"))
    nq = nq.strip()
    if len(nq) < MIN_QUOTE_CHARS:
        return None
    nt, idx = _normalize_with_map(text)
    pos = nt.find(nq)
    if pos >= 0:
        return QuoteMatch(idx[pos], idx[pos + len(nq) - 1] + 1, 1.0, "exact")

    quote_terms = _terms(nq)
    if not quote_terms:
        return None
    head = text[:FUZZY_SEARCH_CHARS]
    spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(head) if m.group().strip()]
    best: QuoteMatch | None = None
    for i in range(len(spans)):
        for width in (1, 2, 3):
            if i + width > len(spans):
                break
            start, end = spans[i][0], spans[i + width - 1][1]
            window = head[start:end]
            if len(_terms(window) & quote_terms) < 0.3 * len(quote_terms):
                continue
            nw, _ = _normalize_with_map(window)
            ratio = SequenceMatcher(None, nq, nw.strip(), autojunk=False).ratio()
            if best is None or ratio > best.score:
                best = QuoteMatch(start, end, ratio, "fuzzy")
    return best if best is not None and best.score >= min_ratio else None


def _snap_start(text: str, lo: int, start: int) -> int:
    """Move *lo* forward to just after a sentence boundary in text[lo:start], if there is one."""
    cut = max(text.rfind(". ", lo, start), text.rfind("\n", lo, start))
    return cut + 2 if cut >= 0 else lo


def _snap_end(text: str, end: int, hi: int) -> int:
    """Move *hi* back to the end of the last sentence in text[end:hi], if there is one."""
    cut = max(text.rfind(". ", end, hi), text.rfind("\n", end, hi))
    return cut + 1 if cut >= 0 else hi


def evidence_window(text: str, start: int, end: int, radius: int = 400) -> str:
    """text[start:end] plus up to *radius* chars each side, trimmed to sentence boundaries;
    an ellipsis marks a side that was cut."""
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    if lo > 0:
        lo = _snap_start(text, lo, start)
    if hi < len(text):
        hi = _snap_end(text, end, hi)
    return ("…" if lo > 0 else "") + text[lo:hi].strip() + ("…" if hi < len(text) else "")


def best_passage(claim: str, text: str, size: int = 800) -> tuple[int, int] | None:
    """(start, end) of the run of consecutive sentences, at most *size* chars, sharing the most
    of the claim's terms; None when even the best shares fewer than 30% of them."""
    claim_terms = _terms(claim)
    if not claim_terms:
        return None
    spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if m.group().strip()]
    best: tuple[float, int, int] | None = None
    for i in range(len(spans)):
        start, end = spans[i]
        j = i
        while True:
            share = len(_terms(text[start:end]) & claim_terms) / len(claim_terms)
            if best is None or share > best[0]:
                best = (share, start, end)
            if j + 1 >= len(spans) or spans[j + 1][1] - start > size:
                break
            j += 1
            end = spans[j][1]
    if best is None or best[0] < 0.3:
        return None
    return best[1], best[2]


def ground(claim: str, quote: str, text: str, radius: int = 400) -> tuple[str, str]:
    """(evidence snippet, grounding) with grounding ``quote`` | ``passage`` | ``none``.

    ``quote``: the model's quote was found in *text*; the snippet is the window around it.
    ``passage``: it was not, but a passage shares most of the claim's terms.
    ``none``: no supporting passage; the snippet is just the head of the text.
    """
    snippet, how, _span = ground_span(claim, quote, text, radius)
    return snippet, how


def ground_span(claim: str, quote: str, text: str,
                radius: int = 400) -> tuple[str, str, str]:
    """``ground`` plus the exact source text the fact rests on: the located quote as it appears
    in *text* (``quote``), the matched passage (``passage``), or "" (``none``)."""
    match = locate_quote(quote, text) if quote and quote.strip() else None
    if match is not None:
        return (evidence_window(text, match.start, match.end, radius), "quote",
                text[match.start:match.end])
    span = best_passage(claim, text)
    if span is not None:
        return evidence_window(text, span[0], span[1], radius=0), "passage", text[span[0]:span[1]]
    return text[:400], "none", ""
