"""Numeric-token helpers: which numbers in a piece of text are not in the evidence.

Shared by the benchmark (``number_grounding`` over a whole report) and the writer's render step
(``ungrounded_numbers`` per sentence), so there is one definition of "a number that counts".
"""
from __future__ import annotations

import re
from typing import Any

_CITE_RE = re.compile(r"\[\d+(?:[\s,\-–]+\d+)*\]")
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%)?")
# Small bare integers ("3 sources", list markers) are incidental more often than they are
# claims, so only integers of at least this size (and any decimal or percentage) count.
MIN_COUNTED_INT = 11


def extract_numbers(text: str, min_int: int = MIN_COUNTED_INT) -> set[str]:
    """Canonical numeric tokens in *text* (commas removed; citation markers ignored)."""
    out: set[str] = set()
    for whole, frac, pct in _NUM_RE.findall(_CITE_RE.sub(" ", text)):
        canon = whole.replace(",", "") + (f".{frac}" if frac else "")
        if not frac and not pct and int(whole.replace(",", "")) < min_int:
            continue
        out.add(canon)
    return out


def number_grounding(report_text: str, corpus_text: str, prompt: str) -> dict[str, Any]:
    """Which of the report's numbers appear in the supplied corpus (or the question).

    A number that is in neither was computed or invented by the model. It is reported as
    *ungrounded*, not as a hallucination: a derived figure (a sum, a difference) also lands
    here, so read the rate as an upper bound on fabricated numbers.
    """
    numbers = extract_numbers(report_text)
    allowed = extract_numbers(corpus_text, min_int=0) | extract_numbers(prompt, min_int=0)
    ungrounded = sorted(numbers - allowed)
    return {
        "numbers": len(numbers),
        "grounded": len(numbers) - len(ungrounded),
        "rate": None if not numbers else round(1 - len(ungrounded) / len(numbers), 4),
        "ungrounded": ungrounded[:10],
    }


def ungrounded_numbers(sentence: str, allowed_text: str) -> set[str]:
    """Numbers in *sentence* (>= MIN_COUNTED_INT, or any decimal / percentage) that do not
    appear anywhere in *allowed_text* (compared as canonical tokens, commas removed)."""
    return extract_numbers(sentence) - extract_numbers(allowed_text, min_int=0)
