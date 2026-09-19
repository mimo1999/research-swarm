"""Small deterministic text helpers shared by grounding, gap fill, the extractor and benchmarks."""
from __future__ import annotations

import re

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was what "
    "when where which who why with does do did between about into than then their there".split()
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Rough per-call character budget for one extraction call: ~4 chars/token, leaving headroom for
# the prompt and the structured-output schema.
MAX_DOC_CHARS = 40_000


def terms(text: str) -> set[str]:
    """Lowercased content words of *text* (stopwords removed): the lexical-overlap tokenizer."""
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower()) if w not in _STOPWORDS}


def split_into_parts(text: str, max_chars: int = MAX_DOC_CHARS) -> list[str]:
    """Split *text* into <= max_chars parts at sentence/paragraph boundaries.

    Sentences are packed greedily into parts so each stays under max_chars, instead of a raw
    character cut that could split mid-sentence. A single "sentence" longer than max_chars (no
    punctuation, e.g. a table dump) is hard-cut. Returns [text] unchanged when it already fits.
    """
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        parts.append(current)
    return parts or [text]
