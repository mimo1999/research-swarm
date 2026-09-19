"""agents/text.py: the sentence-boundary splitter used to pack oversized sources."""
from __future__ import annotations

from research_swarm.agents.text import split_into_parts


def test_splits_oversized_document_into_multiple_parts():
    text = "This is a test sentence about a topic. " * 500          # ~20,000 chars
    parts = split_into_parts(text, max_chars=8_000)

    assert len(parts) >= 2
    assert all(len(part) <= 9_000 for part in parts)
    assert sum(len(p) for p in parts) >= len(text) * 0.95            # nothing silently dropped
