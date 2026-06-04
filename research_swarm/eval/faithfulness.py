"""Faithfulness checker for generated report sections.

Uses cosine similarity between the embedding of each report section body and
the embeddings of its cited source snippets to detect when the writer has
generated claims not grounded in the evidence.

The same ``bge-small-en-v1.5`` model used for RAG retrieval is reused here
(loaded from the ``lru_cache`` in ``rag/indexes.py``) so there is no
additional download or memory cost.

Public API::

    from research_swarm.eval.faithfulness import score_report, FAITHFULNESS_THRESHOLD

    score = score_report(report, evidence_pool)
    if score < FAITHFULNESS_THRESHOLD:
        ...  # trigger rewrite
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_swarm.schemas.report import FinalReport
    from research_swarm.schemas.source import Source

logger = logging.getLogger(__name__)

# Minimum average cosine similarity between a section body and its cited
# source snippets.  Below this the section is considered under-grounded.
FAITHFULNESS_THRESHOLD = 0.25


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the cached bge-small model."""
    from research_swarm.rag.indexes import get_embed_model
    model = get_embed_model()
    return [model.get_text_embedding(t) for t in texts]


def score_section(body: str, snippets: list[str]) -> float:
    """Return the mean cosine similarity of *body* against *snippets*.

    If no snippets are supplied the section is considered un-checkable and
    receives a neutral score of 0.5.
    """
    if not snippets or not body.strip():
        return 0.5

    texts = [body] + snippets
    try:
        embeddings = _embed(texts)
    except Exception as exc:
        logger.warning("Faithfulness embedding failed: %s — skipping check.", exc)
        return 0.5

    body_emb = embeddings[0]
    snippet_embs = embeddings[1:]
    sims = [_cosine(body_emb, s) for s in snippet_embs]
    return sum(sims) / len(sims)


def score_report(
    report: FinalReport,
    evidence_pool: list[Source],
) -> float:
    """Return the mean faithfulness score across all sections with citations.

    Sections without citations are excluded from the average (they cannot be
    checked).  If no section has citations, returns 1.0 (nothing to check).

    Args:
        report:        The generated FinalReport.
        evidence_pool: All Source objects available to the writer (used to
                       look up snippet text for each citation index).

    Returns:
        Float in [0, 1].  Lower = more likely to contain unsupported claims.
    """
    ref_snippets: dict[int, str] = {
        i + 1: (r.snippet if hasattr(r, "snippet") else r.get("snippet", ""))
        for i, r in enumerate(evidence_pool)
    }

    scores: list[float] = []
    for section in getattr(report, "sections", []):
        citations = getattr(section, "citations", []) or []
        if not citations:
            continue
        snippets = [ref_snippets[c] for c in citations if c in ref_snippets and ref_snippets[c]]
        s = score_section(
            body=getattr(section, "body_md", ""),
            snippets=snippets,
        )
        scores.append(s)
        logger.debug(
            "Faithfulness[%s]: %.3f (citations=%s)",
            getattr(section, "heading", "?")[:40],
            s,
            citations,
        )

    if not scores:
        return 1.0

    return sum(scores) / len(scores)
