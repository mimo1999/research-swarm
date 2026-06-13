"""Cross-encoder reranker for RAG retrieval.

Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — a 22 MB model that runs on
CPU and provides substantially better relevance ordering than raw cosine
similarity alone.  The model is loaded once and cached via ``lru_cache``.

Public API::

    from research_swarm.rag.reranker import rerank

    reranked = rerank(query="GAMs for temporal data", chunks=source_dicts, top_k=5)

*chunks* is a list of Source-like dicts that must have at least ``"snippet"``
and ``"url"`` keys (matching the format returned by the retriever tool).
Chunks with an empty snippet are passed through without scoring.
"""
from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@functools.lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load (and cache) the cross-encoder model.

    Downloads ~22 MB to ``~/.cache/huggingface`` on first use.
    Subsequent calls return the cached instance — no re-download.
    """
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model ms-marco-MiniLM-L-6-v2 ...")
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded.")
        return model
    except Exception as exc:
        logger.warning("Could not load cross-encoder (%s) — reranking disabled.", exc)
        return None


_RERANK_MAX_WORDS = 8
"""Skip reranking when the query exceeds this many words.

``ms-marco-MiniLM-L-6-v2`` was trained on short MS MARCO keyword queries
(typically 2–5 words).  Long, sentence-style queries — such as scientific
claims or natural-language questions — are out-of-distribution and the
cross-encoder consistently mis-scores them, reducing nDCG versus dense
retrieval alone.  Returning the dense-ranked list unchanged is safer.
"""


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Return *chunks* sorted by cross-encoder relevance to *query*.

    Args:
        query:   The research sub-question used for retrieval.
        chunks:  List of Source dicts (must have ``"snippet"`` key).
        top_k:   If given, return at most this many chunks.

    Returns:
        A new list ordered by descending relevance score.  The original
        ``"credibility_score"`` values are preserved; a ``"rerank_score"``
        key is added for transparency.
    """
    if not chunks:
        return chunks

    # Long sentence-style queries are out-of-distribution for ms-marco models.
    if len(query.split()) > _RERANK_MAX_WORDS:
        logger.debug(
            "Skipping rerank: query has %d words (> %d threshold).",
            len(query.split()), _RERANK_MAX_WORDS,
        )
        return chunks[:top_k] if top_k else chunks

    model = _get_cross_encoder()
    if model is None:
        # Graceful degradation: return original order if model unavailable.
        return chunks[:top_k] if top_k else chunks

    # Build (query, passage) pairs for the cross-encoder.
    # Prepend title (same signal the dense retriever uses) and let the
    # tokenizer handle truncation at max_length tokens — not [:512] chars
    # which was only ~100 tokens, starving the model of context.
    passages = [
        f"{c.get('title', '')}\n{c.get('snippet', '')}".strip()
        for c in chunks
    ]
    pairs = list(zip([query] * len(chunks), passages))

    try:
        scores: list[float] = model.predict(pairs).tolist()
    except Exception as exc:
        logger.warning("Cross-encoder prediction failed (%s) — using original order.", exc)
        return chunks[:top_k] if top_k else chunks

    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    result = []
    for score, chunk in scored:
        enriched = {**chunk, "rerank_score": round(float(score), 4)}
        result.append(enriched)

    return result[:top_k] if top_k else result
