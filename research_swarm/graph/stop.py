"""Marginal-gain stop signal for the iterative research loop.

The dispatcher calls ``should_stop()`` after every collect phase.  It returns
True when further research rounds are unlikely to add useful information.

Two criteria (either alone can trigger a stop):

1. **Low novelty rate** — the fraction of findings produced in the current
   round that are genuinely new (not just refinements of already-found claims)
   falls below ``novelty_threshold``.  Computed as:
       new_count / max(existing_count, 1)
   where "existing" = the findings that were present *before* this round.

2. **High similarity to existing** — the new findings' claims are, on average,
   very similar (cosine) to the existing findings' claims.  When this exceeds
   ``similarity_threshold`` the workers are saying the same things the previous
   rounds already covered.

Both metrics require at least one prior round; the first round always returns
``(False, "first round")`` regardless of thresholds.

The hard cap (``research_rounds >= max_rounds``) always wins.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_stop(
    pre_dispatch_finding_ids: list[str],
    all_findings: list,
    research_rounds: int,
    max_rounds: int,
    novelty_threshold: float = 0.15,
    similarity_threshold: float = 0.85,
) -> tuple[bool, str]:
    """Return ``(stop: bool, reason: str)``.

    Args:
        pre_dispatch_finding_ids: finding IDs that existed BEFORE the round.
        all_findings:             complete findings list after the round merged.
        research_rounds:          how many rounds have completed (incremented
                                  by collect_node BEFORE this check).
        max_rounds:               hard ceiling from config.
        novelty_threshold:        min new-finding fraction to continue.
        similarity_threshold:     max mean cosine similarity to continue.
    """
    if research_rounds >= max_rounds:
        return True, f"Hard cap: {research_rounds}/{max_rounds} rounds completed"

    pre_ids = set(pre_dispatch_finding_ids or [])
    new_findings = [
        f for f in all_findings
        if (f.id if hasattr(f, "id") else f.get("id", "")) not in pre_ids
    ]

    if not pre_ids:
        # First round — no basis for comparison
        return False, "first round — no comparison basis"

    if not new_findings:
        return True, "no new findings produced in this round"

    novelty_rate = len(new_findings) / max(len(pre_ids), 1)
    if novelty_rate < novelty_threshold:
        return True, (
            f"low novelty rate ({novelty_rate:.2f} < {novelty_threshold}): "
            f"{len(new_findings)} new finding(s) vs {len(pre_ids)} existing"
        )

    # Embedding similarity check (graceful degradation: skip if model unavailable)
    mean_sim = _mean_claim_similarity(new_findings, [
        f for f in all_findings
        if (f.id if hasattr(f, "id") else f.get("id", "")) in pre_ids
    ])
    if mean_sim is not None and mean_sim > similarity_threshold:
        return True, (
            f"high similarity to existing ({mean_sim:.3f} > {similarity_threshold}): "
            "new findings too close to what we already know"
        )

    sim_str = f"{mean_sim:.3f}" if mean_sim is not None else "n/a"
    return False, (
        f"novelty={novelty_rate:.2f}, similarity={sim_str} "
        f"— continuing (round {research_rounds}/{max_rounds})"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_claim_similarity(
    new_findings: list,
    existing_findings: list,
) -> float | None:
    """Return mean cosine similarity between new and existing finding claims.

    Uses the cached bge-small embedding model (no extra download).
    Returns None if the model is unavailable so the caller can skip the check.
    """
    if not new_findings or not existing_findings:
        return None

    try:
        from research_swarm.rag.indexes import get_embed_model
        model = get_embed_model()
    except Exception as exc:
        logger.debug("Stop-signal embedding unavailable (%s) — skipping similarity check.", exc)
        return None

    def _claim(f) -> str:
        return (f.claim if hasattr(f, "claim") else f.get("claim", ""))[:512]

    try:
        new_embs  = [model.get_text_embedding(_claim(f)) for f in new_findings]
        ex_embs   = [model.get_text_embedding(_claim(f)) for f in existing_findings]
    except Exception as exc:
        logger.debug("Embedding computation failed (%s) — skipping similarity check.", exc)
        return None

    from research_swarm.rag.indexes import cosine_similarity

    sims: list[float] = []
    for ne in new_embs:
        for ee in ex_embs:
            sims.append(cosine_similarity(ne, ee))

    return sum(sims) / len(sims) if sims else None
