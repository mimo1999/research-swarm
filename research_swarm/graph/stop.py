"""Marginal-gain stop signal for the iterative research loop.

The dispatcher calls ``should_stop()`` after every collect phase.  It returns
True when further research rounds are unlikely to add useful information.

**Low novelty rate** — the fraction of findings produced in the current
round that are genuinely new (not just refinements of already-found claims)
falls below ``novelty_threshold``.  Computed as:
    new_count / max(existing_count, 1)
where "existing" = the findings that were present *before* this round.

The metric requires at least one prior round; the first round always returns
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
) -> tuple[bool, str]:
    """Return ``(stop: bool, reason: str)``.

    Args:
        pre_dispatch_finding_ids: finding IDs that existed BEFORE the round.
        all_findings:             complete findings list after the round merged.
        research_rounds:          how many rounds have completed (incremented
                                  by collect_node BEFORE this check).
        max_rounds:               hard ceiling from config.
        novelty_threshold:        min new-finding fraction to continue.
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

    return False, (
        f"novelty={novelty_rate:.2f} "
        f"— continuing (round {research_rounds}/{max_rounds})"
    )
