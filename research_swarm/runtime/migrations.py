"""AgentState schema migrations.

When a session is resumed from a checkpoint, its state dict may be missing
fields introduced after the original run.  ``migrate_state`` applies all
necessary migrations in version order so callers always receive a
``CURRENT_SCHEMA_VERSION``-compatible dict without crashing.

Version history:
    0  (implicit) — no schema_version; no model_provider / model_name.
    1  — schema_version added; model_provider / model_name; writer_instructions.
    2  — Phase 4 fields: active_sub_question, active_worker_role,
          research_rounds, pre_dispatch_finding_ids.
          ResearchPlan gains complexity_score + assignments.

Add new versions by appending to the version ladder in ``migrate_state``
and incrementing ``CURRENT_SCHEMA_VERSION``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION: int = 2


def migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *state* upgraded to ``CURRENT_SCHEMA_VERSION``.

    Safe to call on already-current state — it becomes a no-op.
    """
    version = state.get("schema_version", 0)
    if version >= CURRENT_SCHEMA_VERSION:
        return state

    migrated = dict(state)
    if version < 1:
        migrated = _v0_to_v1(migrated)
    if version < 2:
        migrated = _v1_to_v2(migrated)

    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    logger.info(
        "Migrated AgentState from v%d to v%d for session %s",
        version,
        CURRENT_SCHEMA_VERSION,
        state.get("session_id", "?"),
    )
    return migrated


# ---------------------------------------------------------------------------
# Individual migration steps
# ---------------------------------------------------------------------------

def _v0_to_v1(state: dict[str, Any]) -> dict[str, Any]:
    """v0 → v1: fill in fields introduced in the v1 schema."""
    from research_swarm.config import settings  # lazy to avoid circular imports

    patched = dict(state)
    patched.setdefault("model_provider",      settings.default_model_provider)
    patched.setdefault("model_name",          settings.default_model_name)
    patched.setdefault("writer_instructions", None)
    patched.setdefault("human_feedback",      None)
    patched.setdefault("iteration_count",     0)
    patched.setdefault("next_agent",          None)
    patched.setdefault("session_id",          "unknown")
    for list_key in ("findings", "critiques", "messages"):
        if patched.get(list_key) is None:
            patched[list_key] = []
    return patched


def _v1_to_v2(state: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2: add Phase-4 dispatch / stop-signal fields.

    - active_sub_question / active_worker_role: per-worker Send fields;
      None is correct for any non-worker checkpoint.
    - research_rounds: how many dispatch→collect cycles have completed;
      default 0 so migrated sessions start a fresh research loop.
    - pre_dispatch_finding_ids: IDs before last dispatch; empty list safe.
    - ResearchPlan: if a plan exists and lacks complexity_score or assignments,
      backfill sensible defaults so the plan object remains valid.
    """
    patched = dict(state)

    patched.setdefault("active_sub_question",       None)
    patched.setdefault("active_worker_role",         None)
    patched.setdefault("research_rounds",            0)
    patched.setdefault("pre_dispatch_finding_ids",   [])

    # Backfill plan fields if a plan is present as a dict (deserialized checkpoint)
    plan = patched.get("plan")
    if isinstance(plan, dict):
        plan.setdefault("complexity_score", 0.5)
        plan.setdefault("assignments", [])
        patched["plan"] = plan
    # If plan is a Pydantic model the new fields already have defaults — no action needed.

    return patched
