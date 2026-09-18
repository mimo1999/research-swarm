"""Tests for AgentState schema migrations."""
from __future__ import annotations

from research_swarm.runtime.migrations import (
    CURRENT_SCHEMA_VERSION,
    migrate_state,
)


def test_v0_gets_model_defaults():
    """v0 state (no schema_version) must receive model_provider/model_name defaults."""
    state = {"session_id": "old-session", "findings": [], "critiques": [], "messages": []}
    result = migrate_state(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "model_provider" in result
    assert "model_name" in result
    assert result["model_provider"] != ""


def test_idempotent():
    """Applying migrate_state twice must be the same as applying it once."""
    state = {"session_id": "idem"}
    once = migrate_state(state)
    twice = migrate_state(once)
    assert once == twice
