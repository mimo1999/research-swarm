"""Tests for AgentState schema migrations."""
from __future__ import annotations

from research_swarm.runtime.migrations import (
    CURRENT_SCHEMA_VERSION,
    migrate_state,
)


def test_current_state_is_noop():
    """A state at CURRENT_SCHEMA_VERSION should pass through migrate_state unchanged."""
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "model_provider": "ollama",
        "model_name": "minimax-m2.5:cloud",
        "findings": [],
        "critiques": [],
        "messages": [],
        "session_id": "test",
        "research_rounds": 0,
        "pre_dispatch_finding_ids": [],
    }
    result = migrate_state(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert result["model_provider"] == "ollama"
    assert result["model_name"] == "minimax-m2.5:cloud"


def test_v0_gets_model_defaults():
    """v0 state (no schema_version) must receive model_provider/model_name defaults."""
    state = {"session_id": "old-session", "findings": [], "critiques": [], "messages": []}
    result = migrate_state(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "model_provider" in result
    assert "model_name" in result
    assert result["model_provider"] != ""


def test_v0_none_lists_become_empty():
    """None list fields must be coerced to [] so reducers don't crash."""
    state = {"schema_version": 0, "findings": None, "critiques": None, "messages": None}
    result = migrate_state(state)
    assert result["findings"] == []
    assert result["critiques"] == []
    assert result["messages"] == []


def test_v0_writer_instructions_default():
    """writer_instructions must default to None, not raise KeyError."""
    state = {"schema_version": 0, "session_id": "x"}
    result = migrate_state(state)
    assert "writer_instructions" in result
    assert result["writer_instructions"] is None


def test_v0_iteration_count_default():
    """iteration_count must default to 0 for old checkpoints."""
    state = {}
    result = migrate_state(state)
    assert result["iteration_count"] == 0


def test_already_migrated_values_not_overwritten():
    """Existing non-None values must not be replaced by migration defaults."""
    state = {
        "schema_version": 0,
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "iteration_count": 5,
        "findings": [{"id": "x", "claim": "c"}],
    }
    result = migrate_state(state)
    assert result["model_provider"] == "openai"
    assert result["model_name"] == "gpt-4o"
    assert result["iteration_count"] == 5
    assert result["findings"] == [{"id": "x", "claim": "c"}]


def test_idempotent():
    """Applying migrate_state twice must be the same as applying it once."""
    state = {"session_id": "idem"}
    once = migrate_state(state)
    twice = migrate_state(once)
    assert once == twice


def test_v1_to_v2_adds_phase4_fields():
    """v1 state should receive Phase-4 dispatch fields."""
    state = {
        "schema_version": 1,
        "session_id": "s1",
        "model_provider": "ollama",
        "model_name": "test",
    }
    result = migrate_state(state)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert result["research_rounds"] == 0
    assert result["pre_dispatch_finding_ids"] == []
    assert result["active_sub_question"] is None
    assert result["active_worker_role"] is None


def test_v1_to_v2_backfills_plan_dict():
    """If plan is a dict without new fields, migration should add defaults."""
    state = {
        "schema_version": 1,
        "plan": {
            "sub_questions": ["q1"],
            "strategy": "test",
            "required_tools": [],
            # missing complexity_score and assignments
        },
    }
    result = migrate_state(state)
    assert result["plan"]["complexity_score"] == 0.5
    assert result["plan"]["assignments"] == []
