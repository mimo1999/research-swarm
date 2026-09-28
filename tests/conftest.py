"""Shared test fixtures."""
from __future__ import annotations

import pytest

from research_swarm.runtime import trace


@pytest.fixture(autouse=True)
def _no_query_expansion(monkeypatch):
    """Query expansion runs a live probe search before planning; tests stay offline. Tests of
    the expansion itself turn it back on with the probe patched."""
    from research_swarm.config import settings

    monkeypatch.setattr(settings, "query_expansion_enabled", False)
    # The existing writer tests mock one draft call; the sectioned writer (outline -> sections ->
    # review) has its own tests (test_writer_sections.py), which switch it back on.
    monkeypatch.setattr(settings, "writer_mode", "single")
    # Stages on the large model get a ChatOllama for https://ollama.com built directly, bypassing
    # the `_get_tiered_state_llm` patches the node tests use; keep every stage on its tier.
    monkeypatch.setattr(settings, "large_model", "")
    # The deep read fetches paper full text over the network; test_deep_read.py patches the fetch.
    monkeypatch.setattr(settings, "deep_read_papers", 0)
    monkeypatch.setattr(settings, "depth_profiles", {
        depth: {**profile, "deep_read_papers": 0}
        for depth, profile in settings.depth_profiles.items()
    })


@pytest.fixture(autouse=True)
def _no_trace_files():
    """Unit tests must not write trace files into the real data directory."""
    trace.set_enabled(False)
    yield
    trace.set_enabled(True)
