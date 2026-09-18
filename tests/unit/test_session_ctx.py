"""Per-session credential isolation.

These tests exist to pin down the property that makes bring-your-own-key safe:
a credential bound to one session must never be observable from another. The
previous design wrote user keys onto the global ``settings`` singleton, where
any concurrent session's LLM factory would read them.
"""
from __future__ import annotations

import pytest

from research_swarm.runtime import session_ctx
from research_swarm.runtime.session_ctx import (
    SessionCredentials,
    bind_session,
    resolve_api_key,
    resolve_ollama_base_url,
    resolve_ollama_deployment,
    session_scope,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Never let a binding survive into the next test."""
    session_ctx._registry.clear()
    yield
    session_ctx._registry.clear()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_sibling_session_cannot_see_another_key():
    bind_session("alice", SessionCredentials(anthropic_api_key="alice-key"))
    bind_session("bob", SessionCredentials(anthropic_api_key="bob-key"))

    with session_scope("alice"):
        assert resolve_api_key("anthropic") == "alice-key"
    with session_scope("bob"):
        assert resolve_api_key("anthropic") == "bob-key"


def test_binding_does_not_mutate_global_settings():
    """The regression guard: binding must leave the singleton untouched."""
    from research_swarm.config import settings

    before_key = settings.anthropic_api_key.get_secret_value()
    before_url = settings.ollama_base_url
    before_deployment = settings.ollama_deployment

    bind_session(
        "s1",
        SessionCredentials(
            anthropic_api_key="user-key",
            ollama_base_url="http://user-host:11434",
            ollama_deployment="cloud",
        ),
    )
    with session_scope("s1"):
        resolve_api_key("anthropic")
        resolve_ollama_base_url()
        resolve_ollama_deployment()

    assert settings.anthropic_api_key.get_secret_value() == before_key
    assert settings.ollama_base_url == before_url
    assert settings.ollama_deployment == before_deployment


# ---------------------------------------------------------------------------
# Secrets must not be renderable
# ---------------------------------------------------------------------------

def test_credentials_never_render_secret_values():
    creds = SessionCredentials(
        anthropic_api_key="sk-ant-supersecret",
        openai_api_key="sk-openai-supersecret",
        ollama_api_key="ollama-supersecret",
    )
    for render in (repr, str, "{}".format):
        rendered = render(creds)
        assert "supersecret" not in rendered
        assert "sk-ant" not in rendered
        # Presence is still observable for debugging.
        assert "anthropic_api_key=<set:" in rendered
