"""Per-session credential isolation.

These tests exist to pin down the property that makes bring-your-own-key safe:
a credential bound to one session must never be observable from another. The
previous design wrote user keys onto the global ``settings`` singleton, where
any concurrent session's LLM factory would read them.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from research_swarm.runtime import session_ctx
from research_swarm.runtime.session_ctx import (
    SessionCredentials,
    bind_session,
    current_credentials,
    resolve_api_key,
    resolve_ollama_base_url,
    resolve_ollama_deployment,
    session_scope,
    unbind_session,
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

def test_scope_resolves_the_bound_session_key():
    bind_session("s1", SessionCredentials(anthropic_api_key="key-alpha"))
    with session_scope("s1"):
        assert resolve_api_key("anthropic") == "key-alpha"


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


async def test_concurrent_tasks_do_not_cross_contaminate():
    """Interleaved async sessions must each keep their own key.

    Each task gets its own copy of the ambient context, so a ``session_scope``
    entered inside one task is invisible to the others even while suspended.
    """
    bind_session("a", SessionCredentials(anthropic_api_key="key-a"))
    bind_session("b", SessionCredentials(anthropic_api_key="key-b"))

    async def run(session_id: str, expected: str) -> list[str]:
        seen = []
        with session_scope(session_id):
            for _ in range(5):
                seen.append(resolve_api_key("anthropic"))
                await asyncio.sleep(0)  # force interleaving
        assert all(k == expected for k in seen)
        return seen

    a_seen, b_seen = await asyncio.gather(run("a", "key-a"), run("b", "key-b"))
    assert set(a_seen) == {"key-a"}
    assert set(b_seen) == {"key-b"}


def test_unbind_revokes_immediately():
    bind_session("s1", SessionCredentials(anthropic_api_key="key-alpha"))
    unbind_session("s1")
    with session_scope("s1"):
        assert current_credentials().anthropic_api_key == ""


# ---------------------------------------------------------------------------
# Fallback to server configuration
# ---------------------------------------------------------------------------

def test_unbound_context_falls_back_to_server_key():
    """No session bound -> the server's own key, never another user's."""
    from pydantic import SecretStr

    from research_swarm.config import settings

    original = settings.anthropic_api_key
    try:
        settings.anthropic_api_key = SecretStr("server-key")
        assert resolve_api_key("anthropic") == "server-key"
        with session_scope("unbound-session"):
            assert resolve_api_key("anthropic") == "server-key"
    finally:
        settings.anthropic_api_key = original


def test_partial_credentials_fall_back_per_field():
    """A session supplying only an OpenAI key still gets the server's Anthropic one."""
    from pydantic import SecretStr

    from research_swarm.config import settings

    original = settings.anthropic_api_key
    try:
        settings.anthropic_api_key = SecretStr("server-anthropic")
        bind_session("s1", SessionCredentials(openai_api_key="user-openai"))
        with session_scope("s1"):
            assert resolve_api_key("openai") == "user-openai"
            assert resolve_api_key("anthropic") == "server-anthropic"
    finally:
        settings.anthropic_api_key = original


def test_ollama_overrides_resolve_per_session():
    bind_session(
        "s1",
        SessionCredentials(ollama_base_url="http://remote:11434", ollama_deployment="cloud"),
    )
    with session_scope("s1"):
        assert resolve_ollama_base_url() == "http://remote:11434"
        assert resolve_ollama_deployment() == "cloud"
    # Outside the scope, the server defaults apply again.
    from research_swarm.config import settings

    assert resolve_ollama_base_url() == settings.ollama_base_url


# ---------------------------------------------------------------------------
# Secrets must not be renderable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("render", [repr, str, "{}".format])
def test_credentials_never_render_secret_values(render):
    creds = SessionCredentials(
        anthropic_api_key="sk-ant-supersecret",
        openai_api_key="sk-openai-supersecret",
        ollama_api_key="ollama-supersecret",
    )
    rendered = render(creds)
    assert "supersecret" not in rendered
    assert "sk-ant" not in rendered
    # Presence is still observable for debugging.
    assert "anthropic_api_key=<set:" in rendered


def test_credentials_repr_marks_unset_keys():
    assert "anthropic_api_key=<unset>" in repr(SessionCredentials())


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

def test_expired_binding_is_dropped(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(session_ctx.time, "monotonic", lambda: clock["now"])

    bind_session("s1", SessionCredentials(anthropic_api_key="key-alpha"))
    with session_scope("s1"):
        assert resolve_api_key("anthropic") == "key-alpha"

        clock["now"] += session_ctx.CREDENTIAL_TTL_SECONDS + 1
        assert current_credentials().anthropic_api_key == ""

    assert "s1" not in session_ctx._registry


def test_bind_sweeps_other_expired_sessions(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(session_ctx.time, "monotonic", lambda: clock["now"])

    bind_session("stale", SessionCredentials(anthropic_api_key="stale-key"))
    clock["now"] += session_ctx.CREDENTIAL_TTL_SECONDS + 1
    bind_session("fresh", SessionCredentials(anthropic_api_key="fresh-key"))

    assert "stale" not in session_ctx._registry
    assert "fresh" in session_ctx._registry


# ---------------------------------------------------------------------------
# Integration with the LLM factory
# ---------------------------------------------------------------------------

def test_get_agent_llm_uses_the_session_key():
    from research_swarm.agents import base

    bind_session("s1", SessionCredentials(anthropic_api_key="session-key"))
    with patch.object(base, "ChatAnthropic") as mock_cls, session_scope("s1"):
        base.get_agent_llm(provider="anthropic", model="claude-haiku-3-5")

    assert mock_cls.call_args.kwargs["api_key"] == "session-key"


def test_get_agent_llm_ollama_uses_session_url_and_token():
    from research_swarm.agents import base

    bind_session(
        "s1",
        SessionCredentials(ollama_api_key="ollama-token", ollama_base_url="http://remote:11434"),
    )
    with patch("langchain_ollama.ChatOllama") as mock_cls, session_scope("s1"):
        base.get_agent_llm(provider="ollama", model="gemma4:e2b")

    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"] == "http://remote:11434"
    assert kwargs["client_kwargs"] == {"headers": {"Authorization": "Bearer ollama-token"}}


def test_get_agent_llm_ollama_sends_no_auth_header_without_a_token():
    from research_swarm.agents import base

    with patch("langchain_ollama.ChatOllama") as mock_cls:
        base.get_agent_llm(provider="ollama", model="gemma4:e2b")

    assert mock_cls.call_args.kwargs["client_kwargs"] == {}


def test_tiered_standard_model_follows_session_deployment():
    """Two sessions on different deployments must resolve different models."""
    from research_swarm.agents import base
    from research_swarm.config import settings

    bind_session("local-user", SessionCredentials(ollama_deployment="local"))
    bind_session("cloud-user", SessionCredentials(ollama_deployment="cloud"))

    with patch("langchain_ollama.ChatOllama") as mock_cls:
        with session_scope("local-user"):
            base.get_tiered_llm(tier="standard", provider_override="ollama")
        local_model = mock_cls.call_args.kwargs["model"]

        with session_scope("cloud-user"):
            base.get_tiered_llm(tier="standard", provider_override="ollama")
        cloud_model = mock_cls.call_args.kwargs["model"]

    assert local_model == settings.tier_standard_model_local
    assert cloud_model == settings.tier_standard_model_cloud
