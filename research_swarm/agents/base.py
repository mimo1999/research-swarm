"""LLM factory -- returns a LangChain ChatModel for use in agents."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from research_swarm.config import settings
from research_swarm.runtime.session_ctx import (
    resolve_api_key,
    resolve_ollama_base_url,
    resolve_ollama_deployment,
)


def get_tiered_llm(
    tier: str,
    temperature: float = 0.0,
    provider_override: str | None = None,
) -> BaseChatModel:
    """Return a ChatModel for the given cost tier ('fast', 'standard', 'thorough').

    Tiers are configured in ``settings`` as ``tier_{tier}_provider`` and
    ``tier_{tier}_model``.  Falls back to the default provider/model if the
    requested tier is unknown. ``provider_override`` (e.g. the session's
    user-selected provider) takes precedence over the static tier setting
    when given.

    The 'standard' tier (research workers) always resolves to that provider's
    lowest-grade model, since call volume is highest here (once per
    sub-question per tool turn) and workers only need to do tool-calling +
    a short synthesis, not deep reasoning:
      - ollama:    ``tier_standard_model_local``/``_cloud`` based on the
                   session's resolved deployment (local daemon vs.
                   cloud-hosted models need different "smallest" picks)
      - anthropic: ``tier_standard_model_anthropic`` (Haiku)
      - openai:    ``tier_standard_model_openai`` (GPT-5 Nano)
    """
    provider = provider_override or getattr(
        settings, f"tier_{tier}_provider", settings.default_model_provider
    )
    model = getattr(settings, f"tier_{tier}_model", settings.default_model_name)

    if tier == "standard":
        if provider == "ollama":
            model = (
                settings.tier_standard_model_local
                if resolve_ollama_deployment() == "local"
                else settings.tier_standard_model_cloud
            )
        elif provider == "anthropic":
            model = settings.tier_standard_model_anthropic
        elif provider == "openai":
            model = settings.tier_standard_model_openai

    return get_agent_llm(provider=provider, model=model, temperature=temperature)


def get_agent_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
) -> BaseChatModel:
    """Return a ChatModel for the given provider / model.

    Supported providers:
      - ``"anthropic"``  -- Claude via Anthropic API
      - ``"openai"``     -- GPT via OpenAI API
      - ``"ollama"``     -- local model via Ollama (no cloud, no API key)

    Defaults to ``settings.default_model_provider`` and
    ``settings.default_model_name`` when not specified.
    """
    provider = provider or settings.default_model_provider
    model = model or settings.default_model_name

    if provider == "anthropic":
        _key = resolve_api_key("anthropic")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=_key or None,
        )

    if provider == "openai":
        _key = resolve_api_key("openai")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=_key or None,
        )

    if provider == "ollama":
        # langchain_ollama.ChatOllama -- a local daemon needs no credentials.
        # A hosted/cloud daemon does, so pass the session's token as a bearer
        # header when one was supplied.
        from langchain_ollama import ChatOllama  # lazy import
        _key = resolve_api_key("ollama")
        client_kwargs = (
            {"headers": {"Authorization": f"Bearer {_key}"}} if _key else {}
        )
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=resolve_ollama_base_url(),
            client_kwargs=client_kwargs,
        )

    raise ValueError(
        f"Unsupported provider {provider!r}. "
        "Use 'anthropic', 'openai', or 'ollama'."
    )
