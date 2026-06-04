"""LLM factory -- returns a LangChain ChatModel for use in agents."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from research_swarm.config import settings


def get_tiered_llm(
    tier: str,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Return a ChatModel for the given cost tier ('fast', 'standard', 'thorough').

    Tiers are configured in ``settings`` as ``tier_{tier}_provider`` and
    ``tier_{tier}_model``.  Falls back to the default provider/model if the
    requested tier is unknown.
    """
    provider = getattr(settings, f"tier_{tier}_provider", settings.default_model_provider)
    model    = getattr(settings, f"tier_{tier}_model",    settings.default_model_name)
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
        _key = settings.anthropic_api_key.get_secret_value()
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=_key or None,
        )

    if provider == "openai":
        _key = settings.openai_api_key.get_secret_value()
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=_key or None,
        )

    if provider == "ollama":
        # langchain_ollama.ChatOllama -- no API key required,
        # all inference runs locally via the Ollama server.
        from langchain_ollama import ChatOllama  # lazy import
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Unsupported provider {provider!r}. "
        "Use 'anthropic', 'openai', or 'ollama'."
    )
