"""LangSmith tracing helpers for the Streamlit UI.

Tracing configuration lives on the ``Settings`` singleton (loaded from ``.env`` like everything
else in this app) rather than read straight from ``os.environ``: pydantic-settings' ``env_file``
loading parses ``.env`` onto the declared ``Settings`` fields, it does not also populate the
real process environment, so a raw ``os.environ.get("LANGSMITH_API_KEY")`` here would stay empty
even with a real key in ``.env``. For the same reason, the tracer below is given an explicit
``langsmith.Client(api_key=...)`` instead of leaving the key to be picked up from the environment
(LangChain's *global* auto-tracer -- used for any LLM call outside a Send'/graph run this module
doesn't wrap -- does still rely on the real environment, which is a separate, pre-existing gap
this module does not attempt to close).
"""
from __future__ import annotations

import logging
from typing import Any

from research_swarm.config import settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    """Whether tracing is configured for this process (``LANGCHAIN_TRACING_V2=true`` and a
    LangSmith API key, both in ``.env``)."""
    return bool(settings.langchain_tracing_v2 and settings.langsmith_api_key.get_secret_value())


def project_name() -> str:
    return settings.langchain_project


def make_tracer(session_id: str) -> Any | None:
    """A ``LangChainTracer`` for one graph run, tagged with *session_id*, or None if tracing is
    off. Pass it as a callback in that run's config so every node and LLM call under it lands in
    one trace; ``run_url(tracer)`` then gives a direct link to that trace.
    """
    if not tracing_enabled():
        return None
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        from langsmith import Client

        client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        return LangChainTracer(
            project_name=project_name(), tags=[f"session:{session_id}"], client=client,
        )
    except Exception:  # noqa: BLE001 -- tracing must never break a run
        logger.debug("LangSmith tracer setup failed", exc_info=True)
        return None


def run_url(tracer: Any | None) -> str | None:
    """Best-effort link to the just-finished run's page in LangSmith; None if unavailable
    (tracing off, no network, or the run hasn't reached LangSmith yet)."""
    if tracer is None:
        return None
    try:
        return tracer.get_run_url()
    except Exception:  # noqa: BLE001
        logger.debug("LangSmith run_url lookup failed", exc_info=True)
        return None
