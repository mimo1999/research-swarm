"""Per-session LLM call budget guard.

Counts LLM invocations via a LangChain callback handler and raises
``BudgetExceeded`` when the per-session limit is hit.  Each graph node
calls ``get_budget(session_id).check()`` before invoking the LLM so that
a runaway session is terminated gracefully (the supervisor forces a writer
call with whatever findings exist) rather than silently burning through
cloud credits.

Usage in a node::

    from research_swarm.runtime.budget import get_budget, BudgetExceeded

    budget = get_budget(session_id)
    try:
        budget.check()
    except BudgetExceeded:
        # return a graceful partial result
        ...
    llm = get_agent_llm(...).with_callbacks([budget.callback])
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Global registry: session_id -> BudgetGuard
_registry: dict[str, BudgetGuard] = {}
_registry_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised when a session has consumed its LLM call budget."""

    def __init__(self, session_id: str, used: int, limit: int) -> None:
        self.session_id = session_id
        self.used = used
        self.limit = limit
        super().__init__(
            f"Session {session_id!r} exceeded budget: {used}/{limit} LLM calls used."
        )


class _BudgetCallback(BaseCallbackHandler):
    """LangChain callback that increments the session counter on every LLM start."""

    def __init__(self, guard: BudgetGuard) -> None:
        super().__init__()
        self._guard = guard

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:  # noqa: ARG002
        self._guard._increment()

    def on_chat_model_start(  # noqa: ARG002
        self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any
    ) -> None:
        self._guard._increment()


class BudgetGuard:
    """Thread-safe per-session LLM call counter with a configurable hard limit."""

    def __init__(self, session_id: str, limit: int) -> None:
        self.session_id = session_id
        self.limit = limit
        self._count = 0
        self._lock = threading.Lock()
        self.callback = _BudgetCallback(self)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def used(self) -> int:
        with self._lock:
            return self._count

    def check(self) -> None:
        """Raise ``BudgetExceeded`` if the limit has already been reached."""
        with self._lock:
            if self._count >= self.limit:
                raise BudgetExceeded(self.session_id, self._count, self.limit)

    def reset(self) -> None:
        with self._lock:
            self._count = 0

    def __repr__(self) -> str:
        return f"BudgetGuard(session={self.session_id!r}, used={self.used}/{self.limit})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _increment(self) -> None:
        with self._lock:
            self._count += 1
        logger.debug("Budget[%s] %d/%d LLM calls", self.session_id, self._count, self.limit)
        if self._count > self.limit:
            logger.warning(
                "Budget[%s] EXCEEDED: %d/%d calls. Session will be terminated.",
                self.session_id, self._count, self.limit,
            )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_budget(session_id: str, limit: int | None = None) -> BudgetGuard:
    """Return (or create) the ``BudgetGuard`` for *session_id*.

    *limit* is only used when creating a new guard; existing guards keep their
    original limit.  If *limit* is None, ``settings.max_llm_calls`` is used.
    """
    with _registry_lock:
        if session_id not in _registry:
            if limit is None:
                from research_swarm.config import settings  # lazy to avoid circulars
                limit = settings.max_llm_calls
            _registry[session_id] = BudgetGuard(session_id, limit)
        return _registry[session_id]


def clear_budget(session_id: str) -> None:
    """Remove the budget entry for a session (call on session teardown)."""
    with _registry_lock:
        _registry.pop(session_id, None)
