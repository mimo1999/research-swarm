"""Per-session, per-pool LLM call budget guard, plus a session-wide token cap.

Counts LLM invocations via a LangChain callback handler and raises
``BudgetExceeded`` when a pool's call limit -- or the session's token
limit -- is hit.  Each graph node calls ``get_budget(session_id,
pool=...).check()`` before invoking the LLM so that a runaway session is
terminated gracefully rather than silently burning through cloud credits.

Two pools for the CALL-count limit, so that the part of the graph that can
genuinely run away (the dispatch/worker research loop -- multiple rounds,
multiple tool turns per worker) doesn't starve the part that can't
(critic/fact-checker/writer are each one or a few batched calls, not an
open-ended loop). Without this split, a worker-loop overrun exhausts the
*shared* budget before critic ever runs, and every downstream node
force-degrades in the same breath -- producing a completely empty report
even when the worker loop gathered good findings before it ran out.

  "research" -- supervisor, document pass/workers, dispatch/worker loop.
                Bounded by settings.max_llm_calls.
  "review"   -- critic, fact-checker, writer, LLM judge. Bounded by
                settings.max_review_llm_calls, independently of whatever
                the research pool used.

The TOKEN limit (settings.max_tokens_per_session) is deliberately NOT
split the same way: a call's token cost varies wildly with tool-loop
context length and reasoning output, so "N calls" doesn't bound actual
spend the way it does for a pool with predictable per-call cost. It's
checked as one running total across BOTH pools for the session -- see
session_total_tokens() -- because the thing it protects (a shared,
rate-limited account, e.g. Ollama Cloud's allowance) doesn't care which
pool the tokens came from.

Usage in a node::

    from research_swarm.runtime.budget import get_budget, BudgetExceeded

    budget = get_budget(session_id, pool="review")
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

_DEFAULT_POOL = "research"

# Global registry: (session_id, pool) -> BudgetGuard
_registry: dict[tuple[str, str], BudgetGuard] = {}
_registry_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised when a session has consumed its LLM call budget (per-pool) or
    its token budget (session-wide, across pools -- see ``kind``)."""

    def __init__(
        self,
        session_id: str,
        used: int,
        limit: int,
        pool: str = _DEFAULT_POOL,
        *,
        kind: str = "calls",
    ) -> None:
        self.session_id = session_id
        self.used = used
        self.limit = limit
        self.pool = pool
        self.kind = kind  # "calls" (this pool only) or "tokens" (session-wide)
        if kind == "tokens":
            super().__init__(
                f"Session {session_id!r} exceeded its session-wide token budget: "
                f"{used}/{limit} tokens used."
            )
        else:
            super().__init__(
                f"Session {session_id!r} exceeded {pool!r} budget: {used}/{limit} LLM calls used."
            )


class _BudgetCallback(BaseCallbackHandler):
    """LangChain callback that increments the session counter on every LLM start
    and accumulates token usage on every LLM end."""

    def __init__(self, guard: BudgetGuard) -> None:
        super().__init__()
        self._guard = guard

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:  # noqa: ARG002
        self._guard._increment()

    def on_chat_model_start(  # noqa: ARG002
        self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any
    ) -> None:
        self._guard._increment()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._guard._add_tokens(*_extract_token_usage(response))


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Best-effort (input_tokens, output_tokens) extraction from an LLMResult.

    Tries the modern per-message ``usage_metadata`` (set by ChatAnthropic,
    ChatOpenAI, ChatOllama, ...) first, then falls back to the older
    ``llm_output["token_usage"]`` dict some providers still populate instead.
    Returns (0, 0) if neither is present rather than raising -- token
    accounting must never break a graph run.
    """
    input_tokens = 0
    output_tokens = 0
    try:
        for generations in getattr(response, "generations", []) or []:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if usage:
                    input_tokens += usage.get("input_tokens", 0) or 0
                    output_tokens += usage.get("output_tokens", 0) or 0
        if input_tokens == 0 and output_tokens == 0:
            token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0) or 0
            output_tokens = token_usage.get("completion_tokens", 0) or 0
    except Exception:  # noqa: BLE001 - accounting is best-effort, never fatal
        logger.debug("Token usage extraction failed", exc_info=True)
        return 0, 0
    return input_tokens, output_tokens


class BudgetGuard:
    """Thread-safe per-session, per-pool LLM call counter with a hard limit."""

    def __init__(self, session_id: str, limit: int, pool: str = _DEFAULT_POOL) -> None:
        self.session_id = session_id
        self.limit = limit
        self.pool = pool
        self._count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._lock = threading.Lock()
        self.callback = _BudgetCallback(self)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def used(self) -> int:
        with self._lock:
            return self._count

    @property
    def input_tokens(self) -> int:
        with self._lock:
            return self._input_tokens

    @property
    def output_tokens(self) -> int:
        with self._lock:
            return self._output_tokens

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._input_tokens + self._output_tokens

    def check(self) -> None:
        """Raise ``BudgetExceeded`` if this pool's call limit, or the
        session's token limit (across both pools), has been reached."""
        with self._lock:
            if self._count >= self.limit:
                raise BudgetExceeded(self.session_id, self._count, self.limit, self.pool)
        # Token budget is session-wide (spans both pools), so it's read
        # outside this guard's own lock -- see session_total_tokens().
        from research_swarm.config import settings  # lazy to avoid circulars

        total = session_total_tokens(self.session_id)
        if total >= settings.max_tokens_per_session:
            raise BudgetExceeded(
                self.session_id, total, settings.max_tokens_per_session, self.pool, kind="tokens",
            )

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._input_tokens = 0
            self._output_tokens = 0

    def __repr__(self) -> str:
        return (
            f"BudgetGuard(session={self.session_id!r}, pool={self.pool!r}, "
            f"used={self.used}/{self.limit}, tokens={self.total_tokens})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _increment(self) -> None:
        with self._lock:
            self._count += 1
        logger.debug(
            "Budget[%s/%s] %d/%d LLM calls", self.session_id, self.pool, self._count, self.limit,
        )
        if self._count > self.limit:
            logger.warning(
                "Budget[%s/%s] EXCEEDED: %d/%d calls. Session will be terminated.",
                self.session_id, self.pool, self._count, self.limit,
            )

    def _add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_POOL_SETTING: dict[str, str] = {
    "research": "max_llm_calls",
    "review":   "max_review_llm_calls",
}


def session_total_tokens(session_id: str) -> int:
    """Sum input+output tokens across every pool's guard for *session_id*.

    The token budget is session-wide by design (see module docstring), so
    this is what BudgetGuard.check() consults instead of any single pool's
    own counter.
    """
    with _registry_lock:
        guards = [g for (sid, _pool), g in _registry.items() if sid == session_id]
    return sum(g.total_tokens for g in guards)


def get_budget(session_id: str, limit: int | None = None, pool: str = _DEFAULT_POOL) -> BudgetGuard:
    """Return (or create) the ``BudgetGuard`` for *session_id*'s *pool*.

    *limit* is only used when creating a new guard; existing guards keep their
    original limit. If *limit* is None, the pool's own setting is used:
    ``settings.max_llm_calls`` for "research" (the dispatch/worker loop --
    the part that can genuinely iterate across rounds and tool turns),
    ``settings.max_review_llm_calls`` for "review" (critic/fact-checker/
    writer/judge -- a few batched calls, never an open-ended loop). The two
    pools are independent counters: the research loop exhausting its budget
    does not touch the review pool's remaining allowance, so critic/fact-
    checker/writer can still produce a real report from whatever findings
    the research loop gathered before it ran out.
    """
    key = (session_id, pool)
    with _registry_lock:
        if key not in _registry:
            if limit is None:
                from research_swarm.config import settings  # lazy to avoid circulars
                attr = _POOL_SETTING.get(pool, "max_llm_calls")
                limit = getattr(settings, attr)
            _registry[key] = BudgetGuard(session_id, limit, pool)
        return _registry[key]


def clear_budget(session_id: str) -> None:
    """Remove every pool's budget entry for a session (call on session teardown)."""
    with _registry_lock:
        for key in [k for k in _registry if k[0] == session_id]:
            _registry.pop(key, None)
