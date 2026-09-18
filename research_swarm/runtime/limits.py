"""Concurrency limits for LLM calls -- two layers, both needed.

* ``limiter()`` -- a per-run, per-loop ``asyncio.Semaphore`` capping ONE fan-out (e.g. the
  document workers), so a single node can't burst the provider.
* ``llm_slot()`` -- ONE process-wide cap on in-flight LLM requests per provider, applied to every
  call made through ``agents/_utils.ainvoke_with_retry``. Ollama Cloud serves roughly one long
  request at a time per account and answers the rest with 429s (a ~300 s queue timeout, or an
  immediate "too many concurrent requests"), so stages that fan out independently -- workers,
  extraction, verifier batches, paper scoring -- must share one budget.

Per-run limiter usage:

    async with limiter("document_worker", session_id, settings.document_worker_concurrency):
        findings = await extract_facts(...)

Limiters are keyed by (event loop, name, session) because a Semaphore must only be used from
the loop that created it -- Streamlit drives each run through its own ``asyncio.run``. Entries
are weakly held, so one disappears once no branch holds it.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_Key = tuple[str, str]
# loop -> {(name, session): Semaphore}. Both levels are weak, so a finished run's limiters
# vanish once no branch holds them, and a closed loop's whole entry goes with the loop.
# (Keying on id(loop) instead could hand a new loop a semaphore left over from a dead one
# whose id got reused.)
_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.WeakValueDictionary[_Key, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()


def limiter(name: str, session_id: str, limit: int) -> asyncio.Semaphore:
    """The semaphore shared by every branch of *name* in *session_id* on the running loop.

    *limit* only matters when the semaphore is first created; later callers get the same
    object. A non-positive limit is treated as 1.
    """
    per_loop = _limiters.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
    key = (name, session_id)
    sem = per_loop.get(key)
    if sem is None:
        sem = asyncio.Semaphore(max(1, limit))
        per_loop[key] = sem
    return sem


# ---------------------------------------------------------------------------
# Process-wide LLM concurrency cap
# ---------------------------------------------------------------------------
#
# ``limiter()`` above bounds one fan-out; ``llm_slot()`` bounds *every* LLM request the process
# has in flight for a provider, whatever stage issued it. It is a threading semaphore (not an
# asyncio one) on purpose: Streamlit drives each run through its own ``asyncio.run`` and FastAPI
# serves runs from several threads, so an asyncio semaphore would only ever cap one loop.
# Acquisition polls instead of blocking a thread, which keeps waiting tasks cancellable.

# Set by graph/nodes.py::_get_tiered_state_llm when a node builds its LLM. A node makes its
# LLM at the start of its own task and gathered child tasks copy the context, so the values are
# visible at the ``ainvoke`` call without threading them through every call site.
current_provider: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_provider", default="ollama",
)
current_llm_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_session", default=None,
)

_POLL_SECONDS = 0.05
_llm_semaphores: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_llm_semaphores_lock = threading.Lock()


def set_llm_context(provider: str, session_id: str | None) -> None:
    """Record which provider (and session) the current task's LLM calls belong to."""
    current_provider.set(provider)
    current_llm_session.set(session_id)


def llm_limit(provider: str) -> int:
    """Max concurrent LLM requests for *provider* (0 or less = unlimited)."""
    from research_swarm.config import settings

    per_provider = getattr(settings, f"max_concurrent_llm_calls_{provider}", None)
    if per_provider is None:
        per_provider = settings.max_concurrent_llm_calls_ollama
    return int(per_provider)


def _llm_semaphore(provider: str, limit: int) -> threading.BoundedSemaphore:
    # Keyed by size as well as provider so changing the setting at runtime takes effect.
    with _llm_semaphores_lock:
        return _llm_semaphores.setdefault((provider, limit), threading.BoundedSemaphore(limit))


@asynccontextmanager
async def llm_slot(provider: str | None = None) -> AsyncIterator[float]:
    """Hold one of the provider's concurrent-request slots; yields seconds spent waiting.

    Hold it for exactly one request -- never across a retry backoff or while awaiting another
    LLM call -- so slots can't be held hostage and nested acquisition (a deadlock) can't occur.
    """
    provider = provider or current_provider.get()
    limit = llm_limit(provider)
    if limit <= 0:
        yield 0.0
        return
    sem = _llm_semaphore(provider, limit)
    started = time.monotonic()
    while not sem.acquire(blocking=False):
        await asyncio.sleep(_POLL_SECONDS)     # cancellation lands here, before any acquire
    try:
        yield time.monotonic() - started
    finally:
        sem.release()
