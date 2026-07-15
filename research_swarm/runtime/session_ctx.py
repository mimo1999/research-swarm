"""Per-session credentials and connection overrides.

Why this module exists
----------------------
``research_swarm.config.settings`` is a **process-global singleton**.  Anything
written to it is visible to every concurrently-running session in the worker.
That is fine for values loaded once from ``.env``, but it is a credential leak
the moment a request carries its own API key: session A sets
``settings.anthropic_api_key``, session B's LLM factory reads it, and session B's
tokens get billed to A's account.  The same hazard applies to
``ollama_base_url`` / ``ollama_deployment``, which the API previously mutated
per request -- two users on different Ollama deployments would stomp each other
mid-run.

The fix is to keep per-request values off the singleton entirely:

* ``bind_session`` stores a :class:`SessionCredentials` in a registry keyed by
  session id.  Memory only -- these values are never checkpointed, never
  written to ``AgentState``, and never logged.
* ``session_scope`` sets a :class:`contextvars.ContextVar` naming the active
  session.  Each request handler runs in its own asyncio task, and a task
  copies the ambient context at creation time, so concurrent requests cannot
  observe each other's binding.  ``asyncio.to_thread`` and LangGraph's node
  scheduling both propagate context, so graph nodes see the right session.
* The ``resolve_*`` helpers read the active session first and fall back to
  ``settings`` (i.e. the server's own ``.env``) when nothing is bound.

The fallback is deliberate and safe: an unbound context resolves to the
*server's* configured key, never to another user's.  Streamlit and the unit
tests take that path unchanged.
"""
from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

from research_swarm.config import settings

# Bound credentials are dropped this many seconds after binding even if the
# session is never streamed or torn down properly -- a client that POSTs and
# then vanishes must not leave a key resident in memory indefinitely.
CREDENTIAL_TTL_SECONDS = 3600.0


def _redact(value: str) -> str:
    """Render a secret as a non-reversible presence indicator."""
    return f"<set:{len(value)} chars>" if value else "<unset>"


@dataclass(frozen=True)
class SessionCredentials:
    """User-supplied credentials and connection settings for one session.

    Every field is optional; an empty string or ``None`` means "fall back to the
    server's own configuration".  ``__repr__`` is overridden so a stray log
    line, traceback frame, or exception payload can never spill a key.
    """

    anthropic_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    ollama_api_key: str = field(default="", repr=False)
    ollama_base_url: str | None = None
    ollama_deployment: str | None = None

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            # Note: the redacted marker is rendered raw, not via !r -- an
            # f-string conversion would apply to the whole conditional.
            shown = _redact(value) if f.name.endswith("_api_key") else repr(value)
            parts.append(f"{f.name}={shown}")
        return f"SessionCredentials({', '.join(parts)})"

    __str__ = __repr__


_EMPTY = SessionCredentials()

# session_id -> (credentials, bound_at_monotonic)
_registry: dict[str, tuple[SessionCredentials, float]] = {}
_registry_lock = threading.Lock()

_current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "research_swarm_session_id", default=None
)


def _sweep_expired(now: float) -> None:
    """Drop bindings older than the TTL. Caller must hold ``_registry_lock``."""
    expired = [
        sid
        for sid, (_, bound_at) in _registry.items()
        if now - bound_at > CREDENTIAL_TTL_SECONDS
    ]
    for sid in expired:
        _registry.pop(sid, None)


def bind_session(session_id: str, credentials: SessionCredentials) -> None:
    """Register *credentials* for *session_id*, replacing any previous binding."""
    now = time.monotonic()
    with _registry_lock:
        _sweep_expired(now)
        _registry[session_id] = (credentials, now)


def unbind_session(session_id: str) -> None:
    """Forget a session's credentials. Safe to call more than once."""
    with _registry_lock:
        _registry.pop(session_id, None)


def get_session_credentials(session_id: str) -> SessionCredentials:
    """Return the binding for *session_id*, or empty credentials if none/expired."""
    with _registry_lock:
        entry = _registry.get(session_id)
        if entry is None:
            return _EMPTY
        credentials, bound_at = entry
        if time.monotonic() - bound_at > CREDENTIAL_TTL_SECONDS:
            _registry.pop(session_id, None)
            return _EMPTY
        return credentials


@contextmanager
def session_scope(session_id: str | None) -> Iterator[None]:
    """Make *session_id*'s credentials the active ones for this context.

    Wrap the graph run with this so every node, tool, and LLM factory call
    beneath it resolves the right session::

        with session_scope(session_id):
            async for chunk in graph.astream(...):
                ...
    """
    token = _current_session.set(session_id)
    try:
        yield
    finally:
        _current_session.reset(token)


def current_session_id() -> str | None:
    """Return the session id bound to the current context, if any."""
    return _current_session.get()


def current_credentials() -> SessionCredentials:
    """Return the active session's credentials, or empty if nothing is bound."""
    session_id = _current_session.get()
    return get_session_credentials(session_id) if session_id else _EMPTY


# ---------------------------------------------------------------------------
# Resolvers -- session value first, server configuration second
# ---------------------------------------------------------------------------

def resolve_api_key(provider: str) -> str:
    """Return the API key for *provider*: session-supplied, else the server's."""
    credentials = current_credentials()
    if provider == "anthropic":
        return credentials.anthropic_api_key or settings.anthropic_api_key.get_secret_value()
    if provider == "openai":
        return credentials.openai_api_key or settings.openai_api_key.get_secret_value()
    if provider == "ollama":
        # Ollama has no server-side key setting -- a local daemon needs none and
        # the hosted service is authenticated per user.
        return credentials.ollama_api_key
    return ""


def resolve_ollama_base_url() -> str:
    return current_credentials().ollama_base_url or settings.ollama_base_url


def resolve_ollama_deployment() -> str:
    return current_credentials().ollama_deployment or settings.ollama_deployment
