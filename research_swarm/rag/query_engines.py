"""RAG query engine factory.

Returns a vector query engine for a session (similarity search over the
session's Chroma-backed index), with an LLM-synthesised response when Ollama
is reachable and raw source-node retrieval (response_mode="no_text") when
it isn't.

A Router(vector, summary) + SubQuestionQueryEngine stack used to sit in front
of this whenever Ollama was reachable, offering query routing and
decomposition. Removed -- see get_research_query_engine's docstring for why
both layers made retrieval strictly less reliable than the plain vector
engine alone, not just unhelpfully fancy.

All computation is fully local -- embeddings via HuggingFace, LLM via Ollama.
The research agent's own LLM handles final synthesis; this layer only retrieves.
"""
from __future__ import annotations

import logging
import time

import httpx
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.ollama import Ollama

from research_swarm.config import settings
from research_swarm.rag.indexes import load_vector_index
from research_swarm.runtime.session_ctx import resolve_ollama_base_url

logger = logging.getLogger(__name__)

# Prevent LlamaIndex from defaulting to OpenAI when no API key is set.
# Individual engine builders pass an explicit llm= when Ollama is available.
# This must run at import time before any index.as_query_engine() call.
try:
    from llama_index.core import Settings as _LlamaSettings
    from llama_index.core.llms.mock import MockLLM as _MockLLM
    if _LlamaSettings._llm is None:
        _LlamaSettings.llm = _MockLLM()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Ollama probe
# ---------------------------------------------------------------------------

_OLLAMA_PROBE_TTL = 60.0  # seconds between live HTTP checks
# Keyed by base URL: sessions may point at different daemons, and a cached
# "unreachable" for one must not answer for another.
_ollama_probe_cache: dict[str, tuple[bool, float]] = {}


def probe_ollama(base_url: str | None = None) -> bool:
    """Return True if the session's Ollama HTTP endpoint is reachable.

    Results are cached per base URL for ``_OLLAMA_PROBE_TTL`` seconds so a down
    server does not block every researcher tool-call with a 3-second timeout.
    """
    base_url = base_url or resolve_ollama_base_url()
    now = time.monotonic()
    cached = _ollama_probe_cache.get(base_url)
    if cached is not None and now - cached[1] < _OLLAMA_PROBE_TTL:
        return cached[0]
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        result = resp.status_code == 200
    except Exception:
        result = False
    _ollama_probe_cache[base_url] = (result, now)
    return result


def get_local_llm() -> BaseLLM | None:
    """Return an Ollama LLM instance if Ollama is running, else None."""
    base_url = resolve_ollama_base_url()
    if not probe_ollama(base_url):
        logger.warning(
            "Ollama not reachable at %s -- routing and decomposition disabled.",
            base_url,
        )
        return None
    return Ollama(
        model=settings.ollama_model,
        base_url=base_url,
        request_timeout=settings.ollama_timeout,
    )


# ---------------------------------------------------------------------------
# Query engine helpers
# ---------------------------------------------------------------------------

def _vector_query_engine(
    index: VectorStoreIndex,
    llm: BaseLLM | None,
    max_sources: int | None = None,
) -> BaseQueryEngine:
    """Build a vector query engine.

    With LLM  -> response_mode="compact" (LLM summarises retrieved chunks).
    Without   -> response_mode="no_text"  (returns raw source nodes only).

    max_sources overrides settings.max_sources so the engine can use the
    per-session value rather than the process-wide default.
    """
    top_k = max_sources if max_sources is not None else settings.max_sources
    if llm is not None:
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=top_k,
            response_mode="compact",
        )
    # No Ollama LLM — MockLLM is set as the LlamaIndex global default at module
    # import time (see top of file), so as_query_engine() won't try to load OpenAI.
    # response_mode="no_text" means the LLM is never actually invoked.
    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="no_text",
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_research_query_engine(
    session_id: str,
    max_sources: int | None = None,
) -> BaseQueryEngine:
    """Return the query engine for the given session: a plain vector engine.

    max_sources controls the vector store's similarity_top_k.  When None the
    value from settings.max_sources is used, preserving backward compatibility.

    The returned engine is safe to call with `.query(question_str)`.
    Source nodes are always available on the response via `.source_nodes`.

    This used to route through RouterQueryEngine(vector, summary) wrapped in
    a SubQuestionQueryEngine whenever Ollama was reachable. Both extra layers
    were broken in a way that made retrieve_from_rag worse than not having
    them at all, not just unhelpful:

    - The summary engine was always empty. build_summary_index() was called
      with documents=None unconditionally -- nothing in this path ever
      reconstructs the session's ingested documents to populate it, so
      querying it can only ever return nothing.
    - The router's LLMSingleSelector has no way to know that, and routes a
      real fraction of queries into that permanently-empty summary index
      instead of the working vector index -- confirmed directly: a query
      against a session with real, successfully-ingested content got routed
      to "summary_search" and returned zero source nodes, while the same
      query against the plain vector engine returned 4 well-scored matches.
    - SubQuestionQueryEngine.from_defaults() additionally reaches for
      llama-index-question-gen-openai internally regardless of which LLM
      provider is configured. That package isn't installable alongside this
      project's llama-index-core/llama-index-llms-ollama version pins
      (conflicting llama-index-core ranges), so this raised ImportError on
      every single call.

    Both failures reach the calling worker as a swallowed "[Tool error: ...]"
    tool result (via the tool loop's generic exception handler) or a
    confident "Empty Response" with no source nodes -- both indistinguishable
    from "no evidence exists" to a worker with no tool turns left to recover
    with (shallow depth's single required call has nowhere else to go).
    Standard/deep depth's later tool turns masked this by recovering via
    web_search, which is why it went unnoticed outside a single-turn depth.
    Skipping straight to the plain vector engine is strictly more reliable
    than what the router could ever deliver while its summary side stays
    unpopulated -- re-introduce routing/decomposition only once there's an
    actual way to feed real documents into the summary index.
    """
    vector_index = load_vector_index(session_id)
    llm = get_local_llm()
    return _vector_query_engine(vector_index, llm, max_sources=max_sources)
