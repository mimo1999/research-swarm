"""RAG query engine factory.

Returns the best available query engine for a session:

  Ollama reachable -> RouterQueryEngine
      +--- VectorQueryEngine  (specific facts, similarity search)
      +--- SummaryQueryEngine (high-level overview, broad questions)
       +--- with SubQuestionQueryEngine wrapper for complex multi-part queries

  Ollama not reachable -> plain VectorQueryEngine
      (response_mode="no_text" -- returns source nodes; no LLM synthesis)

All computation is fully local -- embeddings via HuggingFace, LLM via Ollama.
The research agent's own LLM handles final synthesis; this layer only retrieves.
"""
from __future__ import annotations

import logging

import httpx
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama

from research_swarm.config import settings
from research_swarm.rag.indexes import build_summary_index, load_vector_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama probe
# ---------------------------------------------------------------------------

def probe_ollama() -> bool:
    """Return True if the local Ollama HTTP endpoint is reachable."""
    try:
        resp = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=3.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_local_llm() -> BaseLLM | None:
    """Return an Ollama LLM instance if Ollama is running, else None."""
    if not probe_ollama():
        logger.warning(
            "Ollama not reachable at %s -- routing and decomposition disabled.",
            settings.ollama_base_url,
        )
        return None
    return Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_timeout,
    )


# ---------------------------------------------------------------------------
# Query engine helpers
# ---------------------------------------------------------------------------

def _vector_query_engine(index: VectorStoreIndex, llm: BaseLLM | None) -> BaseQueryEngine:
    """Build a vector query engine.

    With LLM  -> response_mode="compact" (LLM summarises retrieved chunks).
    Without   -> response_mode="no_text"  (returns raw source nodes only).
    """
    if llm is not None:
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=settings.max_sources,
            response_mode="compact",
        )
    return index.as_query_engine(
        similarity_top_k=settings.max_sources,
        response_mode="no_text",
    )


def _summary_query_engine(summary_index: SummaryIndex, llm: BaseLLM) -> BaseQueryEngine:
    return summary_index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )


def _router_query_engine(
    vector_engine: BaseQueryEngine,
    summary_engine: BaseQueryEngine,
    llm: BaseLLM,
) -> RouterQueryEngine:
    """Combine vector + summary engines under an LLM-driven router."""
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        name="vector_search",
        description=(
            "Useful for answering specific factual questions, retrieving exact claims, "
            "statistics, or detailed information from the research corpus."
        ),
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        name="summary_search",
        description=(
            "Useful for high-level overviews, broad questions about themes, "
            "methodology, or general context across documents."
        ),
    )
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[vector_tool, summary_tool],
    )


def _sub_question_engine(
    router_engine: BaseQueryEngine,
    llm: BaseLLM,
) -> SubQuestionQueryEngine:
    """Wrap the router in a SubQuestionQueryEngine for complex multi-part queries."""
    router_tool = QueryEngineTool.from_defaults(
        query_engine=router_engine,
        name="research_corpus",
        description="The full session research corpus (PDFs, web pages, arXiv papers).",
    )
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[router_tool],
        llm=llm,
        use_async=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_research_query_engine(session_id: str) -> BaseQueryEngine:
    """Return the best available query engine for the given session.

    Checks whether Ollama is reachable and builds accordingly:
      - Ollama UP  -> SubQuestionQueryEngine(RouterQueryEngine(vector, summary))
      - Ollama DOWN -> plain VectorQueryEngine with response_mode="no_text"

    The returned engine is safe to call with `.query(question_str)`.
    Source nodes are always available on the response via `.source_nodes`.
    """
    vector_index = load_vector_index(session_id)
    llm = get_local_llm()

    if llm is None:
        logger.info("Building vector-only query engine for session '%s'.", session_id)
        return _vector_query_engine(vector_index, llm=None)

    logger.info(
        "Ollama available -- building full Router+SubQuestion engine for session '%s'.",
        session_id,
    )

    vector_engine = _vector_query_engine(vector_index, llm)
    summary_index = build_summary_index(session_id, llm, documents=None)
    summary_engine = _summary_query_engine(summary_index, llm)
    router = _router_query_engine(vector_engine, summary_engine, llm)
    return _sub_question_engine(router, llm)
