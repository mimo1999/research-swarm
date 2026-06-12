"""Retriever tool -- wraps a LlamaIndex query engine as a LangChain tool.

The query engine is built in Phase 3 (rag/query_engines.py). This module
provides the LangChain @tool wrapper so the Researcher agent can call it.

Two modes:
  make_retriever_tool(factory)              -- old API; LLM must supply session_id.
  make_retriever_tool(factory, session_id)  -- session_id pre-baked; LLM only provides query.
  build_retriever_tool(session_id=...)      -- real engine factory, session_id pre-baked.
  retriever_stub                            -- no-op placeholder (returns []).
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.schemas.source import Source, SourceType


class RetrieverInput(BaseModel):
    """Full schema -- LLM must supply session_id."""

    query: str = Field(..., description="Natural-language query against the session's RAG index")
    session_id: str = Field(..., description="Session ID to select the correct Chroma collection")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class RetrieverQueryInput(BaseModel):
    """Reduced schema -- session_id is pre-baked into the tool at construction time."""

    query: str = Field(..., description="Natural-language query against the session's RAG index")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


def _do_retrieve(
    get_query_engine: Callable,
    session_id: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Core retrieval logic shared by both tool variants."""
    from research_swarm.rag.reranker import rerank  # lazy import

    engine = get_query_engine(session_id)
    # Fetch more candidates than top_k so the reranker has room to reorder.
    fetch_k = min(top_k * 3, 20)
    response = engine.query(query)

    sources: list[dict] = []
    nodes = getattr(response, "source_nodes", [])[:fetch_k]
    for node in nodes:
        metadata = node.metadata or {}
        source = Source(
            url=metadata.get("url", metadata.get("file_path", "rag://" + session_id)),
            title=metadata.get("title", metadata.get("file_name", "")),
            snippet=node.get_content()[:800],
            source_type=SourceType.retriever,
            credibility_score=float(node.score) if node.score is not None else 0.6,
        )
        sources.append(source.model_dump(mode="json"))

    return rerank(query, sources, top_k=top_k)


def make_retriever_tool(get_query_engine: Callable, session_id: str | None = None):
    """Factory: returns a LangChain tool bound to a specific query-engine factory.

    Args:
        get_query_engine: callable(session_id: str) -> LlamaIndex query engine.
        session_id: when provided the tool's schema omits the session_id field so
            the LLM only needs to supply the query.  The correct collection is
            always used regardless of what the model would have passed.
            When omitted the old schema is used and the LLM must supply session_id.

    Returns:
        A LangChain StructuredTool named 'retrieve_from_rag'.
    """
    if session_id is not None:
        prebaked_id = session_id

        @tool("retrieve_from_rag", args_schema=RetrieverQueryInput)
        def retrieve_from_rag(query: str, top_k: int = 5) -> list[dict]:
            """Query the session-scoped RAG index and return matching Source dicts.

            The index contains previously ingested PDFs, arXiv papers, and URLs.
            Results are re-ranked by a cross-encoder for relevance.
            """
            return _do_retrieve(get_query_engine, prebaked_id, query, top_k)

    else:
        @tool("retrieve_from_rag", args_schema=RetrieverInput)
        def retrieve_from_rag(query: str, session_id: str, top_k: int = 5) -> list[dict]:
            """Query the session-scoped RAG index and return matching Source dicts.

            The index contains previously ingested PDFs, arXiv papers, and URLs.
            Results are re-ranked by a cross-encoder for relevance.
            """
            return _do_retrieve(get_query_engine, session_id, query, top_k)

    return retrieve_from_rag


def build_retriever_tool(max_sources: int | None = None, session_id: str | None = None):
    """Return a LangChain tool backed by the real Phase 3 query engine factory.

    max_sources is forwarded to get_research_query_engine so the vector store's
    similarity_top_k reflects the per-session value rather than the global default.

    session_id, when provided, is pre-baked into the tool so the LLM does not
    need to supply it -- it always queries the correct Chroma collection.

    Import is deferred so that tools/ can be imported before rag/ is fully
    initialised (avoids circular imports at module load time).
    """
    from research_swarm.rag.query_engines import get_research_query_engine

    def _get_engine(sid: str):
        return get_research_query_engine(sid, max_sources=max_sources)

    return make_retriever_tool(_get_engine, session_id=session_id)


# Convenience: a no-op placeholder tool for use before Phase 3 is wired up.
@tool("retrieve_from_rag_stub", args_schema=RetrieverInput)
def retriever_stub(query: str, session_id: str, top_k: int = 5) -> list[dict]:
    """Stub retriever -- returns empty list until Phase 3 is implemented."""
    return []
