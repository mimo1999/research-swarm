"""Retriever tool -- wraps a LlamaIndex query engine as a LangChain tool.

The query engine is built in Phase 3 (rag/query_engines.py). This module
provides the LangChain @tool wrapper so the Researcher agent can call it.

Two modes:
  make_retriever_tool(factory)  -- bind a custom engine factory (used in tests).
  build_retriever_tool()        -- bind the real get_research_query_engine from Phase 3.
  retriever_stub                -- no-op placeholder (returns []).
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.schemas.source import Source, SourceType


class RetrieverInput(BaseModel):
    query: str = Field(..., description="Natural-language query against the session's RAG index")
    session_id: str = Field(..., description="Session ID to select the correct Chroma collection")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


def make_retriever_tool(get_query_engine: Callable):
    """Factory: returns a LangChain tool bound to a specific query-engine factory.

    Args:
        get_query_engine: callable(session_id: str) -> LlamaIndex query engine.
            Will be provided by rag/query_engines.py in Phase 3.

    Returns:
        A LangChain StructuredTool named 'retrieve_from_rag'.
    """

    @tool("retrieve_from_rag", args_schema=RetrieverInput)
    def retrieve_from_rag(query: str, session_id: str, top_k: int = 5) -> list[dict]:
        """Query the session-scoped RAG index and return matching Source dicts.

        The index contains previously ingested PDFs, arXiv papers, and URLs.
        """
        engine = get_query_engine(session_id)
        # LlamaIndex query engines return a Response object with source_nodes
        response = engine.query(query)

        sources: list[dict] = []
        nodes = getattr(response, "source_nodes", [])[:top_k]
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
        return sources

    return retrieve_from_rag


def build_retriever_tool():
    """Return a LangChain tool backed by the real Phase 3 query engine factory.

    Import is deferred so that tools/ can be imported before rag/ is fully
    initialised (avoids circular imports at module load time).
    """
    from research_swarm.rag.query_engines import get_research_query_engine

    return make_retriever_tool(get_research_query_engine)


# Convenience: a no-op placeholder tool for use before Phase 3 is wired up.
@tool("retrieve_from_rag_stub", args_schema=RetrieverInput)
def retriever_stub(query: str, session_id: str, top_k: int = 5) -> list[dict]:
    """Stub retriever -- returns empty list until Phase 3 is implemented."""
    return []
