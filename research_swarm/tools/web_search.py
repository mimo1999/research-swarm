"""Tavily web search tool -- returns a list of Source objects."""
import structlog
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from research_swarm.config import settings
from research_swarm.schemas.source import Source, SourceType

logger = structlog.get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    search_depth: str = Field(
        default="basic",
        description="Tavily search depth: 'basic' or 'advanced'",
    )


def _credibility_score(url: str) -> float:
    """Heuristic credibility based on domain."""
    high = (".edu", ".gov", ".org", "arxiv.org", "nature.com", "science.org")
    low = ("reddit.com", "twitter.com", "x.com", "quora.com")
    if any(d in url for d in high):
        return 0.85
    if any(d in url for d in low):
        return 0.3
    return 0.6


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5, search_depth: str = "basic") -> list[dict]:
    """Search the web using Tavily and return a list of Source dicts.

    Returns JSON-serialisable dicts (not Pydantic objects) so LangGraph
    can place them in the message stream without serialisation issues.
    """
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.warning(
            "web_search failed",
            query=query,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return [
            Source(
                url=f"tavily://search/{query}",
                title="Web search unavailable",
                snippet=f"[Search error: {type(exc).__name__}]",
                source_type=SourceType.web,
                credibility_score=0.0,
            ).model_dump(mode="json")
        ]
    sources: list[dict] = []
    for r in response.get("results", []):
        source = Source(
            url=r.get("url", ""),
            title=r.get("title", ""),
            snippet=r.get("content", "")[:1000],
            source_type=SourceType.web,
            credibility_score=_credibility_score(r.get("url", "")),
        )
        sources.append(source.model_dump(mode="json"))
    return sources
