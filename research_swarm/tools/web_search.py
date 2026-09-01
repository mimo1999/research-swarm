"""Tavily web search tool -- returns a list of Source objects."""
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from research_swarm.config import settings
from research_swarm.schemas.source import Source, SourceType
from research_swarm.utils.security import sanitize_fetched_content

logger = logging.getLogger(__name__)

# Module-level client singleton: avoids re-constructing on every tool call.
# Re-created lazily when the API key changes (SecretStr comparison by value).
_tavily_client: TavilyClient | None = None
_tavily_key_cache: str = ""


def _get_tavily_client() -> TavilyClient:
    global _tavily_client, _tavily_key_cache  # noqa: PLW0603
    current_key = settings.tavily_api_key.get_secret_value()
    if _tavily_client is None or current_key != _tavily_key_cache:
        _tavily_client = TavilyClient(api_key=current_key)
        _tavily_key_cache = current_key
    return _tavily_client


def is_configured() -> bool:
    """Whether a Tavily key is actually set.

    Unlike anthropic/openai/ollama, there's no session-scoped BYOK path for
    Tavily (SessionCredentials has no tavily_api_key field) -- it's
    server-funded or not available at all. Deployments that don't fund it
    (e.g. the Hugging Face Space) should skip web_search entirely rather
    than call it and get back a useless "[Search error: ...]" placeholder
    source for every query. Callers: _get_researcher_tools/fetch_worker_node
    in graph/nodes.py.
    """
    return bool(settings.tavily_api_key.get_secret_value())


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
        client = _get_tavily_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.warning(
            "web_search failed: query=%r error=%s: %s",
            query,
            type(exc).__name__,
            exc,
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
        raw_snippet = r.get("content", "")[:1000]
        source = Source(
            url=r.get("url", ""),
            title=r.get("title", ""),
            snippet=sanitize_fetched_content(raw_snippet),
            source_type=SourceType.web,
            credibility_score=_credibility_score(r.get("url", "")),
        )
        sources.append(source.model_dump(mode="json"))
    return sources
