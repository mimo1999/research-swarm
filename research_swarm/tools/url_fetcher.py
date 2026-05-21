"""URL fetch tool — downloads a webpage and extracts readable text."""
import re
import textwrap

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.schemas.source import Source, SourceType


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchSwarm/1.0; +https://github.com/research-swarm)"
    )
}

_BOILERPLATE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "noscript"]


class URLFetchInput(BaseModel):
    url: str = Field(..., description="URL to fetch and extract text from")
    max_chars: int = Field(
        default=3000, ge=100, le=20000, description="Maximum characters of text to return"
    )


def _extract_title(soup: BeautifulSoup) -> str:
    tag = soup.find("title")
    if tag:
        return tag.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _readability_extract(soup: BeautifulSoup) -> str:
    """Remove boilerplate tags, then prefer <article>/<main> over full body."""
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()

    for selector in ("article", "main", '[role="main"]', ".post-content", ".entry-content"):
        candidate = soup.select_one(selector)
        if candidate:
            return candidate.get_text(separator=" ", strip=True)

    body = soup.find("body")
    return body.get_text(separator=" ", strip=True) if body else soup.get_text(separator=" ", strip=True)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


@tool("fetch_url", args_schema=URLFetchInput)
def fetch_url(url: str, max_chars: int = 3000) -> dict:
    """Fetch a URL and return a Source dict with extracted readable text as the snippet.

    Uses BeautifulSoup readability heuristics to strip navigation and boilerplate.
    """
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return Source(
                url=url,
                title="",
                snippet=f"[Non-HTML content: {content_type}]",
                source_type=SourceType.web,
                credibility_score=0.5,
            ).model_dump(mode="json")

        soup = BeautifulSoup(resp.text, "html.parser")
        title = _extract_title(soup)
        raw_text = _readability_extract(soup)
        text = _collapse_whitespace(raw_text)
        snippet = textwrap.shorten(text, width=max_chars, placeholder="…")

        source = Source(
            url=url,
            title=title,
            snippet=snippet,
            source_type=SourceType.web,
            credibility_score=0.6,
        )
        return source.model_dump(mode="json")

    except httpx.HTTPStatusError as exc:
        return Source(
            url=url,
            title="",
            snippet=f"[HTTP {exc.response.status_code}: {exc.response.reason_phrase}]",
            source_type=SourceType.web,
            credibility_score=0.0,
        ).model_dump(mode="json")
    except Exception as exc:
        return Source(
            url=url,
            title="",
            snippet=f"[Fetch error: {type(exc).__name__}]",
            source_type=SourceType.web,
            credibility_score=0.0,
        ).model_dump(mode="json")
