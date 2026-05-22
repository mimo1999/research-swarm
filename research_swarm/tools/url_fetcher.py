"""URL fetch tool -- downloads a webpage and extracts readable text."""
import re
import textwrap

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.schemas.source import Source, SourceType
from research_swarm.utils.security import sanitize_fetched_content, validate_url

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchSwarm/1.0; +https://github.com/research-swarm)"
    )
}

_BOILERPLATE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "noscript"]
_MAX_REDIRECTS = 5


def _safe_get(url: str, _hops: int = 0) -> httpx.Response:
    """GET *url* following redirects while validating each hop against SSRF rules.

    ``follow_redirects=False`` is used at the httpx level so we can inspect
    every Location header before following it — this prevents an open-redirector
    on a trusted host from bouncing us to an internal address.
    """
    if _hops > _MAX_REDIRECTS:
        raise ValueError(f"Too many redirects (>{_MAX_REDIRECTS})")
    resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=False)
    if resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("location", "")
        if not location:
            return resp  # no Location → return the redirect response as-is
        # Resolve relative redirects against the current URL.
        # Reject protocol-relative URLs (//evil.com/) before urljoin can
        # silently inherit the current scheme.
        if location.startswith("//"):
            raise ValueError(f"Protocol-relative redirect blocked: {location!r}")
        if not location.startswith(("http://", "https://")):
            from urllib.parse import urljoin
            location = urljoin(url, location)
        validate_url(location)  # raises ValueError if the redirect target is internal
        return _safe_get(location, _hops + 1)
    return resp


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
    if body:
        return body.get_text(separator=" ", strip=True)
    return soup.get_text(separator=" ", strip=True)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


@tool("fetch_url", args_schema=URLFetchInput)
def fetch_url(url: str, max_chars: int = 3000) -> dict:
    """Fetch a URL and return a Source dict with extracted readable text as the snippet.

    Uses BeautifulSoup readability heuristics to strip navigation and boilerplate.
    """
    try:
        # Security: block SSRF — reject non-HTTP(S) schemes and private/loopback hosts.
        validate_url(url)

        resp = _safe_get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            # For PDFs, return a useful snippet instead of a dead-end message
            if "application/pdf" in content_type:
                snippet = f"[PDF document — binary content not readable inline. URL: {url}]"
            else:
                snippet = f"[Non-HTML content: {content_type}]"
            return Source(
                url=url,
                title="",
                snippet=snippet,
                source_type=SourceType.web,
                credibility_score=0.3,
            ).model_dump(mode="json")

        soup = BeautifulSoup(resp.text, "html.parser")
        title = _extract_title(soup)
        raw_text = _readability_extract(soup)
        text = _collapse_whitespace(raw_text)
        # Security: strip invisible chars and redact injection patterns.
        text = sanitize_fetched_content(text)
        snippet = textwrap.shorten(text, width=max_chars, placeholder="...")

        source = Source(
            url=url,
            title=title,
            snippet=snippet,
            source_type=SourceType.web,
            credibility_score=0.6,
        )
        return source.model_dump(mode="json")

    except ValueError as exc:
        # Raised by validate_url for blocked schemes / hosts.
        return Source(
            url=url,
            title="",
            snippet=f"[Blocked: {exc}]",
            source_type=SourceType.web,
            credibility_score=0.0,
        ).model_dump(mode="json")
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
