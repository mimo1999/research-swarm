"""arXiv search tool -- queries the arXiv API and returns Source objects."""
import io
import textwrap

import arxiv
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pypdf import PdfReader

from research_swarm.schemas.source import Source, SourceType
from research_swarm.utils.security import sanitize_fetched_content


class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="arXiv search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Max papers to return")
    fetch_pdf_text: bool = Field(
        default=False,
        description="If True, download and extract text from the first page of each PDF",
    )


def _extract_pdf_snippet(pdf_url: str, max_chars: int = 800) -> str:
    """Download a PDF and extract a text snippet from the first two pages."""
    try:
        resp = httpx.get(pdf_url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        text = ""
        for page in reader.pages[:2]:
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                break
        return textwrap.shorten(text.strip(), width=max_chars, placeholder="...")
    except Exception:
        return ""  # non-fatal — abstract snippet is still available


@tool("arxiv_search", args_schema=ArxivSearchInput)
def arxiv_search(
    query: str, max_results: int = 5, fetch_pdf_text: bool = False
) -> list[dict]:
    """Search arXiv for papers matching the query.

    Returns JSON-serialisable Source dicts. Set fetch_pdf_text=True to
    include a snippet extracted from each paper's PDF (slower).
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    sources: list[dict] = []
    try:
        results = list(client.results(search))
    except Exception as exc:
        return [
            Source(
                url=f"arxiv://search/{query}",
                title="arXiv search unavailable",
                snippet=f"[Search error: {type(exc).__name__}]",
                source_type=SourceType.arxiv,
                credibility_score=0.0,
            ).model_dump(mode="json")
        ]
    for result in results:
        abstract_snippet = textwrap.shorten(
            result.summary.replace("\n", " "), width=600, placeholder="..."
        )
        snippet = sanitize_fetched_content(abstract_snippet)
        if fetch_pdf_text:
            pdf_snippet = _extract_pdf_snippet(result.pdf_url)
            if pdf_snippet:
                snippet = sanitize_fetched_content(pdf_snippet)

        source = Source(
            url=result.entry_id,
            title=result.title,
            snippet=snippet,
            source_type=SourceType.arxiv,
            credibility_score=0.85,  # arXiv is a reputable preprint server
        )
        sources.append(source.model_dump(mode="json"))
    return sources
