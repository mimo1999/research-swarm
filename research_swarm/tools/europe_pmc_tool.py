"""Europe PMC search tool -- queries the Europe PMC REST API and returns Source objects.

Unlike pubmed_tool.py, this needs no API key -- Europe PMC's REST API is fully
open (confirmed live: unauthenticated requests to
www.ebi.ac.uk/europepmc/webservices/rest work with no key/token). It also
indexes MEDLINE content (many results carry source="MED", the same corpus
PubMed draws from) plus open-access full text for a subset of results, which
this tool's URL scheme encodes so fetch_worker_node (graph/nodes.py) can
decide whether to attempt a deep full-text fetch -- see its "PMC" vs generic
article-page URL branches below.
"""
from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.schemas.source import Source, SourceType
from research_swarm.utils.security import sanitize_fetched_content

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TIMEOUT = 20.0


class EuropePmcSearchInput(BaseModel):
    query: str = Field(..., description="Europe PMC search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of papers to return")


def _strip_html(text: str) -> str:
    """Europe PMC's abstractText contains literal tags (e.g. "<h4>Background</h4>...")
    unlike PubMed's plain-text abstracts -- strip them before use."""
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def _article_url(result: dict) -> str:
    """Encode what fetch_worker_node needs to decide on a full-text fetch, the
    same way arxiv_tool.py's abs-page URL lets fetch_worker_node derive a PDF
    URL. A "/article/PMC/{pmcid}" URL signals open-access-with-PMCID (a
    full-text attempt is worth making); anything else means abstract-only.
    """
    pmcid = result.get("pmcid")
    if result.get("isOpenAccess") == "Y" and pmcid:
        return f"https://europepmc.org/article/PMC/{pmcid}"
    source = result.get("source", "MED")
    article_id = result.get("id", "")
    return f"https://europepmc.org/article/{source}/{article_id}"


@tool("europe_pmc_search", args_schema=EuropePmcSearchInput)
def europe_pmc_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Europe PMC for biomedical/life-sciences literature.

    Overlaps with pubmed_search (both draw on MEDLINE) but additionally
    surfaces open-access full text for a subset of results -- reach for this
    when an abstract alone (what pubmed_search gives you) isn't enough.
    Returns JSON-serialisable Source dicts.
    """
    try:
        resp = httpx.get(
            _SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": max_results,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
    except Exception as exc:
        logger.warning(
            "europe_pmc_search failed: query=%r error=%s: %s", query, type(exc).__name__, exc
        )
        return [
            Source(
                url=f"europe_pmc://search/{query}",
                title="Europe PMC search unavailable",
                snippet=f"[Search error: {type(exc).__name__}]",
                source_type=SourceType.europe_pmc,
                credibility_score=0.0,
            ).model_dump(mode="json")
        ]

    sources: list[dict] = []
    for result in results:
        abstract = _strip_html(result.get("abstractText", ""))
        if not abstract:
            continue  # no abstract available -- not useful evidence
        journal = (result.get("journalInfo") or {}).get("journal", {}).get("title", "")
        year = result.get("pubYear", "")
        journal_year = " ".join(b for b in (journal, year) if b)
        title = result.get("title") or f"Europe PMC {result.get('id', '')}"
        sources.append(
            Source(
                url=_article_url(result),
                title=f"{title} ({journal_year})" if journal_year else title,
                snippet=sanitize_fetched_content(abstract[:2000]),
                source_type=SourceType.europe_pmc,
                credibility_score=0.9,  # peer-reviewed, indexed biomedical literature
            ).model_dump(mode="json")
        )

    if not sources:
        sources.append(
            Source(
                url=f"europe_pmc://search/{query}",
                title="No Europe PMC results",
                snippet="No abstracts found for this query.",
                source_type=SourceType.europe_pmc,
                credibility_score=0.0,
            ).model_dump(mode="json")
        )
    return sources
