"""PubMed search tool -- queries NCBI E-utilities and returns Source objects.

Biomedical/clinical topics are PubMed's specialty; arXiv is a physics/CS/math
preprint server that barely indexes biomedical literature at all, and generic
web search mixes in low-credibility secondary sources. A worker researching
a medical, clinical, or biological sub-question should reach for this tool
before arxiv_search -- see the ``academic`` role's strategy in workers.py.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_swarm.config import settings
from research_swarm.schemas.source import Source, SourceType
from research_swarm.utils.security import sanitize_fetched_content

logger = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TIMEOUT = 20.0


class PubmedSearchInput(BaseModel):
    query: str = Field(..., description="PubMed search query (supports PubMed syntax)")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of papers to return")


def _api_key_params() -> dict:
    key = settings.ncbi_api_key.get_secret_value()
    return {"api_key": key} if key else {}


def _esearch_pmids(query: str, max_results: int) -> list[str]:
    resp = httpx.get(
        f"{_EUTILS_BASE}/esearch.fcgi",
        params={
            "db": "pubmed", "term": query, "retmode": "json", "retmax": max_results,
            **_api_key_params(),
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _efetch_articles(pmids: list[str]) -> dict[str, dict]:
    """Return {pmid: {"title", "abstract", "journal", "year"}} via EFetch XML."""
    if not pmids:
        return {}
    resp = httpx.get(
        f"{_EUTILS_BASE}/efetch.fcgi",
        params={
            "db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml",
            **_api_key_params(),
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    out: dict[str, dict] = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
        if not pmid:
            continue
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract_parts = ["".join(t.itertext()) for t in article.findall(".//AbstractText")]
        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""
        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""
        out[pmid] = {
            "title": title,
            "abstract": " ".join(p.strip() for p in abstract_parts if p.strip()),
            "journal": journal or "",
            "year": year or "",
        }
    return out


@tool("pubmed_search", args_schema=PubmedSearchInput)
def pubmed_search(query: str, max_results: int = 5) -> list[dict]:
    """Search PubMed for peer-reviewed biomedical/clinical literature.

    Prefer this over arxiv_search for medical, clinical, or biological
    questions -- arXiv does not meaningfully index biomedical literature.
    Returns JSON-serialisable Source dicts.
    """
    try:
        pmids = _esearch_pmids(query, max_results)
        articles = _efetch_articles(pmids)
    except Exception as exc:
        logger.warning(
            "pubmed_search failed: query=%r error=%s: %s", query, type(exc).__name__, exc
        )
        return [
            Source(
                url=f"pubmed://search/{query}",
                title="PubMed search unavailable",
                snippet=f"[Search error: {type(exc).__name__}]",
                source_type=SourceType.pubmed,
                credibility_score=0.0,
            ).model_dump(mode="json")
        ]

    sources: list[dict] = []
    for pmid in pmids:
        art = articles.get(pmid)
        if not art or not art["abstract"]:
            continue  # no abstract available (e.g. letters/errata) -- not useful evidence
        journal_year = " ".join(b for b in (art["journal"], art["year"]) if b)
        title = art["title"] or f"PubMed {pmid}"
        sources.append(
            Source(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                title=f"{title} ({journal_year})" if journal_year else title,
                snippet=sanitize_fetched_content(art["abstract"][:2000]),
                source_type=SourceType.pubmed,
                credibility_score=0.9,  # peer-reviewed, indexed biomedical literature
            ).model_dump(mode="json")
        )

    if not sources:
        sources.append(
            Source(
                url=f"pubmed://search/{query}",
                title="No PubMed results",
                snippet="No abstracts found for this query.",
                source_type=SourceType.pubmed,
                credibility_score=0.0,
            ).model_dump(mode="json")
        )
    return sources
