"""Gap fill: a fixed search -> fetch -> extract step for sub-questions the paper/document pass
left thin. Replaces the tool-calling ReAct worker (an agent loop on a small model made an
unpredictable number of LLM calls, needed thinking on, and produced the 429 bursts).

  1. search the routed tools once (free, no LLM),
  2. rank the results by term overlap with the sub-question and keep the top few,
  3. fetch those web pages (bounded, in parallel; a failed fetch falls back to the search
     snippet) and cut each down to its two most relevant passages,
  4. ONE extraction call turns those passages into grounded findings.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from research_swarm.agents.extractor import extract_facts
from research_swarm.agents.grounding import best_passage
from research_swarm.agents.papers import (
    interleave,
    routed_tools_union,
    search_task,
    tool_registry,
)
from research_swarm.agents.text import terms as _terms
from research_swarm.schemas import Finding

logger = logging.getLogger(__name__)

SEARCH_PER_TOOL = 5
POOL_CAP = 8
FETCH_TIMEOUT_S = 12
FETCH_CHARS = 20_000
PASSAGE_CHARS = 1200
TEXT_CAP = 2500

SourceFn = Callable[[str, str, str], Awaitable[list[dict]]]


def relevant_text(sub_question: str, text: str) -> str:
    """The two passages of *text* most relevant to *sub_question*, joined, at most TEXT_CAP
    chars; the head of the text when nothing shares enough terms with the question."""
    first = best_passage(sub_question, text, size=PASSAGE_CHARS)
    if first is None:
        return text[:TEXT_CAP]
    parts = [text[first[0]:first[1]]]
    rest = text[:first[0]] + " " + text[first[1]:]
    second = best_passage(sub_question, rest, size=PASSAGE_CHARS)
    if second is not None:
        parts.append(rest[second[0]:second[1]])
    return "\n...\n".join(p.strip() for p in parts)[:TEXT_CAP]


async def _fetch_page(url: str) -> str | None:
    from research_swarm.tools import fetch_url

    try:
        page = await asyncio.wait_for(
            asyncio.to_thread(fetch_url.invoke, {"url": url, "max_chars": FETCH_CHARS}),
            timeout=FETCH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - a dead page must not sink the sub-question
        logger.info("Gap fill: fetch failed for %s (%s)", url, exc)
        return None
    snippet = str((page or {}).get("snippet", ""))
    return None if snippet.startswith("[") else snippet


async def web_sources(sub_question: str, query: str, session_id: str, k: int = 3) -> list[dict]:
    """Search, rank and fetch: up to *k* ``{"url", "title", "text", "source_type",
    "credibility_score"}`` dicts for one sub-question."""
    available = tool_registry()
    names = routed_tools_union("other", f"{sub_question} {query}", available)
    results = await search_task(
        sub_question, query, names, available, SEARCH_PER_TOOL, session_id,
    )
    pool = interleave(results, POOL_CAP)
    want = _terms(sub_question)
    pool.sort(key=lambda p: -len(want & _terms(f"{p.get('title', '')} {p.get('snippet', '')}")))
    chosen = pool[:k]

    async def one(item: dict[str, Any]) -> dict:
        text = str(item.get("snippet", ""))
        if item.get("source_type") == "web":
            page = await _fetch_page(item["url"])
            if page:
                text = page
        return {
            "url": item["url"], "title": item.get("title", ""),
            "text": relevant_text(sub_question, text),
            "source_type": item.get("source_type", "web"),
            "credibility_score": item.get("credibility_score", 0.6),
        }

    return list(await asyncio.gather(*(one(i) for i in chosen)))


async def run_gap_fill(
    topic: str, sub_question: str, search_query: str, llm: BaseChatModel, session_id: str,
    source_fn: SourceFn | None = None, scope: str = "",
) -> list[Finding]:
    """Findings for one sub-question from freshly gathered sources (one LLM call); *scope* is
    the question frame's key constraint, used to label each fact direct / background."""
    fn = source_fn or web_sources
    sources = await fn(sub_question, search_query, session_id)
    if not sources:
        return []
    return await extract_facts(
        topic, [sub_question], sources, llm, session_id=session_id, agent="gap_fill",
        scope=scope,
    )
