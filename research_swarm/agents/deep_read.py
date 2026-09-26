"""Deep read: the full text of the top primary papers, reduced to the passages that matter.

The paper scout judges and the extractor reads only title + abstract. The specifics a careful
answer needs -- what exactly a method's exactness covers, variance explained, per-pair results --
live in the paper body, so the only sources that carried them were blogs summarising the paper,
and a report cited those blogs for its strongest claims. This fetches the full text of the best
``settings.deep_read_papers`` arXiv papers among the kept candidates (arXiv's HTML version), keeps
the paragraphs that best match the question and the paper's sub-questions (word overlap, no LLM),
and appends them to the paper's abstract, so extraction quotes the original paper. The paper is
then cited by its arxiv.org/abs URL even when it was found through a mirror.

Costs no extra LLM call: the extraction calls just read a longer source (bounded by
``deep_read_chars`` per paper). A paper whose full text cannot be fetched keeps its abstract.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from research_swarm.agents.papers import is_secondary_source, paper_key
from research_swarm.agents.text import terms
from research_swarm.config import settings
from research_swarm.runtime.trace import trace_event

logger = logging.getLogger(__name__)

ARXIV_HTML = "https://arxiv.org/html/{id}"
ARXIV_ABS = "https://arxiv.org/abs/{id}"
PARAGRAPH_CHARS = 900        # a passage longer than this is cut
MIN_PARAGRAPH_CHARS = 80     # captions, headings and fragments below this are skipped
EXCERPT_HEADER = "Full-text excerpts:"


def _paragraphs(html: str) -> list[str]:
    """Readable paragraphs of an arXiv HTML paper; formulas become their plain-text form."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    for math in soup.find_all("math"):
        # LaTeX alttext, minus the characters the report's markup filter rejects
        math.replace_with(re.sub(r"[\\{}]", "", math.get("alttext") or ""))
    root = soup.select_one("article") or soup.select_one("main") or soup.body or soup
    paragraphs = [" ".join(p.get_text(" ", strip=True).split())
                  for p in root.find_all(["p", "li", "figcaption"])]
    return [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS]


def _fetch_paragraphs(url: str) -> list[str]:
    from research_swarm.tools.url_fetcher import _safe_get
    from research_swarm.utils.security import sanitize_fetched_content, validate_url

    validate_url(url)
    resp = _safe_get(url)
    if resp.status_code != 200:
        return []
    return [sanitize_fetched_content(p) for p in _paragraphs(resp.text)]


def select_passages(paragraphs: list[str], focus: str, budget: int) -> str:
    """The paragraphs sharing most words with *focus*, in document order, within *budget*
    characters; empty when none share any."""
    want = terms(focus)
    scored = sorted(
        ((len(want & terms(p)) / max(1, len(want)), i) for i, p in enumerate(paragraphs)),
        key=lambda s: (-s[0], s[1]),
    )
    chosen: list[int] = []
    used = 0
    for share, i in scored:
        if share <= 0:
            break
        size = min(len(paragraphs[i]), PARAGRAPH_CHARS)
        if used + size > budget:
            continue
        chosen.append(i)
        used += size
    return "\n".join(paragraphs[i][:PARAGRAPH_CHARS] for i in sorted(chosen))


async def deep_read(corpus: list[dict[str, Any]], question: str, frame: Any,
                    session_id: str | None) -> list[dict[str, Any]]:
    """*corpus* (the kept papers, one entry per paper per sub-question) with the full-text
    excerpts of the top primary arXiv papers appended to their abstracts. Entries are copied,
    never mutated."""
    corpus = [dict(p) for p in corpus]
    if settings.deep_read_papers <= 0 or not corpus:
        return corpus
    best: dict[str, dict[str, Any]] = {}
    for p in corpus:
        key = paper_key(p)
        if key is None or is_secondary_source(str(p.get("url", ""))):
            continue
        if key not in best or p.get("score", 0) > best[key].get("score", 0):
            best[key] = p
    top = sorted(best, key=lambda k: -float(best[k].get("score", 0)))[:settings.deep_read_papers]

    focus_extra = " ".join(
        [*(getattr(frame, "define_terms", None) or []),
         getattr(frame, "proof_criterion", "") or "",
         *(getattr(frame, "compare_items", None) or [])]
    )

    async def one(key: str) -> tuple[str, str]:
        arxiv_id = key.split(":", 1)[1]
        try:
            paragraphs = await asyncio.wait_for(
                asyncio.to_thread(_fetch_paragraphs, ARXIV_HTML.format(id=arxiv_id)),
                timeout=settings.deep_read_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 -- an abstract is still usable
            logger.warning("Deep read of %s failed (%s)", key, exc)
            trace_event(session_id, "deep_read.failed", "note", paper=key,
                        error=f"{type(exc).__name__}: {str(exc)[:120]}")
            return key, ""
        sub_questions = {p.get("sub_question", "") for p in corpus if paper_key(p) == key}
        focus = " ".join([question, *sub_questions, focus_extra])
        return key, select_passages(paragraphs, focus, settings.deep_read_chars)

    for key, excerpts in await asyncio.gather(*(one(k) for k in top)):
        trace_event(session_id, "deep_read.paper", "note", paper=key, chars=len(excerpts))
        if not excerpts:
            continue
        for p in corpus:
            if paper_key(p) == key:
                p["snippet"] = f"{p.get('snippet', '')}\n\n{EXCERPT_HEADER}\n{excerpts}"
                p["url"] = ARXIV_ABS.format(id=key.split(":", 1)[1])
                p["source_type"] = "arxiv"
    return corpus
