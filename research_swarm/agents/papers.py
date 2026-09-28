"""Paper scout + paper worker -- abstract-level research without embeddings.

Flow (all sub-questions of a plan are handled together):
  1. ``routed_tools``      -- each sub-question's ``domain`` (set by the supervisor)
                              picks which literature tools are worth searching, so
                              arXiv isn't queried for a clinical question.
  2. ``search_task``       -- one keyword query (also from the supervisor's plan) per
                              tool, all searched concurrently.
  3. ``interleave``        -- per sub-question, results are deduplicated and taken
                              round-robin across tools up to ``paper_max_candidates``.
  4. ``score_pool``        -- one light-LLM call per sub-question rates that pool's
                              title+abstract opening 0-10 against THAT sub-question
                              (calls run concurrently). Only >= threshold survives.
                              Scoring against several sub-questions in one call was
                              tried and collapsed: the model put nearly everything at
                              0 and gave every sub-question the same score.
  5. ``extract_findings``  -- one call per sub-question over its surviving abstracts
                              producing 1-2 findings per paper, each tied to its paper.

All LLM calls here are structured-output and run with thinking off (the node builds the LLM
for the ``paper_scout`` / ``paper_worker`` stages, see ``settings.no_thinking_stages``).
There is no vector store: the "corpus" is just the list of surviving abstracts carried in
graph state.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.config import settings
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas import Finding

logger = logging.getLogger(__name__)

# Title + the opening of the abstract is enough to judge topical relevance; the
# full abstract is only needed at extraction time.
SCORE_ABSTRACT_CHARS = 350
MIN_ABSTRACT_CHARS = 100

# Which tools each domain is worth searching. "other" (industry, policy, ...)
# leans on the web with arXiv as the only scholarly backstop.
DOMAIN_TOOLS: dict[str, tuple[str, ...]] = {
    "biomedical":         ("pubmed", "europe_pmc", "web"),
    "cs_ml_physics_math": ("arxiv", "web"),
    "other":              ("web", "arxiv"),
}


# ---------------------------------------------------------------------------
# 1-3. Routing, search, candidate selection
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was what "
    "when where which who why with does do did between about into than then their there "
    "current state evidence".split()
)


def keyword_query(text: str, max_terms: int = 8) -> str:
    """Cheap no-LLM fallback query: the sub-question's first content words."""
    words = [
        w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text) if w.lower() not in _STOPWORDS
    ]
    return " ".join(words[:max_terms]) or text[:80]


def tool_registry() -> dict[str, Any]:
    """Available literature tools by name (web only when Tavily is configured)."""
    from research_swarm.tools import arxiv_search, europe_pmc_search, pubmed_search, web_search
    from research_swarm.tools.web_search import is_configured as tavily_configured

    tools: dict[str, Any] = {
        "pubmed": pubmed_search, "europe_pmc": europe_pmc_search, "arxiv": arxiv_search,
    }
    if tavily_configured():
        tools["web"] = web_search
    return tools


def routed_tools(domain: str, available: dict[str, Any]) -> list[str]:
    """Tool names to search for *domain*; every available tool if none match."""
    names = [t for t in DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["other"]) if t in available]
    return names or list(available)


_BIOMED_RE = re.compile(
    r"\b(disease|patients?|clinical|trial|therap|drug|dose|cancer|tumou?r|diet|nutrition|vitamin|"
    r"protein|gene|genetic|infection|virus|vaccine|symptom|syndrome|health|medical|mortality|risk|"
    r"obesity|diabetes|cardio|heart|brain|pregnan|child|supplement|food|eating|meat|fat|sugar|"
    r"cholesterol|blood|immune|inflamm)\w*", re.IGNORECASE,
)
_CS_RE = re.compile(
    r"\b(algorithm|neural|model(?:s|ing)?|learning|transformer|llm|dataset|benchmark|compute|gpu|"
    r"quantum|physics|theorem|proof|equation|optimi[sz])\w*", re.IGNORECASE,
)


def routed_tools_union(domain: str, text: str, available: dict[str, Any]) -> list[str]:
    """Tools for *domain*, widened by keywords in *text*, so one wrong LLM ``domain`` label
    cannot keep a health question off PubMed (or a compute question off arXiv).

    The domain's own tools come first, then PubMed / Europe PMC when the text looks biomedical,
    arXiv when it looks like CS/physics/maths, and always web; only available tools are returned.
    """
    wanted = list(DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["other"]))
    if _BIOMED_RE.search(text or ""):
        wanted += ["pubmed", "europe_pmc"]
    if _CS_RE.search(text or ""):
        wanted.append("arxiv")
    wanted.append("web")
    names: list[str] = []
    for name in wanted:
        if name in available and name not in names:
            names.append(name)
    return names or list(available)


def _usable(item: Any) -> bool:
    if not isinstance(item, dict) or not item.get("url"):
        return False
    snippet = str(item.get("snippet", ""))
    if snippet.startswith("[") or snippet.startswith("No abstracts found"):
        return False  # tool error / empty-result placeholder
    return len(snippet) >= MIN_ABSTRACT_CHARS


async def search_task(
    sub_question: str, query: str, tool_names: list[str], available: dict[str, Any],
    per_tool: int, session_id: str,
) -> dict[str, list[dict]]:
    """Run one query against each named tool concurrently -> {tool: usable results}."""

    async def one(name: str) -> tuple[str, list[dict]]:
        t0 = time.perf_counter()
        try:
            res = await asyncio.to_thread(
                available[name].invoke, {"query": query, "max_results": per_tool},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Paper scout: %s failed for %r (%s)", name, query[:60], exc)
            res = []
        res = res if isinstance(res, list) else []
        usable = [r for r in res if _usable(r)]
        trace_event(
            session_id, "paper_scout.search", "tool", tool=name, args={"query": query},
            dur=round(time.perf_counter() - t0, 3), n_results=len(res), n_usable=len(usable),
            sub_question=sub_question[:60],
        )
        return name, usable

    return dict(await asyncio.gather(*(one(n) for n in tool_names)))


_ARXIV_ID_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
# Hosts that mirror arXiv papers under the same id (arxiv.org/abs, /html, /pdf, alphaxiv,
# emergentmind, huggingface papers, ...): one paper found through three mirrors once supplied
# all six facts of a report and crowded out every other paper.
_ARXIV_MIRROR_HINTS = (
    "arxiv", "alphaxiv", "emergentmind", "huggingface.co/papers", "paperswithcode",
)


# Blogs, social posts and topic aggregators: useful leads, but not primary research. A report
# asked to "cite primary research" cited a Medium post as its first reference.
_SECONDARY_HOSTS = (
    "medium.com", "linkedin.com", "substack.com", "reddit.com", "youtube.com", "quora.com",
    "towardsdatascience.com", "dev.to", "hashnode.", "wikipedia.org",
    "emergentmind.com/topics", "alphaxiv.org/overview", "/blog/", "blog.", ".pages.dev",
    "note.com",
)


def is_secondary_source(url: str) -> bool:
    """True for a blog / social / aggregator page rather than a paper or primary source."""
    u = (url or "").lower()
    return any(h in u for h in _SECONDARY_HOSTS)


def paper_key(item: dict) -> str | None:
    """``arxiv:<id>`` for a result that is an arXiv paper or a known mirror of one, else None."""
    url = str(item.get("url", "")).lower()
    if not any(h in url for h in _ARXIV_MIRROR_HINTS):
        return None
    m = _ARXIV_ID_RE.search(url)
    return f"arxiv:{m.group(1)}" if m else None


def prefilter_candidates(
    candidates: list[dict], sub_question: str, query: str, frame: Any, keep: int,
) -> list[dict]:
    """The *keep* most promising candidates, chosen in code before the LLM relevance scorer.

    The scorer's cost grows with every candidate it reads, so the pool it sees stays at
    ``paper_max_candidates``; this lets the search cast a wider net than that for free. Ranked by
    the share of the sub-question's and query's content words in the title + abstract, plus a
    bonus for mentioning the question's scope (question frame) and a penalty for blogs /
    aggregators; ties keep the search engines' own order.
    """
    if len(candidates) <= keep:
        return candidates
    from research_swarm.agents.expansion import scope_hit
    from research_swarm.agents.text import terms

    want = terms(f"{sub_question} {query}")
    has_scope = frame is not None and getattr(frame, "has_constraint", False)

    def score(item: tuple[int, dict]) -> tuple[float, int]:
        i, c = item
        text = f"{c.get('title', '')} {str(c.get('snippet', ''))[:600]}"
        s = len(want & terms(text)) / max(1, len(want))
        if has_scope and scope_hit(text, frame):
            s += 0.5
        if is_secondary_source(str(c.get("url", ""))):
            s -= 0.2
        return (-s, i)

    return [c for _i, c in sorted(enumerate(candidates), key=score)[:keep]]


def interleave(
    ranked_by_tool: dict[str, list[dict]], cap: int, seen: set[str] | None = None,
) -> list[dict]:
    """Dedupe and take results round-robin across tools, best-ranked first, up to *cap*.

    Round-robin (rather than concatenation) keeps one prolific source from
    crowding out the rest of the scoring call's fixed candidate budget. *seen*
    holds urls/titles already taken elsewhere and is updated in place. The same arXiv paper
    reached through different mirrors (``paper_key``) is kept once.
    """
    seen = seen if seen is not None else set()
    lists = [list(v) for v in ranked_by_tool.values()]
    out: list[dict] = []
    rank = 0
    while len(out) < cap and any(rank < len(lst) for lst in lists):
        for lst in lists:
            if rank >= len(lst) or len(out) >= cap:
                continue
            item = lst[rank]
            url = item["url"].strip().lower()
            title = str(item.get("title", "")).strip().lower()
            key = paper_key(item)
            if url in seen or (title and title in seen) or (key and key in seen):
                continue
            seen.add(url)
            if title:
                seen.add(title)
            if key:
                seen.add(key)
            out.append(item)
        rank += 1
    return out


# ---------------------------------------------------------------------------
# 4. Relevance scoring (one call, all sub-questions)
# ---------------------------------------------------------------------------

class PaperScore(BaseModel):
    paper: int = Field(..., description="The paper's number as listed")
    score: int = Field(..., description="Relevance to the sub-question, integer 0-10")


class PaperScores(BaseModel):
    scores: list[PaperScore] = Field(default_factory=list, description="One entry per paper listed")


_SCORE_SYSTEM = (
    "You rate how relevant each paper is to ONE research sub-question (part of a larger "
    "topic), judging ONLY from its title and the opening of its abstract. Give every paper "
    "an integer 0-10:\n"
    "  9-10  directly addresses the sub-question with specific results or data\n"
    "  8     clearly on-topic: the right subject and question, useful evidence\n"
    "  4-7   tangential: same general field but a different population, technology, "
    "condition or question\n"
    "  0-3   unrelated or only shares keywords\n"
    "Use the full range and judge each paper on its own. A paper on a different disease, "
    "model or problem than the one asked about scores 7 or below even if it uses the same "
    "terminology. When a specific scope of the question is given, a paper on the general "
    "subject that does not address that specific scope scores at most 4. "
    "Return one entry per paper."
    + schema_output_instruction(PaperScores)
)


def _scope_lines(frame: Any) -> str:
    """The frame's constraint for the scorer; empty without one."""
    if frame is None or not getattr(frame, "has_constraint", False):
        return ""
    lines = f"Specific scope of the question: {frame.key_constraint}"
    if frame.constraint_terms:
        lines += f" (also called: {'; '.join(frame.constraint_terms)})"
    lines += ".\n"
    if frame.confusable_topics:
        lines += f"NOT the question: {'; '.join(frame.confusable_topics)}.\n"
    return lines


async def score_pool(
    topic: str, sub_question: str, candidates: list[dict], llm: BaseChatModel,
    frame: Any = None,
) -> dict[int, float]:
    """Return {candidate_index: relevance 0..1} for one sub-question; {} on failure.

    With a question *frame* (agents/expansion.py) the scorer is told the question's specific
    scope and its confusable topics: a paper on the general subject that misses the scope scores
    at most 4, below ``paper_topk_floor``.
    """
    if not candidates:
        return {}
    listing = "\n\n".join(
        f"[{n}] {c.get('title', '')}\n{str(c.get('snippet', ''))[:SCORE_ABSTRACT_CHARS]}"
        for n, c in enumerate(candidates, 1)
    )
    msg = HumanMessage(
        content=(
            f"Topic: {topic}\nSub-question: {sub_question}\n{_scope_lines(frame)}\n"
            f"Papers ({len(candidates)}):\n{listing}"
        )
    )
    structured = llm.with_structured_output(PaperScores)
    try:
        result: PaperScores = await ainvoke_with_retry(
            structured, [SystemMessage(content=_SCORE_SYSTEM), msg], agent="paper_scout",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Relevance scoring failed (%s)", exc)
        recovered = recover_from_parse_failure(exc, PaperScores)
        if recovered is None:
            return {}
        result = recovered
    out: dict[int, float] = {}
    for item in result.scores:
        n = item.paper - 1
        if 0 <= n < len(candidates):
            out[n] = max(0, min(10, int(item.score))) / 10
    return out


def select_papers(
    candidates: list[dict], scores: dict[int, float], threshold: float, limit: int,
    min_keep: int = 0, floor: float = 0.0,
) -> list[dict]:
    """Candidates scoring >= *threshold*, best first, at most *limit* -- topped up if too few.

    If fewer than *min_keep* pass the threshold, the best remaining candidates scoring at least
    *floor* are added (best first) until *min_keep* is reached or none qualify. Topped-up papers
    carry ``"topped_up": True`` so traces and callers can tell them from full passes. The
    threshold stays the bar for "clearly relevant"; the top-up only guards against a strict
    scorer leaving a sub-question with almost nothing, and the paper worker still skips any
    abstract that doesn't help. ``min_keep <= 0`` disables it.
    """
    ranked = sorted(
        ((scores[i], c) for i, c in enumerate(candidates) if i in scores),
        key=lambda t: t[0], reverse=True,
    )
    passed = [(sc, c) for sc, c in ranked if sc >= threshold]
    out = [{**c, "score": sc} for sc, c in passed[:limit]]
    want = min(min_keep, limit)
    if want > len(out):
        near = [(sc, c) for sc, c in ranked if floor <= sc < threshold]
        out += [{**c, "score": sc, "topped_up": True} for sc, c in near[: want - len(out)]]
    return out


def select_topk(
    candidates: list[dict], scores: dict[int, float], k: int, floor: float,
) -> list[dict]:
    """The best *k* candidates scoring at least *floor*, best first (ties keep pool order).

    The relevance scorer is a coarse pre-filter (it clumps relevant papers at 7/10), so the
    cut is "the top few above a low floor"; the extractor's "omit if absent" and the verifier
    are the precision stages downstream.
    """
    ranked = sorted(
        ((scores[i], i, c) for i, c in enumerate(candidates) if i in scores),
        key=lambda t: (-t[0], t[1]),
    )
    return [{**c, "score": sc} for sc, _i, c in ranked if sc >= floor][:k]


def choose_papers(candidates: list[dict], scores: dict[int, float],
                  k: int | None = None) -> list[dict]:
    """The scout's selection rule for one scored pool: the best *k* (default
    ``paper_max_per_sub_question``; the scout passes its depth's value) papers scoring at least
    ``paper_topk_floor``."""
    return select_topk(
        candidates, scores, k if k is not None else settings.paper_max_per_sub_question,
        settings.paper_topk_floor,
    )


# ---------------------------------------------------------------------------
# 5. Findings extraction
# ---------------------------------------------------------------------------

async def extract_findings(
    topic: str, sub_question: str, papers: list[dict], llm: BaseChatModel,
    session_id: str | None = None, scope: str = "",
) -> list[Finding]:
    """Grounded findings for one sub-question from its relevance-filtered *papers*: one call of
    the shared extractor over their abstracts (agents/extractor.py), best paper first."""
    if not papers:
        return []
    from research_swarm.agents.extractor import extract_facts

    sources = [
        {"url": p["url"], "title": p.get("title", ""), "text": str(p.get("snippet", "")),
         "source_type": p.get("source_type", "web"),
         "credibility_score": float(p.get("credibility_score", 0.6) or 0.6)}
        for p in papers
    ]
    findings = await extract_facts(
        topic, [sub_question], sources, llm, session_id=session_id, agent="paper_worker",
        scope=scope,
    )
    order = {p["url"]: p.get("score", 0.0) for p in papers}
    findings.sort(
        key=lambda f: (order.get(f.evidence[0].url, 0.0), f.confidence), reverse=True,
    )
    return findings[: settings.paper_max_findings_per_sub_question]
