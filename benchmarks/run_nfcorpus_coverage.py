"""Table C: open-web research coverage on NFCorpus.

Runs the full pipeline OPEN-WEB (real PubMed / Europe PMC / arXiv / web tools, no closed-corpus
patches) at shallow depth on NFCorpus test queries, then measures which of the documents it found
and cited are the ones NFCorpus's qrels call relevant.

    python benchmarks/run_nfcorpus_coverage.py --n 5            # dry run
    python benchmarks/run_nfcorpus_coverage.py --n 100 --update-readme

Caveats (also written under the README table):
  * BEIR's 3,633 abstracts are a *sample* of PubMed. A cited PubMed article outside that sample is
    UNJUDGED, not irrelevant, so precision is computed over judged citations only and the
    unjudged rate is reported next to it.
  * Absolute recall is low by design (~38 relevant docs per query); only comparison between
    configurations, and against the single-search baseline row, is meaningful.
  * NFCorpus grade-1 ("indirectly linked") relevance is noisy, and PubMed content overlaps
    model training data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    RESULTS_ROOT,
    ROOT,
    SEED,
    bootstrap_ci,
    fmt_ci,
    load_beir,
    md_table,
    mean_or_none,
    percentile,
    trace_metrics,
    update_block,
)

_PMID_URL_RES = (
    re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)"),
    re.compile(r"ncbi\.nlm\.nih\.gov/pubmed/(\d+)"),
    re.compile(r"europepmc\.org/article/MED/(\d+)"),
)
# europe_pmc appends "(Journal 2019)" to titles
_TRAILING_JOURNAL_RE = re.compile(r"\s*\([^()]*\d{4}\)\s*$")
PUBMED_TYPES = {"pubmed", "europe_pmc"}
SNIPPET_CHARS = 600
GRADE_CHUNK = 15


# --------------------------------------------------------------------------- #
# ID matching
# --------------------------------------------------------------------------- #

def pmid_from_url(url: str) -> str | None:
    """PMID from a PubMed / Europe PMC (MED) / NCBI-legacy article URL; None for anything else
    (PMC-only Europe PMC records, arXiv, web pages)."""
    for pattern in _PMID_URL_RES:
        m = pattern.search(url or "")
        if m:
            return m.group(1)
    return None


def norm_title(title: str) -> str:
    """Lowercased alphanumeric title with a trailing ``(Journal 2019)`` tag removed."""
    title = _TRAILING_JOURNAL_RE.sub("", title or "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def build_index(corpus: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """{"pmid": pmid -> doc id, "title": normalised title -> doc id} over a BEIR corpus."""
    by_pmid: dict[str, str] = {}
    by_title: dict[str, str] = {}
    for doc_id, doc in corpus.items():
        pmid = pmid_from_url(doc.get("url", ""))
        if pmid:
            by_pmid[pmid] = doc_id
        title = norm_title(doc.get("title", ""))
        if title:
            by_title.setdefault(title, doc_id)
    return {"pmid": by_pmid, "title": by_title}


def match_item(url: str, title: str, source_type: str | None,
               index: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Classify one found/cited item.

    ``status``: ``judged`` (a BEIR corpus doc, ``doc_id`` set), ``unjudged`` (a PubMed article
    that is not in the sample) or ``other`` (web page, arXiv, PMC-only record with no title hit).
    ``method``: ``pmid`` / ``title`` / None.
    """
    pmid = pmid_from_url(url)
    if pmid and pmid in index["pmid"]:
        return {"status": "judged", "doc_id": index["pmid"][pmid], "method": "pmid"}
    doc_id = index["title"].get(norm_title(title))
    if doc_id:
        return {"status": "judged", "doc_id": doc_id, "method": "title"}
    if pmid or (source_type in PUBMED_TYPES):
        return {"status": "unjudged", "doc_id": None, "method": None}
    return {"status": "other", "doc_id": None, "method": None}


def domain_of(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #

def coverage_metrics(
    cited: list[dict[str, Any]],
    qrels: dict[str, int],
    index: dict[str, dict[str, str]],
    candidates: list[dict[str, Any]] | None = None,
    kept: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Metrics for one query. Every list holds ``{"url", "title", "source_type"}`` dicts.

    * judged_precision: distinct cited BEIR docs that are relevant / distinct cited BEIR docs
      (None when nothing cited is judged).
    * unjudged_rate: cited items that are PubMed articles outside the BEIR sample / cited items.
    * graded_gain: sum of the qrel grades (1 or 2) of the distinct relevant docs cited.
    * hit: at least one relevant doc cited.
    * candidate_recall / kept_recall / cited_recall: relevant docs found at each stage / all
      relevant.
    """
    def stage(items: list[dict[str, Any]] | None) -> dict[str, Any]:
        matches = [match_item(i.get("url", ""), i.get("title", ""), i.get("source_type"), index)
                   for i in items or []]
        docs = {m["doc_id"] for m in matches if m["status"] == "judged"}
        return {"matches": matches, "docs": docs, "relevant": {d for d in docs if d in qrels}}

    total_rel = len(qrels)
    c = stage(cited)
    n_cited = len(c["matches"])
    out: dict[str, Any] = {
        "n_cited": n_cited,
        "n_judged": len(c["docs"]),
        "n_relevant_cited": len(c["relevant"]),
        "judged_precision": (len(c["relevant"]) / len(c["docs"])) if c["docs"] else None,
        "unjudged_rate": (sum(m["status"] == "unjudged" for m in c["matches"]) / n_cited)
        if n_cited else None,
        "graded_gain": sum(qrels[d] for d in c["relevant"]),
        "hit": 1.0 if c["relevant"] else 0.0,
        "cited_recall": len(c["relevant"]) / total_rel if total_rel else None,
        "match_methods": {
            m: sum(1 for x in c["matches"] if x["method"] == m) for m in ("pmid", "title")
        },
    }
    if candidates is not None:
        found = len(stage(candidates)["relevant"])
        out["candidate_recall"] = found / total_rel if total_rel else None
    if kept is not None:
        out["kept_recall"] = len(stage(kept)["relevant"]) / total_rel if total_rel else None
    return out


def source_mix(cited: list[dict[str, Any]]) -> dict[str, Any]:
    """Source-type counts, unique-domain count and web fraction of a citation list."""
    by_type: dict[str, int] = {}
    for item in cited:
        kind = str(item.get("source_type") or "web")
        by_type[kind] = by_type.get(kind, 0) + 1
    n = len(cited)
    return {
        "by_type": by_type,
        "unique_domains": len({domain_of(i.get("url", "")) for i in cited if i.get("url")}),
        "web_fraction": (by_type.get("web", 0) / n) if n else None,
    }


def candidates_from_trace(session_id: str) -> list[dict[str, Any]]:
    """Every scout candidate for a run, from its ``paper_scout.candidates`` trace notes."""
    from research_swarm.runtime.trace import trace_path

    path = trace_path(session_id)
    if not path.exists():
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("agent") != "paper_scout.candidates":
            continue
        for url, title, source_type in rec.get("candidates", []):
            if url not in seen:
                seen.add(url)
                out.append({"url": url, "title": title, "source_type": source_type})
    return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def sample_queries(
    queries: dict[str, str], qrels: dict[str, dict[str, int]], n: int,
    min_relevant: int = 5, seed: int = SEED,
) -> list[str]:
    """*n* test query ids with >= *min_relevant* relevant docs, deterministic under *seed*."""
    eligible = sorted(q for q in queries if len(qrels.get(q, {})) >= min_relevant)
    random.Random(seed).shuffle(eligible)
    return eligible[:n]


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def _as_items(sources: list[Any]) -> list[dict[str, Any]]:
    out = []
    for s in sources or []:
        d = s.model_dump(mode="json") if hasattr(s, "model_dump") else dict(s)
        out.append({"url": d.get("url", ""), "title": d.get("title", ""),
                    "source_type": str(d.get("source_type") or "web"),
                    "snippet": str(d.get("snippet", ""))[:SNIPPET_CHARS]})
    return out


def run_baseline(query: str, k: int = 10) -> list[dict[str, Any]]:
    """One raw PubMed search on the query text: what the swarm has to beat."""
    from research_swarm.tools.pubmed_tool import pubmed_search

    try:
        res = pubmed_search.invoke({"query": query, "max_results": k})
    except Exception:  # noqa: BLE001
        return []
    return [{"url": r.get("url", ""), "title": r.get("title", ""), "source_type": "pubmed",
             "snippet": str(r.get("snippet", ""))[:SNIPPET_CHARS]}
            for r in res if isinstance(r, dict) and r.get("url", "").startswith("http")]


async def run_query(qid: str, text: str, qrels: dict[str, int], index: dict, timeout: float,
                    provider: str, model: str, llm_concurrency: int) -> dict[str, Any]:
    from research_swarm.graph.builder import build_graph, get_thread_config
    from research_swarm.runtime.budget import clear_budget
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery

    session_id = f"nfc-{qid}-{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    result: dict[str, Any] = {"query_id": qid, "query": text, "status": "error",
                              "n_relevant_total": len(qrels)}
    try:
        graph = build_graph(interrupt_before_writer=False)
        config = {**get_thread_config(session_id), "max_concurrency": llm_concurrency}
        state0 = {
            "messages": [], "query": ResearchQuery(
                topic=text, depth=ResearchDepth.shallow, max_sources=5, audience="technical"),
            "plan": None, "findings": [], "critiques": [], "draft_report": None,
            "final_report": None, "human_feedback": None, "writer_instructions": None,
            "iteration_count": 0, "next_agent": None, "session_id": session_id,
            "model_provider": provider, "model_name": model, "schema_version": 2,
            "research_rounds": 0, "pre_dispatch_finding_ids": [], "active_sub_question": None,
            "ingested_documents": [],
        }

        async def execute() -> Any:
            async for _ in graph.astream(state0, config, stream_mode="updates"):
                pass
            return (await graph.aget_state(config)).values

        state = await asyncio.wait_for(execute(), timeout=timeout)
        report = state.get("final_report")
        if report is None:
            raise RuntimeError("Graph completed without a final report")
        cited = _as_items(report.references)
        kept = _as_items(state.get("paper_corpus") or [])
        candidates = candidates_from_trace(session_id)
        result.update({
            "status": "ok",
            **coverage_metrics(cited, qrels, index, candidates=candidates, kept=kept),
            "source_mix": source_mix(cited),
            "n_candidates": len(candidates), "n_kept": len(kept),
            "cited": cited,
        })
    except TimeoutError:
        result["status"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        clear_budget(session_id)
        result["seconds"] = round(time.perf_counter() - started, 3)
        result.update(trace_metrics(session_id))
        if result["status"] == "ok" and not result.get("llm_calls"):
            # Every LLM call failed (quota / outage): not a measurement of the pipeline.
            result["status"] = "llm_unavailable"
            result["error"] = "no LLM call succeeded (provider quota or outage?)"
    baseline = run_baseline(text)
    result["baseline"] = {**coverage_metrics(baseline, qrels, index), "cited": baseline}
    return result


# --------------------------------------------------------------------------- #
# Summary / table
# --------------------------------------------------------------------------- #

LLM_METRICS = ["llm_precision", "llm_strict_precision", "llm_gain"]
METRICS = LLM_METRICS + [
    "judged_precision", "unjudged_rate", "graded_gain", "hit", "cited_recall",
]


# --------------------------------------------------------------------------- #
# LLM-judged relevance (qrels only cover NFCorpus's small sample of PubMed)
# --------------------------------------------------------------------------- #

class SourceGrade(BaseModel):
    item: int = Field(..., description="Number of the source")
    grade: int = Field(..., ge=0, le=2)


class SourceGrades(BaseModel):
    grades: list[SourceGrade] = Field(default_factory=list)


_GRADE_SYSTEM = (
    "Grade how useful each source is for answering the health question.\n"
    "2 = directly addresses the question with evidence; 1 = related background; "
    "0 = off-topic or not credible (ads, SEO pages).\n"
    "Return one grade per source."
)


async def judge_sources(query: str, items: list[dict[str, Any]], llm: Any) -> list[int | None]:
    """Grade 0/1/2 per item (None when the judge skipped it or failed), one call per
    ``GRADE_CHUNK`` items."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from research_swarm.agents._utils import (
        ainvoke_with_retry,
        recover_from_parse_failure,
        schema_output_instruction,
    )

    grades: list[int | None] = [None] * len(items)
    structured = llm.with_structured_output(SourceGrades)

    async def chunk(start: int) -> None:
        part = items[start:start + GRADE_CHUNK]
        listing = "\n\n".join(
            f"[{n}] {it.get('title', '')}\n{str(it.get('snippet', ''))[:SNIPPET_CHARS]}"
            for n, it in enumerate(part, 1)
        )
        msgs = [
            SystemMessage(content=_GRADE_SYSTEM + schema_output_instruction(SourceGrades)),
            HumanMessage(content=f"Question: {query}\n\nSources:\n{listing}"),
        ]
        try:
            result = await ainvoke_with_retry(structured, msgs, agent="nfc_judge")
        except Exception as exc:  # noqa: BLE001
            result = recover_from_parse_failure(exc, SourceGrades)
            if result is None:
                return
        for g in result.grades:
            if 1 <= g.item <= len(part):
                grades[start + g.item - 1] = g.grade

    await asyncio.gather(*(chunk(s) for s in range(0, len(items), GRADE_CHUNK)))
    return grades


def llm_metrics(grades: list[int | None]) -> dict[str, float | None]:
    """Precision (grade >= 1), strict precision (grade 2) and mean grade over graded items."""
    got = [g for g in grades if g is not None]
    if not got:
        return {"llm_precision": None, "llm_strict_precision": None, "llm_gain": None}
    return {
        "llm_precision": sum(g >= 1 for g in got) / len(got),
        "llm_strict_precision": sum(g == 2 for g in got) / len(got),
        "llm_gain": sum(got) / len(got),
    }


async def grade_results(results: list[dict[str, Any]], llm: Any) -> None:
    """Add LLM-judged metrics to every ok result and its baseline, in place."""
    for r in results:
        if r.get("status") != "ok":
            continue
        cited_grades = await judge_sources(r["query"], r.get("cited", []), llm)
        r.update(llm_metrics(cited_grades))
        r["cited_grades"] = cited_grades
        base = r.get("baseline") or {}
        base_grades = await judge_sources(r["query"], base.get("cited", []), llm)
        base.update(llm_metrics(base_grades))
        base["cited_grades"] = base_grades


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]

    def col(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    swarm = {m: bootstrap_ci(col(ok, m)) for m in METRICS + ["candidate_recall", "kept_recall"]}
    base_rows = [r["baseline"] for r in results if r.get("baseline")]
    baseline = {m: bootstrap_ci(col(base_rows, m)) for m in METRICS}
    methods = {m: sum(r.get("match_methods", {}).get(m, 0) for r in ok) for m in ("pmid", "title")}
    types: dict[str, int] = {}
    for r in ok:
        for k, v in r.get("source_mix", {}).get("by_type", {}).items():
            types[k] = types.get(k, 0) + v
    secs = [r["seconds"] for r in ok]
    return {
        "queries": len(results), "successful": len(ok),
        "swarm": swarm, "baseline": baseline, "match_methods": methods, "source_types": types,
        "mean_unique_domains": mean_or_none([r["source_mix"]["unique_domains"] for r in ok]),
        "web_fraction": bootstrap_ci(col([r["source_mix"] for r in ok], "web_fraction")),
        "mean_cited": mean_or_none([r["n_cited"] for r in ok]),
        "mean_candidates": mean_or_none([r["n_candidates"] for r in ok]),
        "seconds_p50": percentile(secs, 50), "seconds_p95": percentile(secs, 95),
        "mean_tokens": mean_or_none([r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in ok]),
        "tasks_with_fallbacks": sum(1 for r in ok if r.get("stage_fallbacks")),
    }


def render_table(s: dict[str, Any]) -> str:
    def row(name: str, d: dict[str, Any]) -> list[Any]:
        keys = ("llm_precision", "llm_strict_precision", "llm_gain", "judged_precision",
                "unjudged_rate", "graded_gain", "hit", "cited_recall")
        return [name] + [fmt_ci(d.get(m)) for m in keys]

    header = ["System", "LLM precision", "LLM strict precision", "LLM gain",
              "qrels precision", "Unjudged rate", "qrels gain", "qrels hit", "qrels recall"]
    table = md_table(header, [
        row(f"Swarm, open-web, shallow (n={s['successful']})", s["swarm"]),
        row("Baseline: one PubMed search, top 10", s["baseline"]),
    ])
    stages = md_table(
        ["Stage", "Recall of qrels-relevant docs"],
        [["Scout candidates (before the relevance filter)",
          fmt_ci(s["swarm"].get("candidate_recall"))],
         ["Kept by the relevance filter", fmt_ci(s["swarm"].get("kept_recall"))],
         ["Cited in the report", fmt_ci(s["swarm"].get("cited_recall"))]],
    )
    m = s["match_methods"]
    total = max(1, m["pmid"] + m["title"])
    notes = (
        f"Cited items matched to NFCorpus by PMID {m['pmid'] / total:.0%}, by title "
        f"{m['title'] / total:.0%}. Source mix of citations: {json.dumps(s['source_types'])}; "
        f"{s['mean_unique_domains']} unique domains and {s['mean_cited']} citations per report; "
        f"web-citation fraction {fmt_ci(s['web_fraction'])}. Time p50 {s['seconds_p50']} s, "
        f"p95 {s['seconds_p95']} s; {s['mean_tokens']} tokens per query.\n\n"
        "*Unjudged = a PubMed article outside NFCorpus's 3,633-abstract sample: unknown, not "
        "irrelevant, so precision covers judged citations only. Absolute recall is low by design "
        "(BEIR samples PubMed, ~38 relevant docs per query); compare configurations, not "
        "absolutes. Grade-1 relevance is noisy and PubMed overlaps model training data.*"
    )
    title = "**Table C - open-web coverage, NFCorpus test queries**"
    return "\n\n".join([title, table, stages, notes])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

async def main(args: argparse.Namespace) -> None:
    import research_swarm.graph.nodes as nodes
    from research_swarm.agents.base import get_agent_llm
    from research_swarm.config import settings

    provider, model = "ollama", args.model
    for attr in ("default_model", "tier_fast", "tier_standard", "tier_thorough"):
        if attr == "default_model":
            settings.default_model_provider, settings.default_model_name = provider, model
        else:
            setattr(settings, f"{attr}_provider", provider)
            setattr(settings, f"{attr}_model", model)
    settings.max_sources = 5
    settings.max_llm_calls = 12
    settings.enable_fetch_pass = True
    from run_smoke_benchmark import apply_overrides

    overrides = apply_overrides(args.set)

    def _uniform(tier, temperature=0.0, provider_override=None):
        return get_agent_llm(provider=provider, model=model, temperature=temperature)

    nodes.get_tiered_llm = _uniform

    corpus, queries, qrels = load_beir("nfcorpus", "test")
    index = build_index(corpus)
    ids = sample_queries(queries, qrels, args.n, args.min_relevant)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results_path = RESULTS_ROOT / f"nfcorpus-{run_id}-results.jsonl"
    summary_path = RESULTS_ROOT / f"nfcorpus-{run_id}-summary.json"
    print(f"MODEL ollama/{model}; {len(ids)} NFCorpus queries", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []

    async def guarded(qid: str) -> None:
        async with sem:
            print(f"START {qid}", flush=True)
            r = await run_query(qid, queries[qid], qrels[qid], index, args.timeout,
                                provider, model, args.llm_concurrency)
            results.append(r)
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"DONE  {qid} status={r['status']} seconds={r['seconds']} "
                  f"hit={r.get('hit')} prec={r.get('judged_precision')} "
                  f"unjudged={r.get('unjudged_rate')}", flush=True)

    await asyncio.gather(*(guarded(q) for q in ids))
    if not args.no_judge:
        from research_swarm.agents.base import without_thinking
        from research_swarm.runtime.limits import set_llm_context

        set_llm_context("ollama", "nfc-judge")
        judge = without_thinking(
            get_agent_llm(provider="ollama", model=args.judge_model, temperature=0.0),
            settings.no_thinking_max_tokens,
        )
        await grade_results(results, judge)
        results_path.write_text(
            "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    summary = summarize(results)
    if overrides:
        summary["overrides"] = overrides
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RESULTS {results_path}\nSUMMARY {summary_path}")
    table = render_table(summary)
    print(table)
    if args.update_readme:
        update_block(ROOT / "benchmarks" / "README.md", "TABLE-C", table)
        print("README table block updated")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-relevant", type=int, default=5)
    ap.add_argument("--model", default="nemotron-3-nano:30b-cloud")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--llm-concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=400.0)
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--judge-model", default="gemma4:31b-cloud",
                    help="Ollama model that grades each cited source's relevance (0/1/2)")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM relevance grading")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a settings field for this run (repeatable)")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
