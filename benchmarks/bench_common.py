"""Shared helpers for the benchmark scripts: statistics, dataset loaders, table blocks.

Plain module (``benchmarks/`` is not a package): scripts run from this directory put it on
``sys.path`` automatically; tests load it with ``importlib`` (see tests/unit/test_bench_common.py).
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "benchmarks"
RESULTS_ROOT = ROOT / "data" / "benchmark_results"
SEED = 42


# --------------------------------------------------------------------------- #
# Small numeric helpers (also used by run_smoke_benchmark.py)
# --------------------------------------------------------------------------- #

def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))        # ceil
    return round(ordered[int(rank) - 1], 3)


def rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    values: list[float], n: int = 2000, seed: int = SEED, alpha: float = 0.05,
) -> tuple[float, float, float] | None:
    """(mean, lo, hi): percentile bootstrap CI of the mean. None for no data.

    A single value gives a degenerate (v, v, v) interval -- callers should treat CIs from very
    few tasks as uninformative rather than trust the width.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(statistics.fmean(rng.choices(vals, k=k)) for _ in range(n))
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return mean, lo, hi


def paired_bootstrap_diff(
    a: list[float], b: list[float], n: int = 2000, seed: int = SEED, alpha: float = 0.05,
) -> tuple[float, float, float] | None:
    """(mean(a-b), lo, hi) resampling *pairs* -- for the same tasks under two configurations."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    return bootstrap_ci(diffs, n=n, seed=seed, alpha=alpha)


def fmt_ci(ci: tuple[float, float, float] | None, digits: int = 3) -> str:
    """'0.812 [0.771, 0.850]' or '-' when there is no data."""
    if ci is None:
        return "-"
    mean, lo, hi = ci
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# --------------------------------------------------------------------------- #
# BEIR datasets (SciFact, NFCorpus, ArguAna)
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir(
    name: str, split: str = "test", root: Path | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """(corpus, queries, qrels) for one BEIR dataset split.

    corpus: ``id -> {"title", "text", "url"}`` (url empty when the dataset has none; NFCorpus
    carries its PubMed URL in ``metadata``); queries: ``id -> text``; qrels: ``query id ->
    {doc id: grade}`` restricted to docs present in the corpus and to grade > 0.
    """
    base = (root or DATA_ROOT) / "beir" / name
    corpus = {
        row["_id"]: {
            "title": row.get("title", ""), "text": row.get("text", ""),
            "url": (row.get("metadata") or {}).get("url", ""),
        }
        for row in _read_jsonl(base / "corpus.jsonl")
    }
    queries = {row["_id"]: row["text"] for row in _read_jsonl(base / "queries.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        next(handle)                                         # header
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            if grade > 0 and did in corpus and qid in queries:
                qrels.setdefault(qid, {})[did] = grade
    return corpus, queries, qrels


# --------------------------------------------------------------------------- #
# Lexical helpers (same tokenizer as the worker's source ranker)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=60000)
def _terms_cached(text: str) -> frozenset[str]:
    from research_swarm.agents.text import terms

    return frozenset(terms(text))


def lexical_score(query: str, title: str, text: str) -> float:
    """Query-term overlap (title counts double) -- the no-model baseline ranker."""
    q = _terms_cached(query)
    if not q:
        return 0.0
    return (2 * len(q & _terms_cached(title)) + len(q & _terms_cached(text))) / len(q)


def hard_negatives(
    query: str, corpus: dict[str, dict[str, str]], exclude: set[str], k: int,
) -> list[str]:
    """The *k* corpus ids (outside *exclude*) sharing the most terms with *query*.

    Deterministic: ties break on id. These are the confusable non-relevant documents that make
    a relevance judgement non-trivial, unlike random negatives.
    """
    scored = [
        (lexical_score(query, doc["title"], doc["text"]), did)
        for did, doc in corpus.items() if did not in exclude
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [did for _, did in scored[:k]]


# --------------------------------------------------------------------------- #
# Generated tables inside a README
# --------------------------------------------------------------------------- #

def update_block(path: Path, name: str, markdown: str) -> None:
    """Replace the text between ``<!-- NAME:START -->`` and ``<!-- NAME:END -->`` in *path*,
    adding the block at the end of the file if it isn't there yet. Text outside is untouched."""
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    body = f"{start}\n{markdown.strip()}\n{end}"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: body, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + body + "\n"
    path.write_text(text, encoding="utf-8", newline="")


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join("-" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def trace_metrics(session_id: str) -> dict[str, Any]:
    """Efficiency metrics for one run from its trace file (empty if tracing produced none)."""
    from research_swarm.runtime.trace import trace_path
    from research_swarm.runtime.trace_stats import analyze

    path = trace_path(session_id)
    if not path.exists():
        return {}
    a = analyze(path)
    llm = a["llm"].values()
    return {
        "stage_wall_s": a["stage_wall_s"],
        "llm_calls": int(sum(v.get("calls", 0) for v in llm)),
        "tokens_in": int(sum(v.get("in_tokens", 0) for v in llm)),
        "tokens_out": int(sum(v.get("out_tokens", 0) for v in llm)),
        "reasoning_chars": int(sum(v.get("reasoning_chars", 0) for v in llm)),
        "out_chars": int(sum(v.get("out_chars", 0) for v in llm)),
        "llm_errors": a["quality"]["llm_errors"],
        "peak_llm_concurrency": a["peak_llm_concurrency"],
        "llm_retries": a["llm_retries"],
        "slot_wait_s": a["slot_wait_s"],
        "stage_fallbacks": a["fallbacks"],
        "fallbacks": {
            "synthesis_failures": a["quality"]["synthesis_failures"],
            "incomplete_claims": a["quality"]["incomplete_claims"],
        },
        "verifier_verdicts": a["quality"]["verdicts"],
    }


def isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol)
