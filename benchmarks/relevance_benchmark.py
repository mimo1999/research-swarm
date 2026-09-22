"""Table B: how well does the paper scout's LLM relevance scorer rank relevant abstracts?

Builds candidate pools with known relevance from three datasets, scores each pool with the
PRODUCTION scorer (``research_swarm.agents.papers.score_pool`` + ``select_papers``, LLM built the
way the scout builds it: fast tier, thinking off) and compares it with a word-overlap baseline and
a random baseline.

    python benchmarks/relevance_benchmark.py --n 100                 # full run (~25 min)
    python benchmarks/relevance_benchmark.py --n 5 --dry             # build/inspect pools, no LLM
    python benchmarks/relevance_benchmark.py --evaluate-only <run>   # re-score saved LLM scores

Pools (size 24, like ``settings.paper_max_candidates``; seeded, order shuffled):
  hotpotqa   the question's 10 context docs; gold = its 2 supporting docs (8 distractors)
  nfcorpus   test queries with >= 4 relevant abstracts: up to 6 relevant (grade 2 preferred),
             9 hard negatives (most lexical overlap) + 9 random non-relevant
  scifact    test claims: the gold evidence abstract(s) + hard and random negatives

Metrics per pool (tie-aware -- scores are integers 0-10): nDCG@10, AUC, expected precision@3,
precision/recall at the strict 0.75 threshold and for the selection as shipped (with the top-up
rule), plus a threshold sweep and (``--shuffles 2``) sensitivity to candidate order.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    DATA_ROOT,
    RESULTS_ROOT,
    ROOT,
    SEED,
    bootstrap_ci,
    fmt_ci,
    hard_negatives,
    lexical_score,
    load_beir,
    md_table,
    mean_or_none,
    paired_bootstrap_diff,
    update_block,
)

POOL_SIZE = 24
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
METHODS = ("llm", "lexical", "random")


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #

@dataclass
class Pool:
    id: str
    dataset: str
    topic: str
    sub_question: str
    candidates: list[dict[str, str]]        # {"id", "title", "snippet", "url", "source_type"}
    gains: dict[str, int]                   # candidate id -> relevance grade (> 0 only)


def _candidate(cid: str, title: str, text: str, url: str = "", source_type: str = "pubmed") -> dict:
    return {"id": cid, "title": title, "snippet": text, "url": url, "source_type": source_type}


def build_hotpot_pools(n: int, seed: int = SEED, root: Path | None = None) -> list[Pool]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        (root or DATA_ROOT) / "hotpotqa" / "validation-0000.parquet"
    ).to_pylist()
    rng = random.Random(f"{seed}:hotpotqa")
    pools: list[Pool] = []
    for index in rng.sample(range(len(rows)), min(n, len(rows))):
        row = rows[index]
        gold_titles = set(row["supporting_facts"]["title"])
        cands, gains = [], {}
        for i, (title, sentences) in enumerate(
            zip(row["context"]["title"], row["context"]["sentences"], strict=True)
        ):
            cid = f"{row['id']}:{i}"
            cands.append(_candidate(cid, title, " ".join(sentences), source_type="web"))
            if title in gold_titles:
                gains[cid] = 1
        rng.shuffle(cands)
        pools.append(Pool(
            id=f"hotpotqa-{row['id']}", dataset="hotpotqa", topic=row["question"],
            sub_question=row["question"], candidates=cands, gains=gains,
        ))
    return pools


def _beir_pool(
    dataset: str, qid: str, query: str, corpus: dict[str, dict[str, str]],
    relevant: dict[str, int], chosen: list[str], rng: random.Random,
) -> Pool:
    """*chosen* relevant docs + hard and random negatives (never another relevant doc)."""
    n_neg = POOL_SIZE - len(chosen)
    n_hard = n_neg // 2 if dataset == "scifact" else min(9, n_neg)
    excluded = set(relevant)                        # every relevant doc, chosen or not
    hard = hard_negatives(query, corpus, excluded, n_hard)
    remaining = sorted(d for d in corpus if d not in excluded and d not in set(hard))
    n_random = n_neg - len(hard) if dataset == "scifact" else min(9, n_neg - len(hard))
    rand = rng.sample(remaining, min(n_random, len(remaining)))
    cands = [
        _candidate(did, corpus[did]["title"], corpus[did]["text"], corpus[did]["url"])
        for did in chosen + hard + rand
    ]
    rng.shuffle(cands)
    return Pool(
        id=f"{dataset}-{qid}", dataset=dataset, topic=query, sub_question=query,
        candidates=cands, gains={d: relevant[d] for d in chosen},
    )


def build_nfcorpus_pools(n: int, seed: int = SEED, root: Path | None = None) -> list[Pool]:
    corpus, queries, qrels = load_beir("nfcorpus", "test", root)
    rng = random.Random(f"{seed}:nfcorpus")
    eligible = sorted(q for q, rel in qrels.items() if len(rel) >= 4)
    pools = []
    for qid in rng.sample(eligible, min(n, len(eligible))):
        rel = qrels[qid]
        g2 = sorted(d for d, g in rel.items() if g >= 2)
        g1 = sorted(d for d, g in rel.items() if g < 2)
        top = g2[:3]
        rest = [d for d in g2[3:] + g1]
        chosen = top + rng.sample(rest, min(len(rest), 6 - len(top)))
        pools.append(_beir_pool("nfcorpus", qid, queries[qid], corpus, rel, chosen, rng))
    return pools


def build_scifact_pools(n: int, seed: int = SEED, root: Path | None = None) -> list[Pool]:
    corpus, queries, qrels = load_beir("scifact", "test", root)
    rng = random.Random(f"{seed}:scifact")
    eligible = sorted(qrels)
    pools = []
    for qid in rng.sample(eligible, min(n, len(eligible))):
        rel = qrels[qid]
        pools.append(_beir_pool("scifact", qid, queries[qid], corpus, rel, sorted(rel), rng))
    return pools


BUILDERS = {
    "hotpotqa": build_hotpot_pools, "nfcorpus": build_nfcorpus_pools,
    "scifact": build_scifact_pools,
}


# --------------------------------------------------------------------------- #
# Ranking metrics (tie-aware: equal scores share their positions on average)
# --------------------------------------------------------------------------- #

def _groups(scores: dict[str, float]) -> list[list[str]]:
    """Candidate ids grouped by equal score, best group first."""
    by_score: dict[float, list[str]] = {}
    for cid, s in scores.items():
        by_score.setdefault(s, []).append(cid)
    return [sorted(by_score[s]) for s in sorted(by_score, reverse=True)]


def ndcg_at_k(scores: dict[str, float], gains: dict[str, int], k: int = 10) -> float | None:
    """nDCG@k with linear gain; tied candidates split their positions' gain evenly."""
    dcg, pos = 0.0, 0
    for group in _groups(scores):
        mean_gain = sum(gains.get(c, 0) for c in group) / len(group)
        for p in range(pos, min(pos + len(group), k)):
            dcg += mean_gain / math.log2(p + 2)
        pos += len(group)
        if pos >= k:
            break
    ideal = sorted((gains.get(c, 0) for c in scores), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def auc(scores: dict[str, float], gains: dict[str, int]) -> float | None:
    """P(random relevant outranks random non-relevant); ties count 0.5. None if a class is empty."""
    pos = [scores[c] for c in scores if gains.get(c, 0) > 0]
    neg = [scores[c] for c in scores if gains.get(c, 0) <= 0]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def precision_at_k(scores: dict[str, float], gains: dict[str, int], k: int = 3) -> float | None:
    """Expected precision@k under random tie-breaking."""
    if not scores:
        return None
    taken, expected = 0, 0.0
    for group in _groups(scores):
        take = min(len(group), k - taken)
        rel = sum(1 for c in group if gains.get(c, 0) > 0)
        expected += take * rel / len(group)
        taken += take
        if taken >= k:
            break
    return expected / min(k, len(scores))


def precision_recall(
    kept: set[str], gains: dict[str, int],
) -> tuple[float | None, float | None]:
    """(precision, recall) of a selected set against the relevant docs; precision None if empty."""
    relevant = {c for c, g in gains.items() if g > 0}
    tp = len(kept & relevant)
    return (tp / len(kept) if kept else None), (tp / len(relevant) if relevant else None)


def _ranking(scores: dict[str, float], gains: dict[str, int]) -> dict[str, float | None]:
    return {
        "ndcg10": ndcg_at_k(scores, gains, 10),
        "auc": auc(scores, gains),
        "p_at_3": precision_at_k(scores, gains, 3),
    }


# --------------------------------------------------------------------------- #
# Scoring and per-pool evaluation
# --------------------------------------------------------------------------- #

def shuffled(pool: Pool, shuffle: int, seed: int = SEED) -> list[dict]:
    """The pool's candidates in shuffle #*shuffle*'s order (0 = the pool's own order)."""
    cands = list(pool.candidates)
    if shuffle:
        random.Random(f"{seed}:{pool.id}:{shuffle}").shuffle(cands)
    return cands


async def score_with_llm(pool: Pool, llm: Any, shuffle: int = 0) -> dict[str, Any]:
    """Run the production scorer on one pool; {"scores": {id: score|None}, "failed": bool}."""
    from research_swarm.agents.papers import score_pool

    cands = shuffled(pool, shuffle)
    idx_scores = await score_pool(
        pool.topic, pool.sub_question,
        [{"title": c["title"], "snippet": c["snippet"]} for c in cands], llm,
    )
    return {
        "scores": {c["id"]: idx_scores.get(i) for i, c in enumerate(cands)},
        "failed": not idx_scores,
    }


def shipped_selection(
    pool: Pool, scores: dict[str, float | None], *, threshold: float, min_keep: int, floor: float,
    limit: int, selection: str = "threshold", topk_floor: float = 0.5,
) -> list[dict]:
    """Exactly what the scout keeps for these scores: ``select_papers`` (threshold + top-up) or,
    with ``selection="topk"``, ``select_topk`` (best *limit* at or above *topk_floor*)."""
    from research_swarm.agents.papers import select_papers, select_topk

    cands = pool.candidates
    idx_scores = {
        i: scores[c["id"]] for i, c in enumerate(cands) if scores.get(c["id"]) is not None
    }
    if selection == "topk":
        return select_topk(cands, idx_scores, limit, topk_floor)
    return select_papers(cands, idx_scores, threshold, limit, min_keep=min_keep, floor=floor)


def evaluate_pool(
    pool: Pool, runs: list[dict[str, Any]], *, seed: int = SEED,
    threshold: float = 0.75, min_keep: int = 3, floor: float = 0.6, limit: int = 6,
    selection: str = "threshold", topk_floor: float = 0.5,
) -> dict[str, Any]:
    """All metrics for one pool. *runs* = saved LLM results, one per shuffle (may be empty)."""
    ids = [c["id"] for c in pool.candidates]
    gains = pool.gains
    rng = random.Random(f"{seed}:{pool.id}:random")
    record: dict[str, Any] = {
        "pool": pool.id, "dataset": pool.dataset, "n_cand": len(ids),
        "n_rel": sum(1 for g in gains.values() if g > 0),
    }
    lex = {c["id"]: lexical_score(pool.sub_question, c["title"], c["snippet"])
           for c in pool.candidates}
    record["lexical"] = _ranking(lex, gains)
    record["random"] = _ranking({i: rng.random() for i in ids}, gains)

    first = runs[0] if runs else None
    if first is None or first["failed"]:
        record["failed"] = True
        return record
    record["failed"] = False
    raw = first["scores"]
    record["n_unscored"] = sum(1 for i in ids if raw.get(i) is None)
    scores = {i: (raw.get(i) if raw.get(i) is not None else 0.0) for i in ids}   # omitted = lowest
    record["llm"] = _ranking(scores, gains)

    record["sweep"] = {}
    for t in THRESHOLDS:
        kept = {i for i in ids if scores[i] >= t}
        p, r = precision_recall(kept, gains)
        record["sweep"][str(t)] = {"precision": p, "recall": r, "n": len(kept)}
    strict = shipped_selection(pool, raw, threshold=threshold, min_keep=0, floor=floor, limit=limit)
    shipped = shipped_selection(pool, raw, threshold=threshold, min_keep=min_keep, floor=floor,
                                limit=limit, selection=selection, topk_floor=topk_floor)
    for name, sel in (("strict", strict), ("shipped", shipped)):
        p, r = precision_recall({c["id"] for c in sel}, gains)
        record[name] = {
            "precision": p, "recall": r, "n": len(sel),
            "topped_up": sum(1 for c in sel if c.get("topped_up")),
        }

    if len(runs) > 1 and not runs[1]["failed"]:
        other = {i: (runs[1]["scores"].get(i) or 0.0) for i in ids}
        deltas = [abs(scores[i] - other[i]) for i in ids]
        second = _ranking(other, gains)
        record["sensitivity"] = {
            "mean_abs_delta": sum(deltas) / len(deltas),
            "ndcg_delta": (
                abs(record["llm"]["ndcg10"] - second["ndcg10"])
                if record["llm"]["ndcg10"] is not None and second["ndcg10"] is not None else None
            ),
        }
    return record


# --------------------------------------------------------------------------- #
# Aggregation and rendering
# --------------------------------------------------------------------------- #

def _col(records: list[dict], *path: str) -> list[float]:
    out = []
    for r in records:
        v: Any = r
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
        if v is not None:
            out.append(float(v))
    return out


def summarize(records: list[dict]) -> dict[str, Any]:
    """Per-dataset aggregate (means with 95% bootstrap CIs) of the per-pool records."""
    summary: dict[str, Any] = {"seed": SEED, "datasets": {}}
    for ds in sorted({r["dataset"] for r in records}):
        rs = [r for r in records if r["dataset"] == ds]
        ok = [r for r in rs if not r.get("failed")]
        block: dict[str, Any] = {
            "pools": len(rs), "failed": len(rs) - len(ok),
            "unscored_rate": mean_or_none([r["n_unscored"] / r["n_cand"] for r in ok]),
        }
        for method in METHODS:
            src = ok if method == "llm" else rs
            block[method] = {
                m: bootstrap_ci(_col(src, method, m)) for m in ("ndcg10", "auc", "p_at_3")
            }
        pairs = [(r["llm"]["ndcg10"], r["lexical"]["ndcg10"]) for r in ok
                 if r["llm"]["ndcg10"] is not None and r["lexical"]["ndcg10"] is not None]
        block["llm_minus_lexical_ndcg10"] = (
            paired_bootstrap_diff([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
        )
        for name in ("strict", "shipped"):
            block[name] = {
                "precision": bootstrap_ci(_col(ok, name, "precision")),
                "recall": bootstrap_ci(_col(ok, name, "recall")),
                "mean_kept": mean_or_none(_col(ok, name, "n")),
                "empty_rate": mean_or_none([float(r[name]["n"] == 0) for r in ok]),
                "topped_up_rate": mean_or_none([float(r[name]["topped_up"] > 0) for r in ok]),
            }
        block["sweep"] = {
            str(t): {
                "precision": mean_or_none(_col(ok, "sweep", str(t), "precision")),
                "recall": mean_or_none(_col(ok, "sweep", str(t), "recall")),
                "mean_n": mean_or_none(_col(ok, "sweep", str(t), "n")),
            }
            for t in THRESHOLDS
        }
        block["sensitivity"] = {
            "mean_abs_score_delta": mean_or_none(_col(ok, "sensitivity", "mean_abs_delta")),
            "mean_abs_ndcg_delta": mean_or_none(_col(ok, "sensitivity", "ndcg_delta")),
        }
        summary["datasets"][ds] = block
    return summary


_TABLE_B_NOTE = (
    "Hard negatives are the non-relevant documents with the MOST word overlap with the query, "
    "so the word-overlap baseline is handicapped by construction (on NFCorpus and SciFact it "
    "can fall below random): compare the LLM scorer with the random row as the floor, and with "
    "word overlap only on HotpotQA, whose distractors were not chosen by overlap. NFCorpus "
    "relevance means 'linked from the same article', so many topically relevant abstracts are "
    "unlabelled, and its queries are video headlines, not questions."
)


def render_table_b(summary: dict[str, Any]) -> str:
    """Markdown for README block TABLE-B."""
    rank_rows, sel_rows = [], []
    for ds, b in summary["datasets"].items():
        for method in METHODS:
            m = b[method]
            rank_rows.append([
                ds, b["pools"], {"llm": "**LLM scorer**", "lexical": "word overlap",
                                 "random": "random"}[method],
                fmt_ci(m["ndcg10"]), fmt_ci(m["auc"]), fmt_ci(m["p_at_3"]),
            ])
        shipped_label = (
            "top-k (as configured)" if summary.get("selection") == "topk"
            else "as shipped (+ top-up)"
        )
        for name, label in (("strict", "strict >= 0.75"), ("shipped", shipped_label)):
            s = b[name]
            sel_rows.append([
                ds, label, fmt_ci(s["precision"]), fmt_ci(s["recall"]), s["mean_kept"],
                s["empty_rate"], s["topped_up_rate"] if name == "shipped" else "-",
            ])
    parts = [
        "### Table B - relevance-filter quality (LLM 0-10 scorer on pools with known relevance)",
        "",
        "Mean over pools, 95% bootstrap CI in brackets. Ties are averaged, not broken.",
        "",
        md_table(["Dataset", "Pools", "Ranker", "nDCG@10", "AUC", "Expected P@3"], rank_rows),
        "",
        md_table(
            ["Dataset", "Selection", "Precision", "Recall", "Mean kept", "Empty rate",
             "Topped-up rate"], sel_rows,
        ),
    ]
    parts += ["", _TABLE_B_NOTE]
    diffs = [
        f"{ds}: {fmt_ci(b['llm_minus_lexical_ndcg10'])}"
        for ds, b in summary["datasets"].items() if b.get("llm_minus_lexical_ndcg10")
    ]
    if diffs:
        parts += ["", "Paired nDCG@10 difference, LLM minus word overlap - " + "; ".join(diffs)]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def build_llm(model: str | None = None):
    """The LLM the scout uses: fast tier, thinking off, under the provider concurrency cap."""
    from research_swarm.agents.base import get_tiered_llm, without_thinking
    from research_swarm.config import settings
    from research_swarm.runtime.limits import set_llm_context

    if model:
        settings.tier_fast_provider = "ollama"
        settings.tier_fast_model = model
    llm = without_thinking(get_tiered_llm("fast"), settings.no_thinking_max_tokens)
    set_llm_context(settings.tier_fast_provider, "relevance-bench")
    return llm


def _read_scores(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[(row["pool"], row["shuffle"])] = row
    return out


def evaluate_run(pools: list[Pool], saved: dict, shuffles: int, **kw) -> list[dict]:
    return [
        evaluate_pool(
            p, [saved[(p.id, k)] for k in range(shuffles) if (p.id, k) in saved], **kw,
        )
        for p in pools
    ]


async def main(args: argparse.Namespace) -> None:
    from research_swarm.config import settings

    run_id = args.evaluate_only or args.run_id or time.strftime("%Y%m%d-%H%M%S")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    pools_path = RESULTS_ROOT / f"relevance-{run_id}-pools.jsonl"
    scores_path = RESULTS_ROOT / f"relevance-{run_id}-scores.jsonl"
    selection = args.selection or "topk"
    suffix = "" if selection == "threshold" else f"-{selection}"
    summary_path = RESULTS_ROOT / f"relevance-{run_id}-summary{suffix}.json"

    if args.evaluate_only and pools_path.exists():
        pools = [Pool(**json.loads(line)) for line in
                 pools_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        pools = [p for ds in args.datasets.split(",") for p in BUILDERS[ds](args.n, args.seed)]
        pools_path.write_text(
            "\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in pools) + "\n",
            encoding="utf-8",
        )
    print(f"{len(pools)} pools ({', '.join(sorted({p.dataset for p in pools}))})", flush=True)
    for ds in sorted({p.dataset for p in pools}):
        rows = [p for p in pools if p.dataset == ds]
        print(f"  {ds}: {len(rows)} pools, mean size "
              f"{sum(len(p.candidates) for p in rows) / len(rows):.1f}, mean relevant "
              f"{sum(len(p.gains) for p in rows) / len(rows):.1f}", flush=True)
    if args.dry:
        return

    saved = _read_scores(scores_path)
    if not args.evaluate_only:
        llm = build_llm(args.model)
        print(f"scorer: {settings.tier_fast_provider}/{settings.tier_fast_model}", flush=True)
        todo = [(p, k) for p in pools for k in range(args.shuffles) if (p.id, k) not in saved]
        for i, (pool, k) in enumerate(todo, 1):
            t0 = time.perf_counter()
            result = await score_with_llm(pool, llm, k)
            row = {"pool": pool.id, "shuffle": k, **result}
            saved[(pool.id, k)] = row
            with scores_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(todo)}] {pool.id} shuffle={k} failed={result['failed']} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    records = evaluate_run(
        pools, saved, args.shuffles, seed=args.seed, threshold=settings.relevance_threshold,
        min_keep=settings.paper_min_per_sub_question, floor=settings.relevance_floor,
        limit=settings.paper_max_per_sub_question, selection=selection,
        topk_floor=settings.paper_topk_floor,
    )
    summary = summarize(records)
    summary["selection"] = selection
    summary["run_id"] = run_id
    summary["scorer"] = f"{settings.tier_fast_provider}/{settings.tier_fast_model}"
    summary["records"] = records
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = render_table_b(summary)
    print("\n" + table)
    print(f"\nSUMMARY {summary_path}")
    if args.update_readme:
        update_block(ROOT / "benchmarks" / "README.md", "TABLE-B", table)
        print("README table block updated")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hotpotqa,nfcorpus,scifact")
    ap.add_argument("--n", type=int, default=100, help="pools per dataset")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--shuffles", type=int, default=1,
                    help="score each pool this many times in different candidate orders")
    ap.add_argument("--model", default=None, help="override the fast-tier scorer model")
    ap.add_argument("--run-id", default=None, help="resume/extend an existing run")
    ap.add_argument("--evaluate-only", default=None, metavar="RUN_ID",
                    help="recompute metrics from a run's saved scores (no LLM calls)")
    ap.add_argument("--selection", choices=["threshold", "topk"], default=None,
                    help="selection rule for the as-shipped rows (default: topk); "
                         " with --evaluate-only this re-scores saved scores for free")
    ap.add_argument("--dry", action="store_true", help="build pools and print stats only")
    ap.add_argument("--update-readme", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
