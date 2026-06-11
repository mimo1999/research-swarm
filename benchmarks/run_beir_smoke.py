"""Evaluate dense retrieval and reranking across all three BEIR subsets.

Datasets (downloaded by download_datasets.ps1):
  scifact   -- scientific claim verification (5,183 docs, short-ish claims)
  nfcorpus  -- medical/nutrition queries (3,633 docs, graded relevance 0-2)
  arguana   -- argument counter-retrieval (8,674 docs, very long queries)

For each dataset the script compares:
  - Dense BGE (bge-small-en-v1.5) cosine retrieval
  - + Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    (automatically skipped for queries > 8 words — see reranker.py)

Usage:
    poetry run python benchmarks/run_beir_smoke.py
    poetry run python benchmarks/run_beir_smoke.py --datasets scifact nfcorpus
    poetry run python benchmarks/run_beir_smoke.py --queries 50
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from research_swarm.rag.indexes import get_embed_model
from research_swarm.rag.reranker import rerank

SEED = 42
DEFAULT_QUERY_COUNT = 100
BEIR_ROOT = Path("data/benchmarks/beir")
RESULTS_ROOT = Path("data/benchmark_results")
ALL_DATASETS = ["scifact", "nfcorpus", "arguana"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_qrels(dataset_root: Path) -> dict[str, dict[str, int]]:
    """Load relevance judgments from qrels/test.tsv."""
    qrels: dict[str, dict[str, int]] = {}
    tsv = dataset_root / "qrels" / "test.tsv"
    lines = tsv.read_text(encoding="utf-8").splitlines()[1:]  # skip header
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        query_id, corpus_id, score = parts[0], parts[1], parts[2]
        qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def normalize(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed_batches(texts: list[str], *, query: bool, batch_size: int = 64) -> np.ndarray:
    model = get_embed_model()
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if query:
            rows.extend(model.get_query_embedding(t) for t in batch)
        else:
            rows.extend(model.get_text_embedding_batch(batch))
    return normalize(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / max(len(relevant), 1)


def ndcg_at(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2 ** relevance.get(doc_id, 0) - 1) / math.log2(idx + 2)
        for idx, doc_id in enumerate(ranked[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** s - 1) / math.log2(idx + 2) for idx, s in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def means(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    query_count: int,
    rng: random.Random,
) -> dict:
    root = BEIR_ROOT / name
    corpus_rows = read_jsonl(root / "corpus.jsonl")
    all_queries = {row["_id"]: row["text"] for row in read_jsonl(root / "queries.jsonl")}
    qrels = load_qrels(root)

    # Sample query IDs that have at least one relevant document in test qrels.
    eligible = [qid for qid in qrels if qid in all_queries]
    sample_size = min(query_count, len(eligible))
    query_ids = rng.sample(sorted(eligible), sample_size)
    queries = {qid: all_queries[qid] for qid in query_ids}

    print(
        f"  [{name}] {len(corpus_rows):,} corpus docs, "
        f"{sample_size} / {len(eligible)} eligible queries sampled"
    )

    # Corpus embeddings
    corpus_ids = [row["_id"] for row in corpus_rows]
    corpus_texts = [f"{row.get('title', '')}\n{row['text']}" for row in corpus_rows]
    print(f"  [{name}] Embedding corpus ...", flush=True)
    t0 = time.perf_counter()
    corpus_emb = embed_batches(corpus_texts, query=False)
    print(f"  [{name}] Corpus embedded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Query embeddings
    query_texts = [queries[qid] for qid in query_ids]
    query_emb = embed_batches(query_texts, query=True)

    # Similarities
    similarities = query_emb @ corpus_emb.T

    dense_metrics: list[dict] = []
    reranked_metrics: list[dict] = []
    details: list[dict] = []
    skipped_rerank = 0

    for row_idx, qid in enumerate(query_ids):
        # Top-20 candidates by dense score
        cand_idx = np.argpartition(similarities[row_idx], -20)[-20:]
        cand_idx = cand_idx[np.argsort(similarities[row_idx, cand_idx])[::-1]]
        dense_ids = [corpus_ids[i] for i in cand_idx]

        relevance = qrels[qid]
        relevant = {k for k, v in relevance.items() if v > 0}

        dense = {
            "recall@5":  recall_at(dense_ids, relevant, 5),
            "recall@10": recall_at(dense_ids, relevant, 10),
            "ndcg@10":   ndcg_at(dense_ids, relevance, 10),
        }

        # Build source dicts for the reranker
        chunks = [
            {
                "url": f"beir://{name}/{corpus_ids[i]}",
                "title": corpus_rows[i].get("title", ""),
                "snippet": corpus_rows[i]["text"],
                "credibility_score": float(similarities[row_idx, i]),
            }
            for i in cand_idx
        ]
        query_text = queries[qid]
        reranked_chunks = rerank(query_text, chunks, top_k=10)
        reranked_ids = [c["url"].rsplit("/", 1)[-1] for c in reranked_chunks]

        # Track whether the reranker ran or was skipped
        if len(query_text.split()) > 8:
            skipped_rerank += 1

        reranked = {
            "recall@5":  recall_at(reranked_ids, relevant, 5),
            "recall@10": recall_at(reranked_ids, relevant, 10),
            "ndcg@10":   ndcg_at(reranked_ids, relevance, 10),
        }

        dense_metrics.append(dense)
        reranked_metrics.append(reranked)
        details.append({
            "query_id":   qid,
            "query":      query_text[:120],
            "word_count": len(query_text.split()),
            "dense":      dense,
            "reranked":   reranked,
        })

    mean_words = float(np.mean([len(queries[qid].split()) for qid in query_ids]))
    return {
        "dataset":          f"beir/{name}",
        "queries":          sample_size,
        "corpus_documents": len(corpus_rows),
        "mean_query_words": round(mean_words, 1),
        "rerank_skipped":   skipped_rerank,
        "dense":            means(dense_metrics),
        "reranked":         means(reranked_metrics),
        "details":          details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    rng = random.Random(SEED)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    all_results: list[dict] = []
    total_start = time.perf_counter()

    for name in args.datasets:
        if not (BEIR_ROOT / name).exists():
            print(f"SKIP {name}: not found at {BEIR_ROOT / name}")
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        result = evaluate_dataset(name, args.queries, rng)
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        all_results.append(result)

        # Print per-dataset summary
        print(
            f"  [{name}] mean_query_words={result['mean_query_words']:.1f}  "
            f"rerank_skipped={result['rerank_skipped']}/{result['queries']}"
        )
        print(f"  [{name}] Dense   nDCG@10={result['dense']['ndcg@10']:.4f}  "
              f"R@5={result['dense']['recall@5']:.4f}  R@10={result['dense']['recall@10']:.4f}")
        print(f"  [{name}] Reranked nDCG@10={result['reranked']['ndcg@10']:.4f}  "
              f"R@5={result['reranked']['recall@5']:.4f}  R@10={result['reranked']['recall@10']:.4f}")
        delta = result["reranked"]["ndcg@10"] - result["dense"]["ndcg@10"]
        print(f"  [{name}] Reranker delta: {delta:+.4f}")

    total_elapsed = time.perf_counter() - total_start

    # Save detailed results per dataset
    for result in all_results:
        ds_name = result["dataset"].split("/")[-1]
        path = RESULTS_ROOT / f"beir-{ds_name}-{run_id}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"RESULTS {path}")

    # Save combined summary (no per-query details)
    summary = {
        "seed": SEED,
        "queries_per_dataset": args.queries,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "datasets": [
            {k: v for k, v in r.items() if k != "details"}
            for r in all_results
        ],
    }
    summary_path = RESULTS_ROOT / f"beir-summary-{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSUMMARY {summary_path}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEIR dense + reranking evaluation")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Which BEIR subsets to evaluate (default: all 3)",
    )
    p.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="Number of queries to sample per dataset (default: 100)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
