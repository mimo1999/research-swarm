"""Side-by-side BEIR evaluation: dense (BGE-small) vs ms-marco-MiniLM vs bge-reranker-base vs mxbai-rerank-xsmall.

Does NOT modify any production code — results only.

Memory strategy
---------------
Corpus embeddings are cached to disk (data/benchmark_results/emb_cache/) so
re-runs skip the expensive embedding step.

Usage:
    poetry run python benchmarks/run_beir_reranker_compare.py
    poetry run python benchmarks/run_beir_reranker_compare.py --datasets scifact nfcorpus
    poetry run python benchmarks/run_beir_reranker_compare.py --queries 50
    poetry run python benchmarks/run_beir_reranker_compare.py --no-cache
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from research_swarm.rag.indexes import get_embed_model

SEED = 42
DEFAULT_QUERY_COUNT = 100
BEIR_ROOT = Path("data/benchmarks/beir")
RESULTS_ROOT = Path("data/benchmark_results")
EMB_CACHE_ROOT = Path("data/benchmark_results/emb_cache")
ALL_DATASETS = ["scifact", "nfcorpus", "arguana"]

_MSMARCO_GUARD = 8  # skip reranking for queries longer than this many words

# (result key, print label, HF model id, CrossEncoder max_length)
RERANKER_MODELS = [
    ("msmarco", "ms-marco-MiniLM", "cross-encoder/ms-marco-MiniLM-L-6-v2", 512),
    ("bge", "bge-reranker-base", "BAAI/bge-reranker-base", 512),
    ("mxbai", "mxbai-rerank-xsmall", "mixedbread-ai/mxbai-rerank-xsmall-v1", 512),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_qrels(dataset_root: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in dataset_root.joinpath("qrels", "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return qrels


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def normalize(matrix: np.ndarray) -> np.ndarray:
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
    return normalize(np.asarray(rows, dtype=np.float32))


def load_or_embed_corpus(
    name: str,
    corpus_rows: list[dict],
    use_cache: bool,
) -> tuple[list[str], np.ndarray]:
    n = len(corpus_rows)
    EMB_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    emb_path = EMB_CACHE_ROOT / f"beir_{name}_{n}.npy"
    ids_path = EMB_CACHE_ROOT / f"beir_{name}_{n}_ids.json"

    if use_cache and emb_path.exists() and ids_path.exists():
        print(f"  [{name}] Loading cached corpus embeddings ({n:,} docs) ...", flush=True)
        corpus_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        corpus_emb = np.load(str(emb_path))
        print(f"  [{name}] Cache loaded.", flush=True)
        return corpus_ids, corpus_emb

    corpus_ids = [row["_id"] for row in corpus_rows]
    corpus_texts = [f"{row.get('title', '')}\n{row['text']}" for row in corpus_rows]
    print(f"  [{name}] Embedding corpus ({n:,} docs) ...", flush=True)
    t0 = time.perf_counter()
    corpus_emb = embed_batches(corpus_texts, query=False)
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] Embedding done in {elapsed:.1f}s - saving to cache.", flush=True)
    np.save(str(emb_path), corpus_emb)
    ids_path.write_text(json.dumps(corpus_ids), encoding="utf-8")
    return corpus_ids, corpus_emb


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
# Cross-encoder pass (shared by ms-marco and bge-reranker-base)
# ---------------------------------------------------------------------------

def run_cross_encoder_pass(
    label: str,
    model_name: str,
    max_length: int,
    query_texts: list[str],
    all_chunks: list[list[dict]],
    dense_ids_list: list[list[str]],
    relevant_list: list[set[str]],
    relevance_list: list[dict[str, int]],
    guard_words: int,
) -> tuple[dict[str, float], int]:
    print(f"\n  --- {label} pass ---", flush=True)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name, max_length=max_length)

    all_pairs: list[tuple[str, str]] = []
    skip_flags: list[bool] = []

    for q_text, chunks in zip(query_texts, all_chunks):
        if len(q_text.split()) > guard_words:
            skip_flags.append(True)
        else:
            skip_flags.append(False)
            for c in chunks:
                passage = f"{c.get('title', '')}\n{c.get('snippet', '')}".strip()
                all_pairs.append((q_text, passage))

    all_scores: list[float] = []
    if all_pairs:
        print(f"  Scoring {len(all_pairs)} pairs ...", flush=True)
        t0 = time.perf_counter()
        all_scores = model.predict(all_pairs).tolist()
        print(f"  Done in {time.perf_counter()-t0:.1f}s.", flush=True)

    del model
    gc.collect()

    metrics: list[dict] = []
    skipped = 0
    pair_cursor = 0

    for q_idx, (chunks, dense_ids) in enumerate(zip(all_chunks, dense_ids_list)):
        if skip_flags[q_idx]:
            reranked_ids = dense_ids[:10]
            skipped += 1
        else:
            n = len(chunks)
            scores = all_scores[pair_cursor : pair_cursor + n]
            pair_cursor += n
            ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
            reranked_ids = [c["url"].rsplit("/", 1)[-1] for _, c in ranked[:10]]

        metrics.append({
            "recall@5":  recall_at(reranked_ids, relevant_list[q_idx], 5),
            "recall@10": recall_at(reranked_ids, relevant_list[q_idx], 10),
            "ndcg@10":   ndcg_at(reranked_ids, relevance_list[q_idx], 10),
        })

    return means(metrics), skipped


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(name: str, query_count: int, rng: random.Random, use_cache: bool) -> dict:
    root = BEIR_ROOT / name
    corpus_rows = read_jsonl(root / "corpus.jsonl")
    all_queries = {row["_id"]: row["text"] for row in read_jsonl(root / "queries.jsonl")}
    qrels = load_qrels(root)

    eligible = [qid for qid in qrels if qid in all_queries]
    sample_size = min(query_count, len(eligible))
    query_ids = rng.sample(sorted(eligible), sample_size)
    queries = {qid: all_queries[qid] for qid in query_ids}

    print(
        f"  [{name}] {len(corpus_rows):,} corpus docs, "
        f"{sample_size}/{len(eligible)} queries sampled",
        flush=True,
    )

    # Step 1: corpus embeddings (cached).
    corpus_ids, corpus_emb = load_or_embed_corpus(name, corpus_rows, use_cache)

    # Step 2: query embeddings + dense retrieval — collect everything for reranking.
    query_texts = [queries[qid] for qid in query_ids]
    query_emb = embed_batches(query_texts, query=True)
    similarities = query_emb @ corpus_emb.T

    dense_metrics: list[dict] = []
    all_chunks: list[list[dict]] = []
    dense_ids_list: list[list[str]] = []
    relevance_list: list[dict[str, int]] = []
    relevant_list: list[set[str]] = []

    for row_idx, qid in enumerate(query_ids):
        cand_idx = np.argpartition(similarities[row_idx], -20)[-20:]
        cand_idx = cand_idx[np.argsort(similarities[row_idx, cand_idx])[::-1]]
        dense_ids = [corpus_ids[i] for i in cand_idx]

        relevance = qrels[qid]
        relevant = {k for k, v in relevance.items() if v > 0}

        dense_metrics.append({
            "recall@5":  recall_at(dense_ids, relevant, 5),
            "recall@10": recall_at(dense_ids, relevant, 10),
            "ndcg@10":   ndcg_at(dense_ids, relevance, 10),
        })

        chunks = [
            {
                "url":     f"beir://{name}/{corpus_ids[i]}",
                "title":   corpus_rows[i].get("title", ""),
                "snippet": corpus_rows[i]["text"],
            }
            for i in cand_idx
        ]
        all_chunks.append(chunks)
        dense_ids_list.append(dense_ids)
        relevance_list.append(relevance)
        relevant_list.append(relevant)

    # Step 3: each reranker model, built/scored/unloaded in turn.
    reranker_results: dict[str, dict] = {}
    for key, label, model_name, max_length in RERANKER_MODELS:
        means_, skipped = run_cross_encoder_pass(
            label, model_name, max_length,
            query_texts, all_chunks, dense_ids_list, relevant_list, relevance_list,
            guard_words=_MSMARCO_GUARD,
        )
        reranker_results[key] = {"means": means_, "guard_fires": skipped}

    mean_words = float(np.mean([len(q.split()) for q in query_texts]))
    result = {
        "dataset":          f"beir/{name}",
        "queries":          sample_size,
        "corpus_documents": len(corpus_rows),
        "mean_query_words": round(mean_words, 1),
        "dense":            means(dense_metrics),
    }
    for key, _label, _model_name, _max_length in RERANKER_MODELS:
        result[f"{key}_reranked"] = reranker_results[key]["means"]
        result[f"{key}_guard_fires"] = reranker_results[key]["guard_fires"]
    return result


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
        result = evaluate_dataset(name, args.queries, rng, use_cache=not args.no_cache)
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        all_results.append(result)

        d = result["dense"]
        guard_str = "  ".join(
            f"{label} guard fires={result[f'{key}_guard_fires']}/{result['queries']}"
            for key, label, _mn, _ml in RERANKER_MODELS
        )
        print(f"\n  [{name}] mean_query_words={result['mean_query_words']:.1f}  {guard_str}")
        print(f"  [{name}] {'Method':<28} {'nDCG@10':>8}  {'R@5':>7}  {'R@10':>7}  {'Delta nDCG@10':>13}")
        print(f"  [{name}] {'Dense (BGE-small)':<28} {d['ndcg@10']:>8.4f}  {d['recall@5']:>7.4f}  {d['recall@10']:>7.4f}  {'baseline':>13}")
        for key, label, _mn, _ml in RERANKER_MODELS:
            m = result[f"{key}_reranked"]
            print(f"  [{name}] {'+ ' + label:<28} {m['ndcg@10']:>8.4f}  {m['recall@5']:>7.4f}  {m['recall@10']:>7.4f}  {m['ndcg@10']-d['ndcg@10']:>+13.4f}")

    total_elapsed = time.perf_counter() - total_start

    summary = {
        "seed": SEED,
        "queries_per_dataset": args.queries,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "datasets": all_results,
    }
    summary_path = RESULTS_ROOT / f"beir-reranker-compare-{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY -- nDCG@10")
    header = f"  {'Dataset':<12}  {'Dense':>7}"
    for _key, label, _mn, _ml in RERANKER_MODELS:
        header += f"  {label:>13}  {'Delta':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        d = r["dense"]["ndcg@10"]
        row = f"  {r['dataset'].split('/')[-1]:<12}  {d:>7.4f}"
        for key, _label, _mn, _ml in RERANKER_MODELS:
            m = r[f"{key}_reranked"]["ndcg@10"]
            row += f"  {m:>13.4f}  {m-d:>+7.4f}"
        print(row)
    print(f"\nResults saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--queries", type=int, default=DEFAULT_QUERY_COUNT)
    p.add_argument("--no-cache", action="store_true", help="Force re-embedding even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
