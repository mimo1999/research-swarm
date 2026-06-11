"""Evaluate dense retrieval and reranking on 100 BEIR SciFact test queries."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from research_swarm.rag.indexes import get_embed_model
from research_swarm.rag.reranker import rerank

SEED = 42
QUERY_COUNT = 100
ROOT = Path("data/benchmarks/beir/scifact")
RESULTS_ROOT = Path("data/benchmark_results")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
            rows.extend(model.get_query_embedding(text) for text in batch)
        else:
            rows.extend(model.get_text_embedding_batch(batch))
    return normalize(rows)


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    lines = (ROOT / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        query_id, corpus_id, score = line.split("\t")
        qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / max(len(relevant), 1)


def ndcg_at(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2 ** relevance.get(doc_id, 0) - 1) / math.log2(index + 2)
        for index, doc_id in enumerate(ranked[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(index + 2) for index, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    started = time.perf_counter()
    corpus = read_jsonl(ROOT / "corpus.jsonl")
    queries = {row["_id"]: row["text"] for row in read_jsonl(ROOT / "queries.jsonl")}
    qrels = load_qrels()
    query_ids = random.Random(SEED).sample(sorted(qrels), QUERY_COUNT)

    corpus_ids = [row["_id"] for row in corpus]
    corpus_texts = [f"{row.get('title', '')}\n{row['text']}" for row in corpus]
    corpus_embeddings = embed_batches(corpus_texts, query=False)
    query_embeddings = embed_batches([queries[qid] for qid in query_ids], query=True)
    similarities = query_embeddings @ corpus_embeddings.T

    dense_metrics = []
    reranked_metrics = []
    details = []
    for row_index, query_id in enumerate(query_ids):
        candidate_indices = np.argpartition(similarities[row_index], -20)[-20:]
        candidate_indices = candidate_indices[
            np.argsort(similarities[row_index, candidate_indices])[::-1]
        ]
        dense_ids = [corpus_ids[index] for index in candidate_indices]
        relevance = qrels[query_id]
        relevant = set(relevance)
        dense = {
            "recall@5": recall_at(dense_ids, relevant, 5),
            "recall@10": recall_at(dense_ids, relevant, 10),
            "ndcg@10": ndcg_at(dense_ids, relevance, 10),
        }
        chunks = [
            {
                "url": f"beir://scifact/{corpus_ids[index]}",
                "title": corpus[index].get("title", ""),
                "snippet": corpus[index]["text"],
                "credibility_score": float(similarities[row_index, index]),
            }
            for index in candidate_indices
        ]
        reranked_ids = [
            item["url"].rsplit("/", 1)[-1]
            for item in rerank(queries[query_id], chunks, top_k=10)
        ]
        reranked = {
            "recall@5": recall_at(reranked_ids, relevant, 5),
            "recall@10": recall_at(reranked_ids, relevant, 10),
            "ndcg@10": ndcg_at(reranked_ids, relevance, 10),
        }
        dense_metrics.append(dense)
        reranked_metrics.append(reranked)
        details.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "dense": dense,
                "reranked": reranked,
            }
        )

    def means(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: round(float(np.mean([row[key] for row in rows])), 4)
            for key in rows[0]
        }

    result = {
        "dataset": "beir/scifact",
        "seed": SEED,
        "queries": QUERY_COUNT,
        "corpus_documents": len(corpus),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dense": means(dense_metrics),
        "reranked": means(reranked_metrics),
        "details": details,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULTS_ROOT / f"beir-scifact-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))
    print(f"RESULTS {path}")


if __name__ == "__main__":
    main()
