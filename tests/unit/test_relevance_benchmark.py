"""benchmarks/relevance_benchmark.py: pools, tie-aware ranking metrics, evaluation, tables."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("relevance_benchmark", _DIR / "relevance_benchmark.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["relevance_benchmark"] = rb          # dataclass annotations resolve via sys.modules
_spec.loader.exec_module(rb)


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #

class TestNdcg:


    def test_ties_share_positions_on_average(self):
        # one relevant doc tied with another at the top: expected DCG is the mean over both slots
        got = rb.ndcg_at_k({"rel": 5, "other": 5, "z": 0}, {"rel": 1})
        assert got == pytest.approx(0.5 + 0.5 / math.log2(3))


# --------------------------------------------------------------------------- #
# Pools (synthetic corpora)
# --------------------------------------------------------------------------- #

def _toy_beir(tmp_path: Path, name: str, n_docs: int, qrels: dict[str, dict[str, int]]) -> Path:
    base = tmp_path / "beir" / name
    (base / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": f"D{i}", "title": f"title {i} topic{i % 7}", "text": f"text about topic{i % 7} {i}",
         "metadata": {"url": f"http://pubmed/{i}"}}
        for i in range(n_docs)
    ]
    (base / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus), encoding="utf-8")
    (base / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": q, "text": f"query {q} topic3"}) for q in qrels),
        encoding="utf-8",
    )
    lines = ["query-id\tcorpus-id\tscore"] + [
        f"{q}\t{d}\t{g}" for q, rel in qrels.items() for d, g in rel.items()
    ]
    (base / "qrels" / "test.tsv").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


class TestBeirPools:
    def test_nfcorpus_pool_shape_and_no_false_negatives(self, tmp_path):
        rel = {f"D{i}": (2 if i < 4 else 1) for i in range(10)}          # 10 relevant docs
        root = _toy_beir(tmp_path, "nfcorpus", 120, {"Q1": rel, "Q2": {"D50": 1}})   # Q2: < 4 rel
        pools = rb.build_nfcorpus_pools(5, root=root)

        assert [p.id for p in pools] == ["nfcorpus-Q1"]                  # Q2 ineligible
        pool = pools[0]
        ids = [c["id"] for c in pool.candidates]
        assert len(ids) == len(set(ids)) and len(ids) == 6 + 9 + 9
        assert len(pool.gains) == 6
        assert sorted(pool.gains.values(), reverse=True)[:3] == [2, 2, 2]  # grade 2 preferred
        # relevant docs that were NOT chosen must not appear as negatives
        assert not (set(ids) - set(pool.gains)) & set(rel)
        assert all(g > 0 for g in pool.gains.values())


# --------------------------------------------------------------------------- #
# Scoring + evaluation
# --------------------------------------------------------------------------- #


def _run(scores: dict, failed=False):
    return {"pool": "p1", "shuffle": 0, "scores": scores, "failed": failed}
