"""--set overrides, the fixed ablation subset and the paired run comparison."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load("run_smoke_benchmark")
subset = _load("make_ablation_subset")
cmp = _load("compare_runs")


# --- --set -----------------------------------------------------------------


def test_apply_overrides_sets_and_rejects_unknown(monkeypatch):
    from research_swarm.config import settings

    monkeypatch.setattr(settings, "paper_max_candidates", 24)
    assert smoke.apply_overrides(["paper_max_candidates=7"]) == {"paper_max_candidates": 7}
    assert settings.paper_max_candidates == 7
    with pytest.raises(SystemExit):
        smoke.apply_overrides(["definitely_not_a_setting=1"])


# --- subset ----------------------------------------------------------------

def _tasks(n_per=40):
    out = []
    for fam in ("alce", "hotpotqa", "scifact"):
        for i in range(n_per):
            out.append({"id": f"{fam}-{i}", "dataset": f"{fam}/x", "prompt": "", "evidence": [],
                        "expected": [], "metadata": {}})
    return out


# --- compare_runs ----------------------------------------------------------

def _run(scores, seconds=10.0):
    return {
        f"t{i}": {
            "res": {"answer_score": s, "seconds": seconds, "false_refuted": 1,
                    "answer_findings": 2 if i else 0, "unanswerable": False, "abstained": False},
            "claims": {"claim_metrics": {"faithfulness": s}, "critical_error": False},
        }
        for i, s in enumerate(scores)
    }


def test_compare_flags_better_and_pairs_only_shared():
    base = _run([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cand = _run([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    del cand["t0"]                                          # not shared -> excluded
    out = cmp.compare(base, cand)
    assert out["tasks"] == 5
    row = next(r for r in out["rows"] if r["metric"] == "answer_score")
    assert row["n"] == 5 and row["diff"][0] == pytest.approx(0.4) and row["verdict"] == "BETTER"
