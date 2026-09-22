"""benchmarks/score_claims.py and benchmarks/judge_validation.py."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


sc = _load("score_claims")
jv = _load("judge_validation")

from research_swarm.eval.claims import ClaimVerdict  # noqa: E402


def _task(id="t1", dataset="hotpotqa/bridge", expected=("Paris",)):
    return sc.smoke.BenchmarkTask(
        id=id, dataset=dataset, prompt="q?", metadata={}, expected=list(expected),
        evidence=[{"title": "Louvre", "text": "The Louvre is in Paris."},
                  {"title": "Berlin", "text": "Berlin is in Germany."}],
    )


def _result(task_id="t1", dataset="hotpotqa/bridge", **kw):
    base = {
        "task_id": task_id, "dataset": dataset, "status": "ok", "model": "gen-model",
        "worker_model": "gen-model", "seconds": 10.0, "answer_score": 1.0,
        "normalized_answer_score": 1.0, "unanswerable": False, "abstained": False,
        "support_doc_recall": 1.0,
        "report": {
            "title": "T", "exec_summary": "Paris hosts the Louvre [1].",
            "sections": [{"heading": "S", "body_md": "Berlin is in Germany [2].", "citations": []}],
            "references": [{"url": f"benchmark://{task_id}/0"}, {"url": f"benchmark://{task_id}/1"}],
        },
    }
    base.update(kw)
    return base


def _verdict(n, cited="yes", corpus="yes", cw=True, contradicted=False):
    return ClaimVerdict(claim=n, checkworthy=cw, cited_support=cited, corpus_support=corpus,
                        contradicted=contradicted)


class TestIndependence:
    def test_a_model_may_not_grade_its_own_reports(self):
        with pytest.raises(SystemExit, match="also generated"):
            sc.check_independent("gen-model", [_result()])


class TestCriticalError:
    def test_scifact_direction_flip(self):
        task = _task(dataset="scifact", expected=("SUPPORT",))
        m = {"n_contradicted": 0}
        assert sc.is_critical_error(task, "Verdict: CONTRADICT.", m) == (True, True)
        assert sc.is_critical_error(task, "Verdict: SUPPORT.", m) == (False, False)


class TestMainResumable:
    def _setup(self, tmp_path):
        tasks = [_task("a"), _task("b")]
        (tmp_path / "smoke-x-tasks.json").write_text(
            json.dumps([sc.smoke.asdict(t) for t in tasks]), encoding="utf-8")
        results = tmp_path / "smoke-x-results.jsonl"
        results.write_text("\n".join(json.dumps(_result(t.id)) for t in tasks), encoding="utf-8")
        return results

    def _args(self, results, **kw):
        base = dict(results=str(results), tasks=None, judge_model="judge-x", out=None, limit=None,
                    concurrency=2, allow_same_model=False, summarize_only=False,
                    update_readme=False)
        return argparse.Namespace(**{**base, **kw})

    def test_scores_each_task_once_and_resumes(self, tmp_path):
        results = self._setup(tmp_path)
        scored: list[str] = []

        async def fake_score(task, result, llm):
            scored.append(task.id)
            return {"task_id": task.id, "dataset": task.dataset, "critical_error": False,
                    "verdict_flip": False, "documents": [], "claims": [],
                    "claim_metrics": {m: None for m in sc.CLAIM_METRICS} | {
                        "n_contradicted": 0, "n_checkworthy": 0, "n_claims": 0}}

        with patch.object(sc, "build_judge", return_value=MagicMock()), \
             patch.object(sc, "score_task", fake_score):
            asyncio.run(sc.main(self._args(results)))
            assert sorted(scored) == ["a", "b"]
            asyncio.run(sc.main(self._args(results)))          # second run: nothing left to do
        assert sorted(scored) == ["a", "b"]

        claims = tmp_path / "smoke-x-claims.jsonl"
        assert len(claims.read_text(encoding="utf-8").splitlines()) == 2
        assert (tmp_path / "smoke-x-claims-summary.json").exists()


# --------------------------------------------------------------------------- #
# judge_validation
# --------------------------------------------------------------------------- #

class TestCohenKappa:


    def test_known_value(self):
        # observed 0.7; expected = 0.5*0.6 + 0.5*0.4 = 0.5  ->  kappa = 0.4
        a = ["yes"] * 5 + ["no"] * 5
        b = ["yes"] * 4 + ["no"] + ["yes"] * 2 + ["no"] * 3
        assert jv.cohen_kappa(a, b) == pytest.approx(0.4)
