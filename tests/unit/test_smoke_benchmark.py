"""Tests for the closed-corpus adaptation of benchmarks/run_smoke_benchmark.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "run_smoke_benchmark.py"
_spec = importlib.util.spec_from_file_location("smoke_benchmark", _PATH)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["smoke_benchmark"] = smoke     # dataclasses resolves annotations via sys.modules
_spec.loader.exec_module(smoke)


def _task(dataset="hotpotqa/bridge", expected=None, evidence=None) -> smoke.BenchmarkTask:
    return smoke.BenchmarkTask(
        id="t1", dataset=dataset, prompt="q?", metadata={},
        evidence=evidence if evidence is not None else [{"title": "A", "text": "Paris is in France."}],
        expected=expected if expected is not None else ["Paris"],
    )


class TestScifactScoring:

    def test_first_named_verdict_wins(self):
        assert smoke._scifact_verdict("SUPPORT, though some might say CONTRADICT") == "SUPPORT"


class TestNumberGrounding:
    def test_numbers_absent_from_corpus_and_question_are_ungrounded(self):
        g = smoke.number_grounding(
            "It rose 45% in 2019 to 8,000 units [1].",
            corpus_text="Revenue rose 45% in 2019.", prompt="Units sold?",
        )
        assert g["numbers"] == 3 and g["grounded"] == 2
        assert g["ungrounded"] == ["8000"]
        assert g["rate"] == pytest.approx(0.6667, abs=1e-3)


def _hotpot_task() -> smoke.BenchmarkTask:
    return smoke.BenchmarkTask(
        id="h1", dataset="hotpotqa/bridge", prompt="Which city hosts the Louvre?",
        expected=["Paris"], metadata={"gold_docs": ["Louvre", "Paris"]},
        evidence=[
            {"title": "Louvre", "text": "The Louvre is a museum in Paris opened in 1793."},
            {"title": "Paris", "text": "Paris is the capital of France."},
            {"title": "Berlin", "text": "Berlin is the capital of Germany."},
        ],
    )


def _metrics(task, report, findings, critiques):
    return smoke.compute_task_metrics(task, smoke._task_documents(task), report, findings, critiques)


def _report(summary="Paris hosts the Louvre [1].", refs=("benchmark://h1/0", "benchmark://h1/2")):
    return {
        "title": "Louvre", "exec_summary": summary,
        "sections": [{"heading": "h", "body_md": "It opened in 1793 [1] and in 1900 [2].",
                      "citations": [1]}],
        "references": [{"url": u} for u in refs],
    }


class TestComputeTaskMetrics:


    def test_localization_finds_answer_lost_after_extraction(self):
        findings = [
            {"id": "f1", "claim": "The Louvre is in Paris.", "confidence": 0.9, "evidence_urls": []},
            {"id": "f2", "claim": "Paris is the capital.", "confidence": 0.05, "evidence_urls": []},
            {"id": "f3", "claim": "Berlin is in Germany.", "confidence": 0.9, "evidence_urls": []},
        ]
        critiques = [
            {"finding_id": "f1", "verdict": "weak"},
            {"finding_id": "f1", "verdict": "refuted"},       # latest verdict wins
            {"finding_id": "f3", "verdict": "refuted"},       # not answer-bearing -> not counted
        ]
        report = _report(summary="The corpus does not say.")
        report["sections"] = []                                # the answer never reached the report
        m = _metrics(_hotpot_task(), report, findings, critiques)

        assert m["finding_recall"] == 1.0 and m["answer_score"] == 0.0
        assert m["synthesis_loss"] == 1.0
        assert m["answer_findings"] == 2                       # f1 and f2 mention Paris
        assert m["false_refuted"] == 1                         # f1 (latest verdict: refuted)
        assert m["writer_dropped"] == 2                        # f1 refuted, f2 confidence < 0.1
