"""benchmarks/run_nfcorpus_coverage.py: ID matching, judged/unjudged accounting, sampling."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nc = _load("run_nfcorpus_coverage")

CORPUS = {
    "MED-1": {"title": "Statins and Breast Cancer Survival", "text": "x",
              "url": "http://www.ncbi.nlm.nih.gov/pubmed/111"},
    "MED-2": {"title": "Olive Oil and Heart Disease", "text": "x",
              "url": "http://www.ncbi.nlm.nih.gov/pubmed/222"},
    "MED-3": {"title": "Green Tea Trial", "text": "x",
              "url": "http://www.ncbi.nlm.nih.gov/pubmed/333"},
}
INDEX = nc.build_index(CORPUS)


def item(url, title="", st="pubmed"):
    return {"url": url, "title": title, "source_type": st}


# --- URL / title normalisation ------------------------------------------------


def test_match_title_fallback_for_pmc_only_record():
    m = nc.match_item("https://europepmc.org/article/PMC/PMC5",
                      "Olive oil and heart disease (Circulation 2018)", "europe_pmc", INDEX)
    assert (m["status"], m["doc_id"], m["method"]) == ("judged", "MED-2", "title")


# --- per-query metrics --------------------------------------------------------

QRELS = {"MED-1": 2, "MED-2": 1, "MED-3": 1}     # 3 relevant


def test_coverage_metrics_judged_and_unjudged_accounting():
    cited = [
        item("https://pubmed.ncbi.nlm.nih.gov/111"),      # relevant, grade 2
        item("https://pubmed.ncbi.nlm.nih.gov/999"),      # unjudged
        item("https://example.com/x", "Blog", "web"),     # other
    ]
    m = nc.coverage_metrics(cited, QRELS, INDEX)
    assert m["n_cited"] == 3 and m["n_judged"] == 1
    assert m["judged_precision"] == 1.0
    assert m["unjudged_rate"] == pytest.approx(1 / 3)
    assert m["graded_gain"] == 2
    assert m["hit"] == 1.0
    assert m["cited_recall"] == pytest.approx(1 / 3)


# --- trace parsing / sampling / summary ---------------------------------------


# --- LLM-judged relevance ---------------------------------------------------------------------
