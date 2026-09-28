"""SciFact rationale scoring (benchmarks/score_rationales.py): quote -> sentence mapping, gold
pairs, fact sets and the reference points."""
from __future__ import annotations

import pytest

from benchmarks import score_rationales as sr

ABS_A = ["Aspirin lowers fever in adults.", "It was tested in 200 patients.",
         "Side effects were rare."]
ABS_B = ["Ibuprofen is an NSAID.", "Its effect on fever was not measured."]
CORPUS = {"11": {"doc_id": 11, "abstract": ABS_A}, "22": {"doc_id": 22, "abstract": ABS_B}}


def test_quote_maps_to_the_sentences_it_covers():
    assert sr.quote_sentences("tested in 200 patients", ABS_A) == {1}
    # spans a boundary: both sentences are selected
    assert sr.quote_sentences("in adults. It was tested", ABS_A) == {0, 1}
    # touching only a few characters of the next sentence does not select it
    assert sr.quote_sentences("lowers fever in adults. It", ABS_A) == {0}
    # fuzzy match (one word changed) still maps
    assert sr.quote_sentences("Side effects were very rare", ABS_A) == {2}
    assert sr.quote_sentences("completely unrelated words about volcanoes", ABS_A) is None


def test_gold_pairs_index_the_supplied_abstracts_only():
    claim = {"evidence": {"22": [{"sentences": [1], "label": "CONTRADICT"}],
                          "99": [{"sentences": [0], "label": "SUPPORT"}]}}
    assert sr.gold_pairs(claim, ["11", "22"]) == {(1, 1)}


def _result(task_id, facts, critiques=()):
    return {"task_id": task_id, "status": "ok", "finding_details": facts,
            "critique_details": list(critiques)}


def _fact(fid, task_id, doc, quote, grounding="quote", confidence=0.9):
    return {"id": fid, "quote": quote, "grounding": grounding, "confidence": confidence,
            "evidence_urls": [f"benchmark://{task_id}/{doc}"]}


def test_score_fact_sets_and_reference_points():
    claims = {
        1: {"id": 1, "cited_doc_ids": [11, 22],
            "evidence": {"11": [{"sentences": [0, 1], "label": "SUPPORT"}]}},
        2: {"id": 2, "cited_doc_ids": [22], "evidence": {}},              # NOT_ENOUGH_INFO
    }
    results = [
        _result("scifact-1", [
            _fact("a", "scifact-1", 0, "Aspirin lowers fever in adults."),          # gold (0,0)
            _fact("b", "scifact-1", 1, "Ibuprofen is an NSAID.", grounding="passage"),  # wrong
            _fact("c", "scifact-1", 0, "Side effects were rare."),                  # refuted
            _fact("d", "scifact-1", 0, "nothing like this appears anywhere at all"),  # unmapped
        ], critiques=[{"finding_id": "c", "verdict": "refuted"}]),
        _result("scifact-2", [_fact("e", "scifact-2", 0, "Ibuprofen is an NSAID.")]),
        _result("hotpotqa-9", []),                                                  # ignored
    ]
    s = sr.score(results, claims, CORPUS, n_boot=200)
    assert s["tasks"] == 2
    ext = s["sets"]["extracted"]
    # task 1 picks (0,0) (0,2) (1,0) -> tp 1, fp 2, fn 1; task 2 (NEI) picks (0,0) -> fp 1
    assert (ext["tp"], ext["fp"], ext["fn"]) == (1, 3, 1)
    assert ext["precision"] == pytest.approx(0.25) and ext["recall"] == pytest.approx(0.5)
    assert ext["task_hit_rate"] == 1.0 and ext["unmapped_quotes"] == 1
    assert ext["nei_sentences_per_task"] == 1.0
    verbatim = s["sets"]["verbatim"]                     # drops the passage-grounded fact b
    assert (verbatim["tp"], verbatim["fp"]) == (1, 2)
    kept = s["sets"]["kept"]                             # drops the refuted fact c
    assert (kept["tp"], kept["fp"]) == (1, 2)
    everything = s["sets"]["all sentences"]              # recall 1 by construction
    assert everything["recall"] == 1.0
    assert everything["precision"] == pytest.approx(2 / 7)   # 2 gold of 5 + 2 sentences
    rnd = s["sets"]["random, same count"]                # 3 picks of 5 sentences, 2 gold
    assert rnd["tp"] == pytest.approx(3 * 2 / 5)
    assert "Fact set" in sr.markdown(s)


def test_paired_diff_on_common_tasks():
    a = {"task_ids": ["t1", "t2"], "rows": {"extracted": [(2, 0, 0), (1, 1, 0)]}}
    b = {"task_ids": ["t2", "t1", "t3"], "rows": {"extracted": [(0, 2, 1), (1, 1, 1), (5, 0, 0)]}}
    d = sr.paired_diff(a, b, "extracted", n=200)
    # a: tp 3 fp 1 fn 0 -> P .75 R 1 ; b on t1+t2: tp 1 fp 3 fn 2 -> P .25 R 1/3
    assert d["precision"][0] == pytest.approx(0.5)
    assert d["recall"][0] == pytest.approx(2 / 3)
