"""agents/verifier.py: policy table, no-LLM refutation, missing verdicts, batching, conflicts."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_swarm.agents import verifier as vf
from research_swarm.config import settings
from research_swarm.schemas import Finding, ResearchQuery, Source
from research_swarm.schemas.critique import CritiqueVerdict


def _finding(i, grounding="quote", conf=0.6):
    return Finding(
        id=f"f{i}", claim=f"claim {i}", sub_question="sq", grounding=grounding, confidence=conf,
        evidence=[Source(url=f"http://x/{i}", title=f"T{i}", snippet=f"evidence {i}")],
    )


def _state(findings):
    return {"findings": findings, "session_id": "s",
            "query": ResearchQuery(topic="the question", depth="shallow")}


def _llm(verdicts_by_call):
    """Each ainvoke returns the next VerifyBatch in the list."""
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=verdicts_by_call)
    return llm


def _v(n, verdict="supported", conflicts=()):
    return vf.FactVerdict(fact=n, verdict=verdict, conflicts_with=list(conflicts))


def _set_cap(monkeypatch, cap):
    """The writer cap for these tests' shallow runs (a depth profile overrides the global)."""
    monkeypatch.setattr(settings, "depth_profiles", {"shallow": {"max_facts_for_writer": cap}})


@pytest.fixture(autouse=True)
def _cap(monkeypatch):
    _set_cap(monkeypatch, 30)


# --- policy table -------------------------------------------------------------

def test_policy_table():
    table = [
        ("supported", "quote", CritiqueVerdict.supported, 0.9),
        ("supported", "passage", CritiqueVerdict.supported, 0.75),
        ("partial", "quote", CritiqueVerdict.weak, 0.5),
        ("unsupported", "passage", CritiqueVerdict.refuted, 0.1),
        ("unsupported", "quote", CritiqueVerdict.weak, 0.3),   # verbatim quote: hedge, do not hide
    ]
    for verdict, grounding, crit, conf in table:
        new, c = vf._apply(_finding(1, grounding), verdict)
        assert c.verdict == crit and new.confidence == conf and c.finding_id == "f1", verdict
        assert c.reasoning == f"verifier:{verdict}"


# --- run_verifier ---------------------------------------------------------------

async def test_ungrounded_facts_are_refuted_without_an_llm_call():
    llm = _llm([])
    findings, critiques, _ = await vf.run_verifier(
        _state([_finding(1, "none"), _finding(2, "none")]), llm)
    assert {c.verdict for c in critiques} == {CritiqueVerdict.refuted}
    assert all(f.confidence == 0.1 for f in findings)
    llm.with_structured_output.return_value.ainvoke.assert_not_awaited()


async def test_missing_verdict_counts_as_partial_not_refuted():
    llm = _llm([vf.VerifyBatch(verdicts=[_v(1)])])                    # nothing for fact 2
    _, critiques, _ = await vf.run_verifier(_state([_finding(1), _finding(2)]), llm)
    by = {c.finding_id: c.verdict for c in critiques}
    assert by == {"f1": CritiqueVerdict.supported, "f2": CritiqueVerdict.weak}


async def test_conflicts_are_deduplicated_unordered_pairs():
    llm = _llm([vf.VerifyBatch(verdicts=[_v(1, conflicts=[2]), _v(2, conflicts=[1])])])
    _, _, conflicts = await vf.run_verifier(_state([_finding(1), _finding(2)]), llm)
    assert conflicts == [["f1", "f2"]]


async def test_cap_keeps_best_grounded_and_lowers_the_rest_below_writer_bar(monkeypatch):
    _set_cap(monkeypatch, 2)
    fs = [_finding(1, "passage", 0.5), _finding(2, "quote", 0.6), _finding(3, "quote", 0.9),
          _finding(4, "none", 0.9)]
    llm = _llm([vf.VerifyBatch(verdicts=[_v(1), _v(2)])])
    findings, critiques, _ = await vf.run_verifier(_state(fs), llm)
    capped = [f for f in findings if f.confidence == 0.05]
    assert {f.id for f in capped} == {"f1", "f4"}                    # not refuted, just dropped
    assert {c.finding_id for c in critiques} == {"f2", "f3"}
