"""Per-depth research profiles (settings.depth_profiles) and the gap-fill worker cap."""
from __future__ import annotations

from research_swarm.config import settings
from research_swarm.graph.nodes import _research_targets
from research_swarm.schemas import Finding, ResearchPlan, ResearchQuery
from research_swarm.schemas.query import ResearchDepth


def test_profiles_scale_with_depth_and_fall_back_to_globals(monkeypatch):
    monkeypatch.setattr(settings, "paper_max_candidates", 24)
    subs = [settings.for_depth("sub_questions", d) for d in ("shallow", "standard", "deep")]
    assert subs == [3, 5, 7]
    assert settings.for_depth("sub_questions", ResearchDepth.deep) == 7      # enum accepted
    for d in ("shallow", "standard", "deep"):                               # no paid-for fact
        assert settings.for_depth("max_facts_for_writer", d) == 6 * settings.for_depth(
            "sub_questions", d)                                             # is dropped
    assert settings.for_depth("paper_max_candidates", None) == 24          # no depth: global
    monkeypatch.setattr(settings, "depth_profiles", {"shallow": {"sub_questions": 2}})
    assert settings.for_depth("paper_max_candidates", "shallow") == 24     # key not in profile
    assert settings.max_research_rounds("shallow") == settings.research_rounds


def _state(depth: str, covered: dict[str, int]) -> dict:
    sqs = ["q1", "q2", "q3", "q4"]
    findings = [Finding(claim=f"fact about {sq}", evidence=[], confidence=0.8, sub_question=sq)
                for sq, n in covered.items() for _ in range(n)]
    return {"plan": ResearchPlan(sub_questions=sqs, strategy="s"), "findings": findings,
            "critiques": [], "research_rounds": 0, "session_id": "t",
            "query": ResearchQuery(topic="t", depth=depth)}


def test_gap_fill_workers_capped_per_depth_least_covered_first(monkeypatch):
    monkeypatch.setattr(settings, "min_grounded_facts", 2)
    monkeypatch.setattr(settings, "depth_profiles", {"shallow": {"gap_fill_workers": 2},
                                                     "deep": {"gap_fill_workers": 5}})
    covered = {"q1": 1, "q3": 0, "q4": 1}          # q2 has 0 too; all four under the bar of 2
    assert _research_targets(_state("shallow", covered)) == ["q2", "q3"]
    assert sorted(_research_targets(_state("deep", covered))) == ["q1", "q2", "q3", "q4"]
