"""Section-by-section writer (agents/writer_sections.py) and the writer's own model endpoint.

The fake model answers by schema: the outline, each section and the review are separate calls,
so each can be scripted (or made to fail) independently."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from research_swarm.agents import writer as writer_mod
from research_swarm.agents.writer_render import DraftSentence, WriterDraft
from research_swarm.agents.writer_sections import (
    ReportOutline,
    ReviewPass,
    SectionPlan,
    SectionText,
    SentenceFix,
)
from research_swarm.config import settings
from research_swarm.schemas import Critique, CritiqueVerdict, Finding, Source
from tests.unit.test_graph import _make_plan, _make_state


class FakeLLM:
    """``with_structured_output(schema).ainvoke(messages)`` -> the value scripted for that schema
    (a callable gets the messages; an Exception is raised)."""

    def __init__(self, by_schema: dict):
        self.by_schema = by_schema
        self.calls: list[tuple[str, list]] = []

    def with_structured_output(self, schema, **_kw):
        outer = self

        class _Runnable:
            async def ainvoke(self, messages, *_a, **_k):
                outer.calls.append((schema.__name__, messages))
                value = outer.by_schema[schema.__name__]
                if isinstance(value, Exception):
                    raise value
                return value(messages) if callable(value) else value

        return _Runnable()

    def prompts(self, schema_name: str) -> list[str]:
        return [msgs[1].content for name, msgs in self.calls if name == schema_name]


def _fact(i, claim, sq):
    return Finding(id=f"f{i}", claim=claim, sub_question=sq, confidence=0.9,
                   evidence=[Source(url=f"https://arxiv.org/abs/2600.0000{i}", title=f"Paper {i}",
                                    snippet=claim)])


def _state(audience="academic"):
    plan = _make_plan(2)
    sq1, sq2 = plan.sub_questions
    facts = [_fact(1, "Method A reuses the cache for identical architectures.", sq1),
             _fact(2, "Method B maps caches between model sizes with 73% accuracy retention.", sq2),
             _fact(3, "Method C recomputes the first 4 layers and reuses the rest.", sq2)]
    q = _make_state()["query"].model_copy(update={"audience": audience, "topic": "Can caches move?"})
    return _make_state(plan=plan, findings=facts, query=q, critiques=[
        Critique(finding_id=f.id, verdict=CritiqueVerdict.supported, reasoning="r") for f in facts
    ])


def _outline(**kw):
    base = dict(
        title="Cache transfer", direct_answer="Only approximately.", answer_facts=[2],
        stance="partial", limitations="",
        sections=[SectionPlan(heading="Previous Work", purpose="prior reuse", facts=[1]),
                  SectionPlan(heading="Experiments", purpose="transfer results", facts=[2])],
    )
    base.update(kw)
    return ReportOutline(**base)


def _section(messages):
    """Scripted section text, keyed by the section being written."""
    user = messages[1].content
    heading = user.split("Section to write now: ")[1].split("\n")[0]
    return SectionText(sentences={
        "Previous Work": [DraftSentence(
            text="Method A reuses the cache for identical architectures.", facts=[1])],
        "Experiments": [
            DraftSentence(text="Method B maps caches with 73% accuracy retention.", facts=[2]),
            DraftSentence(text="Method B makes transfer exact and lossless.", facts=[2]),
            DraftSentence(text="Method C recomputes the first 4 layers.", facts=[3]),
        ],
        "Discussion": [DraftSentence(
            text="Reuse therefore hinges on how closely the two models match.", facts=[1, 2])],
        "Abstract": [DraftSentence(
            text="Caches move between models only approximately, within one family.", facts=[2])],
    }.get(heading, []))


# Reading order: Abstract S1 | Previous Work S2 | Experiments S3 S4 S5 | Discussion S6
_REVIEW = ReviewPass(
    fixes=[SentenceFix(sentence=4, problem="overclaim", fix="")],          # delete the overclaim
    direct_answer="Not exactly: transfer between different models is approximate.",
    answer_facts=[2], stance="partial",
    summary=[DraftSentence(text="Reuse works across identical architectures only.", facts=[1])],
)


@pytest.fixture(autouse=True)
def _sectioned(monkeypatch):
    monkeypatch.setattr(settings, "writer_mode", "sectioned")


def _prompt_for(llm: FakeLLM, heading: str) -> str:
    return next(p for p in llm.prompts("SectionText") if f"Section to write now: {heading}\n" in p)


def _facts_part(prompt: str) -> str:
    return prompt.split("Facts for this section:")[1]


async def test_fixed_sections_are_written_in_order_each_seeing_the_ones_before():
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section, "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(_state(), llm)

    # evidence sections first, then the interpretive one, then the overview -- one call each
    written = [p.split("Section to write now: ")[1].split("\n")[0]
               for p in llm.prompts("SectionText")]
    assert written == ["Previous Work", "Experiments", "Discussion", "Abstract"]
    # ... but rendered in the report type's reading order
    assert [s.heading for s in report.sections] == [
        "Abstract", "Previous Work", "Experiments", "Discussion"]
    # each section sees what was written before it
    assert "Sections already written:\n(none yet)" in _prompt_for(llm, "Previous Work")
    exp = _prompt_for(llm, "Experiments")
    assert "## Previous Work\nMethod A reuses the cache for identical architectures. [F1]" in exp
    abstract = _prompt_for(llm, "Abstract")
    assert "## Discussion" in abstract and "## Experiments" in abstract
    # evidence sections see only their own facts (with URLs); F3, left out of the outline, went
    # to the section holding its sub-question's facts
    prev = _facts_part(_prompt_for(llm, "Previous Work"))
    assert "F1 " in prev and "F2 " not in prev
    assert "F2 " in _facts_part(exp) and "F3 " in _facts_part(exp)
    assert "https://arxiv.org/abs/2600.00002" in _facts_part(exp)
    # the interpretive section gets the facts the evidence sections cited
    assert "F3 " in _facts_part(_prompt_for(llm, "Discussion"))

    body = " ".join(s.body_md for s in report.sections)
    assert "exact and lossless" not in body                          # deleted by the review
    assert "recomputes the first 4 layers" in body                   # the leftover fact is used
    assert report.exec_summary.startswith("**Answer:** Not exactly")  # the review's answer
    review_prompt = llm.prompts("ReviewPass")[0]
    assert "S4 [F2] Method B makes transfer exact and lossless." in review_prompt
    assert "Evidence:" in review_prompt


async def test_review_fix_replaces_the_sentence_text_and_facts():
    review = _REVIEW.model_copy(update={"fixes": [SentenceFix(
        sentence=3, problem="overclaim", fix="Method B retains 73% accuracy.", facts=[2])]})
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section, "ReviewPass": review})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    body = " ".join(s.body_md for s in report.sections)
    assert "Method B retains 73% accuracy" in body and "maps caches with" not in body


async def test_outline_failure_falls_back_to_the_single_call_draft():
    single = WriterDraft(title="single", direct_answer="Single-call answer.", answer_facts=[1],
                         stance="answered", summary=[DraftSentence(
                             text="Method A reuses the cache for identical architectures.",
                             facts=[1])])
    llm = FakeLLM({"ReportOutline": ValueError("bad json"), "WriterDraft": single})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    assert report.title == "single"
    assert [n for n, _ in llm.calls] == ["ReportOutline", "WriterDraft"]


async def test_review_failure_keeps_the_unreviewed_draft():
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section,
                   "ReviewPass": ValueError("timeout")})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    assert report.exec_summary.startswith("**Answer:** Only approximately")   # outline's answer
    assert report.sections


async def test_executive_audience_skips_the_section_calls():
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section, "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(_state("executive"), llm)
    assert "SectionText" not in [n for n, _ in llm.calls]
    assert report.sections == [] and "identical architectures" in report.exec_summary


async def test_outline_facts_go_only_to_evidence_sections_one_section_each():
    # kv-report-922f84d8: abstract and discussion were each given all 20 facts; the review then
    # deleted the experiments section as a duplicate.
    outline = _outline(sections=[
        SectionPlan(heading="Abstract", facts=[1, 2, 3]),
        SectionPlan(heading="Previous Work", facts=[1, 2]),
        SectionPlan(heading="Experiments", facts=[2, 3]),
        SectionPlan(heading="Discussion", facts=[1, 2, 3]),
    ])
    llm = FakeLLM({"ReportOutline": outline, "SectionText": _section, "ReviewPass": _REVIEW})
    await writer_mod.run_attributed_writer(_state(), llm)
    prev = _facts_part(_prompt_for(llm, "Previous Work"))
    exp = _facts_part(_prompt_for(llm, "Experiments"))
    assert "F1 " in prev and "F2 " in prev
    assert "F2 " not in exp and "F3 " in exp                      # F2 already in Previous Work


async def test_general_audience_uses_the_outline_headings():
    outline = _outline(sections=[SectionPlan(heading="What was tried", facts=[1, 2, 3])])
    llm = FakeLLM({"ReportOutline": outline, "SectionText": lambda m: SectionText(sentences=[
        DraftSentence(text="Method A reuses the cache for identical architectures.", facts=[1])]),
        "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(_state("general"), llm)
    assert [s.heading for s in report.sections] == ["What was tried"]


async def test_a_looping_review_without_an_answer_still_applies_its_fixes():
    # kv-report-6f1e7764: the reviewer repeated fixes 106 times, never wrote direct_answer or
    # stance, and the whole review failed to parse.
    looping = ReviewPass(fixes=[SentenceFix(sentence=4, problem="overclaim", fix="")] * 50)
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section, "ReviewPass": looping})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    body = " ".join(s.body_md for s in report.sections)
    assert "exact and lossless" not in body                          # the fix still applied
    assert report.exec_summary.startswith("**Answer:** Only approximately")   # outline answer


def test_review_accepts_summary_lines_given_as_plain_strings():
    # kv-report-12329a50: summary came back as ["The study investigates ...", ...] and the whole
    # review failed to parse.
    review = ReviewPass.model_validate({
        "direct_answer": "Not exactly.", "stance": "partial",
        "summary": ["The study investigates transfer.", {"text": "Cited line.", "facts": [2]}],
    })
    assert [s.text for s in review.summary] == ["The study investigates transfer.", "Cited line."]
    assert review.summary[0].facts == [] and review.summary[1].facts == [2]


def test_review_schema_puts_the_answer_before_the_fixes():
    fields = list(ReviewPass.model_fields)
    assert fields.index("direct_answer") < fields.index("fixes")


async def test_an_overloaded_evidence_section_is_rebalanced(monkeypatch):
    # kv-report-6f1e7764: the outline gave all 36 facts to Previous Work and none to Experiments.
    from research_swarm.agents import writer_sections

    monkeypatch.setattr(writer_sections, "MAX_FACTS_PER_SECTION", 1)
    outline = _outline(sections=[SectionPlan(heading="Previous Work", facts=[1, 2, 3])])
    llm = FakeLLM({"ReportOutline": outline, "SectionText": _section, "ReviewPass": _REVIEW})
    await writer_mod.run_attributed_writer(_state(), llm)
    prev = _facts_part(_prompt_for(llm, "Previous Work"))
    exp = _facts_part(_prompt_for(llm, "Experiments"))
    assert prev.strip() and exp.strip()                  # both sections now have facts
    # a sub-question's facts move together (F2 and F3 share one)
    assert ("F2 " in prev) == ("F3 " in prev) and ("F1 " in prev) != ("F2 " in prev)


async def test_a_looping_section_is_deduplicated_and_capped():
    from research_swarm.agents import writer_sections

    loop = [DraftSentence(text="Method A reuses the cache.", facts=[1])] * 150 + [
        DraftSentence(text=f"Distinct point {w}.", facts=[1]) for w in "abcdefghijklmnop"]
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": SectionText(sentences=loop),
                   "ReviewPass": _REVIEW})
    await writer_mod.run_attributed_writer(_state(), llm)
    review_prompt = llm.prompts("ReviewPass")[0]
    per_section = review_prompt.split("## Previous Work")[1].split("##")[0]
    assert per_section.count("Method A reuses the cache.") == 1
    assert per_section.count("\nS") <= writer_sections.MAX_SECTION_SENTENCES


# --- strong claims: the review's ruling, else the code rule ------------------------------

def _strong_section(messages):
    user = messages[1].content
    heading = user.split("Section to write now: ")[1].split("\n")[0]
    return SectionText(sentences={
        "Previous Work": [DraftSentence(
            text="Method A reuses the cache for identical architectures.", facts=[1])],
        "Experiments": [
            DraftSentence(text="Since the rotation is orthogonal, transfer is lossless.",
                          facts=[2]),                                               # S2
            DraftSentence(text="Method C recomputes the first 4 layers.", facts=[3]),   # S3
        ],
    }.get(heading, []))


def _ruled(**kw):
    base = dict(direct_answer="Only approximately.", answer_facts=[2], stance="partial")
    base.update(kw)
    return ReviewPass(**base)


async def test_an_unsupported_strong_claim_is_rewritten_by_the_review():
    from research_swarm.agents.writer_sections import ClaimCheck

    review = _ruled(claim_checks=[ClaimCheck(
        sentence=2, level="component", supported=False,
        fix="The rotation step is exactly invertible; the mapping as a whole is approximate.")])
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _strong_section,
                   "ReviewPass": review})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    body = " ".join(s.body_md for s in report.sections)
    assert "transfer is lossless" not in body and "exactly invertible" in body
    assert "recomputes the first 4 layers" in body
    # the reviewer was told which sentences are strong claims (S1 says "identical")
    assert "Strong claims (each needs a claim_checks entry): S1, S2" in \
        llm.prompts("ReviewPass")[0]


async def test_an_unruled_strong_claim_falls_back_to_the_code_rule():
    # the review skips S3; "lossless" is not in F2's evidence, so the sentence is dropped
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _strong_section,
                   "ReviewPass": _ruled()})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    body = " ".join(s.body_md for s in report.sections)
    assert "transfer is lossless" not in body and "recomputes the first 4 layers" in body


async def test_a_failed_review_still_applies_the_code_rule_to_strong_claims():
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _strong_section,
                   "ReviewPass": ValueError("timeout")})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    body = " ".join(s.body_md for s in report.sections)
    assert "transfer is lossless" not in body
    # "identical" IS in F1's own evidence, so that strong claim is kept
    assert "identical architectures" in body


# --- comparison table ---------------------------------------------------------------------

def _state_comparing(items):
    from research_swarm.schemas.frame import QuestionFrame

    state = _state()
    state["plan"] = state["plan"].model_copy(update={"frame": QuestionFrame(compare_items=items)})
    return state


async def test_the_comparison_table_is_rendered_and_checked_in_code():
    from research_swarm.agents.writer_render import (
        ComparisonTable,
        TableCell,
        TableRow,
    )

    table = ComparisonTable(columns=["Recomputed", "Accuracy"], rows=[
        TableRow(item="partial reuse", cells=[
            TableCell(text="first 4 layers", facts=[3]),
            TableCell(text="lossless", facts=[])]),                       # uncited strong claim
        TableRow(item="transformation", cells=[
            TableCell(text="nothing", facts=[]),                          # definitional: kept
            TableCell(text="73% retention", facts=[2])]),
        TableRow(item="made up row", cells=[TableCell(text="x")]),        # not a compare item
    ])
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section,
                   "ComparisonTable": table, "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(
        _state_comparing(["partial reuse", "transformation"]), llm)
    headings = [s.heading for s in report.sections]
    assert headings.index("Comparison") == headings.index("Discussion") - 1   # before synthesis
    md = next(s.body_md for s in report.sections if s.heading == "Comparison")
    assert md.splitlines()[0] == "| Item | Recomputed | Accuracy |"
    assert "| partial reuse | first 4 layers [" in md and "| not established |" in md
    assert "| transformation | nothing | 73% retention [" in md
    assert "made up row" not in md
    # the table call saw the sections already written
    assert "## Experiments" in llm.prompts("ComparisonTable")[0]


async def test_a_table_cell_listing_fact_numbers_is_not_rendered():
    # kv-audit-4ac8fd0d: a "Key evidence" column read "F1, F2, F3, ... [3]".
    from research_swarm.agents.writer_render import ComparisonTable, TableCell, TableRow

    table = ComparisonTable(columns=["Key evidence"], rows=[TableRow(
        item="partial reuse", cells=[TableCell(text="F1, F2, F3", facts=[1, 2, 3])])])
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section,
                   "ComparisonTable": table, "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(_state_comparing(["partial reuse"]), llm)
    md = next(s.body_md for s in report.sections if s.heading == "Comparison")
    assert "F1" not in md and "| partial reuse | not established |" in md


async def test_no_comparison_items_means_no_table_call():
    llm = FakeLLM({"ReportOutline": _outline(), "SectionText": _section, "ReviewPass": _REVIEW})
    report = await writer_mod.run_attributed_writer(_state(), llm)
    assert "ComparisonTable" not in [n for n, _ in llm.calls]
    assert "Comparison" not in [s.heading for s in report.sections]


def test_planner_residue_is_removed_from_the_methodology():
    from research_swarm.agents.writer_render import _methodology

    plan = _make_plan(1).model_copy(update={"strategy": "First, survey.\nnext_agent:dispatch"})
    assert _methodology(plan) == "First, survey."


# --- the writer's own model --------------------------------------------------------------

@pytest.fixture
def _large_model(monkeypatch):
    monkeypatch.setattr(settings, "large_model_provider", "ollama")
    monkeypatch.setattr(settings, "large_model", "nemotron-3-nano:30b-cloud")
    monkeypatch.setattr(settings, "large_model_ollama_base_url", "https://ollama.com")
    monkeypatch.setattr(settings, "large_model_stages", ["supervisor", "writer"])


@pytest.mark.parametrize("agent", ["writer", "supervisor"])
def test_large_model_stages_use_their_own_endpoint_and_pool(_large_model, agent):
    from research_swarm.graph import nodes

    with patch.object(nodes, "set_llm_context") as ctx:
        llm = nodes._get_tiered_state_llm(_make_state(), "thorough", agent=agent)
    assert llm.model == "nemotron-3-nano:30b-cloud"
    assert llm.base_url == "https://ollama.com"
    assert llm.reasoning is False                        # both are no-thinking stages
    assert ctx.call_args[0][0] == "ollama_cloud"


def test_other_stages_keep_their_tier_model(_large_model):
    from research_swarm.graph import nodes

    llm = nodes._get_tiered_state_llm(_make_state(), "fast", agent="verifier")
    assert llm.model == settings.tier_fast_model and "ollama.com" not in str(llm.base_url)


def test_a_research_stage_moves_to_the_large_model_by_config_alone(_large_model, monkeypatch):
    from research_swarm.graph import nodes

    monkeypatch.setattr(settings, "large_model_stages", ["supervisor", "writer", "paper_scout"])
    # the scout's label carries a suffix; the stage is what counts
    llm = nodes._get_tiered_state_llm(_make_state(), "fast", agent="paper_scout")
    assert llm.model == "nemotron-3-nano:30b-cloud"
    other = nodes._get_tiered_state_llm(_make_state(), "standard", agent="paper_worker[sq]")
    assert other.model != "nemotron-3-nano:30b-cloud"


def test_empty_large_model_uses_the_tier(_large_model, monkeypatch):
    from research_swarm.graph import nodes

    monkeypatch.setattr(settings, "large_model", "")
    llm = nodes._get_tiered_state_llm(_make_state(), "thorough", agent="writer")
    assert llm.model == settings.tier_thorough_model
