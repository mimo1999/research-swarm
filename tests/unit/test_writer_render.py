"""agents/writer_render.py: citation assembly, sentence dropping, direct answer, claims round trip."""
from __future__ import annotations

from research_swarm.agents.writer_render import (
    DraftSection,
    DraftSentence,
    WriterDraft,
    deterministic_report,
    ground_free_form_report,
    render_report,
)
from research_swarm.eval.claims import split_claims
from research_swarm.schemas import FinalReport, Finding, ReportSection, Source


def _fact(i, claim, url=None, snippet=""):
    return Finding(
        id=f"f{i}", claim=claim, sub_question="sq",
        evidence=[Source(url=url or f"http://x/{i}", title=f"T{i}", snippet=snippet or claim)],
    )


FACTS = [
    _fact(1, "Metformin lowered HbA1c by 1.2% in 1,200 adults."),
    _fact(2, "Berlin is the capital of Germany."),
    _fact(3, "The trial ran for 24 months.", url="http://x/1"),       # same url as fact 1
]


def _sent(text, *facts):
    return DraftSentence(text=text, facts=list(facts))


def _draft(**kw):
    base = dict(title="T", direct_answer="Yes, it lowers HbA1c.", answer_facts=[1],
                stance="answered", summary=[], sections=[], limitations="")
    base.update(kw)
    return WriterDraft(**base)


def _render(draft, facts=FACTS):
    return render_report(draft, facts, "Does metformin work?")


def test_references_are_numbered_by_first_use_and_shared_by_url():
    draft = _draft(answer_facts=[2], sections=[DraftSection(
        heading="A", sentences=[_sent("Metformin lowered HbA1c by 1.2% overall.", 1, 2),
                                _sent("The trial ran for 24 months in total.", 3)])])
    report, stats = _render(draft)
    assert [r.url for r in report.references] == ["http://x/2", "http://x/1"]   # F2 first, then F1
    body = report.sections[0].body_md
    assert "1.2% overall [1, 2]." in body                # marker before the final period
    assert "in total [2]." in body                       # fact 3 shares fact 1's url -> ref 2
    assert report.sections[0].citations == [1, 2]
    assert stats["n_references"] == 2


def test_sentence_with_ungrounded_number_is_dropped_but_grounded_one_kept():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("Metformin lowered HbA1c by 45% in adults.", 1),
        _sent("Metformin lowered HbA1c by 1.2% in adults.", 1)])])
    report, stats = _render(draft)
    assert "45%" not in report.sections[0].body_md and "1.2%" in report.sections[0].body_md
    assert stats["dropped_ungrounded_number"] == 1


def test_uncited_sentences_transition_kept_numeric_or_long_dropped():
    long = " ".join(["word"] * 30) + "."
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("Overall, the picture is mixed."), _sent("About 40 trials exist."),
        _sent(long), _sent("Metformin lowered HbA1c by 1.2% here.", 1)])])
    report, stats = _render(draft)
    body = report.sections[0].body_md
    assert "picture is mixed" in body and "40 trials" not in body and "word word" not in body
    assert stats["kept_transitions"] == 1 and stats["dropped_uncited"] == 2


def test_exec_summary_starts_with_the_direct_answer_and_cites_it():
    report, _ = _render(_draft(summary=[_sent("Berlin is the capital of Germany.", 2)],
                               sections=[]))
    assert report.exec_summary.startswith("**Answer:** Yes, it lowers HbA1c [1].")
    assert "Berlin is the capital of Germany [2]." in report.exec_summary


def test_empty_render_is_flagged_for_fallback():
    draft = _draft(summary=[_sent("Invented 999 numbers here.", 1)],
                   sections=[DraftSection(heading="A", sentences=[_sent("Nothing 12345 cited.")])])
    _, stats = _render(draft)
    assert stats["empty"] is True


def test_rendered_report_round_trips_through_split_claims_with_citations():
    draft = _draft(summary=[_sent("Berlin is the capital of Germany.", 2)], sections=[
        DraftSection(heading="Results", sentences=[
            _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1),
            _sent("The trial ran for 24 months in total.", 3)])])
    report, _ = _render(draft)
    claims = split_claims(report.model_dump())
    by_text = {c.text: c.citations for c in claims}
    assert by_text["Metformin lowered HbA1c by 1.2% in 1,200 adults."] == [1]
    assert by_text["The trial ran for 24 months in total."] == [1]     # same url as fact 1
    assert all(c.citations for c in claims if "Answer" not in c.text and "Berlin" not in c.text)


# --- safety net for the free-form fallback writer -------------------------------------------

def test_ground_free_form_report_strips_an_invented_number_but_keeps_grounded_text():
    report = FinalReport(
        title="t",
        exec_summary="Metformin lowered HbA1c by 45% overall.",
        sections=[ReportSection(
            heading="Results",
            body_md="Metformin lowered HbA1c by 1.2% in 1,200 adults. It was well tolerated.",
        )],
    )
    grounded, stats = ground_free_form_report(report, FACTS, "does metformin work?")
    assert "45%" not in grounded.exec_summary
    assert "1.2%" in grounded.sections[0].body_md and "well tolerated" in grounded.sections[0].body_md
    assert stats["dropped_ungrounded_number"] == 1 and stats["empty"] is False


def test_ground_free_form_report_drops_a_section_left_empty_and_flags_empty_overall():
    report = FinalReport(
        title="t", exec_summary="Invented 99999 units of something.",
        sections=[ReportSection(heading="A", body_md="Another invented 88888 figure.")],
    )
    grounded, stats = ground_free_form_report(report, FACTS, "topic")
    assert grounded.sections == [] and stats["empty"] is True


def test_deterministic_report_has_no_llm_call_and_is_automatically_grounded():
    report = deterministic_report(FACTS, "does metformin work?")
    assert report.exec_summary.startswith("**Answer:** Metformin lowered HbA1c")
    assert len(report.sections) == 1                        # all facts share sub_question "sq"
    assert {r.url for r in report.references} == {"http://x/1", "http://x/2"}  # f3 shares f1's url
    body = report.sections[0].body_md
    for f in FACTS:
        assert f.claim in body


def test_deterministic_report_with_no_evidence_url_gets_no_citation_marker():
    fact = Finding(id="n1", claim="A claim with no url.", sub_question="sq", evidence=[])
    report = deterministic_report([fact], "topic")
    assert "[None]" not in report.sections[0].body_md and "A claim with no url." in report.sections[0].body_md


# --- checks added after the first sectioned-writer report (kv-report-001e1841) ---------------

def test_a_sentence_repeated_in_a_later_section_is_dropped():
    # Two section calls with overlapping facts wrote "Experiments" and "Discussion" identically.
    same = [_sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1),
            _sent("The trial ran for 24 months in total.", 3)]
    draft = _draft(sections=[DraftSection(heading="Experiments", sentences=same),
                             DraftSection(heading="Discussion", sentences=list(same))])
    report, stats = _render(draft)
    assert [s.heading for s in report.sections] == ["Experiments"]   # empty Discussion dropped
    assert stats["dropped_duplicate"] == 2


def test_a_near_duplicate_summary_sentence_is_dropped():
    draft = _draft(summary=[_sent("Metformin lowered HbA1c by 1.2% among 1,200 adults.", 1)],
                   sections=[DraftSection(heading="A", sentences=[
                       _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)])])
    report, stats = _render(draft)
    assert "among" not in report.exec_summary and stats["dropped_duplicate"] == 1


def test_a_sentence_with_source_markup_is_dropped():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("Across six pairs, four retain T01--T02 of standalone accuracy.", 1),
        _sent("Metformin lowered HbA1c by 1.2% in adults.", 1)])])
    report, stats = _render(draft)
    assert "T01" not in report.sections[0].body_md and stats["dropped_markup"] == 1


def test_bulk_citations_are_trimmed_to_the_facts_that_match():
    facts = [_fact(i, f"Unrelated eviction paper number {i}.") for i in range(1, 6)]
    facts.append(_fact(6, "Cross-model transfer retains most accuracy within a family."))
    draft = _draft(answer_facts=[6], sections=[DraftSection(heading="A", sentences=[
        _sent("Cross-model transfer retains most accuracy within a family.", 1, 2, 3, 4, 5, 6)])])
    report, stats = render_report(draft, facts, "q")
    assert report.sections[0].citations == [1]          # only F6, which the answer cited first
    assert stats["trimmed_citations"] == 1


def test_arxiv_mirrors_share_one_reference_listed_under_arxiv_org():
    facts = [_fact(1, "Transfer retains most accuracy.", url="https://www.alphaxiv.org/abs/2608.03893"),
             _fact(2, "Transfer is faster than prefill.", url="https://arxiv.org/html/2608.03893v1"),
             _fact(3, "Another paper.", url="https://arxiv.org/abs/2411.02820")]
    draft = _draft(answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
        _sent("Transfer is faster than prefill.", 2), _sent("Another paper.", 3)])])
    report, stats = render_report(draft, facts, "q")
    assert [r.url for r in report.references] == [
        "https://arxiv.org/html/2608.03893v1", "https://arxiv.org/abs/2411.02820"]
    assert "faster than prefill [1]" in report.sections[0].body_md
    assert stats["merged_mirror_refs"] == 1


def test_invented_author_attribution_is_stripped_but_a_real_one_kept():
    # kv-report-3c177505 credited four different papers to "(Qin et al.)".
    facts = [_fact(1, "KVLink concatenates the KV caches of retrieved documents."),
             _fact(2, "Qin et al. report a closed-form ridge mapper.")]
    draft = _draft(answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
        _sent("KVLink concatenates the KV caches of retrieved documents (Qin et al.)", 1),
        _sent("A closed-form ridge mapper is reported (Qin et al.).", 2)])])
    report, stats = render_report(draft, facts, "q")
    body = report.sections[0].body_md
    assert "retrieved documents [1]." in body                   # stripped, and given a full stop
    assert "(Qin et al.) [2]." in body                          # the name is in F2's evidence
    assert stats["stripped_attribution"] == 1


def test_sentence_without_final_punctuation_gets_a_full_stop():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("Metformin lowered HbA1c by 1.2% in adults", 1),
        _sent("The trial ran for 24 months", 3)])])
    body = _render(draft)[0].sections[0].body_md
    assert "in adults [1]. The trial" in body


def test_writer_meta_language_is_dropped():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("Dataset sizes are not detailed in the provided facts."),
        _sent("Metformin lowered HbA1c by 1.2% in adults.", 1)])])
    report, stats = _render(draft)
    assert "provided facts" not in report.sections[0].body_md and stats["dropped_meta"] == 1


def test_secondary_sources_are_labelled_and_ordered_after_primary_ones():
    from research_swarm.agents import writer as writer_mod
    from research_swarm.agents.papers import is_secondary_source
    from research_swarm.schemas import Critique, CritiqueVerdict
    from tests.unit.test_graph import _make_plan, _make_state

    assert is_secondary_source("https://medium.com/@x/cross-model-kv")
    assert is_secondary_source("https://www.emergentmind.com/topics/kv-cache-reuse-strategy")
    assert not is_secondary_source("https://arxiv.org/abs/2411.02820")
    plan = _make_plan(1)
    blog = Finding(id="b", claim="Blog claim.", sub_question=plan.sub_questions[0], confidence=0.9,
                   evidence=[Source(url="https://medium.com/@x/post", title="Blog")])
    paper = Finding(id="p", claim="Paper claim.", sub_question=plan.sub_questions[0],
                    confidence=0.8, evidence=[Source(url="https://arxiv.org/abs/1", title="P")])
    state = _make_state(plan=plan, findings=[blog, paper], critiques=[
        Critique(finding_id=i, verdict=CritiqueVerdict.supported, reasoning="r") for i in "bp"])
    facts, _ = writer_mod._select_facts(state)
    assert [f.id for f in facts] == ["p", "b"]          # primary first despite lower confidence


def test_a_duplicate_is_removed_from_the_abstract_not_the_detailed_section():
    # kv-report-d73f5d2f: the Abstract restated results and the Experiments section was emptied.
    line = _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)
    draft = _draft(sections=[DraftSection(heading="Abstract", sentences=[line]),
                             DraftSection(heading="Experiments", sentences=[line])])
    report, stats = _render(draft)
    assert [s.heading for s in report.sections] == ["Experiments"]
    assert stats["dropped_duplicate"] == 1


def test_latex_subscripts_are_markup():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        _sent("The pair has matched KV when nkvs=nkvtn_{kv}^s=n_{kv}^t holds.", 1),
        _sent("Metformin lowered HbA1c by 1.2% in adults.", 1)])])
    report, stats = _render(draft)
    assert "_{kv}" not in report.sections[0].body_md and stats["dropped_markup"] == 1


def test_scope_claim_is_backed_by_a_fact_using_another_phrasing_of_the_scope():
    from research_swarm.schemas.frame import QuestionFrame

    frame = QuestionFrame(key_constraint="across different LLMs",
                          constraint_terms=["cross-model"], topic="Can a KV cache move?")
    facts = [_fact(1, "Cross-model KV transfer retains most accuracy.")]
    draft = _draft(answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
        _sent("Reuse across different LLMs retains most accuracy.", 1)])])
    report, stats = render_report(draft, facts, "q", frame=frame)
    assert report.sections and stats["dropped_scope_overclaim"] == 0


def test_the_direct_answer_is_cut_to_two_sentences():
    draft = _draft(direct_answer="One. Two is here. Three adds more. Four ends it.",
                   sections=[DraftSection(heading="A", sentences=[
                       _sent("Metformin lowered HbA1c by 1.2% in adults.", 1)])])
    report, stats = _render(draft)
    assert "Three" not in report.exec_summary and "Two is here" in report.exec_summary
    assert stats["answer_truncated"] is True


def test_a_blog_citation_is_dropped_when_a_paper_backs_the_same_sentence():
    facts = [_fact(1, "Transfer retains most accuracy.", url="https://arxiv.org/abs/2608.03893"),
             _fact(2, "Transfer retains most accuracy.", url="https://www.youtube.com/watch?v=x")]
    draft = _draft(answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
        _sent("Transfer retains most accuracy.", 1, 2)])])
    report, stats = render_report(draft, facts, "q")
    assert [r.url for r in report.references] == ["https://arxiv.org/abs/2608.03893"]
    assert stats["dropped_secondary_citations"] == 1


# --- audience-specific report shapes ---------------------------------------------------------

_ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth"]


def test_executive_audience_drops_sections_and_caps_the_summary():
    draft = _draft(
        summary=[_sent(f"{word} supporting point holds true overall.") for word in _ORDINALS],
        sections=[DraftSection(heading="A", sentences=[
            _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)])],
    )
    report, stats = render_report(draft, FACTS, "Does metformin work?", audience="executive")
    assert report.sections == []                      # drafted sections are dropped entirely
    # answer + at most EXECUTIVE_MAX_SUMMARY_SENTENCES kept
    from research_swarm.agents.writer_render import EXECUTIVE_MAX_SUMMARY_SENTENCES
    kept = sum(1 for word in _ORDINALS if f"{word} supporting point" in report.exec_summary)
    assert kept == EXECUTIVE_MAX_SUMMARY_SENTENCES
    assert stats["executive_summary_truncated"] is True


def test_executive_audience_with_short_summary_is_not_flagged_truncated():
    draft = _draft(summary=[_sent("One short point.")], sections=[])
    report, stats = render_report(draft, FACTS, "Does metformin work?", audience="executive")
    assert report.sections == [] and stats["executive_summary_truncated"] is False


def test_technical_audience_canonicalizes_headings_by_keyword():
    draft = _draft(sections=[
        DraftSection(heading="The Problem", sentences=[
            _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)]),
        DraftSection(heading="What the papers found", sentences=[
            _sent("The trial ran for 24 months in total.", 3)]),
        DraftSection(heading="Some unrelated heading", sentences=[
            _sent("Berlin is the capital of Germany.", 2)]),
    ])
    report, _ = render_report(draft, FACTS, "topic", audience="technical")
    headings = [s.heading for s in report.sections]
    assert "Problem Statement" in headings
    assert "Some unrelated heading" in headings         # no keyword match -> left as-is


def test_academic_audience_canonicalizes_headings_by_keyword():
    draft = _draft(sections=[
        DraftSection(heading="Abstract of the study", sentences=[
            _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)]),
        DraftSection(heading="Discussion and caveats", sentences=[
            _sent("The trial ran for 24 months in total.", 3)]),
    ])
    report, _ = render_report(draft, FACTS, "topic", audience="academic")
    headings = [s.heading for s in report.sections]
    assert headings == ["Abstract", "Discussion"]


def test_general_audience_leaves_headings_unchanged():
    draft = _draft(sections=[DraftSection(heading="A quirky magazine-style heading", sentences=[
        _sent("Metformin lowered HbA1c by 1.2% in 1,200 adults.", 1)])])
    report, _ = render_report(draft, FACTS, "topic", audience="general")
    assert report.sections[0].heading == "A quirky magazine-style heading"


def test_deterministic_report_for_executive_has_no_sections():
    report = deterministic_report(FACTS, "does metformin work?", audience="executive")
    assert report.sections == []
    assert report.exec_summary.startswith("**Answer:**")
    for f in FACTS[:3]:
        assert f.claim in report.exec_summary
