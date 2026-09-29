"""Question parsing (content vs answer format), the claim verdict, and their use by the planner
and the attributed writer."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from research_swarm.agents import verdict as vd
from research_swarm.agents import writer as writer_mod
from research_swarm.agents.question import (
    is_meta_sub_question,
    parse_question,
)
from research_swarm.agents.writer_render import (
    DraftSection,
    DraftSentence,
    WriterDraft,
    render_report,
)
from research_swarm.schemas import (
    Critique,
    CritiqueVerdict,
    Finding,
    ResearchPlan,
    ResearchQuery,
    Source,
)
from research_swarm.schemas.worker import SubQuestionAssignment
from tests.unit.test_graph import _make_plan, _make_state

SCIFACT = ("Using only the supplied scientific abstracts, classify the claim as exactly SUPPORT, "
           "CONTRADICT, or NOT_ENOUGH_INFO, then explain the verdict: Cold exposure reduces BAT "
           "recruitment.")


# --- parse_question ------------------------------------------------------------------------

def test_scifact_prompt_splits_into_claim_instruction_and_labels():
    spec = parse_question(SCIFACT)
    assert spec.content == "Cold exposure reduces BAT recruitment."
    assert spec.instruction.startswith("Using only the supplied scientific abstracts")
    assert spec.labels == ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
    assert spec.is_claim_check
    assert spec.label_for("insufficient") == "NOT_ENOUGH_INFO"


def test_plain_topics_are_left_whole():
    for topic in (
        "Metformin: does it reduce all-cause mortality in type 2 diabetes?",  # colon, no instruction
        "Effects of intermittent fasting on LDL cholesterol",
        "",
    ):
        spec = parse_question(topic)
        assert spec.content == topic.strip() and spec.instruction == "" and spec.labels == ()


# --- verdict aggregation ------------------------------------------------------------------------

SPEC = parse_question(SCIFACT)


def test_aggregate():
    cases = [
        ({}, "NOT_ENOUGH_INFO", []),
        ({1: "unrelated", 2: "unrelated"}, "NOT_ENOUGH_INFO", []),
        ({1: "supports", 2: "unrelated"}, "SUPPORT", [1]),
        ({1: "contradicts", 2: "contradicts", 3: "supports"}, "CONTRADICT", [1, 2]),
        ({1: "supports", 2: "contradicts"}, "NOT_ENOUGH_INFO", []),   # mixed tie -> insufficient
    ]
    for relations, label, deciding in cases:
        v = vd.aggregate(relations, SPEC)
        assert v.label == label and v.deciding_facts == deciding, relations


def _fact(i, claim, url=None):
    return Finding(id=f"f{i}", claim=claim, sub_question="sq", confidence=0.9,
                   evidence=[Source(url=url or f"http://x/{i}", title=f"T{i}", snippet=claim)])


def _rel_llm(*pairs, fail=False):
    llm = MagicMock()
    if fail:
        llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=ValueError("x"))
    else:
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=vd.ClaimRelations(
            relations=[vd.FactRelation(fact=n, relation=r) for n, r in pairs]))
    return llm


async def test_decide_verdict_uses_relations_and_ignores_out_of_range():
    facts = [_fact(1, "Cold exposure increased BAT recruitment in mice.")]
    v = await vd.decide_verdict(SPEC, facts, _rel_llm((1, "contradicts"), (7, "supports")))
    assert v.label == "CONTRADICT" and v.deciding_facts == [1]


# --- render with a verdict ----------------------------------------------------------------------

FACTS = [_fact(1, "Cold exposure increased BAT recruitment in adult mice."),
         _fact(2, "BAT activity rose after two weeks at 4 C.")]


def _draft(answer="SUPPORT. Cold exposure reduces BAT recruitment.", sections=None, summary=None):
    return WriterDraft(
        title="t", direct_answer=answer, answer_facts=[1], stance="answered",
        summary=summary or [], sections=sections or [DraftSection(heading="Evidence", sentences=[
            DraftSentence(text="Cold exposure increased BAT recruitment in adult mice.", facts=[1]),
            DraftSentence(text="The abstracts classify the claim as SUPPORT overall.", facts=[1]),
        ])],
    )


def test_conflicting_draft_answer_is_replaced_and_conflicting_sentences_dropped():
    verdict = vd.Verdict(label="CONTRADICT", role="contradict", deciding_facts=[1])
    report, stats = render_report(_draft(), FACTS, "claim", verdict=verdict, labels=SPEC.labels)
    assert report.exec_summary.startswith(
        "**Answer:** CONTRADICT: the verified facts directly contradict the claim [1]")
    body = report.sections[0].body_md
    assert "SUPPORT" not in body and "increased BAT recruitment" in body
    assert stats["answer_replaced"] is True and stats["dropped_conflicting_verdict"] == 1


def test_summary_sentences_repeated_in_sections_are_dropped():
    s = "Cold exposure increased BAT recruitment in adult mice."
    draft = _draft(summary=[DraftSentence(text=s, facts=[1]),
                            DraftSentence(text="BAT activity rose after two weeks at 4 C.",
                                          facts=[2])])
    report, stats = render_report(draft, FACTS, "claim")
    assert report.exec_summary.count("increased BAT recruitment") == 0
    assert "BAT activity rose" in report.exec_summary and stats["dropped_duplicate"] == 1


# --- supervisor: format instructions stay out of the plan ------------------------------------

async def test_supervisor_plans_the_content_and_drops_meta_sub_questions():
    from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor

    sqs = ["How do the abstracts classify the claim as SUPPORT or CONTRADICT?",
           "Does cold exposure change BAT recruitment in mammals?"]
    plan = ResearchPlan(
        sub_questions=sqs, strategy="s", required_tools=[],
        assignments=[SubQuestionAssignment(sub_question=q)
                     for q in sqs])
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=SupervisorDecision(reasoning="r", next_agent="dispatch", plan=plan))
    state = _make_state(query=ResearchQuery(topic=SCIFACT))
    decision = await run_supervisor(state, llm)
    assert decision.plan.sub_questions == [sqs[1]]
    assert [a.sub_question for a in decision.plan.assignments] == [sqs[1]]
    prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert "Research topic: Cold exposure reduces BAT recruitment." in prompt
    assert "do NOT plan sub-questions about it" in prompt


# --- attributed writer end to end (mocked) -----------------------------------------------------

async def test_attributed_writer_runs_the_verdict_for_claim_checks():
    state = _make_state(
        query=ResearchQuery(topic=SCIFACT), plan=_make_plan(1),
        findings=[_fact(1, "Cold exposure increased BAT recruitment in adult mice.")],
        critiques=[Critique(finding_id="f1", verdict=CritiqueVerdict.supported, reasoning="r")],
    )
    draft = _draft()
    rel = vd.ClaimRelations(relations=[vd.FactRelation(fact=1, relation="contradicts")])

    async def fake_retry(structured, messages, **kw):
        return rel if kw.get("agent") == "verdict" else draft

    orig_w, orig_v = writer_mod.ainvoke_with_retry, vd.ainvoke_with_retry
    writer_mod.ainvoke_with_retry = vd.ainvoke_with_retry = fake_retry
    try:
        report = await writer_mod.run_attributed_writer(state, MagicMock())
    finally:
        writer_mod.ainvoke_with_retry, vd.ainvoke_with_retry = orig_w, orig_v
    assert report.exec_summary.startswith("**Answer:** CONTRADICT")


def test_title_naming_a_different_verdict_is_replaced():
    draft = _draft()
    draft.title = "Evidence SUPPORT for cold exposure"
    verdict = vd.Verdict(label="CONTRADICT", role="contradict", deciding_facts=[1])
    report, _ = render_report(draft, FACTS, "Cold exposure reduces BAT recruitment.",
                              verdict=verdict, labels=SPEC.labels)
    assert report.title == "Claim check: Cold exposure reduces BAT recruitment."


def test_ordinary_topics_are_not_parsed_as_instructions_or_labels():
    """'label' in a title, ALL-CAPS 'CPU or GPU', 'classification' as subject matter."""
    for topic in ("Machine learning label noise: a survey of robust training",
                  "Compare CPU or GPU inference cost for transformers",
                  "Deep learning for skin lesion classification"):
        spec = parse_question(topic)
        assert spec.content == topic and spec.instruction == "" and spec.labels == (), topic
        for sq in ("How much does GPU inference cost per token?",
                   "What datasets are used for lesion classification?"):
            assert not is_meta_sub_question(sq, spec)


async def test_supervisor_keeps_subject_sub_questions_of_an_ordinary_topic():
    from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor

    sqs = ["How much does GPU inference cost per token?", "How does CPU inference scale?"]
    plan = ResearchPlan(
        sub_questions=sqs, strategy="s", required_tools=[],
        assignments=[SubQuestionAssignment(sub_question=q)
                     for q in sqs])
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=SupervisorDecision(reasoning="r", next_agent="dispatch", plan=plan))
    topic = "Compare CPU or GPU inference cost for transformers"
    decision = await run_supervisor(_make_state(query=ResearchQuery(topic=topic)), llm)
    assert decision.plan.sub_questions == sqs
    prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert f"Research topic: {topic}" in prompt


def test_no_evidence_yes_no_verdict_states_insufficiency_and_asserts_no_label():
    spec = parse_question("Answer YES or NO: does coffee raise blood pressure?")
    verdict = vd.aggregate({}, spec)
    assert verdict.role == "insufficient" and verdict.label == ""
    draft = _draft(answer="YES. Coffee raises blood pressure.", sections=[DraftSection(
        heading="Evidence", sentences=[
            DraftSentence(text="Some trials answer YES to this question.", facts=[1]),
            DraftSentence(text="Cold exposure increased BAT recruitment in adult mice.", facts=[1]),
        ])])
    report, stats = render_report(draft, FACTS, "does coffee raise blood pressure?",
                                  verdict=verdict, labels=spec.labels)
    assert report.exec_summary.startswith("**Answer:** The verified facts do not directly test")
    assert "YES" not in report.exec_summary and "YES" not in report.sections[0].body_md
    assert stats["answer_replaced"] is True and stats["dropped_conflicting_verdict"] == 1
