"""Synthesis (agents/synthesis.py): one call citing sentence IDs, audited by the code render."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from research_swarm.agents import synthesis as sy
from research_swarm.agents.packet import build_packet
from research_swarm.schemas import ResearchQuery

CLAIM_TOPIC = ("Using only the supplied scientific abstracts, classify the claim as exactly "
               "SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO, then explain the verdict: "
               "Dexamethasone decreases the risk of postoperative bleeding.")
SOURCES = [
    {"url": "benchmark://t/0", "title": "Dexamethasone trial",
     "text": "Children received dexamethasone or placebo. Bleeding occurred in 24% with "
             "dexamethasone versus 4% with placebo. Dexamethasone increased the risk of "
             "postoperative bleeding."},
    {"url": "benchmark://t/1", "title": "Unrelated", "text": "Tonsillectomy is common."},
]


def _llm(result=None, error=None):
    llm = MagicMock()
    call = AsyncMock(side_effect=error) if error else AsyncMock(return_value=result)
    llm.with_structured_output.return_value.ainvoke = call
    return llm


async def _state(topic=CLAIM_TOPIC):
    packet = await build_packet(topic, [], SOURCES, 1000)
    return {"session_id": "t", "query": ResearchQuery(topic=topic, depth="shallow"),
            "plan": None, "evidence_packet": packet.to_dict()}


def test_normalize_ids():
    assert sy.normalize_ids(["[s1.2]", "S1.3, S2.1", "S1.2", "F4"]) == ["S1.2", "S1.3", "S2.1"]


async def test_cited_verdict_leads_and_only_cited_sentences_become_findings():
    result = sy.Synthesis(
        title="Dexamethasone and bleeding", stance="answered",
        direct_answer="CONTRADICT: dexamethasone increased postoperative bleeding.",
        answer_ids=["S1.3"], verdict="CONTRADICT", verdict_ids=["S1.2", "S1.3"],
        sections=[sy.SynthesisSection(heading="Evidence", sentences=[
            sy.CitedSentence(text="Bleeding occurred in 24% with dexamethasone versus 4% with "
                                  "placebo.", ids=["S1.2"])])],
    )
    report, cited = await sy.run_synthesis(await _state(), _llm(result))
    assert "CONTRADICT" in report.exec_summary
    assert [f.quote for f in cited] == [
        "Bleeding occurred in 24% with dexamethasone versus 4% with placebo.",
        "Dexamethasone increased the risk of postoperative bleeding.",
    ]
    assert [r.url for r in report.references] == ["benchmark://t/0"]   # only cited sources


async def test_uncited_or_unknown_verdict_becomes_insufficient():
    result = sy.Synthesis(title="t", stance="answered", direct_answer="SUPPORT.",
                          verdict="SUPPORT", verdict_ids=["S9.9"])        # unknown ID only
    state = await _state()
    packet = sy.EvidencePacket.from_dict(state["evidence_packet"])
    from research_swarm.agents.question import parse_question

    verdict = sy.audit_verdict(result, parse_question(CLAIM_TOPIC), packet)
    assert verdict.label == "NOT_ENOUGH_INFO" and verdict.role == "insufficient"
    draft, stats = sy.to_draft(result, packet)
    assert stats == {"cited_ids": 0, "unknown_ids": 0}                 # verdict_ids not in draft
    result2 = sy.Synthesis(title="t", stance="answered", direct_answer="x", answer_ids=["S9.9"])
    assert sy.to_draft(result2, packet)[1] == {"cited_ids": 0, "unknown_ids": 1}
    unlisted = sy.Synthesis(title="t", stance="answered", direct_answer="x", verdict="MAYBE",
                            verdict_ids=["S1.3"])
    assert sy.audit_verdict(unlisted, parse_question(CLAIM_TOPIC), packet).role == "insufficient"


async def test_parse_failure_and_empty_packet_fall_back_without_an_llm_report():
    report, cited = await sy.run_synthesis(await _state(), _llm(error=ValueError("bad json")))
    assert report.sections and cited == []
    empty = {"session_id": "t", "query": ResearchQuery(topic="q"), "plan": None,
             "evidence_packet": None}
    llm = _llm()
    report, cited = await sy.run_synthesis(empty, llm)
    assert cited == [] and not llm.with_structured_output.return_value.ainvoke.called


async def test_fenced_reply_is_parsed_from_the_raw_text():
    """The parser returned nothing for valid JSON inside a ```json fence (a live gemma4 reply)."""
    good = sy.Synthesis(title="t", stance="answered", direct_answer="CONTRADICT: it increased.",
                        answer_ids=["S1.3"], verdict="CONTRADICT", verdict_ids=["S1.3"])
    raw = MagicMock(content="```json\n" + good.model_dump_json() + "\n```")
    report, cited = await sy.run_synthesis(
        await _state(), _llm({"raw": raw, "parsed": None, "parsing_error": ValueError("x")}))
    assert "CONTRADICT" in report.exec_summary and [f.quote for f in cited] == [
        "Dexamethasone increased the risk of postoperative bleeding."]


async def test_insufficient_answer_citing_nothing_is_the_report_not_a_fallback():
    result = sy.Synthesis(title="t", stance="insufficient", verdict="NOT_ENOUGH_INFO",
                          direct_answer="NOT_ENOUGH_INFO: no sentence tests this claim.")
    report, cited = await sy.run_synthesis(await _state(), _llm(result))
    assert "NOT_ENOUGH_INFO" in report.exec_summary and cited == []
