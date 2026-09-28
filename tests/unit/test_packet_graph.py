"""Packet-path graph (pipeline_mode="packet"): supplied sources, no planning call, one synthesis
call in the writer's slot, review pause before it."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents.synthesis import CitedSentence, Synthesis, SynthesisSection
from research_swarm.graph.builder import build_graph, get_thread_config
from research_swarm.schemas import ResearchQuery

DOCS = [{"url": "doc://1", "title": "Trial", "text": "Aspirin lowered fever by 1.5 degrees "
         "in 200 adults. Side effects were rare.", "source_type": "pdf"}]
RESULT = Synthesis(
    title="Aspirin and fever", stance="answered",
    direct_answer="Aspirin lowered fever by 1.5 degrees in adults.", answer_ids=["S1.1"],
    sections=[SynthesisSection(heading="Evidence", sentences=[
        CitedSentence(text="In 200 adults, aspirin lowered fever by 1.5 degrees.",
                      ids=["S1.1"])])],
)


def _state(session_id):
    return {"messages": [], "query": ResearchQuery(topic="Does aspirin lower fever?"),
            "plan": None, "findings": [], "critiques": [], "draft_report": None,
            "final_report": None, "human_feedback": None, "writer_instructions": None,
            "iteration_count": 0, "next_agent": None, "session_id": session_id,
            "ingested_documents": DOCS}


def _fake_llm():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=RESULT)
    return llm


async def test_packet_graph_runs_supplied_sources_with_one_synthesis_call():
    llm = _fake_llm()
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=llm), \
         patch("research_swarm.graph.nodes.run_supervisor", AsyncMock()) as planner:
        graph = build_graph(interrupt_before_writer=False, pipeline_mode="packet")
        config = get_thread_config("pk1")
        async for _ in graph.astream(_state("pk1"), config, stream_mode="updates"):
            pass
        values = (await graph.aget_state(config)).values
    assert not planner.called                                   # sources supplied: no planning
    assert values["plan"].sub_questions == ["Does aspirin lower fever?"]
    assert values["evidence_packet"]["stats"]["fit"] == "whole"
    report = values["final_report"]
    assert "1.5 degrees" in report.exec_summary and report.references[0].url == "doc://1"
    assert [f.quote for f in values["findings"]] == [
        "Aspirin lowered fever by 1.5 degrees in 200 adults."]
    # exactly one structured call, and it was the synthesis
    llm.with_structured_output.assert_called_once_with(Synthesis, include_raw=True)


async def test_packet_graph_pauses_before_synthesis():
    llm = _fake_llm()
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=llm):
        graph = build_graph(interrupt_before_writer=True, pipeline_mode="packet")
        config = get_thread_config("pk2")
        async for _ in graph.astream(_state("pk2"), config, stream_mode="updates"):
            pass
        snapshot = await graph.aget_state(config)
    assert snapshot.next == ("writer",)
    assert snapshot.values["evidence_packet"]["sentences"]           # the packet to review
    assert not llm.with_structured_output.called


def test_unknown_pipeline_mode_is_an_error():
    with pytest.raises(ValueError):
        build_graph(pipeline_mode="agents")
