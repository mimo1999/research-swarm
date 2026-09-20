"""Evidence-first pipeline wiring: graph routing, document packing, topic plumbing, the attributed
writer and its fallbacks. LLMs are mocked; no provider is touched."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from research_swarm.agents import writer as writer_mod
from research_swarm.agents.writer_render import DraftSection, DraftSentence, WriterDraft
from research_swarm.config import settings
from research_swarm.schemas import Critique, CritiqueVerdict, FinalReport, Finding, Source
from research_swarm.schemas.query import ResearchQuery
from tests.unit.test_graph import _make_plan, _make_state


def _finding(i, claim, sq="Sub-question 1", conf=0.75, url=None):
    return Finding(id=f"f{i}", claim=claim, sub_question=sq, confidence=conf,
                   evidence=[Source(url=url or f"http://x/{i}", title=f"T{i}", snippet=claim)])


def _crit(fid, verdict):
    return Critique(finding_id=fid, verdict=verdict, reasoning="r")


# --- document pass routing ---------------------------------------------------------------

def _docs(n):
    return [{"url": f"http://d/{i}", "title": f"D{i}", "text": "small paragraph " * 20}
            for i in range(n)]


def _route(docs):
    from research_swarm.graph.nodes import route_from_document_pass

    state = _make_state(plan=_make_plan(2), ingested_documents=docs)
    with patch.object(settings, "enable_fetch_pass", False):
        return route_from_document_pass(state)


def test_ten_small_docs_pack_into_one_send_with_topic():
    sends = _route(_docs(10))
    assert len(sends) == 1 and sends[0].node == "document_worker_node"
    arg = sends[0].arg
    assert len(arg["active_batch"]) == 10 and arg["topic"] == "AI safety"
    assert arg["sub_questions_snapshot"] == ["Sub-question 1", "Sub-question 2"]


async def test_document_worker_node_batch_path_uses_the_shared_extractor():
    from research_swarm.graph.nodes import document_worker_node

    fake = AsyncMock(return_value=[_finding(1, "a fact")])
    state = _make_state(active_batch=[{"url": "u", "title": "t", "text": "x"}],
                        sub_questions_snapshot=["sq"], topic="THE Q")
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=MagicMock()), \
         patch("research_swarm.graph.nodes._check_budget", return_value=None), \
         patch("research_swarm.agents.extractor.extract_facts", fake):
        out = await document_worker_node(state)
    assert len(out["findings"]) == 1
    assert fake.await_args.args[0] == "THE Q" and fake.await_args.args[1] == ["sq"]


# --- graph routing after collect -----------------------------------------------------------

async def _run_graph():
    from langgraph.checkpoint.memory import MemorySaver

    import research_swarm.graph.nodes as nodes
    from research_swarm.graph.builder import _serde, build_graph, get_thread_config
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery

    order: list[str] = []
    plan = _make_plan(1)
    report = FinalReport(title="r", exec_summary="s")

    def stub(name, update):
        async def fn(state):
            order.append(name)
            return update
        return fn

    patches = [
        patch.object(nodes, "supervisor_node", stub("supervisor", {
            "next_agent": "dispatch", "plan": plan, "messages": []})),
        patch.object(nodes, "paper_scout_node", stub("scout", {"messages": []})),
        patch.object(nodes, "worker_node", stub("worker", {"messages": []})),
        patch.object(nodes, "collect_node", stub("collect", {
            "next_agent": "verifier", "research_rounds": 1, "messages": []})),
        patch.object(nodes, "verifier_node", stub("verifier", {"messages": []})),
        patch.object(nodes, "writer_node", stub("writer", {
            "final_report": report, "draft_report": report, "messages": []})),
        patch.object(settings, "enable_fetch_pass", False),
    ]
    for p in patches:
        p.start()
    try:
        session = f"v2-{uuid.uuid4().hex[:6]}"
        graph = build_graph(checkpointer=MemorySaver(serde=_serde), interrupt_before_writer=False)
        initial = _make_state(session_id=session, query=ResearchQuery(
            topic="t", depth=ResearchDepth.shallow, max_sources=3, audience="technical"))
        async for _ in graph.astream(initial, get_thread_config(session), stream_mode="updates"):
            pass
    finally:
        for p in patches:
            p.stop()
    return order


async def test_collect_routes_to_verifier_then_writer():
    order = await _run_graph()
    assert order[-2:] == ["verifier", "writer"]


async def test_verifier_node_returns_findings_critiques_and_conflicts():
    from research_swarm.graph.nodes import verifier_node

    f = _finding(1, "c")
    verdict = (_crit("f1", CritiqueVerdict.supported))
    fake = AsyncMock(return_value=([f], [verdict], [["f1", "f2"]]))
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=MagicMock()), \
         patch("research_swarm.graph.nodes._check_budget", return_value=None), \
         patch("research_swarm.agents.verifier.run_verifier", fake):
        out = await verifier_node(_make_state(findings=[f]))
    assert out["findings"] == [f] and out["critiques"] == [verdict]
    assert out["fact_conflicts"] == [["f1", "f2"]]


# --- writer ---------------------------------------------------------------------------------

def _llm_returning(draft):
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=draft)
    return llm


def _writer_state(**kw):
    findings = [_finding(1, "Metformin lowered HbA1c by 1.2% in adults."),
                _finding(2, "Refuted fact about placebo."),
                _finding(3, "Partial fact about kidneys.", conf=0.5)]
    critiques = [_crit("f1", CritiqueVerdict.supported), _crit("f2", CritiqueVerdict.refuted),
                 _crit("f3", CritiqueVerdict.weak)]
    return _make_state(plan=_make_plan(1), findings=findings, critiques=critiques, **kw)


def _draft(**kw):
    base = dict(title="T", direct_answer="Yes, it lowers HbA1c.", answer_facts=[1],
                stance="answered", summary=[], sections=[DraftSection(
                    heading="A", sentences=[DraftSentence(
                        text="Metformin lowered HbA1c by 1.2% in adults.", facts=[1])])])
    base.update(kw)
    return WriterDraft(**base)


async def test_attributed_writer_renders_citations_and_hides_refuted_facts():
    llm = _llm_returning(_draft())
    report = await writer_mod.run_attributed_writer(_writer_state(), llm)
    assert report.exec_summary.startswith("**Answer:** Yes, it lowers HbA1c [1].")
    assert [r.url for r in report.references] == ["http://x/1"]
    prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert "Refuted fact" not in prompt                       # refuted -> never shown
    assert "F1 [Q1] (supported)" in prompt and "(partial)" in prompt
    assert "AI safety" in prompt                              # the question is in the prompt


async def test_attributed_writer_prompt_carries_the_audiences_structure_guidance():
    """The audience dropdown (query.audience) must reach the writer's system prompt as the
    corresponding report-shape instructions, not just the bare word -- see writer.py's
    ``structure_guidance``/``_AUDIENCE_STRUCTURE``."""
    # A summary sentence (not just a section one) so the render still has content for the
    # "executive" audience, whose sections are dropped by design -- otherwise it would hit
    # the empty-render fallback and call run_writer through the same undiscriminating mock.
    draft_with_summary = _draft(summary=[DraftSentence(
        text="Metformin lowered HbA1c by 1.2% in adults.", facts=[1])])
    for audience, needle in [
        ("academic", "Previous Work"),
        ("technical", "Proposed Solutions"),
        ("executive", "Leave `sections` EMPTY"),
        ("general", "no dry academic headings"),
    ]:
        llm = _llm_returning(draft_with_summary)
        state = _writer_state(query=ResearchQuery(topic="AI safety", audience=audience))
        await writer_mod.run_attributed_writer(state, llm)
        system_prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][0].content
        assert needle in system_prompt, f"missing {needle!r} for audience={audience!r}"


async def test_attributed_writer_executive_audience_renders_no_sections():
    draft = _draft(summary=[DraftSentence(text="A short executive point.", facts=[1])])
    state = _writer_state(query=ResearchQuery(topic="AI safety", audience="executive"))
    report = await writer_mod.run_attributed_writer(state, _llm_returning(draft))
    assert report.sections == []


async def test_attributed_writer_falls_back_to_legacy_on_parse_failure():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=ValueError("bad json"))
    legacy = AsyncMock(return_value=FinalReport(title="legacy", exec_summary="x"))
    with patch.object(writer_mod, "run_writer", legacy):
        report = await writer_mod.run_attributed_writer(_writer_state(), llm)
    assert report.title == "legacy" and legacy.await_count == 1


async def test_attributed_writer_falls_back_when_nothing_survives_rendering():
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="Invented 99999 things happened.", facts=[1])])])
    legacy = AsyncMock(return_value=FinalReport(title="legacy", exec_summary="x"))
    with patch.object(writer_mod, "run_writer", legacy):
        report = await writer_mod.run_attributed_writer(_writer_state(), _llm_returning(draft))
    assert report.title == "legacy"


async def test_free_form_writer_fallback_builds_a_cited_report_without_refuted_facts():
    """run_writer is the attributed writer's fallback, so it must work on its own."""
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=FinalReport(title="t", exec_summary="s"))
    report = await writer_mod.run_writer(_writer_state(), llm)
    assert {r.url for r in report.references} == {"http://x/1", "http://x/3"}   # f2 is refuted
    user = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert "Research question (the report must answer THIS):\nAI safety" in user
    assert "Refuted fact" not in user


async def test_free_form_writer_survives_an_llm_failure_with_a_fallback_report():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=ValueError("bad"))
    report = await writer_mod.run_writer(_writer_state(), llm)
    assert "fallback mode" in report.limitations
    assert any("Metformin" in sec.body_md for sec in report.sections)


async def test_attributed_writer_grounds_the_free_form_fallback_instead_of_hallucinating():
    """Regression for a real incident (local gemma3:270m): the attributed draft rendered empty,
    fell back to run_writer, which invented a wrong, ungrounded number ('100°F (37°C)' for a
    fact that actually said '100°C or 212°F'). The fallback must be grounded like the primary
    path, not bypass it."""
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="Invented 99999 things happened.", facts=[1])])])
    hallucinated = FinalReport(
        title="Boiling Point of Water",
        exec_summary="The boiling point of water at sea level is approximately 100°F (37°C).",
    )
    legacy = AsyncMock(return_value=hallucinated)
    finding = _finding(1, "The boiling point of water at sea level is 100°C or 212°F.")
    state = _make_state(
        plan=_make_plan(1), findings=[finding],
        critiques=[_crit("f1", CritiqueVerdict.supported)])
    with patch.object(writer_mod, "run_writer", legacy):
        report = await writer_mod.run_attributed_writer(state, _llm_returning(draft))
    # The invented "100°F (37°C)" is gone; since nothing grounded survived the free-form
    # fallback, the report falls through to the fact's own verbatim (correct) claim text.
    assert "37" not in report.exec_summary and "100°F" not in report.exec_summary
    assert report.exec_summary == "**Answer:** " + finding.claim
    assert legacy.await_count == 1
