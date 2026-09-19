"""agents/gap_fill.py and its graph wiring (coverage gate, worker mode, search query)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from research_swarm.agents import gap_fill as gf
from research_swarm.config import settings
from research_swarm.schemas import Finding, Source
from tests.unit.test_graph import _make_plan, _make_state

LONG_PAGE = (
    "Cookie banner and navigation words. " * 30
    + "Statins reduce LDL cholesterol by about 30% in adults with high risk. "
    + "Boilerplate footer text again. " * 30
)


def _item(n, source_type="web", snippet=None):
    return {"url": f"http://p/{n}", "title": f"Title {n}", "source_type": source_type,
            "snippet": snippet or ("statins cholesterol adults " * 10),
            "credibility_score": 0.7}


# --- relevant_text ------------------------------------------------------------------------

def test_relevant_text_picks_the_matching_passage_and_is_capped():
    out = gf.relevant_text("Do statins reduce LDL cholesterol in adults?", LONG_PAGE)
    assert "Statins reduce LDL cholesterol by about 30%" in out
    assert len(out) <= gf.TEXT_CAP


# --- web_sources ----------------------------------------------------------------------------

async def test_web_sources_fetches_web_pages_but_not_scholarly_abstracts():
    results = {"web": [_item(1, "web")], "pubmed": [_item(2, "pubmed", "abstract text " * 20)]}
    fetched = []

    async def fake_fetch(url):
        fetched.append(url)
        return LONG_PAGE

    with patch.object(gf, "tool_registry", return_value={"web": 1, "pubmed": 1}), \
         patch.object(gf, "search_task", AsyncMock(return_value=results)), \
         patch.object(gf, "_fetch_page", fake_fetch):
        out = await gf.web_sources("Do statins reduce LDL cholesterol?", "statins ldl", "s")
    assert fetched == ["http://p/1"]                         # pubmed item skipped fetching
    by_url = {s["url"]: s for s in out}
    assert "Statins reduce LDL" in by_url["http://p/1"]["text"]
    assert by_url["http://p/2"]["text"].startswith("abstract text")
    assert all(len(s["text"]) <= gf.TEXT_CAP for s in out)


async def test_fetch_failure_falls_back_to_the_search_snippet():
    with patch.object(gf, "tool_registry", return_value={"web": 1}), \
         patch.object(gf, "search_task", AsyncMock(return_value={"web": [_item(1, "web")]})), \
         patch.object(gf, "_fetch_page", AsyncMock(return_value=None)):
        out = await gf.web_sources("statins cholesterol adults", "q", "s")
    assert out[0]["text"].startswith("statins cholesterol adults")


# --- run_gap_fill ------------------------------------------------------------------------------

async def test_run_gap_fill_uses_the_source_hook_and_one_extraction_call():
    sources = [{"url": "u", "title": "t", "text": "x", "source_type": "web"}]
    src_fn = AsyncMock(return_value=sources)
    extract = AsyncMock(return_value=["finding"])
    with patch.object(gf, "extract_facts", extract):
        out = await gf.run_gap_fill("TOPIC", "sq", "kw", MagicMock(), "s", source_fn=src_fn)
    assert out == ["finding"]
    src_fn.assert_awaited_once_with("sq", "kw", "s")
    assert extract.await_args.args[:3] == ("TOPIC", ["sq"], sources)
    assert extract.await_args.kwargs["agent"] == "gap_fill"


# --- graph wiring -------------------------------------------------------------------------------

def _finding(i, sq, grounding="quote"):
    return Finding(id=f"f{i}", claim=f"c{i}", sub_question=sq, grounding=grounding,
                   evidence=[Source(url=f"http://x/{i}", snippet="s")])


def test_round0_targets_need_min_grounded_facts_per_sub_question():
    from research_swarm.graph.nodes import _research_targets

    plan = _make_plan(2)
    sq0, sq1 = plan.sub_questions
    findings = [_finding(1, sq0), _finding(2, sq1), _finding(3, sq1)]
    with patch.object(settings, "min_grounded_facts", 2):
        assert _research_targets(_make_state(plan=plan, findings=findings)) == [sq0]


async def test_worker_node_gap_fills_through_the_source_hook():
    from research_swarm.graph.nodes import worker_node

    state = _make_state(active_sub_question="sq", search_query="kw")
    gap = AsyncMock(return_value=[_finding(1, "sq")])
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=MagicMock()),          patch("research_swarm.graph.nodes._check_budget", return_value=None),          patch("research_swarm.agents.gap_fill.run_gap_fill", gap):
        out = await worker_node(state)
    assert len(out["findings"]) == 1 and gap.await_count == 1
    assert gap.await_args.kwargs["source_fn"] is not None
