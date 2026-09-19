"""Unit tests for the paper scout / paper worker (relevance-filtered abstract corpus)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents.papers import (
    PaperScore,
    PaperScores,
    _usable,
    extract_findings,
    interleave,
    score_pool,
    search_task,
)

ABSTRACT = "A randomized trial of 100 patients found a significant effect. " * 3


def _paper(n: int, **kw) -> dict:
    return {"url": f"https://p/{n}", "title": f"Paper {n}", "snippet": ABSTRACT,
            "source_type": "pubmed", "credibility_score": 0.9, **kw}


def _llm_returning(result) -> MagicMock:
    """A mock chat model whose (reasoning-disabled) structured output returns *result*."""
    llm = MagicMock()
    llm.model_copy.return_value = llm
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=result)
    return llm


def _fake_tool(results=None, error: Exception | None = None) -> MagicMock:
    t = MagicMock()
    if error:
        t.invoke.side_effect = error
    else:
        t.invoke.return_value = results
    return t


class TestUsable:
    def test_rejects_error_and_empty_placeholders(self):
        assert not _usable({"url": "u", "snippet": "[Search error: HTTPStatusError]" + "x" * 200})
        assert not _usable({"url": "u", "snippet": "No abstracts found for this query." + "x" * 200})


class TestInterleave:

    def test_dedupes_by_url_and_title_and_shares_seen(self):
        seen: set[str] = set()
        ranked = {
            "a": [_paper(1), {**_paper(2), "url": "https://p/1"}],          # same url as #1
            "b": [{**_paper(3), "title": "paper 1"}, _paper(4)],            # same title as #1
        }
        out = interleave(ranked, cap=10, seen=seen)
        assert [p["url"] for p in out] == ["https://p/1", "https://p/4"]
        # a later call with the shared `seen` won't hand the same papers out again
        assert interleave({"a": [_paper(1), _paper(9)]}, cap=10, seen=seen) == [_paper(9)]


class TestScorePool:
    @pytest.mark.asyncio
    async def test_maps_one_based_paper_numbers_and_scales_to_unit_range(self):
        llm = _llm_returning(PaperScores(scores=[
            PaperScore(paper=1, score=9),
            PaperScore(paper=2, score=15),     # clamped to 10
            PaperScore(paper=3, score=-3),     # clamped to 0
            PaperScore(paper=9, score=10),     # out of range -> ignored
        ]))
        scores = await score_pool("topic", "sq", [_paper(0), _paper(1), _paper(2)], llm)
        assert scores == {0: 0.9, 1: 1.0, 2: 0.0}


class TestSearchTask:

    @pytest.mark.asyncio
    async def test_placeholders_dropped_and_failing_tool_does_not_sink_others(self):
        err = {"url": "arxiv://search/q", "title": "x", "snippet": "[Search error: X]" + "z" * 200}
        tools = {
            "pubmed": _fake_tool([_paper(1), err]),
            "europe_pmc": _fake_tool(error=RuntimeError("network")),
        }
        out = await search_task("sq", "q", ["pubmed", "europe_pmc"], tools, 8, "sess")

        assert [p["url"] for p in out["pubmed"]] == ["https://p/1"]
        assert out["europe_pmc"] == []


class TestPaperScoutNode:
    def _tasks(self):
        return [
            {"sub_question": "sq-bio", "search_query": "kw bio", "domain": "biomedical"},
            {"sub_question": "sq-cs", "search_query": "kw cs", "domain": "cs_ml_physics_math"},
        ]

    async def _run(self, tools, score_side_effect, min_per_sq=1):
        """Run the scout with score_pool mocked; returns (result, score_pool mock).

        *min_per_sq* is ``settings.paper_min_per_sub_question``: a sub-question retries the tools
        its routing skipped while fewer non-web papers than this survive (1 = "none survived").
        """
        from research_swarm.config import settings
        from research_swarm.graph.nodes import paper_scout_node
        from tests.unit.test_graph import _make_state

        state = _make_state(scout_tasks=self._tasks())
        score_mock = AsyncMock(side_effect=score_side_effect)
        with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=MagicMock()),              patch.object(settings, "paper_min_per_sub_question", min_per_sq), \
             patch("research_swarm.agents.papers.tool_registry", return_value=tools), \
             patch("research_swarm.agents.papers.score_pool", score_mock):
            return await paper_scout_node(state), score_mock

    @pytest.mark.asyncio
    async def test_routes_tools_and_scores_each_sub_question_against_its_own_pool(self):
        tools = {
            "pubmed": _fake_tool([_paper(1)]), "europe_pmc": _fake_tool([_paper(2)]),
            "arxiv": _fake_tool([_paper(3)]), "web": _fake_tool([_paper(4)]),
        }
        calls: list[tuple[str, list[str]]] = []

        async def fake_score(topic, sq, pool, llm, frame=None):
            calls.append((sq, [p["url"] for p in pool]))
            return {i: 0.9 for i in range(len(pool))}

        out, _ = await self._run(tools, fake_score)

        by_sq = dict(calls)
        assert set(by_sq) == {"sq-bio", "sq-cs"}                       # one call per sub-question
        assert set(by_sq["sq-bio"]) == {"https://p/1", "https://p/2", "https://p/4"}
        assert set(by_sq["sq-cs"]) == {"https://p/3", "https://p/4"}    # p/4 (web) scored per sq
        tools["arxiv"].invoke.assert_called_once()                     # cs sub-question only
        assert tools["pubmed"].invoke.call_count == 1                  # biomedical only
        assert {p["sub_question"] for p in out["paper_corpus"]} == {"sq-bio", "sq-cs"}


    @pytest.mark.asyncio
    async def test_retries_skipped_tools_when_only_web_papers_survive(self):
        # The old rule retried only when NOTHING was kept, so a few web hits kept a health
        # question off PubMed forever. Web-only survivors must now trigger the retry.
        tools = {
            "pubmed": _fake_tool([_paper(1)]), "europe_pmc": _fake_tool([]),
            "arxiv": _fake_tool([]), "web": _fake_tool([_paper(4)]),
        }

        async def fake_score(topic, sq, pool, llm, frame=None):
            return {i: 0.9 for i in range(len(pool))}

        # both sub-questions use domain routing that includes web; force web-only survivors
        for p in (tools["web"],):
            p.invoke.return_value = [{**_paper(4), "source_type": "web"}]
        out, score_mock = await self._run(tools, fake_score, min_per_sq=1)
        assert score_mock.await_count >= 3                 # initial pools plus at least one retry


    @pytest.mark.asyncio
    async def test_no_retry_when_a_non_web_paper_survived_even_below_the_top_up_minimum(self):
        # min_per_sq=3 (the shipped top-up minimum): one PubMed / arXiv paper per sub-question is
        # enough, retrying the skipped tools for every sub-question would double scoring calls.
        tools = {
            "pubmed": _fake_tool([_paper(1)]), "europe_pmc": _fake_tool([]),
            "arxiv": _fake_tool([_paper(3)]), "web": _fake_tool([]),
        }

        async def fake_score(topic, sq, pool, llm, frame=None):
            return {i: 0.9 for i in range(len(pool))}

        out, score_mock = await self._run(tools, fake_score, min_per_sq=3)
        assert score_mock.await_count == 2                 # one per sub-question, no retry
        assert len(out["paper_corpus"]) == 2


class TestRoutingUnion:
    AVAIL = {"pubmed": 1, "europe_pmc": 1, "arxiv": 1, "web": 1}

    def test_cs_label_with_health_text_still_reaches_pubmed(self):
        from research_swarm.agents.papers import routed_tools_union

        names = routed_tools_union("cs_ml_physics_math", "saturated fat studies", self.AVAIL)
        assert {"pubmed", "europe_pmc", "arxiv", "web"} <= set(names)


class TestSelection:
    POOL = [{"url": f"u{i}"} for i in range(5)]

    def test_select_topk_orders_by_score_floors_and_caps(self):
        from research_swarm.agents.papers import select_topk

        scores = {0: 0.4, 1: 0.7, 2: 0.9, 3: 0.7, 4: 0.5}
        out = select_topk(self.POOL, scores, k=3, floor=0.5)
        assert [p["url"] for p in out] == ["u2", "u1", "u3"]          # ties keep pool order
        assert all("score" in p for p in out)
        assert [p["url"] for p in select_topk(self.POOL, scores, k=9, floor=0.7)] == [
            "u2", "u1", "u3"]


class TestExtractFindingsModes:

    @pytest.mark.asyncio
    async def test_multi_mode_uses_the_shared_extractor(self):
        from research_swarm.agents.extractor import ExtractedFact, Extraction

        papers = [{**_paper(0), "score": 0.9,
                   "snippet": "The trial enrolled 100 patients and found a significant effect."}]
        fact = ExtractedFact(source=1, sub_question=1, claim="100 patients enrolled",
                             quote="The trial enrolled 100 patients")
        llm = _llm_returning(Extraction(facts=[fact]))
        out = await extract_findings("t", "sq", papers, llm)
        assert [f.claim for f in out] == ["100 patients enrolled"]
        assert out[0].sub_question == "sq" and out[0].grounding == "quote"
