"""Deep read (agents/deep_read.py): full-text excerpts of the top primary arXiv papers."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from research_swarm.agents import deep_read as dr
from research_swarm.config import settings
from research_swarm.schemas.frame import QuestionFrame

_HTML = """<html><body><nav>arXiv menu</nav><article>
<h1>Cross-Model KV Cache Transfer</h1>
<p>We map the source cache into the target model's format with a per-head ridge mapper.</p>
<p>Since <math alttext="\\Theta_{R}">x</math> is orthogonal, the inversion of the rotation is
exact; the ridge mapping itself is approximate and fitted on a calibration corpus.</p>
<p>Across six matched-KV pairs, four retain 73-98% of standalone accuracy on HellaSwag.</p>
<p>Short.</p>
<p>We thank the reviewers and our funding agencies for their generous support of this work.</p>
</article></body></html>"""


def test_paragraphs_keep_readable_text_and_formula_text():
    paragraphs = dr._paragraphs(_HTML)
    assert not any("arXiv menu" in p or p == "Short." for p in paragraphs)
    assert any("Theta_R is orthogonal" in p for p in paragraphs)     # alttext, braces removed


def test_select_passages_keeps_the_relevant_paragraphs_in_document_order():
    paragraphs = dr._paragraphs(_HTML)
    text = dr.select_passages(paragraphs, "is the orthogonal inversion exact; accuracy retention",
                              budget=400)
    assert "orthogonal" in text and "funding" not in text
    assert len(text) <= 400 + 1
    assert dr.select_passages(paragraphs, "photosynthesis chlorophyll", budget=400) == ""


def _corpus():
    return [
        {"sub_question": "sq1", "url": "https://www.alphaxiv.org/abs/2608.03893",
         "title": "Cross-Model KV Cache Transfer", "snippet": "Abstract A.", "score": 0.9},
        {"sub_question": "sq2", "url": "https://arxiv.org/html/2608.03893v1",
         "title": "Cross-Model KV Cache Transfer", "snippet": "Abstract A.", "score": 0.7},
        {"sub_question": "sq1", "url": "https://arxiv.org/abs/2411.02820",
         "title": "DroidSpeak", "snippet": "Abstract B.", "score": 0.8},
        {"sub_question": "sq1", "url": "https://arxiv.org/abs/2501.00001",
         "title": "Low-ranked paper", "snippet": "Abstract C.", "score": 0.5},
        {"sub_question": "sq1", "url": "https://medium.com/@x/post-2608.03893",
         "title": "Blog", "snippet": "Blog text.", "score": 0.99},
    ]


@pytest.fixture
def _two_papers(monkeypatch):
    monkeypatch.setattr(settings, "deep_read_papers", 2)
    monkeypatch.setattr(settings, "deep_read_chars", 2000)


async def test_top_primary_papers_get_excerpts_and_their_arxiv_url(_two_papers):
    fetched: list[str] = []

    def fake_fetch(url):
        fetched.append(url)
        return dr._paragraphs(_HTML)

    original = _corpus()
    with patch.object(dr, "_fetch_paragraphs", fake_fetch):
        out = await dr.deep_read(original, "is KV transfer exact", QuestionFrame(), "s")
    assert sorted(fetched) == ["https://arxiv.org/html/2411.02820",
                               "https://arxiv.org/html/2608.03893"]     # top 2, blog skipped
    mirror, html, droid, low, blog = out
    assert dr.EXCERPT_HEADER in mirror["snippet"] and dr.EXCERPT_HEADER in html["snippet"]
    assert mirror["url"] == html["url"] == "https://arxiv.org/abs/2608.03893"   # canonical
    assert dr.EXCERPT_HEADER not in low["snippet"] and blog["snippet"] == "Blog text."
    assert original[0]["snippet"] == "Abstract A."                      # input not mutated


async def test_a_failed_fetch_keeps_the_abstract(_two_papers):
    def boom(url):
        raise ConnectionError("offline")

    with patch.object(dr, "_fetch_paragraphs", boom):
        out = await dr.deep_read(_corpus(), "q", None, "s")
    assert [p["snippet"] for p in out] == [p["snippet"] for p in _corpus()]


async def test_deep_read_off_changes_nothing():
    out = await dr.deep_read(_corpus(), "q", None, "s")          # conftest sets 0 papers
    assert out == _corpus()
