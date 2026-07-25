"""Unit tests for the document worker (single-shot full-document extraction).

All LLM calls are mocked. No API keys or network access required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_llm(return_value=None, side_effect=None):
    mock_llm = MagicMock()
    if side_effect is not None:
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=return_value)
    return mock_llm


# ---------------------------------------------------------------------------
# _split_into_parts
# ---------------------------------------------------------------------------

class TestSplitIntoParts:
    def test_returns_single_part_when_under_threshold(self):
        from research_swarm.agents.document_worker import _split_into_parts

        text = "Short document. " * 10
        parts = _split_into_parts(text, max_chars=10_000)
        assert parts == [text]

    def test_splits_oversized_document_into_multiple_parts(self):
        from research_swarm.agents.document_worker import _split_into_parts

        # Well-formed sentences so SentenceSplitter has real boundaries to work with.
        text = "This is a test sentence about a topic. " * 500  # ~20,000 chars
        parts = _split_into_parts(text, max_chars=8_000)

        assert len(parts) >= 2
        for part in parts:
            # Individual sentence chunks can occasionally push a part slightly
            # over -- allow some slack rather than asserting a hard cap.
            assert len(part) <= 9_000
        # No content silently dropped.
        assert sum(len(p) for p in parts) >= len(text) * 0.95

    def test_split_parts_do_not_cut_words_mid_token(self):
        from research_swarm.agents.document_worker import _split_into_parts

        text = "Alpha beta gamma delta epsilon. " * 400
        parts = _split_into_parts(text, max_chars=6_000)
        for part in parts:
            assert not part.startswith(" ")


# ---------------------------------------------------------------------------
# run_document_worker
# ---------------------------------------------------------------------------

class TestRunDocumentWorker:
    @pytest.mark.asyncio
    async def test_extracts_finding_for_matching_sub_question(self):
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        llm = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(
                sub_question="What is the effect size?",
                claim="The effect size was 0.45.",
                confidence=0.8,
                quote="effect size of 0.45 was observed",
            ),
        ]))

        findings = await run_document_worker(
            document={"url": "https://paper.com", "title": "A Paper", "source_type": "pdf"},
            part_index=0, part_total=1,
            part_text="... effect size of 0.45 was observed ...",
            sub_questions=["What is the effect size?", "What is the sample size?"],
            llm=llm,
        )

        assert len(findings) == 1
        f = findings[0]
        assert f.sub_question == "What is the effect size?"
        assert f.claim == "The effect size was 0.45."
        assert f.confidence == pytest.approx(0.8)
        assert f.evidence[0].url == "https://paper.com"
        assert f.evidence[0].source_type == "pdf"

    @pytest.mark.asyncio
    async def test_snaps_sub_question_to_canonical_casing(self):
        """The LLM's returned sub_question string may differ in case/whitespace
        from the plan's -- it must be snapped to the exact canonical string so
        downstream exact-match logic (dispatch/rework) still finds it."""
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        llm = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(
                sub_question="  what is the EFFECT size?  ",
                claim="claim", confidence=0.5,
            ),
        ]))

        findings = await run_document_worker(
            document={"url": "https://paper.com"},
            part_index=0, part_total=1, part_text="text",
            sub_questions=["What is the effect size?"],
            llm=llm,
        )

        assert findings[0].sub_question == "What is the effect size?"

    @pytest.mark.asyncio
    async def test_drops_items_not_matching_any_sub_question(self):
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        llm = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(sub_question="Some unrelated question?", claim="c", confidence=0.5),
        ]))

        findings = await run_document_worker(
            document={"url": "https://paper.com"},
            part_index=0, part_total=1, part_text="text",
            sub_questions=["What is the effect size?"],
            llm=llm,
        )

        assert findings == []

    @pytest.mark.asyncio
    async def test_no_sub_questions_returns_empty_without_calling_llm(self):
        from research_swarm.agents.document_worker import run_document_worker

        llm = _mock_llm(return_value=None)
        findings = await run_document_worker(
            document={"url": "https://paper.com"},
            part_index=0, part_total=1, part_text="text",
            sub_questions=[],
            llm=llm,
        )
        assert findings == []
        llm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_list(self):
        from research_swarm.agents.document_worker import run_document_worker

        llm = _mock_llm(side_effect=Exception("model unavailable"))
        findings = await run_document_worker(
            document={"url": "https://paper.com"},
            part_index=0, part_total=1, part_text="text",
            sub_questions=["Q?"],
            llm=llm,
        )
        assert findings == []

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self):
        """A document that doesn't address any sub-question returns no findings."""
        from research_swarm.agents.document_worker import DocumentExtraction, run_document_worker

        llm = _mock_llm(return_value=DocumentExtraction(items=[]))
        findings = await run_document_worker(
            document={"url": "https://paper.com"},
            part_index=0, part_total=1, part_text="text",
            sub_questions=["Q?"],
            llm=llm,
        )
        assert findings == []

    @pytest.mark.asyncio
    async def test_two_parts_of_same_document_addressing_same_sub_question_get_same_id(self):
        """route_from_document_pass fans out every part of an oversized
        document as an independent Send in the same graph step -- there's no
        prior-round state to look up an existing finding's id against the way
        researcher.py/workers.py do for re-research. Without a deterministic
        id, two parts covering the same sub-question would produce two
        separate Finding objects that both survive the merge-by-id reducer
        instead of colliding into one."""
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        document = {"url": "https://paper.com", "title": "A Paper", "source_type": "pdf"}
        sub_questions = ["What is the effect size?"]

        llm_part1 = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(
                sub_question="What is the effect size?", claim="From part 1.", confidence=0.7,
            ),
        ]))
        llm_part2 = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(
                sub_question="What is the effect size?", claim="From part 2.", confidence=0.6,
            ),
        ]))

        (finding1,) = await run_document_worker(
            document=document, part_index=0, part_total=2, part_text="part one text",
            sub_questions=sub_questions, llm=llm_part1,
        )
        (finding2,) = await run_document_worker(
            document=document, part_index=1, part_total=2, part_text="part two text",
            sub_questions=sub_questions, llm=llm_part2,
        )

        assert finding1.id == finding2.id

    @pytest.mark.asyncio
    async def test_different_sub_questions_get_different_ids(self):
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        document = {"url": "https://paper.com"}
        llm = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(sub_question="Q1?", claim="c1", confidence=0.5),
            DocumentFindingItem(sub_question="Q2?", claim="c2", confidence=0.5),
        ]))

        findings = await run_document_worker(
            document=document, part_index=0, part_total=1, part_text="text",
            sub_questions=["Q1?", "Q2?"], llm=llm,
        )

        assert findings[0].id != findings[1].id

    @pytest.mark.asyncio
    async def test_documents_without_a_url_get_distinct_ids(self):
        """A document with no URL can't be deduped against -- it must fall
        back to a fresh id per finding rather than colliding with an
        unrelated urlless document's finding for the same sub-question."""
        from research_swarm.agents.document_worker import (
            DocumentExtraction,
            DocumentFindingItem,
            run_document_worker,
        )

        sub_questions = ["Q?"]
        llm1 = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(sub_question="Q?", claim="c1", confidence=0.5),
        ]))
        llm2 = _mock_llm(return_value=DocumentExtraction(items=[
            DocumentFindingItem(sub_question="Q?", claim="c2", confidence=0.5),
        ]))

        (finding1,) = await run_document_worker(
            document={}, part_index=0, part_total=1, part_text="text",
            sub_questions=sub_questions, llm=llm1,
        )
        (finding2,) = await run_document_worker(
            document={}, part_index=0, part_total=1, part_text="text",
            sub_questions=sub_questions, llm=llm2,
        )

        assert finding1.id != finding2.id
