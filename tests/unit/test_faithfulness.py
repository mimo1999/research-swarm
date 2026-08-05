"""Unit tests for eval/faithfulness.py -- section-level and report-level scoring."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_swarm.schemas.report import FinalReport, ReportSection
from research_swarm.schemas.source import Source, SourceType


def _mock_embed_model(vectors: dict[str, list[float]]):
    """Return a mock embed model whose get_text_embedding looks up by exact text."""
    model = MagicMock()
    model.get_text_embedding.side_effect = lambda t: vectors[t]
    return model


def _patch_embed_model(vectors: dict[str, list[float]]):
    return patch(
        "research_swarm.rag.indexes.get_embed_model",
        return_value=_mock_embed_model(vectors),
    )


class TestScoreSections:
    def test_section_without_citations_is_excluded(self):
        from research_swarm.eval.faithfulness import score_sections
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="No cites", body_md="body", citations=[])],
        )
        assert score_sections(report, []) == []

    def test_high_similarity_scores_near_one(self):
        from research_swarm.eval.faithfulness import score_sections
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="H1", body_md="claim text", citations=[1])],
        )
        refs = [Source(url="https://x.com", snippet="snippet text", source_type=SourceType.web)]
        vectors = {"claim text": [1.0, 0.0], "snippet text": [1.0, 0.0]}
        with _patch_embed_model(vectors):
            result = score_sections(report, refs)
        assert len(result) == 1
        assert result[0]["heading"] == "H1"
        assert result[0]["score"] == pytest.approx(1.0)
        assert result[0]["citations"] == [1]

    def test_orthogonal_embeddings_score_near_zero(self):
        from research_swarm.eval.faithfulness import score_sections
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="H1", body_md="claim text", citations=[1])],
        )
        refs = [Source(url="https://x.com", snippet="snippet text", source_type=SourceType.web)]
        vectors = {"claim text": [1.0, 0.0], "snippet text": [0.0, 1.0]}
        with _patch_embed_model(vectors):
            result = score_sections(report, refs)
        assert result[0]["score"] == pytest.approx(0.0, abs=1e-9)

    def test_missing_citation_index_is_silently_skipped(self):
        from research_swarm.eval.faithfulness import score_sections
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="H1", body_md="claim text", citations=[1, 99])],
        )
        refs = [Source(url="https://x.com", snippet="snippet text", source_type=SourceType.web)]
        vectors = {"claim text": [1.0, 0.0], "snippet text": [1.0, 0.0]}
        with _patch_embed_model(vectors):
            result = score_sections(report, refs)
        assert result[0]["score"] == pytest.approx(1.0)

    def test_embedding_failure_degrades_to_neutral_score(self):
        from research_swarm.eval.faithfulness import score_sections
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[ReportSection(heading="H1", body_md="claim text", citations=[1])],
        )
        refs = [Source(url="https://x.com", snippet="snippet text", source_type=SourceType.web)]
        with patch("research_swarm.rag.indexes.get_embed_model", side_effect=Exception("no model")):
            result = score_sections(report, refs)
        assert result[0]["score"] == pytest.approx(0.5)


class TestScoreReport:
    def test_mean_of_section_scores(self):
        from research_swarm.eval.faithfulness import score_report
        report = FinalReport(
            title="T", exec_summary="S",
            sections=[
                ReportSection(heading="H1", body_md="a", citations=[1]),
                ReportSection(heading="H2", body_md="b", citations=[1]),
            ],
        )
        refs = [Source(url="https://x.com", snippet="a", source_type=SourceType.web)]
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        with _patch_embed_model(vectors):
            score = score_report(report, refs)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_no_citable_sections_returns_one(self):
        from research_swarm.eval.faithfulness import score_report
        report = FinalReport(title="T", exec_summary="S", sections=[])
        assert score_report(report, []) == 1.0
