"""Unit tests for Phase 3 RAG layer — all heavy dependencies mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def session_id(tmp_path, monkeypatch):
    """Override data_dir so Chroma is written to a temp directory.

    sessions_dir is a @property computed from data_dir, so we only patch
    data_dir here — and additionally patch the _session_chroma_path helper
    in both ingestion and indexes so they resolve inside tmp_path.
    """
    from research_swarm import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)

    def _fake_chroma_path(sid: str):
        p = tmp_path / "sessions" / sid / "chroma"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Both ingestion and indexes now import from rag._chroma — patch the
    # canonical location so both modules pick up the fake automatically.
    monkeypatch.setattr("research_swarm.rag._chroma.session_chroma_path", _fake_chroma_path)
    return "test-session-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chroma_collection(count: int = 0):
    coll = MagicMock()
    coll.count.return_value = count
    return coll


def _fake_chroma_client(collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def _fake_embed_model():
    """Return a mock HuggingFaceEmbedding that produces 384-dim zero vectors."""
    model = MagicMock()
    model.get_text_embedding.return_value = [0.0] * 384
    model.get_text_embedding_batch.return_value = [[0.0] * 384]
    return model


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

class TestIngestionPipeline:

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_collection_count(self, mock_vi, mock_chroma_cls, session_id, tmp_path):
        coll = _fake_chroma_collection(count=7)
        mock_chroma_cls.return_value = _fake_chroma_client(coll)

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        assert pipeline.collection_count() == 7

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_text_returns_chunk_count(self, mock_vi, mock_chroma_cls, session_id):
        coll = _fake_chroma_collection()
        mock_chroma_cls.return_value = _fake_chroma_client(coll)
        mock_vi.from_documents.return_value = MagicMock()

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        embed = _fake_embed_model()

        n = pipeline.ingest_text(
            "Hello world. " * 50,
            {"url": "https://example.com", "title": "Test", "source_type": "web"},
            embed,
        )
        assert n >= 1  # at least one chunk was created
        mock_vi.from_documents.assert_called()

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_empty_text_returns_zero(self, mock_vi, mock_chroma_cls, session_id):
        coll = _fake_chroma_collection()
        mock_chroma_cls.return_value = _fake_chroma_client(coll)

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        n = pipeline.ingest_text("", {"url": "x"}, _fake_embed_model())
        assert n == 0
        mock_vi.from_documents.assert_not_called()

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_source_dict_skips_error_snippets(self, mock_vi, mock_chroma_cls, session_id):
        mock_chroma_cls.return_value = _fake_chroma_client(_fake_chroma_collection())

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        # Snippet starting with "[" signals an error from url_fetcher / pdf_loader
        n = pipeline.ingest_source_dict(
            {"url": "x", "snippet": "[HTTP 404: Not Found]"},
            _fake_embed_model(),
        )
        assert n == 0

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_source_dict_good_snippet(self, mock_vi, mock_chroma_cls, session_id):
        mock_chroma_cls.return_value = _fake_chroma_client(_fake_chroma_collection())
        mock_vi.from_documents.return_value = MagicMock()

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        n = pipeline.ingest_source_dict(
            {
                "url": "https://arxiv.org/abs/0001",
                "title": "Great Paper",
                "snippet": "We present a novel approach to machine learning. " * 20,
                "source_type": "arxiv",
                "credibility_score": 0.85,
            },
            _fake_embed_model(),
        )
        assert n >= 1

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_url_calls_fetch_url(self, mock_vi, mock_chroma_cls, session_id):
        mock_chroma_cls.return_value = _fake_chroma_client(_fake_chroma_collection())
        mock_vi.from_documents.return_value = MagicMock()

        fetch_result = {
            "url": "https://example.com",
            "title": "Example",
            "snippet": "This is a long article about AI. " * 30,
            "source_type": "web",
            "credibility_score": 0.6,
        }

        with patch("research_swarm.tools.url_fetcher.fetch_url") as mock_fetch:
            mock_fetch.invoke.return_value = fetch_result
            from research_swarm.rag.ingestion import IngestionPipeline
            pipeline = IngestionPipeline(session_id)
            pipeline.ingest_url("https://example.com", _fake_embed_model())
            mock_fetch.invoke.assert_called_once()

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_pdf_calls_load_pdf(self, mock_vi, mock_chroma_cls, session_id):
        mock_chroma_cls.return_value = _fake_chroma_client(_fake_chroma_collection())
        mock_vi.from_documents.return_value = MagicMock()

        pdf_result = {
            "url": "file:///tmp/paper.pdf",
            "title": "Research Paper",
            "total_pages": 2,
            "chunks": [
                {"page": 1, "text": "Introduction to the study. " * 20, "char_count": 500},
                {"page": 2, "text": "Methodology details here. " * 20, "char_count": 480},
            ],
            "snippet": "Introduction to the study.",
            "source_type": "pdf",
            "credibility_score": 0.7,
        }

        with patch("research_swarm.tools.pdf_loader.load_pdf") as mock_load:
            mock_load.invoke.return_value = pdf_result
            from research_swarm.rag.ingestion import IngestionPipeline
            pipeline = IngestionPipeline(session_id)
            n = pipeline.ingest_pdf("/tmp/paper.pdf", _fake_embed_model())
            assert n >= 1
            mock_load.invoke.assert_called_once()

    @patch("research_swarm.rag.ingestion.chromadb.PersistentClient")
    @patch("research_swarm.rag.ingestion.VectorStoreIndex")
    def test_ingest_source_dicts_bulk(self, mock_vi, mock_chroma_cls, session_id):
        mock_chroma_cls.return_value = _fake_chroma_client(_fake_chroma_collection())
        mock_vi.from_documents.return_value = MagicMock()

        from research_swarm.rag.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(session_id)
        sources = [
            {"url": f"https://example.com/{i}", "snippet": f"Content chunk {i} " * 30}
            for i in range(3)
        ]
        n = pipeline.ingest_source_dicts(sources, _fake_embed_model())
        assert n >= 3  # at least one chunk per source

    def test_collection_name_is_safe(self):
        from research_swarm.rag.ingestion import _collection_name
        # UUIDs with hyphens
        name = _collection_name("550e8400-e29b-41d4-a716-446655440000")
        assert all(c.isalnum() or c == "-" for c in name)
        assert len(name) <= 63
        assert len(name) >= 3

    def test_collection_name_special_chars(self):
        from research_swarm.rag.ingestion import _collection_name
        name = _collection_name("session with spaces & symbols!")
        assert all(c.isalnum() or c == "-" for c in name)


# ---------------------------------------------------------------------------
# indexes.py — get_embed_model (cached singleton)
# ---------------------------------------------------------------------------

class TestGetEmbedModel:
    def test_returns_huggingface_embedding(self):
        # Only test that it returns a HuggingFaceEmbedding without loading the model
        with patch("research_swarm.rag.indexes.HuggingFaceEmbedding") as mock_hf:
            mock_hf.return_value = MagicMock()
            # Clear lru_cache so our mock is used
            from research_swarm.rag import indexes
            indexes.get_embed_model.cache_clear()
            model = indexes.get_embed_model()
            mock_hf.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")
            assert model is not None
            # Second call should be cached
            indexes.get_embed_model()
            mock_hf.assert_called_once()  # still once

    def test_uses_config_model_name(self, monkeypatch):
        from research_swarm import config
        monkeypatch.setattr(config.settings, "embed_model_name", "custom/model")
        with patch("research_swarm.rag.indexes.HuggingFaceEmbedding") as mock_hf:
            mock_hf.return_value = MagicMock()
            from research_swarm.rag import indexes
            indexes.get_embed_model.cache_clear()
            indexes.get_embed_model()
            mock_hf.assert_called_once_with(model_name="custom/model")
            indexes.get_embed_model.cache_clear()  # restore


class TestLoadVectorIndex:
    @patch("research_swarm.rag.indexes.chromadb.PersistentClient")
    @patch("research_swarm.rag.indexes.VectorStoreIndex")
    @patch("research_swarm.rag.indexes.get_embed_model")
    def test_loads_from_vector_store(self, mock_embed, mock_vi, mock_chroma, tmp_path, monkeypatch):
        def _fake_path(sid):
            p = tmp_path / "sessions" / sid / "chroma"
            p.mkdir(parents=True, exist_ok=True)
            return p
        monkeypatch.setattr("research_swarm.rag._chroma.session_chroma_path", _fake_path)

        mock_embed.return_value = _fake_embed_model()
        mock_chroma.return_value = _fake_chroma_client(_fake_chroma_collection())
        mock_vi.from_vector_store.return_value = MagicMock()

        from research_swarm.rag.indexes import load_vector_index
        index = load_vector_index("sess-xyz")
        mock_vi.from_vector_store.assert_called_once()
        assert index is not None


# ---------------------------------------------------------------------------
# query_engines.py — probe_ollama and get_local_llm
# ---------------------------------------------------------------------------

class TestProbeOllama:
    def setup_method(self):
        # Reset the module-level TTL cache so each test gets a fresh HTTP call.
        import research_swarm.rag.query_engines as qe
        qe._ollama_probe_cache = None

    def test_returns_true_when_ollama_up(self):
        with patch("research_swarm.rag.query_engines.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            from research_swarm.rag.query_engines import probe_ollama
            assert probe_ollama() is True

    def test_returns_false_on_connection_error(self):
        import httpx
        with patch("research_swarm.rag.query_engines.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("refused")
            from research_swarm.rag.query_engines import probe_ollama
            assert probe_ollama() is False

    def test_returns_false_on_non_200(self):
        with patch("research_swarm.rag.query_engines.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=503)
            from research_swarm.rag.query_engines import probe_ollama
            assert probe_ollama() is False


class TestGetLocalLLM:
    def test_returns_ollama_when_up(self):
        with patch("research_swarm.rag.query_engines.probe_ollama", return_value=True):
            with patch("research_swarm.rag.query_engines.Ollama") as mock_ollama:
                mock_ollama.return_value = MagicMock()
                from research_swarm.rag.query_engines import get_local_llm
                llm = get_local_llm()
                assert llm is not None
                mock_ollama.assert_called_once()

    def test_returns_none_when_down(self):
        with patch("research_swarm.rag.query_engines.probe_ollama", return_value=False):
            from research_swarm.rag.query_engines import get_local_llm
            llm = get_local_llm()
            assert llm is None


class TestGetResearchQueryEngine:
    @patch("research_swarm.rag.query_engines.load_vector_index")
    @patch("research_swarm.rag.query_engines.get_local_llm")
    def test_returns_vector_engine_when_no_ollama(self, mock_llm, mock_vi):
        mock_llm.return_value = None
        mock_index = MagicMock()
        mock_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_engine
        mock_vi.return_value = mock_index

        from research_swarm.rag.query_engines import get_research_query_engine
        engine = get_research_query_engine("sess-1")
        # Should have called as_query_engine with response_mode="no_text"
        call_kwargs = mock_index.as_query_engine.call_args.kwargs
        assert call_kwargs.get("response_mode") == "no_text"
        assert engine is mock_engine

    @patch("research_swarm.rag.query_engines.load_vector_index")
    @patch("research_swarm.rag.query_engines.get_local_llm")
    @patch("research_swarm.rag.query_engines.build_summary_index")
    @patch("research_swarm.rag.query_engines.RouterQueryEngine")
    @patch("research_swarm.rag.query_engines.SubQuestionQueryEngine")
    def test_returns_subq_engine_when_ollama_up(
        self, mock_subq, mock_router, mock_summary, mock_llm_fn, mock_vi
    ):
        mock_llm_fn.return_value = MagicMock()  # Ollama available
        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = MagicMock()
        mock_vi.return_value = mock_index

        mock_summary_index = MagicMock()
        mock_summary_index.as_query_engine.return_value = MagicMock()
        mock_summary.return_value = mock_summary_index

        mock_router.return_value = MagicMock()
        mock_subq_engine = MagicMock()
        mock_subq.from_defaults.return_value = mock_subq_engine

        from research_swarm.rag.query_engines import get_research_query_engine
        engine = get_research_query_engine("sess-2")
        mock_subq.from_defaults.assert_called_once()
        assert engine is mock_subq_engine
