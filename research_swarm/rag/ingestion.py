"""RAG ingestion pipeline -- converts PDFs, URLs, and arXiv results into
LlamaIndex Documents and stores them in a session-scoped ChromaDB collection.

All computation is local:
  - Text extraction : existing tools (pdf_loader, url_fetcher, arxiv_tool)
  - Chunking        : LlamaIndex SentenceSplitter (CPU)
  - Embeddings      : HuggingFaceEmbedding (CPU) -- see rag/indexes.py
  - Vector store    : ChromaDB persisted to data/sessions/<session_id>/chroma/
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from research_swarm.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_chroma_path(session_id: str) -> Path:
    path = settings.sessions_dir / session_id / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collection_name(session_id: str) -> str:
    # Chroma collection names must be 3-63 chars, alphanumeric + hyphens.
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in session_id)
    return f"rs-{safe}"[:63]


def _make_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Ingest documents from various sources into a session-scoped Chroma collection.

    Usage::

        pipeline = IngestionPipeline(session_id="abc123")
        pipeline.ingest_pdf("/path/to/paper.pdf")
        pipeline.ingest_url("https://example.com/article")
        pipeline.ingest_source_dict(arxiv_result_dict)
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._splitter = _make_splitter()
        self._chroma_client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # Chroma setup
    # ------------------------------------------------------------------

    def _get_chroma_collection(self) -> chromadb.Collection:
        if self._collection is None:
            path = _session_chroma_path(self.session_id)
            self._chroma_client = chromadb.PersistentClient(path=str(path))
            self._collection = self._chroma_client.get_or_create_collection(
                name=_collection_name(self.session_id),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def collection_count(self) -> int:
        """Return the number of chunks currently stored in this session."""
        return self._get_chroma_collection().count()

    # ------------------------------------------------------------------
    # Core: text -> Documents -> nodes -> Chroma
    # ------------------------------------------------------------------

    def _text_to_documents(self, text: str, metadata: dict) -> list[Document]:
        """Wrap raw text in a LlamaIndex Document, then split into nodes."""
        if not text or not text.strip():
            return []
        doc = Document(text=text, metadata=metadata)
        nodes = self._splitter.get_nodes_from_documents([doc])
        # Convert nodes back to Documents so VectorStoreIndex can handle them
        return [
            Document(
                text=node.get_content(),
                metadata={**metadata, "chunk_index": i},
            )
            for i, node in enumerate(nodes)
        ]

    def _insert_documents(self, documents: list[Document], embed_model) -> int:
        """Embed and insert documents into Chroma. Returns chunk count inserted."""
        if not documents:
            return 0
        collection = self._get_chroma_collection()
        store = ChromaVectorStore(chroma_collection=collection)
        storage_ctx = StorageContext.from_defaults(vector_store=store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_ctx,
            embed_model=embed_model,
            show_progress=False,
        )
        logger.info(
            "Ingested %d chunks into session '%s'", len(documents), self.session_id
        )
        return len(documents)

    # ------------------------------------------------------------------
    # Public ingest methods
    # ------------------------------------------------------------------

    def ingest_text(
        self, text: str, metadata: dict, embed_model
    ) -> int:
        """Ingest arbitrary text with metadata. Returns chunk count."""
        docs = self._text_to_documents(text, metadata)
        return self._insert_documents(docs, embed_model)

    def ingest_pdf(self, file_path: str, embed_model) -> int:
        """Extract text from a local PDF and ingest all page chunks."""
        from research_swarm.tools.pdf_loader import load_pdf

        result = load_pdf.invoke({"file_path": file_path})
        if not result.get("chunks"):
            logger.warning("PDF '%s': no chunks extracted -- %s", file_path, result.get("snippet"))
            return 0

        total = 0
        for chunk in result["chunks"]:
            page_text = chunk.get("text", "").strip()
            if not page_text:
                continue
            metadata = {
                "url": result.get("url", file_path),
                "title": result.get("title", ""),
                "source_type": "pdf",
                "page": chunk.get("page", 0),
                "credibility_score": result.get("credibility_score", 0.7),
            }
            docs = self._text_to_documents(page_text, metadata)
            total += self._insert_documents(docs, embed_model)
        return total

    def ingest_url(self, url: str, embed_model, max_chars: int = 8000) -> int:
        """Fetch a URL and ingest its readable text content."""
        from research_swarm.tools.url_fetcher import fetch_url

        result = fetch_url.invoke({"url": url, "max_chars": max_chars})
        snippet = result.get("snippet", "")
        if snippet.startswith("["):
            logger.warning("URL '%s': fetch error -- %s", url, snippet)
            return 0

        metadata = {
            "url": result.get("url", url),
            "title": result.get("title", ""),
            "source_type": "web",
            "credibility_score": result.get("credibility_score", 0.6),
        }
        return self.ingest_text(snippet, metadata, embed_model)

    def ingest_source_dict(self, source_dict: dict, embed_model) -> int:
        """Ingest from a Source dict (as returned by web_search / arxiv_search tools)."""
        text = source_dict.get("snippet", "")
        if not text or text.startswith("["):
            return 0
        metadata = {
            "url": source_dict.get("url", ""),
            "title": source_dict.get("title", ""),
            "source_type": source_dict.get("source_type", "web"),
            "credibility_score": source_dict.get("credibility_score", 0.5),
        }
        return self.ingest_text(text, metadata, embed_model)

    def ingest_source_dicts(self, source_dicts: list[dict], embed_model) -> int:
        """Bulk-ingest a list of Source dicts. Returns total chunks inserted."""
        return sum(self.ingest_source_dict(s, embed_model) for s in source_dicts)
