"""RAG index builders — create and load session-scoped LlamaIndex indexes.

Two indexes per session:
  - VectorStoreIndex  : backed by ChromaDB; used for specific fact retrieval.
  - SummaryIndex      : in-memory list index; used for high-level doc overviews.
    (Built only when Ollama is available; see query_engines.py.)

All computation is local — HuggingFaceEmbedding runs on CPU, no cloud calls.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import chromadb
from llama_index.core import Settings as LISettings
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from research_swarm.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singletons — embedding model is expensive to load; cache after first load
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    """Return a cached HuggingFaceEmbedding instance (BAAI/bge-small-en-v1.5).

    Downloads the model on first call (~130 MB) to ~/.cache/huggingface.
    All subsequent calls return the cached instance — no re-download.
    """
    kwargs: dict = {"model_name": settings.embed_model_name}
    if settings.embed_cache_dir:
        kwargs["cache_folder"] = settings.embed_cache_dir
    logger.info("Loading embed model '%s' …", settings.embed_model_name)
    model = HuggingFaceEmbedding(**kwargs)
    logger.info("Embed model loaded.")
    return model


# ---------------------------------------------------------------------------
# Chroma helpers (shared with ingestion.py)
# ---------------------------------------------------------------------------

def _session_chroma_path(session_id: str) -> Path:
    path = settings.sessions_dir / session_id / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collection_name(session_id: str) -> str:
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in session_id)
    return f"rs-{safe}"[:63]


def _get_chroma_store(session_id: str) -> ChromaVectorStore:
    path = _session_chroma_path(session_id)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=_collection_name(session_id),
        metadata={"hnsw:space": "cosine"},
    )
    return ChromaVectorStore(chroma_collection=collection)


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def load_vector_index(session_id: str) -> VectorStoreIndex:
    """Load (or create empty) a VectorStoreIndex from the session's Chroma store.

    This is cheap — it connects to the existing persisted collection rather
    than re-embedding anything.
    """
    embed_model = get_embed_model()
    store = _get_chroma_store(session_id)
    index = VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=embed_model,
    )
    logger.info("VectorStoreIndex loaded for session '%s'.", session_id)
    return index


def build_summary_index(
    session_id: str,
    llm: BaseLLM,
    *,
    documents=None,
) -> SummaryIndex:
    """Build an in-memory SummaryIndex from all documents already in the session.

    Requires a local LLM (Ollama) for synthesis at query time; the index itself
    is built without LLM calls (documents are stored as-is).

    Args:
        session_id: Session identifier.
        llm: A LlamaIndex-compatible LLM instance (e.g. Ollama).
        documents: Optional list of llama_index Documents. If None, the index
            is empty — useful as a placeholder when no LLM is available.
    """
    docs = documents or []
    # Temporarily configure LlamaIndex global settings for this index build
    LISettings.llm = llm
    LISettings.embed_model = get_embed_model()

    index = SummaryIndex.from_documents(docs, show_progress=False)
    logger.info(
        "SummaryIndex built with %d documents for session '%s'.", len(docs), session_id
    )
    return index
