"""Shared ChromaDB path and collection-name helpers for RAG modules.

Both ``ingestion.py`` and ``indexes.py`` need identical logic for resolving
the per-session Chroma directory and sanitising a session ID into a valid
Chroma collection name.  Keeping a single source of truth here prevents the
two from drifting apart.
"""
from __future__ import annotations

from pathlib import Path

from research_swarm.config import settings


def session_chroma_path(session_id: str) -> Path:
    """Return (and create) the ChromaDB persist directory for *session_id*."""
    path = settings.sessions_dir / session_id / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def collection_name(session_id: str) -> str:
    """Return a Chroma-safe collection name for *session_id*.

    Chroma requires names that are 3–63 characters, alphanumeric + hyphens,
    and must start/end with an alphanumeric character.
    """
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in session_id)
    return f"rs-{safe}"[:63]
