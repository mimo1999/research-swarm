"""Tests for SQLite database connection and persistence layer.

Covers:
  - builder._db_path()         directory creation and path correctness
  - builder.make_async_checkpointer()  aiosqlite connection, AsyncSqliteSaver
  - AsyncSqliteSaver            round-trip read/write via graph.ainvoke
  - persistence.sessions._db_path()
  - persistence.sessions.list_sessions()
  - persistence.sessions.delete_session()

All tests are fully offline — no API keys, no LLMs, no network.
Temporary directories (tmp_path) keep every test isolated from one another
and from the real data/ directory.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CHECKPOINTS_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id           TEXT NOT NULL,
    checkpoint_ns       TEXT NOT NULL DEFAULT '',
    checkpoint_id       TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type                TEXT,
    checkpoint          BLOB,
    metadata            BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""


def _uuid6(when: str) -> str:
    """A UUIDv6 checkpoint id for an ISO timestamp, like LangGraph's."""
    from datetime import datetime
    dt = datetime.fromisoformat(when).replace(tzinfo=UTC)
    ticks = int((dt - datetime(1582, 10, 15, tzinfo=UTC)).total_seconds() * 10_000_000)
    h = f"{ticks:015x}"
    return f"{h[:8]}-{h[8:12]}-6{h[12:]}-8000-000000000000"


def _seed_checkpoints(db_path: Path, rows: list[dict]) -> None:
    """Write synthetic checkpoint rows directly via sqlite3."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CHECKPOINTS_DDL)
    for r in rows:
        conn.execute(
            """
            INSERT INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id)
            VALUES (?, '', ?)
            """,
            (r["thread_id"], _uuid6(r["created_at"]) if "created_at" in r else r["checkpoint_id"]),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# AsyncSqliteSaver round-trip (write checkpoint → read it back)
# ---------------------------------------------------------------------------

class TestAsyncSqliteSaverRoundTrip:
    @pytest.mark.asyncio
    async def test_checkpoint_survives_reconnect(self, tmp_path, monkeypatch):
        """State written by graph.ainvoke must be readable from a fresh connection."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.config import settings
        from research_swarm.graph.builder import build_graph, get_thread_config
        from research_swarm.schemas import ResearchQuery

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        db_file = tmp_path / "checkpoints" / "sessions.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        session_id = "round-trip-session"
        try:
            _nodes.supervisor_node = fake_supervisor
            # First connection: write
            conn1  = await aiosqlite.connect(str(db_file))
            saver1 = AsyncSqliteSaver(conn1)
            graph1 = build_graph(checkpointer=saver1, interrupt_before_writer=False)
            config = get_thread_config(session_id)
            initial = {
                "messages": [], "query": ResearchQuery(topic="persist test", audience="general"),
                "plan": None, "findings": [], "critiques": [],
                "draft_report": None, "final_report": None,
                "human_feedback": None, "iteration_count": 0,
                "next_agent": None, "session_id": session_id,
            }
            await graph1.ainvoke(initial, config)
            await conn1.close()

            # Second connection: read back
            conn2  = await aiosqlite.connect(str(db_file))
            saver2 = AsyncSqliteSaver(conn2)
            graph2 = build_graph(checkpointer=saver2, interrupt_before_writer=False)
            snap   = await graph2.aget_state(config)
            await conn2.close()
        finally:
            _nodes.supervisor_node = orig

        assert snap is not None, "No snapshot found after reconnect"
        assert snap.values["session_id"] == session_id
        assert snap.values["iteration_count"] == 1


# ---------------------------------------------------------------------------
# persistence.sessions.list_sessions()
# ---------------------------------------------------------------------------

class TestListSessions:
    def _patch_db(self, monkeypatch, tmp_path):
        """Redirect sessions.py to use a temp DB path."""
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)


    def test_returns_one_session_per_thread(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "sess-aaa", "checkpoint_id": "ckpt-1", "created_at": "2024-06-01T10:00:00"},
            {"thread_id": "sess-aaa", "checkpoint_id": "ckpt-2", "created_at": "2024-06-01T10:05:00"},
            {"thread_id": "sess-bbb", "checkpoint_id": "ckpt-1", "created_at": "2024-06-02T09:00:00"},
        ])
        from research_swarm.persistence.sessions import list_sessions
        result = list_sessions()
        assert len(result) == 2
        ids = {s.thread_id for s in result}
        assert ids == {"sess-aaa", "sess-bbb"}


# ---------------------------------------------------------------------------
# persistence.sessions.delete_session()
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def _patch_db(self, monkeypatch, tmp_path):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)


    def test_deletes_all_checkpoints_for_thread(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "del-sess", "checkpoint_id": "c1"},
            {"thread_id": "del-sess", "checkpoint_id": "c2"},
            {"thread_id": "del-sess", "checkpoint_id": "c3"},
            {"thread_id": "keep-sess", "checkpoint_id": "c1"},
        ])
        from research_swarm.persistence.sessions import delete_session
        n = delete_session("del-sess")
        assert n == 3

    def test_does_not_delete_other_sessions(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "del-me",   "checkpoint_id": "c1"},
            {"thread_id": "keep-me",  "checkpoint_id": "c1"},
            {"thread_id": "keep-me",  "checkpoint_id": "c2"},
        ])
        from research_swarm.persistence.sessions import delete_session, list_sessions
        delete_session("del-me")
        remaining = list_sessions()
        assert len(remaining) == 1
        assert remaining[0].thread_id == "keep-me"


# ---------------------------------------------------------------------------
# persistence.sessions.prune_expired_sessions()
# ---------------------------------------------------------------------------

class TestPruneExpiredSessions:
    """prune_expired_sessions() backs space_mode's startup cleanup pass."""

    def _patch_db(self, monkeypatch, tmp_path):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)


    def test_deletes_sessions_older_than_retention(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        # created_at is parsed as UTC; "old" is far enough in the past that
        # any retention window this test uses (seconds) has long expired.
        _seed_checkpoints(db, [
            {"thread_id": "ancient", "checkpoint_id": "c1", "created_at": "2000-01-01T00:00:00"},
        ])
        from research_swarm.persistence.sessions import list_sessions, prune_expired_sessions
        deleted = prune_expired_sessions(retention_seconds=60, max_sessions=40)
        assert deleted == 1
        assert list_sessions() == []
