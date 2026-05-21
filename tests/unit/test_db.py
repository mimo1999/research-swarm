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
    created_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""


def _seed_checkpoints(db_path: Path, rows: list[dict]) -> None:
    """Write synthetic checkpoint rows directly via sqlite3."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(_CHECKPOINTS_DDL)
    for r in rows:
        conn.execute(
            """
            INSERT INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, created_at)
            VALUES (?, '', ?, ?)
            """,
            (r["thread_id"], r["checkpoint_id"], r.get("created_at", "2024-01-01T00:00:00")),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# builder._db_path()
# ---------------------------------------------------------------------------

class TestDbPath:
    def test_returns_path_inside_data_dir(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.graph.builder import _db_path
        p = _db_path()
        assert p == tmp_path / "checkpoints" / "sessions.db"

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path / "nested" / "deep")
        from research_swarm.graph.builder import _db_path
        p = _db_path()
        assert p.parent.exists(), "checkpoints/ directory was not created"

    def test_successive_calls_are_idempotent(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.graph.builder import _db_path
        p1 = _db_path()
        p2 = _db_path()
        assert p1 == p2


# ---------------------------------------------------------------------------
# builder.make_async_checkpointer()
# ---------------------------------------------------------------------------

class TestMakeAsyncCheckpointer:
    @pytest.mark.asyncio
    async def test_returns_async_sqlite_saver(self, tmp_path, monkeypatch):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.graph.builder import make_async_checkpointer
        saver = await make_async_checkpointer()
        try:
            assert isinstance(saver, AsyncSqliteSaver)
        finally:
            await saver.conn.close()

    @pytest.mark.asyncio
    async def test_connection_is_open(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.graph.builder import make_async_checkpointer
        saver = await make_async_checkpointer()
        try:
            # A live aiosqlite connection responds to execute
            cursor = await saver.conn.execute("SELECT 1")
            row = await cursor.fetchone()
            assert row[0] == 1
        finally:
            await saver.conn.close()

    @pytest.mark.asyncio
    async def test_database_file_is_created_on_disk(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.graph.builder import make_async_checkpointer
        saver = await make_async_checkpointer()
        await saver.conn.close()
        db_file = tmp_path / "checkpoints" / "sessions.db"
        assert db_file.exists(), "SQLite file was not created on disk"

    @pytest.mark.asyncio
    async def test_checkpoints_table_is_created_after_first_use(self, tmp_path, monkeypatch):
        """AsyncSqliteSaver.setup() is called lazily on first write.
        Verify the table exists after writing a checkpoint via graph.ainvoke.
        """

        import research_swarm.graph.nodes as _nodes
        from research_swarm.config import settings
        from research_swarm.graph.builder import (
            build_graph,
            get_thread_config,
            make_async_checkpointer,
        )
        from research_swarm.schemas import ResearchQuery

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        orig = _nodes.supervisor_node
        try:
            _nodes.supervisor_node = fake_supervisor
            saver = await make_async_checkpointer()
            graph  = build_graph(checkpointer=saver, interrupt_before_writer=False)
            config = get_thread_config("tbl-test")
            initial = {
                "messages": [], "query": ResearchQuery(topic="test", audience="general"),
                "plan": None, "findings": [], "critiques": [],
                "draft_report": None, "final_report": None,
                "human_feedback": None, "iteration_count": 0,
                "next_agent": None, "session_id": "tbl-test",
            }
            await graph.ainvoke(initial, config)
        finally:
            _nodes.supervisor_node = orig
            await saver.conn.close()

        # Inspect the file via a plain sqlite3 connection
        db_file = tmp_path / "checkpoints" / "sessions.db"
        raw = sqlite3.connect(str(db_file))
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        raw.close()
        assert "checkpoints" in tables, "checkpoints table was not created"


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
# persistence.sessions._db_path()
# ---------------------------------------------------------------------------

class TestPersistenceDbPath:
    """_db_path() reads settings.data_dir at call-time, so plain monkeypatching works
    without any module reload (which causes test-isolation issues)."""

    def test_path_is_under_data_dir(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        from research_swarm.persistence.sessions import _db_path
        p = _db_path()
        assert p == tmp_path / "checkpoints" / "sessions.db"

    def test_creates_parent_dir(self, tmp_path, monkeypatch):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path / "new_dir")
        from research_swarm.persistence.sessions import _db_path
        p = _db_path()
        assert p.parent.exists()


# ---------------------------------------------------------------------------
# persistence.sessions.list_sessions()
# ---------------------------------------------------------------------------

class TestListSessions:
    def _patch_db(self, monkeypatch, tmp_path):
        """Redirect sessions.py to use a temp DB path."""
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)

    def test_returns_empty_when_db_missing(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        from research_swarm.persistence.sessions import list_sessions
        result = list_sessions()
        assert result == []

    def test_returns_empty_when_table_missing(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        # Create the file but don't add the checkpoints table
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(str(db)).close()
        from research_swarm.persistence.sessions import list_sessions
        result = list_sessions()
        assert result == []

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

    def test_step_count_is_correct(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "sess-x", "checkpoint_id": "ckpt-1"},
            {"thread_id": "sess-x", "checkpoint_id": "ckpt-2"},
            {"thread_id": "sess-x", "checkpoint_id": "ckpt-3"},
        ])
        from research_swarm.persistence.sessions import list_sessions
        (sess,) = list_sessions()
        assert sess.step_count == 3

    def test_sessions_ordered_most_recent_first(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "old-sess", "checkpoint_id": "c1", "created_at": "2024-01-01T00:00:00"},
            {"thread_id": "new-sess", "checkpoint_id": "c1", "created_at": "2024-12-31T23:59:59"},
        ])
        from research_swarm.persistence.sessions import list_sessions
        result = list_sessions()
        assert result[0].thread_id == "new-sess"
        assert result[1].thread_id == "old-sess"

    def test_updated_at_is_most_recent_checkpoint(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [
            {"thread_id": "ts-sess", "checkpoint_id": "c1", "created_at": "2024-03-01T08:00:00"},
            {"thread_id": "ts-sess", "checkpoint_id": "c2", "created_at": "2024-03-01T09:30:00"},
        ])
        from research_swarm.persistence.sessions import list_sessions
        (sess,) = list_sessions()
        assert sess.updated_at is not None
        assert sess.updated_at.hour == 9
        assert sess.updated_at.minute == 30

    def test_null_created_at_parses_as_none(self, tmp_path, monkeypatch):
        """Rows with NULL created_at should not crash the parser."""
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        conn.execute(_CHECKPOINTS_DDL)
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, created_at) VALUES (?, '', ?, NULL)",
            ("null-ts-sess", "c1"),
        )
        conn.commit()
        conn.close()
        from research_swarm.persistence.sessions import list_sessions
        result = list_sessions()
        assert len(result) == 1
        assert result[0].created_at is None
        assert result[0].updated_at is None


# ---------------------------------------------------------------------------
# persistence.sessions.delete_session()
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def _patch_db(self, monkeypatch, tmp_path):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)

    def test_returns_zero_when_db_missing(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        from research_swarm.persistence.sessions import delete_session
        assert delete_session("ghost-session") == 0

    def test_returns_zero_when_table_missing(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(str(db)).close()   # file exists, no tables
        from research_swarm.persistence.sessions import delete_session
        assert delete_session("ghost-session") == 0

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

    def test_returns_zero_for_nonexistent_thread(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [{"thread_id": "other", "checkpoint_id": "c1"}])
        from research_swarm.persistence.sessions import delete_session
        assert delete_session("no-such-session") == 0

    def test_double_delete_is_safe(self, tmp_path, monkeypatch):
        self._patch_db(monkeypatch, tmp_path)
        db = tmp_path / "checkpoints" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _seed_checkpoints(db, [{"thread_id": "once", "checkpoint_id": "c1"}])
        from research_swarm.persistence.sessions import delete_session
        assert delete_session("once") == 1
        assert delete_session("once") == 0   # idempotent — second call is safe


# ---------------------------------------------------------------------------
# persistence.sessions.get_session_state()
# ---------------------------------------------------------------------------

class TestGetSessionState:
    """get_session_state() loads a persisted graph snapshot from AsyncSqliteSaver."""

    def _patch_db(self, monkeypatch, tmp_path):
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "data_dir", tmp_path)

    def test_returns_none_for_nonexistent_thread(self, tmp_path, monkeypatch):
        """Querying a thread that was never written returns None (or empty dict)."""
        self._patch_db(monkeypatch, tmp_path)
        from research_swarm.persistence.sessions import get_session_state
        result = get_session_state("thread-does-not-exist")
        # LangGraph may return {} for an unknown thread — either falsy value is acceptable.
        assert not result  # None or empty dict are both acceptable

    def test_returns_state_after_graph_run(self, tmp_path, monkeypatch):
        """State written by graph.ainvoke is readable via get_session_state."""
        import asyncio

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        import research_swarm.graph.nodes as _nodes
        from research_swarm.graph.builder import build_graph, get_thread_config
        from research_swarm.schemas import ResearchQuery

        self._patch_db(monkeypatch, tmp_path)
        db_file = tmp_path / "checkpoints" / "sessions.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)

        session_id = "gs-round-trip"
        initial = {
            "messages": [], "query": ResearchQuery(topic="state load test"),
            "plan": None, "findings": [], "critiques": [],
            "draft_report": None, "final_report": None,
            "human_feedback": None, "iteration_count": 0,
            "next_agent": None, "session_id": session_id,
        }

        async def fake_supervisor(state):
            return {"next_agent": "end", "iteration_count": 1, "messages": []}

        async def _write_checkpoint():
            orig = _nodes.supervisor_node
            try:
                _nodes.supervisor_node = fake_supervisor
                conn = await aiosqlite.connect(str(db_file))
                saver = AsyncSqliteSaver(conn)
                graph = build_graph(checkpointer=saver, interrupt_before_writer=False)
                config = get_thread_config(session_id)
                await graph.ainvoke(initial, config)
            finally:
                _nodes.supervisor_node = orig
                await conn.close()

        # Write the checkpoint (asyncio.run creates its own isolated event loop)
        asyncio.run(_write_checkpoint())

        # Now read it back via the synchronous helper
        from research_swarm.persistence.sessions import get_session_state
        state = get_session_state(session_id)

        assert state is not None, "get_session_state must return the saved state"
        assert state.get("session_id") == session_id
        assert state.get("iteration_count") == 1
