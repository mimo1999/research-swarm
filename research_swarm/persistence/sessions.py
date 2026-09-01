"""SQLite session management -- list, load, and delete past research sessions."""
from __future__ import annotations

import asyncio
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_swarm.config import settings


def _db_path() -> Path:
    p = settings.data_dir / "checkpoints" / "sessions.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class SessionSummary:
    thread_id: str
    created_at: datetime | None
    updated_at: datetime | None
    step_count: int
    has_report: bool


def list_sessions() -> list[SessionSummary]:
    """Return a list of all past sessions stored in the checkpoint database.

    Reads directly from the LangGraph SqliteSaver schema
    (table: checkpoints, columns: thread_id, checkpoint_ns, checkpoint_id,
     parent_checkpoint_id, type, checkpoint, metadata, created_at).
    """
    db = _db_path()
    if not db.exists():
        return []

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        # The LangGraph SqliteSaver table is named 'checkpoints'
        cur = conn.execute(
            """
            SELECT
                thread_id,
                MIN(created_at) AS first_seen,
                MAX(created_at) AS last_seen,
                COUNT(*)        AS steps
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_seen DESC
            """
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return []
    finally:
        conn.close()

    def _parse_ts(raw: str | None) -> datetime | None:
        if not raw:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    summaries: list[SessionSummary] = []
    for row in rows:
        summaries.append(
            SessionSummary(
                thread_id=row["thread_id"],
                created_at=_parse_ts(row["first_seen"]),
                updated_at=_parse_ts(row["last_seen"]),
                step_count=row["steps"],
                has_report=False,
            )
        )
    return summaries


def get_session_state(thread_id: str) -> dict[str, Any] | None:
    """Load the latest state snapshot for a session from the persistent SQLite store.

    Opens a fresh AsyncSqliteSaver connection so the caller does not need to
    manage a long-lived checkpointer reference.

    Uses ``asyncio.get_event_loop().run_until_complete()`` when a loop is
    already running (e.g. inside Streamlit with nest_asyncio applied) to
    avoid the "cannot run a new event loop" error that ``asyncio.run()``
    raises in that context.
    """
    from research_swarm.graph.builder import build_graph, get_thread_config, make_async_checkpointer

    async def _load() -> dict[str, Any] | None:
        saver = await make_async_checkpointer()
        try:
            graph    = build_graph(checkpointer=saver)
            config   = get_thread_config(thread_id)
            snapshot = await graph.aget_state(config)
        finally:
            await saver.conn.close()
        if snapshot is None:
            return None
        raw = dict(snapshot.values) if hasattr(snapshot, "values") else None
        # An empty dict means the thread was never written — treat as not found.
        if not raw:
            return None
        from research_swarm.runtime.migrations import migrate_state
        return migrate_state(raw)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # nest_asyncio is active (Streamlit context) — submit to running loop
            future = asyncio.run_coroutine_threadsafe(_load(), loop)
            return future.result(timeout=30)
        return loop.run_until_complete(_load())
    except RuntimeError:
        # No current event loop — create one
        return asyncio.run(_load())


def delete_session(thread_id: str) -> int:
    """Delete all checkpoints and session data for *thread_id*.

    Removes:
    - All rows in the ``checkpoints`` and ``writes`` tables for this thread.
    - The session's ChromaDB directory (``data/sessions/<thread_id>/chroma``).

    Returns the number of checkpoint rows deleted.
    """
    db = _db_path()
    rows_deleted = 0

    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            # Delete checkpoints first, in its own commit so the deletion is
            # persisted even when the writes table doesn't exist yet (older DBs).
            try:
                cur = conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                rows_deleted = cur.rowcount
                conn.commit()
            except sqlite3.OperationalError:
                pass

            # Also clean the writes table (LangGraph stores pending writes there).
            # This table was added later and may not exist in older databases, so
            # handle OperationalError independently from the checkpoints deletion.
            try:
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                conn.commit()
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()

    # Remove the Chroma vector store for this session
    chroma_dir = settings.sessions_dir / thread_id
    if chroma_dir.exists():
        try:
            shutil.rmtree(chroma_dir)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Could not remove session directory %s: %s", chroma_dir, exc
            )

    return rows_deleted


def prune_expired_sessions(retention_seconds: int, max_sessions: int) -> int:
    """Delete sessions older than *retention_seconds* or beyond *max_sessions*.

    Intended for hosted/multi-tenant deployments (settings.space_mode) where
    no one is around to click "delete session" and the underlying storage is
    typically ephemeral anyway -- see the space_* fields on Settings. Deletes:
      - every session last touched more than retention_seconds ago, then
      - the oldest remaining sessions past max_sessions, if still over the cap.

    Sessions with no parseable updated_at timestamp are treated as expired
    (deleted) rather than kept indefinitely, since a corrupt/unreadable
    timestamp is more likely stale data than a session still in use.

    Returns the number of sessions deleted.
    """
    now = datetime.now(timezone.utc)
    sessions = list_sessions()  # already ordered newest-first (last_seen DESC)

    deleted = 0
    kept: list[SessionSummary] = []
    for s in sessions:
        age = (now - s.updated_at).total_seconds() if s.updated_at else None
        if age is None or age > retention_seconds:
            delete_session(s.thread_id)
            deleted += 1
        else:
            kept.append(s)

    # kept is still newest-first; anything past max_sessions is the oldest excess.
    for s in kept[max_sessions:]:
        delete_session(s.thread_id)
        deleted += 1

    return deleted
