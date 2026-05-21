"""SQLite session management -- list, load, and delete past research sessions."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
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
                COUNT(*) AS steps
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

    summaries: list[SessionSummary] = []
    for row in rows:
        def _parse_dt(val: str | None) -> datetime | None:
            if not val:
                return None
            try:
                return datetime.fromisoformat(str(val)).replace(tzinfo=UTC)
            except ValueError:
                return None

        summaries.append(
            SessionSummary(
                thread_id=row["thread_id"],
                created_at=_parse_dt(row["first_seen"]),
                updated_at=_parse_dt(row["last_seen"]),
                step_count=row["steps"],
                has_report=False,   # updated below if possible
            )
        )
    return summaries


def get_session_state(thread_id: str) -> dict[str, Any] | None:
    """Load the latest state snapshot for a session from the persistent SQLite store.

    Opens a fresh AsyncSqliteSaver connection so the caller does not need to
    manage a long-lived checkpointer reference.
    """
    import asyncio

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
        return dict(snapshot.values) if hasattr(snapshot, "values") else None

    return asyncio.run(_load())


def delete_session(thread_id: str) -> int:
    """Delete all checkpoints for a session. Returns number of rows deleted."""
    db = _db_path()
    if not db.exists():
        return 0
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        conn.commit()
        return cur.rowcount
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()
