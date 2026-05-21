"""Assemble and compile the research swarm LangGraph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import research_swarm.graph.nodes as _nodes
from research_swarm.config import settings
from research_swarm.graph.edges import route_from_supervisor
from research_swarm.schemas.state import AgentState


def _db_path() -> Path:
    """Return the path to the SQLite checkpoint database."""
    db_dir = settings.data_dir / "checkpoints"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


async def make_async_checkpointer():
    """Create the production AsyncSqliteSaver.

    Must be awaited (or run via asyncio.run) in a context where nest_asyncio
    has already been applied (e.g. inside app.py after nest_asyncio.apply()).
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(_db_path()))
    return AsyncSqliteSaver(conn)


def build_graph(checkpointer=None, interrupt_before_writer: bool = True):
    """Build and compile the StateGraph.

    All graph nodes are async, so the caller is responsible for supplying an
    async-compatible checkpointer (AsyncSqliteSaver in production, MemorySaver
    in tests).  Passing no checkpointer falls back to an in-memory MemorySaver
    -- state is not persisted across Python processes in that case.

    Args:
        checkpointer: Any LangGraph checkpointer.  Omit only for quick tests
            or throwaway runs; use ``make_async_checkpointer()`` for the
            production persistent store.
        interrupt_before_writer: If True, the graph pauses before the writer
            node so the user can review findings (HITL).

    Returns:
        A compiled LangGraph (CompiledStateGraph).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    sg = StateGraph(AgentState)

    # Register nodes via module reference -- evaluated at build_graph() call time so
    # unittest.mock patches applied before calling build_graph() are picked up.
    sg.add_node("supervisor",   _nodes.supervisor_node)
    sg.add_node("researcher",   _nodes.researcher_node)
    sg.add_node("critic",       _nodes.critic_node)
    sg.add_node("fact_checker", _nodes.fact_checker_node)
    sg.add_node("writer",       _nodes.writer_node)

    # Entry point
    sg.add_edge(START, "supervisor")

    # Supervisor is the central router
    sg.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "critic": "critic",
            "fact_checker": "fact_checker",
            "writer": "writer",
            END: END,
        },
    )

    # All worker nodes return to supervisor
    sg.add_edge("researcher", "supervisor")
    sg.add_edge("critic", "supervisor")
    sg.add_edge("fact_checker", "supervisor")

    # Writer terminates the graph
    sg.add_edge("writer", END)

    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if interrupt_before_writer:
        compile_kwargs["interrupt_before"] = ["writer"]

    return sg.compile(**compile_kwargs)


def get_thread_config(session_id: str) -> dict:
    """Return the config dict required for graph.invoke / graph.astream."""
    return {"configurable": {"thread_id": session_id}}
