"""Assemble and compile the research swarm LangGraph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

import research_swarm.graph.nodes as _nodes
from research_swarm.config import settings
from research_swarm.graph.edges import route_from_collect, route_from_critic, route_from_supervisor
from research_swarm.schemas.state import AgentState

# ---------------------------------------------------------------------------
# Serializer with all project schema types registered.
#
# LangGraph 1.2+ warns (and will soon error) when deserializing unregistered
# Pydantic types from msgpack checkpoints.  We declare every type the graph
# stores so the serde layer can round-trip them safely.
# ---------------------------------------------------------------------------
_SCHEMA_MODULES: list[tuple[str, str]] = [
    ("research_swarm.schemas.query",    "ResearchDepth"),
    ("research_swarm.schemas.query",    "ResearchQuery"),
    ("research_swarm.schemas.worker",   "WorkerRole"),
    ("research_swarm.schemas.worker",   "SubQuestionAssignment"),
    ("research_swarm.schemas.plan",     "ResearchPlan"),
    ("research_swarm.schemas.finding",  "Finding"),
    ("research_swarm.schemas.source",   "SourceType"),
    ("research_swarm.schemas.source",   "Source"),
    ("research_swarm.schemas.critique", "CritiqueVerdict"),
    ("research_swarm.schemas.critique", "Critique"),
    ("research_swarm.schemas.report",   "ReportSection"),
    ("research_swarm.schemas.report",   "FinalReport"),
    ("research_swarm.schemas.report",   "ReportQualityScore"),
    ("research_swarm.schemas.judge",    "JudgeVerdict"),
    ("research_swarm.schemas.judge",    "LLMJudgeResult"),
]

_serde = JsonPlusSerializer(allowed_msgpack_modules=_SCHEMA_MODULES)


def _db_path() -> Path:
    """Return the path to the SQLite checkpoint database."""
    db_dir = settings.data_dir / "checkpoints"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "sessions.db"


async def make_async_checkpointer():
    """Create the production AsyncSqliteSaver with registered schema types.

    Must be awaited (or run via asyncio.run) in a context where nest_asyncio
    has already been applied (e.g. inside app.py after nest_asyncio.apply()).
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(_db_path()))
    cp = AsyncSqliteSaver(conn, serde=_serde)
    # AsyncSqliteSaver 3.x creates its own bare jsonplus_serde for legacy
    # json-format checkpoints. Patch it with our registered serde so that
    # reading old json-format state also goes through the allowlist.
    if hasattr(cp, "jsonplus_serde"):
        cp.jsonplus_serde = _serde
    return cp


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
        checkpointer = MemorySaver(serde=_serde)

    sg = StateGraph(AgentState)

    # ── Node registration ──────────────────────────────────────────────────
    # Evaluated at build_graph() call time so unittest.mock patches work.
    sg.add_node("supervisor",           _nodes.supervisor_node)
    sg.add_node("document_pass_node",   _nodes.document_pass_node)
    sg.add_node("document_worker_node", _nodes.document_worker_node)
    sg.add_node("dispatch_node",        _nodes.dispatch_node)
    sg.add_node("worker_node",          _nodes.worker_node)
    sg.add_node("collect_node",         _nodes.collect_node)
    sg.add_node("critic",               _nodes.critic_node)
    sg.add_node("fact_checker",         _nodes.fact_checker_node)
    sg.add_node("writer",               _nodes.writer_node)
    # Legacy researcher node kept so old tests / checkpoints continue to work
    sg.add_node("researcher",           _nodes.researcher_node)

    # ── Entry ──────────────────────────────────────────────────────────────
    sg.add_edge(START, "supervisor")

    # supervisor → document_pass_node (plan created; routing is deterministic)
    sg.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"document_pass_node": "document_pass_node", END: END},
    )

    # document_pass_node → [document_worker_node × N] via Send fan-out, or a
    # bounce straight to dispatch_node when there are no ingested documents
    sg.add_conditional_edges(
        "document_pass_node",
        _nodes.route_from_document_pass,   # returns list[Send]
    )

    # All parallel document workers converge at dispatch_node (outputs
    # merged by the findings reducer) for the normal round-0 sub-question
    # dispatch, which now skips any sub-question the document pass answered.
    sg.add_edge("document_worker_node", "dispatch_node")

    # dispatch_node → [worker_node × N]  via Send fan-out
    sg.add_conditional_edges(
        "dispatch_node",
        _nodes.route_from_dispatch,   # returns list[Send]
    )

    # All parallel workers converge at collect_node (outputs merged by reducers)
    sg.add_edge("worker_node", "collect_node")

    # collect_node → dispatch_node (re-research) OR critic (stop)
    sg.add_conditional_edges(
        "collect_node",
        route_from_collect,
        {"dispatch_node": "dispatch_node", "critic": "critic"},
    )

    # ── Post-critic pipeline (deterministic) ──────────────────────────────
    # critic → dispatch_node (rework weak/refuted findings, capped) OR fact_checker
    sg.add_conditional_edges(
        "critic",
        route_from_critic,
        {"dispatch_node": "dispatch_node", "fact_checker": "fact_checker"},
    )
    sg.add_edge("fact_checker", "writer")
    sg.add_edge("writer",       END)

    # ── Legacy researcher loop (for backward-compat) ──────────────────────
    sg.add_edge("researcher", "collect_node")

    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if interrupt_before_writer:
        compile_kwargs["interrupt_before"] = ["writer"]

    return sg.compile(**compile_kwargs)


def get_thread_config(session_id: str) -> dict:
    """Return the config dict required for graph.invoke / graph.astream."""
    return {"configurable": {"thread_id": session_id}}
