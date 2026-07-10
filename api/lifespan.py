from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from research_swarm.graph.builder import build_graph, make_async_checkpointer

from api.runs import shutdown_all_runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = await make_async_checkpointer()
    app.state.checkpointer = checkpointer
    app.state.graphs = {
        True: build_graph(checkpointer=checkpointer, interrupt_before_writer=True),
        False: build_graph(checkpointer=checkpointer, interrupt_before_writer=False),
    }
    yield
    # Cancel in-flight runs before closing the checkpointer connection they
    # write to. Their progress survives in the checkpoint and is resumable.
    await shutdown_all_runs()
    await checkpointer.conn.close()
