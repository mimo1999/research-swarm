"""FastAPI entry point for the Research Swarm React frontend.

Run with:  poetry run uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.lifespan import lifespan
from api.routes import config, documents, research, sessions

app = FastAPI(title="Research Swarm API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router)
app.include_router(sessions.router)
app.include_router(config.router)
app.include_router(documents.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
