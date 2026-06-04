"""Worker role definitions for heterogeneous parallel research dispatch."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class WorkerRole(StrEnum):
    """Research perspective assigned to each parallel worker."""
    general    = "general"    # balanced web + arxiv + RAG
    academic   = "academic"   # prioritises arXiv, DOI sources, peer-reviewed papers
    industry   = "industry"   # prioritises web search, company blogs, case studies
    skeptic    = "skeptic"    # actively seeks counter-evidence and limitations
    benchmark  = "benchmark"  # seeks quantitative comparisons, metrics, evaluations


class SubQuestionAssignment(BaseModel):
    """Maps a single sub-question to the worker role best suited to answer it."""
    sub_question: str  = Field(..., description="The research sub-question")
    worker_role:  WorkerRole = Field(
        default=WorkerRole.general,
        description="Which worker perspective should answer this sub-question",
    )
