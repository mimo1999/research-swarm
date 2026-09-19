"""Worker role definitions for heterogeneous parallel research dispatch."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from research_swarm.utils.compat import StrEnum


class WorkerRole(StrEnum):
    """Research perspective assigned to each parallel worker."""
    general    = "general"    # balanced web + arxiv + pubmed
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
    search_query: str = Field(
        default="",
        description=(
            "3-8 word keyword query for literature search (the field's standard terms, "
            "NOT a full sentence)"
        ),
    )
    domain: Literal["biomedical", "cs_ml_physics_math", "other"] = Field(
        default="other",
        description=(
            "Which literature this sub-question lives in: biomedical (medicine, biology, "
            "clinical), cs_ml_physics_math (computer science, ML, physics, math, engineering "
            "preprints), or other (industry, business, policy, everything else)"
        ),
    )
