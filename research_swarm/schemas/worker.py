"""Per-sub-question search assignment produced by the planner."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestionAssignment(BaseModel):
    """Search query and literature domain for one sub-question."""
    sub_question: str  = Field(..., description="The research sub-question")
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
