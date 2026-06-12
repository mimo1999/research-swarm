from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CritiqueVerdict(StrEnum):
    supported = "supported"
    weak = "weak"
    refuted = "refuted"


class Critique(BaseModel):
    # Optional so Pydantic accepts null from the LLM; critic.py always overwrites
    # with the real UUID via model_copy before appending to the critiques list.
    finding_id: str | None = Field(None, description="ID of the Finding being critiqued")
    verdict: CritiqueVerdict = Field(..., description="Evaluation verdict")
    reasoning: str = Field(..., description="Explanation for the verdict")
    suggested_followup: str = Field(
        default="",
        description="Follow-up research question if verdict is weak or refuted",
    )
