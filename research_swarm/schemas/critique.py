from enum import Enum
from pydantic import BaseModel, Field


class CritiqueVerdict(str, Enum):
    supported = "supported"
    weak = "weak"
    refuted = "refuted"


class Critique(BaseModel):
    finding_id: str = Field(..., description="ID of the Finding being critiqued")
    verdict: CritiqueVerdict = Field(..., description="Evaluation verdict")
    reasoning: str = Field(..., description="Explanation for the verdict")
    suggested_followup: str = Field(
        default="",
        description="Follow-up research question if verdict is weak or refuted",
    )
