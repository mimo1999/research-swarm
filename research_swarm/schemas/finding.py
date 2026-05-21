from uuid import uuid4

from pydantic import BaseModel, Field

from .source import Source


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str = Field(..., description="The research claim or fact discovered")
    evidence: list[Source] = Field(
        default_factory=list, description="Sources supporting this claim"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the claim (updated by fact-checker)",
    )
    sub_question: str = Field(
        default="", description="The sub-question this finding addresses"
    )
