from __future__ import annotations

from pydantic import BaseModel, Field

from research_swarm.utils.compat import StrEnum


class JudgeVerdict(StrEnum):
    approve = "approve"
    revise = "revise"
    reject = "reject"


class LLMJudgeResult(BaseModel):
    """Independent LLM review of a generated report.

    Distinct from ``ReportQualityScore`` (embedding cosine similarity) -- this
    is a model reading the report itself and judging things embeddings can't:
    whether it answers the right question, covers every sub-question, reads
    coherently, and cites sources that actually exist.
    """

    coherence: int = Field(
        ..., ge=1, le=5, description="Clarity and logical organization of the writing"
    )
    relevance: int = Field(
        ..., ge=1, le=5, description="How directly the report addresses the research topic"
    )
    completeness: int = Field(
        ..., ge=1, le=5, description="Fraction of sub-questions substantively addressed"
    )
    citation_quality: int = Field(
        ..., ge=1, le=5,
        description="Whether claims are backed by [N] citations pointing at real, listed refs",
    )
    verdict: JudgeVerdict = Field(..., description="Overall recommendation")
    reasoning: str = Field(..., description="Concrete gaps found, or 'no significant issues'")

    @property
    def overall(self) -> float:
        criteria = self.coherence + self.relevance + self.completeness + self.citation_quality
        return round(criteria / 4, 2)
