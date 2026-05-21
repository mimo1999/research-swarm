from pydantic import BaseModel, Field
from .source import Source


class ReportSection(BaseModel):
    heading: str = Field(..., description="Section heading")
    body_md: str = Field(..., description="Section body in Markdown")
    citations: list[int] = Field(
        default_factory=list,
        description="1-based indices into the FinalReport.references list",
    )


class ReportQualityScore(BaseModel):
    faithfulness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of claims supported by cited sources",
    )
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average cosine similarity between sub-questions and sections",
    )
    completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of sub-questions addressed in the report",
    )

    @property
    def overall(self) -> float:
        return round((self.faithfulness + self.relevance + self.completeness) / 3, 3)


class FinalReport(BaseModel):
    title: str = Field(..., description="Report title")
    exec_summary: str = Field(..., description="Executive summary in Markdown")
    sections: list[ReportSection] = Field(default_factory=list)
    references: list[Source] = Field(
        default_factory=list,
        description="Ordered reference list (1-based citation index)",
    )
    methodology: str = Field(
        default="", description="Description of research methodology"
    )
    limitations: str = Field(
        default="", description="Known limitations of this research"
    )
    quality_score: ReportQualityScore | None = Field(
        default=None, description="Attached after eval phase"
    )
