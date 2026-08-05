from pydantic import BaseModel, Field

from .judge import LLMJudgeResult
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
    # relevance/completeness default to None, not 0.0 -- nothing in the
    # codebase computes them yet (only faithfulness is). A 0.0 default looked
    # identical to "computed and genuinely zero", so the UI rendered a
    # misleading "Relevance: 0%" / "Completeness: 0%" for every report the
    # moment writer.py started always attaching a quality_score. None means
    # "not computed"; report_view.py shows that distinctly instead of a score.
    relevance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average cosine similarity between sub-questions and sections",
    )
    completeness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of sub-questions addressed in the report",
    )

    @property
    def overall(self) -> float:
        """Average of whichever dimensions have actually been computed.

        Averaging in an uncomputed dimension as 0.0 would silently deflate
        this score (e.g. faithfulness=0.9 with relevance/completeness
        uncomputed used to report overall=0.3) -- only populated scores
        count, so `overall` still means "how good are the dimensions we
        actually measured", not "how many dimensions did we measure".
        """
        dims = (self.faithfulness, self.relevance, self.completeness)
        scores = [s for s in dims if s is not None]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 3)


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
    llm_judge: LLMJudgeResult | None = Field(
        default=None, description="Independent LLM review, attached after the writer runs"
    )
