from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(
        ..., description="Decomposed sub-questions to answer"
    )
    strategy: str = Field(
        ..., description="High-level strategy for conducting the research"
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tool names needed (e.g. web_search, arxiv, retriever)",
    )
