from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchDepth(StrEnum):
    shallow = "shallow"
    standard = "standard"
    deep = "deep"


class ResearchQuery(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The research topic or question",
    )
    depth: ResearchDepth = Field(
        default=ResearchDepth.standard,
        description="How deeply to research the topic",
    )
    max_sources: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum number of sources to retrieve",
    )
    audience: str = Field(
        default="general",
        description="Target audience (e.g. general, technical, academic, executive)",
    )
