from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    web = "web"
    arxiv = "arxiv"
    pdf = "pdf"
    retriever = "retriever"


class Source(BaseModel):
    url: str = Field(..., description="URL or identifier for the source")
    title: str = Field(default="", description="Title of the source document")
    snippet: str = Field(default="", description="Relevant excerpt from the source")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_type: SourceType = Field(default=SourceType.web)
    credibility_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated credibility (0=low, 1=high)",
    )
