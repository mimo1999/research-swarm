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
        description="Confidence in the claim (set by the verifier)",
    )
    sub_question: str = Field(
        default="", description="The sub-question this finding addresses"
    )
    grounding: str = Field(
        default="unknown",
        description=(
            "How the evidence snippet was obtained: quote (the model's quote was found in the "
            "source), passage (best lexical passage for the claim), none (no supporting text), "
            "unknown (a path that does not ground). Treat unknown like passage."
        ),
    )
    quote: str = Field(
        default="",
        description=(
            "The exact source text the fact rests on, as located in the source: the model's "
            "quote (grounding=quote) or the matched passage (grounding=passage); empty otherwise. "
            "Kept for evaluation against gold evidence (benchmarks/score_rationales.py)."
        ),
    )
    relevance: str = Field(
        default="unknown",
        description=(
            "Whether the fact answers the question as asked: direct, background (context on the "
            "general subject, outside the question's specific scope), off_topic, or unknown. "
            "Set by the extractor, revised by the verifier."
        ),
    )
