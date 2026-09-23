"""QuestionFrame: what makes a research question specific (see agents/expansion.py)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class QuestionFrame(BaseModel):
    interpretation: str = Field(default="", description="One sentence: what is actually asked")
    key_constraint: str = Field(
        default="",
        description=(
            "The qualifier that separates this question from its general subject, e.g. "
            "'across different LLMs'. Empty if the question has none."
        ),
    )
    constraint_terms: list[str] = Field(
        default_factory=list,
        description="2-5 phrasings of the key constraint as the literature words it",
    )
    confusable_topics: list[str] = Field(
        default_factory=list,
        description="Adjacent topics that share keywords but are NOT this question",
    )
    define_terms: list[str] = Field(
        default_factory=list,
        description="Strict qualifiers in the question that set a bar the answer must meet, "
                    "e.g. 'lossless' -- not the subject's own names",
    )
    proof_criterion: str = Field(
        default="",
        description="For a strict qualifier: what evidence would establish it (e.g. identical "
                    "outputs vs the reference, not just high accuracy), and what does NOT. Empty "
                    "if the question has no strict qualifier",
    )
    compare_items: list[str] = Field(
        default_factory=list,
        description="Items the question explicitly asks to distinguish or compare, as named in "
                    "the question. Empty if it asks for no comparison",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2-3 keyword queries (3-8 words) for the whole question, each containing "
                    "the key constraint",
    )
    # Set by code: the probe search's hits for the literal question, added to every
    # sub-question's candidate pool by the paper scout. Hidden from the expander's schema.
    probe_hits: SkipJsonSchema[list[dict[str, Any]]] = Field(default_factory=list)
    # Set by code: the question it was built from. Its general-subject words ("KV cache") are
    # ignored when matching scope phrases, so "KV cache sharing across models" is not matched by
    # any text that merely mentions KV caches.
    topic: SkipJsonSchema[str] = ""

    @property
    def has_constraint(self) -> bool:
        return bool(self.key_constraint.strip())

    def scope_phrases(self) -> list[str]:
        """The key constraint and its phrasings (what ``scope_hit`` looks for)."""
        out: list[str] = []
        seen: set[str] = set()
        for item in [self.key_constraint, *self.constraint_terms]:
            text = " ".join(str(item).split())
            if text and text.lower() not in seen:
                seen.add(text.lower())
                out.append(text)
        return out
