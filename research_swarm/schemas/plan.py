from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from .frame import QuestionFrame
from .worker import SubQuestionAssignment, WorkerRole


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(
        ..., description="Decomposed sub-questions to answer"
    )
    strategy: str = Field(
        ..., description="High-level strategy for conducting the research"
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tool names needed (e.g. web_search, arxiv, pubmed)",
    )
    complexity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated complexity 0–1. "
            "0 = single-fact lookup, 1 = deep multi-faceted investigation. "
            "Used to determine worker_count = ceil(score * max_workers_per_depth)."
        ),
    )
    assignments: list[SubQuestionAssignment] = Field(
        default_factory=list,
        description=(
            "Per-sub-question worker-role assignments. "
            "When empty the dispatch node falls back to WorkerRole.general for all questions."
        ),
    )

    # Set in code by the supervisor (agents/expansion.py), never by the planning LLM: the
    # question's distinguishing constraint, enforced by every later stage. Defaulted, so older
    # checkpoints still load. SkipJsonSchema keeps it out of the schema the planner is shown.
    frame: SkipJsonSchema[QuestionFrame | None] = None

    def role_for(self, sub_question: str) -> WorkerRole:
        """Return the assigned WorkerRole for *sub_question*, defaulting to general."""
        for a in self.assignments:
            if a.sub_question.strip().lower() == sub_question.strip().lower():
                return a.worker_role
        return WorkerRole.general

    def assignment_for(self, sub_question: str) -> SubQuestionAssignment | None:
        """Return the assignment for *sub_question* (case/whitespace-insensitive), if any."""
        key = sub_question.strip().lower()
        for a in self.assignments:
            if a.sub_question.strip().lower() == key:
                return a
        return None
