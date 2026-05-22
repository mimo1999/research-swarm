from typing import Annotated, Literal, NotRequired

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .critique import Critique
from .finding import Finding
from .plan import ResearchPlan
from .query import ResearchQuery
from .report import FinalReport


def _add_list(existing: list, new: list) -> list:
    """Reducer that appends new items to the existing list."""
    return existing + new


def _merge_findings(existing: list, new: list) -> list:
    """Merge findings by id -- new items with matching ids overwrite existing ones.

    This lets the fact-checker return updated Finding objects (same id,
    revised confidence) without duplicating the list.
    """
    merged: dict = {}
    for f in existing:
        key = f["id"] if isinstance(f, dict) else f.id
        merged[key] = f
    for f in new:
        key = f["id"] if isinstance(f, dict) else f.id
        merged[key] = f
    return list(merged.values())


AgentName = Literal[
    "supervisor", "researcher", "critic", "writer", "fact_checker", "human", "end"
]


class AgentState(TypedDict):
    # Conversation history (uses built-in add_messages reducer)
    messages: Annotated[list, add_messages]

    # Core research objects
    query: ResearchQuery | None
    plan: ResearchPlan | None

    # Findings: merge-by-id so fact_checker can overwrite confidence
    findings: Annotated[list[Finding], _merge_findings]
    # Critiques: append-only (one critique per finding per pass)
    critiques: Annotated[list[Critique], _add_list]

    # Reports
    draft_report: FinalReport | None
    final_report: FinalReport | None

    # Human-in-the-loop feedback strings:
    #   human_feedback      -- consumed by the researcher for re-research passes
    #   writer_instructions -- consumed by the writer for report revisions (HITL)
    human_feedback: str | None
    writer_instructions: NotRequired[str | None]

    # Routing & control
    iteration_count: int
    next_agent: AgentName | None

    # Session identifier for persistence
    session_id: str

    # Per-run model settings; omitted in tests and older checkpoints.
    model_provider: NotRequired[str]
    model_name: NotRequired[str]
