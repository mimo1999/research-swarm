from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from .query import ResearchQuery
from .plan import ResearchPlan
from .finding import Finding
from .critique import Critique
from .report import FinalReport


def _add_list(existing: list, new: list) -> list:
    """Reducer that appends new items to the existing list."""
    return existing + new


def _merge_findings(existing: list, new: list) -> list:
    """Merge findings by id — new items with matching ids overwrite existing ones.

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

    # Human-in-the-loop feedback string
    human_feedback: str | None

    # Routing & control
    iteration_count: int
    next_agent: AgentName | None

    # Session identifier for persistence
    session_id: str
