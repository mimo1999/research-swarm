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
    "supervisor", "researcher", "critic", "writer", "fact_checker",
    "dispatch", "collect", "human", "end",
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
    #   human_feedback      -- consumed by the dispatcher for re-research passes
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

    # Incremented when AgentState fields change in a breaking way.
    # Older checkpoints that lack this field are treated as version 0.
    schema_version: NotRequired[int]

    # --- Phase 4: parallel dispatch fields ---

    # Per-worker state injected via Send; cleared after each dispatch round.
    active_sub_question: NotRequired[str | None]
    active_worker_role:  NotRequired[str | None]

    # How many dispatch→workers→collect cycles have completed.
    research_rounds: NotRequired[int]

    # Finding IDs present just before the most recent dispatch round.
    # collect_node uses this to identify which findings are newly produced.
    pre_dispatch_finding_ids: NotRequired[list[str]]

    # How many times each sub-question (normalised, lowercased) has been sent
    # back for rework after a weak/refuted critique verdict. Capped by
    # settings.max_rework_attempts so one persistently-bad finding can't
    # consume unbounded rework rounds. Incremented in collect_node once a
    # rework round completes; read by critic_node (loop-back decision) and
    # _research_targets (dispatch/route_from_dispatch fan-out).
    rework_counts: NotRequired[dict[str, int]]

    # --- Document pass: one-time full-document extraction, no Chroma ---

    # User-uploaded documents ingested before the graph starts, each
    # {"url", "title", "text", "source_type"}. Populated once in app.py.
    # Consumed exactly once by document_pass_node/route_from_document_pass
    # (fans out one worker per document, or per size-bounded slice of an
    # oversized one) before round-0 dispatch -- not re-processed on later
    # rounds since the documents themselves don't change mid-session.
    ingested_documents: NotRequired[list[dict]]

    # Per-document-worker state injected via Send; cleared after that
    # worker's single call (mirrors active_sub_question/active_worker_role
    # above, but for the document pass instead of the sub-question dispatch).
    active_document:        NotRequired[dict | None]
    active_doc_part_text:   NotRequired[str | None]
    active_doc_part_index:  NotRequired[int]
    active_doc_part_total:  NotRequired[int]
    sub_questions_snapshot: NotRequired[list[str]]
