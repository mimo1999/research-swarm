from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from .critique import Critique
from .finding import Finding
from .frame import QuestionFrame
from .plan import ResearchPlan
from .query import ResearchQuery
from .report import FinalReport


def _add_list(existing: list, new: list) -> list:
    """Reducer that appends new items to the existing list."""
    return existing + new


def _merge_findings(existing: list, new: list) -> list:
    """Merge findings by id -- new items with matching ids overwrite existing ones.

    This lets the verifier return updated Finding objects (same id,
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


def _last_value(existing, new):  # noqa: ARG001
    """Reducer for next_agent: tolerate multiple writes within one step.

    Without an Annotated reducer, next_agent is a plain LastValue channel,
    which raises InvalidUpdateError the instant more than one write lands on
    it in a single step -- even when every write carries the identical value.
    That's exactly what happens when >=2 parallel worker_node/
    document_worker_node branches (Send-fanned in the same step) each
    independently detect the same exhausted budget pool and return
    {"next_agent": "writer", ...}: a real, reproducible crash instead of the
    intended graceful degrade. Last-write-wins is safe here because nothing
    in this graph ever wants two *different* concurrent next_agent values to
    both take effect -- routing is always one node's decision, so tolerating
    concurrent writes (which in practice always agree) just removes the crash.
    """
    return new


AgentName = Literal[
    "supervisor", "verifier", "writer", "dispatch", "collect", "human", "end",
]


class AgentState(TypedDict):
    # Conversation history (uses built-in add_messages reducer)
    messages: Annotated[list, add_messages]

    # Core research objects
    query: ResearchQuery | None
    plan: ResearchPlan | None

    # Findings: merge-by-id so the verifier can overwrite confidence
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
    # Annotated with a reducer (not a plain LastValue field) so that
    # >=1 concurrent Send-fanned branches writing the same value in one
    # step (e.g. several workers all hitting an exhausted budget at once)
    # don't crash the run -- see _last_value.
    next_agent: Annotated[AgentName | None, _last_value]

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

    # How many dispatch→workers→collect cycles have completed.
    research_rounds: NotRequired[int]

    # Finding IDs present just before the most recent dispatch round.
    # collect_node uses this to identify which findings are newly produced.
    pre_dispatch_finding_ids: NotRequired[list[str]]

    # --- Document pass: one-time extraction over uploaded documents ---

    # User-uploaded documents ingested before the graph starts, each
    # {"url", "title", "text", "source_type"}. Populated once in app.py.
    # Consumed exactly once by document_pass_node/route_from_document_pass
    # (packed into extraction batches, one call each) before round-0 dispatch --
    # not re-processed on later rounds since the documents don't change mid-session.
    ingested_documents: NotRequired[list[dict]]

    sub_questions_snapshot: NotRequired[list[str]]

    # --- Paper scout pass: relevance-filtered abstract corpus ---

    # Papers that passed the relevance threshold, each {"sub_question", "url",
    # "title", "snippet" (abstract), "source_type", "score", "credibility_score"}.
    # Append-only: one paper_scout_node per sub-question writes its own slice.
    # Consumed by paper_worker_node, which turns them into findings.
    paper_corpus: NotRequired[Annotated[list[dict], _add_list]]

    # Send payload for paper_scout_node: one {"sub_question", "search_query",
    # "domain"} per plan sub-question. The scout handles all of them in one
    # node so candidate papers are deduplicated and scored in a single pass.
    scout_tasks: NotRequired[list[dict]]

    # --- Evidence-first pipeline (v2) ---

    # The user's research question. Send payloads carry only what they list, so the extraction
    # workers get it through this key; the writer/verifier read it from ``query.topic``.
    topic: NotRequired[str]

    # One packed batch of sources for a single extraction call each
    # {"url", "title", "text", "source_type", "credibility_score"}.
    active_batch: NotRequired[list[dict]]

    # Search query the plan assigned to the sub-question a gap-fill worker handles.
    search_query: NotRequired[str]

    # Question frame (agents/expansion.py) for Send-fanned nodes that do not receive ``plan``
    # (the paper scout), and its key constraint for extraction workers (``scope``).
    frame: NotRequired[QuestionFrame | None]
    scope: NotRequired[str]

    # Set (to the reviewer's text, possibly "") by graph.rework.request_rework when a reviewer
    # asks for more research at the HITL pause; None otherwise. While set, dispatch targets the
    # weakly answered sub-questions; collect_node clears it after that round.
    rework_instructions: NotRequired[str | None]

    # The evidence packet (agents/packet.py, EvidencePacket.to_dict()) in pipeline mode "packet":
    # the numbered source sentences the synthesis call reads. Written once by packet_node.
    evidence_packet: NotRequired[dict | None]

    # Finding-id pairs the verifier found in direct conflict; the writer presents both sides.
    # Overwritten (plain LastValue), written once by verifier_node.
    fact_conflicts: NotRequired[list[list[str]]]
