"""Supervisor agent -- decides which agent to invoke next."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from research_swarm.agents._utils import ainvoke_with_retry, schema_output_instruction
from research_swarm.agents.expansion import (
    distinctive_phrases,
    expand_question,
    frame_prompt_block,
    probe,
    scope_hit,
)
from research_swarm.agents.question import QuestionSpec, is_meta_sub_question, parse_question
from research_swarm.config import settings
from research_swarm.runtime.limits import current_llm_session
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas import ResearchPlan
from research_swarm.schemas.frame import QuestionFrame
from research_swarm.schemas.state import AgentName

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState


# Sub-question count by depth — controls how many sub-questions the LLM
# generates in the initial research plan (1 worker per sub-question). Fixed
# exactly, not a ceiling: "AT MOST N" gave the model no pressure to actually
# use the budget -- observed producing a single sub-question at standard
# depth for a topic explicitly asking to compare two named techniques,
# silently dropping one side of the comparison entirely. An exact count
# removes that decision from the model altogether.
# standard reverted 5 -> 4: raising it (alongside more tool turns per worker)
# pushed a live run's total LLM-call usage to 49 against a 40 budget. Fewer
# workers keeps a session's call volume predictable.
# The counts live in settings.depth_profiles ("sub_questions"). Each sub-question costs one
# relevance-scoring call and one extraction call (plus a gap-fill worker when its coverage is
# thin): about 55 s of local gemma4 time, ~10 s on the cloud model.


def _sub_questions_for(depth: str) -> int:
    return settings.for_depth("sub_questions", depth)


def _build_system_prompt(depth: str = "standard") -> str:
    """Return a supervisor system prompt focused on plan creation only."""
    n_sq = _sub_questions_for(depth)
    return (
        "You are the Supervisor of a multi-agent research system.\n"
        "Your ONLY job in this call is to create the initial research plan.\n\n"
        "Worker roles available for assignment:\n"
        "  general   -- balanced web + arXiv + RAG research\n"
        "  academic  -- prioritises peer-reviewed papers and arXiv pre-prints\n"
        "  industry  -- prioritises real-world deployment and case studies\n"
        "  skeptic   -- actively seeks counter-evidence and known failure modes\n"
        "  benchmark -- seeks quantitative comparisons and empirical metrics\n\n"
        f"Generate EXACTLY {n_sq} sub-questions (depth = {depth!r}).\n"
        "If the topic asks to compare, contrast, or evaluate differences between two or "
        "more named things (e.g. \"X vs Y\", \"differences between A and B\"), your "
        "sub-questions MUST cover each thing individually AND their direct comparison -- "
        "never collapse a comparison topic into sub-questions about only one side. For "
        "shallow depth (a single sub-question), phrase that one question to address the "
        "comparison directly (e.g. \"How do X and Y differ in Z?\") rather than "
        "researching only one of the things being compared.\n"
        "Assign a worker role to each sub-question based on the angle that best answers it.\n"
        "For each assignment also set `search_query` -- a 3-8 word keyword query in the "
        "field's standard terminology (NOT a sentence) that would retrieve the best papers "
        "for that sub-question -- and `domain`: \"biomedical\" (medicine, biology, clinical), "
        "\"cs_ml_physics_math\" (computer science, ML, physics, math, engineering) or "
        "\"other\" (industry, business, policy, everything else).\n"
        "For shallow depth, always use role 'general'.\n"
        "If a key constraint is given below, every sub-question and every search_query must "
        "stay within it (include its wording); never plan sub-questions about the topics "
        "listed as NOT this question. If terms to define are given and there is more than one "
        "sub-question, make one sub-question define them in the question's context.\n"
        "Set complexity_score 0.0–1.0 (0 = single-fact, 1 = deep multi-faceted).\n\n"
        "Always set `plan`. Leave `next_agent` as \"dispatch\"."
        + schema_output_instruction(SupervisorDecision)
    )


def _has_plan(decision: SupervisorDecision) -> bool:
    return decision.plan is not None and bool(decision.plan.sub_questions)


def _drop_meta_sub_questions(plan: ResearchPlan, spec: QuestionSpec) -> ResearchPlan:
    """Remove sub-questions about the answer format (naming a label, "classify ..."); if none
    are left, research the question's content directly."""
    keep = [sq for sq in plan.sub_questions if not is_meta_sub_question(sq, spec)]
    if len(keep) == len(plan.sub_questions):
        return plan
    trace_event(
        current_llm_session.get(), "supervisor.meta_dropped", "note",
        dropped=[sq for sq in plan.sub_questions if sq not in keep],
    )
    if not keep:
        keep = [spec.content]
    kept = {sq.strip().lower() for sq in keep}
    assignments = [a for a in plan.assignments if a.sub_question.strip().lower() in kept]
    if not assignments and plan.assignments:
        # keep the first assignment's role/query for the replacement question
        assignments = [plan.assignments[0].model_copy(update={"sub_question": keep[0]})]
    return plan.model_copy(update={"sub_questions": keep, "assignments": assignments})


def _enforce_plan(plan: ResearchPlan, frame: QuestionFrame, n_sq: int) -> ResearchPlan:
    """Code-side guarantees on the planner's output (a 2B planner ignores prompt rules):

    * at most *n_sq* sub-questions (the depth's budget; "EXACTLY 1" once came back as 3);
    * every sub-question and every search query stays within the frame's key constraint -- a
      sub-question missing it gets `` (<constraint>)`` appended, a query gets the shortest
      constraint phrasing appended. (The KV-cache run lost "between different LLMs" from all
      three queries and so never searched for cross-model work.)

    The frame is attached to the plan so later stages can enforce it too.
    """
    from research_swarm.agents.papers import keyword_query
    from research_swarm.schemas.worker import SubQuestionAssignment

    session_id = current_llm_session.get()
    sub_questions = list(plan.sub_questions)
    if n_sq > 0 and len(sub_questions) > n_sq:
        trace_event(session_id, "supervisor.truncated", "note",
                    dropped=sub_questions[n_sq:], kept=n_sq)
        sub_questions = sub_questions[:n_sq]

    by_key = {a.sub_question.strip().lower(): a for a in plan.assignments}
    if not frame.has_constraint:
        kept = {sq.strip().lower() for sq in sub_questions}
        return plan.model_copy(update={
            "sub_questions": sub_questions, "frame": frame,
            "assignments": [a for a in plan.assignments
                            if a.sub_question.strip().lower() in kept],
        })
    shortest = min(distinctive_phrases(frame) or [frame.key_constraint], key=len)
    changed_sq: list[str] = []
    changed_q: list[str] = []
    new_sqs: list[str] = []
    assignments: list[SubQuestionAssignment] = []
    for sq in sub_questions:
        assignment = by_key.get(sq.strip().lower()) or SubQuestionAssignment(sub_question=sq)
        new_sq = sq
        query = assignment.search_query.strip() or keyword_query(sq)
        if not scope_hit(sq, frame):
            new_sq = f"{sq.rstrip()} ({frame.key_constraint})"
            changed_sq.append(new_sq)
        if not scope_hit(query, frame):
            query = f"{query} {shortest}"
            changed_q.append(query)
        new_sqs.append(new_sq)
        assignments.append(assignment.model_copy(
            update={"sub_question": new_sq, "search_query": query},
        ))
    if changed_sq or changed_q:
        trace_event(session_id, "supervisor.scope_enforced", "note",
                    sub_questions=changed_sq, queries=changed_q,
                    constraint=frame.key_constraint)
    return plan.model_copy(update={
        "sub_questions": new_sqs, "assignments": assignments, "frame": frame,
    })


_PLAN_FIELDS = ("sub_questions", "strategy", "required_tools", "complexity_score", "assignments")


class SupervisorDecision(BaseModel):
    # Defaulted like next_agent: it is only logged, and nemotron once omitted it, failing an
    # otherwise complete plan into the one-question fallback.
    reasoning:  str          = Field(default="", description="Brief explanation")
    # Defaulted: code always routes to dispatch after planning (run_supervisor overrides it), and
    # nemotron once omitted it, which failed an otherwise good plan into the fallback.
    next_agent: AgentName    = Field(default="dispatch", description="Which agent to run next")
    plan:       ResearchPlan | None = Field(
        default=None,
        description="The research plan. ALWAYS provide it: without it no research happens",
    )

    @model_validator(mode="before")
    @classmethod
    def _lift_plan_fields(cls, data: Any) -> Any:
        """Accept plan fields written at the top level (nemotron put `assignments` and
        `strategy` beside `plan` instead of inside it): move them into `plan` where it lacks
        them, or build `plan` from them when it is missing."""
        if not isinstance(data, dict):
            return data
        loose = {k: data[k] for k in _PLAN_FIELDS if k in data}
        if not loose:
            return data
        plan = data.get("plan")
        if isinstance(plan, dict):
            merged = {**loose, **{k: v for k, v in plan.items() if v not in (None, [], "")}}
        elif plan is None and "sub_questions" in loose:
            merged = {"strategy": "", **loose}
        else:
            return data
        rest = {k: v for k, v in data.items() if k not in _PLAN_FIELDS}
        return {**rest, "plan": merged}


async def run_supervisor(state: AgentState, llm: BaseChatModel) -> SupervisorDecision:
    """Create the initial research plan (LLM call) or route deterministically.

    After Phase 4 the supervisor is invoked only once — at session start — to
    produce the plan with sub-question assignments.  All subsequent routing
    decisions are handled deterministically inside collect_node and the graph
    edge functions, so the LLM is never called again for routing purposes.
    """
    # If a plan already exists, return a deterministic decision without any LLM call.
    if state.get("plan") is not None:
        return SupervisorDecision(
            reasoning="Plan already exists; routing to dispatch.",
            next_agent="dispatch",
        )

    # --- Plan creation via LLM ---
    query     = state.get("query")
    depth_str = str(
        query.depth.value if hasattr(query.depth, "value") else query.depth
    ) if query else "standard"

    # Plan the subject matter only: an answer-format instruction ("classify as SUPPORT /
    # CONTRADICT ...") in the topic used to become research sub-questions about classification.
    spec = parse_question(query.topic if query else "")
    format_note = (
        f"Answer format (for the final writer only; do NOT plan sub-questions about it): "
        f"{spec.instruction}\n" if spec.instruction else ""
    )
    # The question frame: probe-search the literal question, then extract its distinguishing
    # constraint (agents/expansion.py). An empty frame changes nothing downstream.
    frame = QuestionFrame()
    if settings.query_expansion_enabled and spec.content:
        session_id = current_llm_session.get()
        hits = await probe(spec.content, session_id)
        frame = await expand_question(spec.content, hits, llm, session_id)
    n_sq = _sub_questions_for(depth_str)
    structured_llm = llm.with_structured_output(SupervisorDecision)
    messages = [
        SystemMessage(content=_build_system_prompt(depth_str)),
        HumanMessage(
            content=(
                f"Research topic: {spec.content or 'unknown'}\n"
                + format_note
                + frame_prompt_block(frame)
                + f"Depth: {depth_str}\n"
                f"Audience: {query.audience if query else 'general'}\n\n"
                "Create the research plan now."
            )
        ),
    ]
    try:
        decision = await ainvoke_with_retry(structured_llm, messages, agent="supervisor")
        if not _has_plan(decision):
            # Valid JSON but no `plan` (seen with thinking off: the model treats the plan as
            # optional). Without a plan the graph skips all research and the writer emits an empty
            # report, so ask once more, explicitly, before falling back.
            logger.warning("Supervisor returned no plan -- retrying once with a reminder.")
            trace_event(current_llm_session.get(), "supervisor.retry_no_plan", "note")
            reminder = HumanMessage(
                content=(
                    "Your previous answer had no `plan`. Answer again with the JSON object and "
                    "a complete `plan` (sub_questions, strategy, assignments): it is required."
                )
            )
            decision = await ainvoke_with_retry(
                structured_llm, [*messages, reminder], agent="supervisor",
            )
        if not _has_plan(decision):
            raise ValueError("supervisor returned no usable plan after a retry")
        # Enforce next_agent = "dispatch" regardless of what the LLM returned.
        return SupervisorDecision(
            reasoning=decision.reasoning,
            next_agent="dispatch",
            plan=_enforce_plan(_drop_meta_sub_questions(decision.plan, spec), frame, n_sq),
        )
    except Exception as exc:
        topic = spec.content or "research topic"
        # Loud on purpose: a one-question fallback plan silently degrades the whole run, so it
        # is logged at ERROR, traced, and named in the node's message (reasoning) below.
        logger.error(
            "Supervisor LLM failed (%s: %s) — using fallback plan.",
            type(exc).__name__, exc,
        )
        trace_event(
            current_llm_session.get(), "supervisor.fallback", "note",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        from research_swarm.schemas.worker import SubQuestionAssignment
        return SupervisorDecision(
            reasoning=(
                f"FALLBACK PLAN: the planning LLM call failed ({type(exc).__name__}); "
                "researching the topic as a single question."
            ),
            next_agent="dispatch",
            plan=_enforce_plan(ResearchPlan(
                sub_questions=[topic],
                strategy="Direct research of the main topic",
                required_tools=["web_search"],
                complexity_score=0.3,
                assignments=[SubQuestionAssignment(
                    sub_question=topic,
                    search_query=(frame.search_queries or [""])[0],
                )],
            ), frame, 1),
        )



