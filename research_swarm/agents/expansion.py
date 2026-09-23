"""Question frame: what makes THIS question specific, extracted once before planning.

A run on "Can we losslessly migrate KV cache from one LLM to another" produced a report about
KV-cache compression: the 2B planner kept "between different LLMs" in its sub-questions but
dropped it from every search query, and no later stage could notice. The frame names that
distinguishing constraint (and how the literature phrases it) so code can enforce it at every
stage -- planning (``supervisor._enforce_plan``), search and scoring (``papers``), coverage
(``nodes._research_targets``), verification and writing -- instead of trusting one LLM call to
carry it through.

  1. ``probe``: search the literal question (no LLM call) so the expander sees the field's own
     terminology in real titles -- a small model may not know "cross-model KV cache transfer".
  2. ``expand_question``: one short structured call -> ``QuestionFrame``.
  3. ``scope_hit``: the tolerant lexical check every stage uses to ask "does this text talk about
     the constraint?".

An empty frame (expansion off, failed, or a question with no qualifier) makes every stage behave
exactly as before.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.agents._utils import (
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.text import terms
from research_swarm.config import settings
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas.frame import QuestionFrame

logger = logging.getLogger(__name__)

__all__ = ["QuestionFrame", "expand_question", "frame_prompt_block", "mentions_any",
           "normalize_frame", "probe", "scope_hit", "subject_stems", "distinctive_phrases"]

MAX_ITEMS = 5
MAX_COMPARE_ITEMS = 6
_PLACEHOLDER_RE = re.compile(
    r"^\W*(none|n/?a|null|nil|no|nothing|empty|not applicable|no (specific |key )?"
    r"(constraint|qualifier)s?( given| specified)?)?\W*$",
    re.IGNORECASE,
)
# A constraint phrasing "matches" a text when this share of its content words appear in it.
SCOPE_MATCH_SHARE = 0.6
# Words are compared by prefix so "models" matches "model" and "sharing" matches "shared".
_PREFIX = 5


_SYSTEM_PROMPT = (
    "You analyse a research question before it is planned and searched.\n"
    "Identify what makes it SPECIFIC: the qualifier that a report on the general subject would\n"
    "miss. Example: for 'Can a KV cache from one LLM be reused by a different LLM?' the subject\n"
    "is KV caching and the key constraint is 'across different LLMs'; KV-cache compression and\n"
    "same-model cache migration are confusable topics, not the question.\n"
    "Fields:\n"
    "- interpretation: one sentence stating what is actually asked.\n"
    "- key_constraint: that qualifier in a few words (empty if the question has none).\n"
    "- constraint_terms: 2-5 ways the literature phrases the constraint. Prefer wording you see\n"
    "  in the search-result titles below.\n"
    "- confusable_topics: 1-4 adjacent topics that share keywords but are not the question.\n"
    "- define_terms: strict qualifiers in the question that set a bar the answer must meet\n"
    "  (e.g. 'lossless', 'guaranteed', 'always', 'causes', 'safe'). NOT the subject's own\n"
    "  names such as 'KV cache' or 'LLM'. Empty if there are none.\n"
    "- proof_criterion: if there are define_terms, 1-3 sentences on what evidence would\n"
    "  establish them and what would not. Distinguish the levels a result can reach: an exact\n"
    "  step inside a method, equality of the outputs with the reference, and equality of the\n"
    "  system's behaviour. Say that approximate evidence (high accuracy, variance explained,\n"
    "  'negligible' loss, speedups) does not establish a strict qualifier. Empty otherwise.\n"
    "- compare_items: the items the question explicitly asks to distinguish or compare, named\n"
    "  as in the question (e.g. ['direct reuse', 'transformation']). Empty if none.\n"
    "- search_queries: 2-3 keyword queries (3-8 words, not sentences) for the whole question,\n"
    "  each containing the key constraint."
    + schema_output_instruction(QuestionFrame)
)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _stems(text: str) -> set[str]:
    return {w[:_PREFIX] for w in terms(text)}


def subject_stems(frame: QuestionFrame | None) -> set[str]:
    """Words of the question that are its general subject, not its constraint."""
    if frame is None or not frame.topic:
        return set()
    return _stems(frame.topic) - _stems(frame.key_constraint)


def mentions_any(text: str, phrases: list[str], ignore: set[str] | None = None) -> bool:
    """True when some phrase has at least ``SCOPE_MATCH_SHARE`` of its content words in *text*
    (prefix-matched), not counting the *ignore* words (the question's general subject). A phrase
    made only of ignored words carries no scope information and never matches: the expander once
    offered "KV Cache Migration" as a phrasing of "from one LLM to another", which would have let
    every KV-cache paper count as in scope."""
    have = _stems(text)
    for phrase in phrases:
        want = _stems(phrase) - (ignore or set())
        if want and len(want & have) >= SCOPE_MATCH_SHARE * len(want):
            return True
    return False


def distinctive_phrases(frame: QuestionFrame) -> list[str]:
    """Scope phrasings that say something beyond the question's general subject (the key
    constraint first); what the planner may append to a query that lost the scope."""
    subject = subject_stems(frame)
    return [p for p in frame.scope_phrases() if _stems(p) - subject]


def scope_hit(text: str, frame: QuestionFrame | None) -> bool:
    """True when *text* talks about the frame's constraint (``mentions_any`` over its scope
    phrases, ignoring the question's subject words). Always True for a frame with no constraint,
    so callers can apply it unconditionally."""
    if frame is None or not frame.has_constraint:
        return True
    return mentions_any(text, frame.scope_phrases(), subject_stems(frame))


def normalize_frame(frame: QuestionFrame, topic: str) -> QuestionFrame:
    """Code post-processing: dedupe, cap lists, and fill the fields downstream stages rely on."""
    from research_swarm.agents.papers import keyword_query

    constraint = " ".join(frame.key_constraint.split())
    if _PLACEHOLDER_RE.match(constraint):
        # A small model writes "none" instead of leaving the field empty; treating that as a
        # constraint appended " (none)" to a boiling-point sub-question.
        constraint = ""
    constraint_terms = _dedupe(frame.constraint_terms)[:MAX_ITEMS]
    queries = _dedupe(frame.search_queries)[:MAX_ITEMS]
    if constraint and not constraint_terms:
        constraint_terms = [constraint]
    if constraint and not queries:
        queries = [f"{keyword_query(topic, 6)} {constraint}"]
    normalized = QuestionFrame(
        interpretation=" ".join(frame.interpretation.split()),
        key_constraint=constraint,
        constraint_terms=constraint_terms,
        confusable_topics=_dedupe(frame.confusable_topics)[:MAX_ITEMS],
        define_terms=_dedupe(frame.define_terms)[:MAX_ITEMS],
        # a proof criterion only means something for a strict qualifier
        proof_criterion=(
            " ".join(frame.proof_criterion.split())
            if frame.define_terms and not _PLACEHOLDER_RE.match(frame.proof_criterion) else ""
        ),
        compare_items=[i for i in _dedupe(frame.compare_items)
                       if not _PLACEHOLDER_RE.match(i)][:MAX_COMPARE_ITEMS],
        search_queries=queries if constraint else [],
        topic=topic,
    )
    # A "not the question" topic that is itself about the constraint is the question: nemotron
    # once listed "cross-model KV cache sharing" as confusable for a cross-model question, which
    # told the relevance scorer to reject exactly the papers wanted (DroidSpeak, ICaRus, KVLink).
    if normalized.has_constraint:
        normalized.confusable_topics = [
            t for t in normalized.confusable_topics if not scope_hit(t, normalized)
        ]
    return normalized


async def probe(topic: str, session_id: str | None) -> list[dict]:
    """Search the literal question on the routed tools (no LLM call); at most
    ``settings.probe_results`` deduplicated hits. Empty on any failure."""
    from research_swarm.agents.papers import (
        interleave,
        keyword_query,
        routed_tools_union,
        search_task,
        tool_registry,
    )

    try:
        available = tool_registry()
        names = routed_tools_union("other", topic, available)
        ranked = await search_task(
            topic, keyword_query(topic, max_terms=10), names, available,
            settings.fetch_pass_results_per_tool, session_id or "default",
        )
        return interleave(ranked, settings.probe_results)
    except Exception as exc:  # noqa: BLE001 -- the probe is an aid, never a blocker
        logger.warning("Expansion probe failed (%s)", exc)
        return []


async def expand_question(
    topic: str, probe_hits: list[dict], llm: BaseChatModel, session_id: str | None,
) -> QuestionFrame:
    """One structured call -> normalized ``QuestionFrame``; an empty frame on failure."""
    titles = "\n".join(
        f"[{n}] {h.get('title', '')}" for n, h in enumerate(probe_hits, 1) if h.get("title")
    ) or "(no results)"
    user = HumanMessage(content=(
        f"Research question: {topic}\n\nSearch-result titles for the literal question:\n{titles}"
    ))
    try:
        frame: Any = await ainvoke_with_retry(
            llm.with_structured_output(QuestionFrame),
            [SystemMessage(content=_SYSTEM_PROMPT), user],
            session_id=session_id, agent="expansion",
        )
    except Exception as exc:  # noqa: BLE001
        frame = recover_from_parse_failure(exc, QuestionFrame)
        if frame is None:
            logger.error("Query expansion failed (%s) -- planning without a frame.", exc)
            trace_event(session_id, "expansion.fallback", "note",
                        error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return QuestionFrame()
    if not isinstance(frame, QuestionFrame):
        trace_event(session_id, "expansion.fallback", "note", error="wrong output type")
        return QuestionFrame()
    frame = normalize_frame(frame, topic)
    trace_event(session_id, "expansion.frame", "note",
                **frame.model_dump(exclude={"probe_hits"}),
                probe_titles=[h.get("title", "") for h in probe_hits])
    return frame.model_copy(update={"probe_hits": list(probe_hits)})


def frame_prompt_block(frame: QuestionFrame | None) -> str:
    """The frame as planner / writer context; empty string for an empty frame."""
    if frame is None or not (frame.has_constraint or frame.interpretation):
        return ""
    lines = []
    if frame.interpretation:
        lines.append(f"Interpretation: {frame.interpretation}")
    if frame.has_constraint:
        lines.append(
            f"Key constraint (every sub-question and search_query must stay within it): "
            f"{frame.key_constraint}"
        )
    if frame.constraint_terms:
        lines.append(f"Terms the literature uses: {'; '.join(frame.constraint_terms)}")
    if frame.confusable_topics:
        lines.append(
            f"NOT this question (do not plan sub-questions about these): "
            f"{'; '.join(frame.confusable_topics)}"
        )
    if frame.define_terms:
        lines.append(f"Terms to define precisely: {'; '.join(frame.define_terms)}")
    if frame.proof_criterion:
        lines.append(f"What would establish them: {frame.proof_criterion}")
    if frame.compare_items:
        lines.append(f"Items to compare (cover each): {'; '.join(frame.compare_items)}")
    return "\n".join(lines) + "\n"
