"""Writer agent -- synthesises validated findings into a FinalReport."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.agents._utils import (
    _field,
    _latest_verdicts,
    ainvoke_with_retry,
    schema_output_instruction,
)
from research_swarm.runtime.trace import trace_event
from research_swarm.schemas import FinalReport, ReportSection, Source
from research_swarm.schemas.critique import CritiqueVerdict

if TYPE_CHECKING:
    from research_swarm.schemas.state import AgentState

logger = logging.getLogger(__name__)

def _fixed_structure(kind_of_report: str, audience: str) -> str:
    """Structure guidance for a report type with fixed sections (writer_render.REPORT_SECTIONS),
    so the single-call and sectioned writers can never disagree about the sections."""
    from research_swarm.agents.writer_render import REPORT_SECTIONS

    lines = [f"Write this as {kind_of_report}. Use exactly these section headings, in this "
             "order, spelled verbatim so they render correctly:"]
    for i, spec in enumerate(REPORT_SECTIONS[audience], 1):
        lines.append(f'  {i}. "{spec.heading}" -- {spec.purpose}')
    lines.append("Do not write a Citations or References section yourself -- the reference list "
                 "is generated automatically. Omit a section that no fact supports rather than "
                 "filling it with unrelated material.")
    return "\n".join(lines)


# Report shape per audience. Headings stay free text in the schema (WriterDraft/ReportSection);
# this only steers what the model writes into them -- writer_render.py's _canonicalize_heading
# and the executive section-drop are the code-side backstop for a model that doesn't comply
# exactly. Selected via the "Audience" dropdown in app.py (ResearchQuery.audience).
_AUDIENCE_STRUCTURE: dict[str, str] = {
    "academic": _fixed_structure("an academic paper", "academic"),
    "technical": _fixed_structure("a technical report", "technical"),
    "executive": (
        "Write this as a one-minute executive summary. Leave `sections` EMPTY. Put everything "
        "-- the direct answer plus the 3-5 most decision-relevant supporting points -- into "
        "`summary`, as one dense paragraph. No headings, no per-topic breakdown."
    ),
    "general": (
        "Write this as a general-audience article: flowing prose, no dry academic headings. "
        "Pick short, plain-language section headings that describe what each part covers (not "
        "restated sub-questions), and write in an engaging, accessible tone while staying "
        "accurate to the evidence."
    ),
}


def structure_guidance(audience: str) -> str:
    """The report-shape instructions for *audience* (writer.py's audience dropdown values),
    falling back to the general/article shape for anything unrecognized."""
    return _AUDIENCE_STRUCTURE.get(
        (audience or "general").strip().lower(), _AUDIENCE_STRUCTURE["general"],
    )


_SYSTEM_PROMPT = """\
You are an expert Research Writer. Produce a comprehensive, well-structured report
from the provided research findings.

Guidelines:
  - The FIRST sentence of exec_summary must directly answer the research question.
  - If the question asks for a specific answer format (a label such as SUPPORT /
    CONTRADICT / NOT_ENOUGH_INFO, a yes/no, a name, a number, a list), give the answer
    in exactly that format in that first sentence.
  - Only say the evidence is insufficient if NO finding addresses the question. If
    findings partially answer it, give the best-supported answer and state what is
    uncertain.
  - Cite sources using [N] notation where N is the 1-based index in the references list.
  - Be accurate: only include claims supported by the evidence.
  - Preserve exact numbers from the findings below -- effect sizes, percentages,
    sample sizes, p-values, confidence intervals, dosages, durations. If a finding
    states "-3.5 points (95% CI -6.7 to -0.3; p=0.03)", write that, not "showed
    improvement". Collapsing a quantitative result into a qualitative gist during
    synthesis is a bigger loss than any single missing detail below -- the numbers
    ARE the finding.
  - Cite precisely, not in bulk. Each finding below lists its sources as
    "[N] Title" so you can tell which source plausibly covers which specific
    number or sub-claim. When a finding bundles several distinct facts from
    different sources, attach each sentence only the citation(s) that actually
    support IT -- do not tack the finding's entire source list onto every
    sentence derived from it. If you genuinely cannot tell which source backs a
    specific number, cite only the source(s) whose title plausibly covers that
    claim rather than citing all of them defensively.
  - Before presenting two quantitative results side by side as a comparison
    (e.g. "X vs Y", a before/after, a table row), verify they were actually
    measured under comparable conditions -- same model size/scale, same
    benchmark or task, same evaluation setup. Findings may bundle numbers
    gathered from sources covering different scales or setups; if the
    conditions don't match, either compare only the matched-scale figures or
    state plainly that the comparison isn't apples-to-apples (e.g. "not
    directly comparable -- measured on a much smaller model") instead of
    presenting mismatched figures as if they were equivalent.
  - Ground every claim in the source excerpts. Each reference below is listed
    with an excerpt of its actual text. A section is only as good as its
    support in those excerpts -- if an excerpt doesn't back a sentence you
    want to write, either cite a source that does, qualify the sentence, or
    drop it. Do not extrapolate beyond what the excerpts say.
  - Acknowledge limitations honestly.
  - Incorporate any human feedback provided below.
  - Leave the `references` array EMPTY ([]) — it is populated programmatically
    from the collected sources; do not re-list them.

Report structure for this audience:
{structure_guidance}

Audience: {audience}
Human feedback: {human_feedback}
"""

_FINDINGS_TEMPLATE = (
    "Research question (the report must answer THIS):\n"
    "{topic}\n\n"
    "Research findings ({n} total):\n\n"
    "{findings_text}\n\n"
    "Sub-questions to cover:\n"
    "{sub_questions}\n\n"
    "All sources referenced, with an excerpt of each:\n"
    "{sources_text}\n\n"
    "Write the final report now."
)

# Appended after .format() so the JSON braces don't clash with str.format()
_FINDINGS_JSON_SUFFIX = schema_output_instruction(FinalReport)


def _collect_references(findings: list) -> list[Source]:
    """Deduplicate sources across all findings; return ordered reference list."""
    seen_urls: set[str] = set()
    refs: list[Source] = []
    for f in findings:
        evidence = f.evidence if hasattr(f, "evidence") else f.get("evidence", [])
        for e in evidence:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                refs.append(e if hasattr(e, "url") else Source(**e))
    return refs


# Per-reference cap on how much snippet text goes into the writer's prompt.
# Each source's excerpt goes into the writer's prompt so its claims can be
# grounded in the actual source text rather than just titles.
SNIPPET_CHAR_LIMIT = 400


def _format_sources(references: list[Source]) -> str:
    """Render the reference list with a truncated excerpt of each source."""
    lines = []
    for i, r in enumerate(references, 1):
        snippet = (r.snippet or "").strip().replace("\n", " ")
        if len(snippet) > SNIPPET_CHAR_LIMIT:
            snippet = snippet[:SNIPPET_CHAR_LIMIT].rstrip() + "..."
        line = f"[{i}] {r.url} -- {r.title}"
        if snippet:
            line += f'\n    Excerpt: "{snippet}"'
        lines.append(line)
    return "\n".join(lines)


def _format_findings(findings: list, references: list[Source]) -> str:
    """Format findings for the writer prompt, one source-per-citation.

    Each evidence source is listed as "[N] Title" rather than a bare number
    cluster -- the writer's prompt instructs it to attach only the citation(s)
    that plausibly cover a given sentence/number, which requires being able to
    tell sources apart by more than an index. A bare "[2],[3],[4],[5],[6]"
    block gives it nothing to distinguish between them, and it just re-attaches
    the whole cluster to every sentence derived from the finding.
    """
    ref_lookup = {r.url: (i + 1, r.title) for i, r in enumerate(references)}
    lines = []
    for i, f in enumerate(findings, 1):
        claim = f.claim if hasattr(f, "claim") else f.get("claim", "")
        confidence = f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.5)
        sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
        evidence = f.evidence if hasattr(f, "evidence") else f.get("evidence", [])
        cite_parts = []
        for e in evidence[:5]:
            url = e.url if hasattr(e, "url") else e.get("url", "")
            if url in ref_lookup:
                num, title = ref_lookup[url]
                cite_parts.append(f"[{num}] {title}" if title else f"[{num}]")
        sources_line = "; ".join(cite_parts) if cite_parts else "(no sources)"
        lines.append(
            f"{i}. [{sub_q}] {claim} (confidence={confidence:.2f})\n   Sources: {sources_line}"
        )
    return "\n".join(lines)


def _conflict_note(state: AgentState, findings: list) -> str:
    """Prompt suffix naming findings the verifier saw contradict each other (by the numbers the
    legacy findings list uses); empty when there are none."""
    number_of = {_field(f, "id", ""): i for i, f in enumerate(findings, 1)}
    pairs = [
        f"{number_of[a]} vs {number_of[b]}"
        for a, b in (state.get("fact_conflicts") or [])
        if a in number_of and b in number_of
    ]
    if not pairs:
        return ""
    return (
        "\n\nConflicting findings (present both sides, do not pick one silently): "
        + "; ".join(pairs) + "."
    )


async def run_writer(
    state: AgentState,
    llm: BaseChatModel,
) -> FinalReport:
    """Generate a FinalReport from validated findings."""
    findings: list = state.get("findings") or []
    critiques: list = state.get("critiques") or []
    query = state.get("query")
    plan = state.get("plan")
    # writer_instructions is the dedicated HITL channel for report revisions.
    # Fall back to human_feedback for backwards compatibility with checkpoints
    # that pre-date the writer_instructions field.
    human_feedback = (
        state.get("writer_instructions")
        or state.get("human_feedback")
        or "None provided."
    )

    refuted_ids = {
        fid
        for fid, verdict in _latest_verdicts(critiques).items()
        if verdict == CritiqueVerdict.refuted.value
    }

    valid_findings = [
        f for f in findings
        if _field(f, "id", "") not in refuted_ids
        and _field(f, "confidence", 0) >= 0.1  # low bar — writer acknowledges uncertainty
    ]

    if not valid_findings:
        logger.warning("Writer: no valid findings -- producing empty report.")
        return FinalReport(
            title=f"Research Report: {query.topic if query else 'Unknown'}",
            exec_summary="Insufficient evidence was gathered to produce a report.",
        )

    references = _collect_references(valid_findings)

    sources_text = _format_sources(references)
    findings_text = _format_findings(valid_findings, references)
    sub_questions = "\n".join(
        f"  - {q}" for q in (plan.sub_questions if plan else [])
    )

    audience = query.audience if query else "general"
    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(
            audience=audience,
            structure_guidance=structure_guidance(audience),
            human_feedback=human_feedback,
        )
    )
    user_msg = HumanMessage(
        content=_FINDINGS_TEMPLATE.format(
            topic=query.topic if query else "",
            n=len(valid_findings),
            findings_text=findings_text,
            sub_questions=sub_questions or "  (none)",
            sources_text=sources_text or "  (none)",
        ) + _conflict_note(state, valid_findings) + _FINDINGS_JSON_SUFFIX
    )

    structured_llm = llm.with_structured_output(FinalReport)

    try:
        report: FinalReport = await ainvoke_with_retry(
            structured_llm, [system_msg, user_msg], agent="writer",
            session_id=state.get("session_id"),
        )
        # References are always set programmatically — the LLM is instructed to
        # leave them empty (saves output tokens and avoids hallucinated URLs).
        report = report.model_copy(update={"references": references})
    except Exception as exc:
        logger.error("Writer structured output failed: %s", exc)
        trace_event(
            state.get("session_id"), "writer.fallback", "note",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        # Fallback: build a minimal report manually
        report = FinalReport(
            title=f"Research Report: {query.topic if query else 'Topic'}",
            exec_summary="\n".join(
                f"- {f.claim if hasattr(f, 'claim') else f.get('claim','')}"
                for f in valid_findings[:5]
            ),
            sections=[
                ReportSection(
                    heading=(
                        f.sub_question
                        if hasattr(f, "sub_question")
                        else f.get("sub_question", "Finding")
                    ),
                    body_md=f.claim if hasattr(f, "claim") else f.get("claim", ""),
                    citations=[],
                )
                for f in valid_findings
            ],
            references=references,
            methodology=plan.strategy if plan else "",
            limitations="Report generated in fallback mode due to LLM error.",
        )

    return report


# ---------------------------------------------------------------------------
# Attributed writer: claim-level draft, citations assembled by code
# ---------------------------------------------------------------------------

_ATTRIBUTED_SYSTEM = """\
You write a research report from numbered, verified facts. Every factual sentence you
write must list the fact numbers (F#) it is based on in `facts`. Do not put [n] markers
in the text; citations are added automatically from `facts`.

Rules:
- direct_answer answers the research question in 1-2 sentences. If the question asks for
  a specific format (a label such as SUPPORT / CONTRADICT / NOT_ENOUGH_INFO, yes/no, a
  name, a number, a list), give exactly that format first.
- stance = insufficient ONLY if no fact addresses the question. If some facts do, answer
  with them (stance = answered or partial) and say what is uncertain.
- If the question can have several correct answers (who / which / what ... , "name the ..."),
  give EVERY distinct answer the facts support: list them all in direct_answer and give each
  its own sentence (with its fact numbers) in the sections. Do not stop at the first one.
- Never repeat a sentence: the summary is a 1-3 sentence overview, the sections hold the detail.
- Use only the facts given. Copy numbers exactly. Never compute new numbers.
- A sentence with no fact behind it may only be a short transition with no facts or
  numbers in it.
- Facts marked (partial) must be stated with hedging ("suggests", "in one study").
- If facts conflict, present both sides with their fact numbers.
- Prefer primary research: cite a fact marked (secondary source) only when no primary fact
  states the same thing. Never add author names that are not in the evidence text.
{scope_rules}
Report structure for this audience:
{structure_guidance}

Audience: {audience}. Human feedback: {human_feedback}
"""

_SCOPE_RULES = """\
- The question's specific scope is: {scope}. Facts marked (direct) address it; facts marked
  (background) are context about the general subject only.
- answer_facts may only be direct facts. Never present a background fact as answering the
  question or as evidence about {scope}.
- If no direct fact answers the question (or a sub-question), say plainly that the retrieved
  evidence does not address it. Do not fill the gap with background material.
- If the question sets a strict bar (e.g. "losslessly", "guaranteed", "always"), answer "yes"
  only when a fact states that bar is met. Otherwise say what the evidence does show (for
  example an approximate or partial result) and that the strict bar is not established.
"""

_PROOF_RULES = """\
- Strict requirement in the question: {terms}. What would establish it: {criterion}
- Say which level each result reaches: an exact step INSIDE a method (e.g. an exactly invertible
  matrix or rotation), equal outputs versus the reference, or equal behaviour. A step being exact
  does not make the whole method exact. High accuracy, accuracy retention, variance explained,
  "negligible" quality loss and speedups are approximate results: never present them as meeting
  the strict requirement, and never treat them as synonyms for it.
- Sweeping claims ("only", "never", "always", "all", "guarantees", "proves") need a source that
  states them; otherwise say what the evidence shows and where it stops.
"""

_ANALYSIS_RULE = """\
- `analysis` (optional): for a conceptual or theoretical question, 1-4 sentences of reasoning
  from general principles, with no citations and no numbers. It is shown as reasoning, not as
  evidence; leave it empty otherwise.
"""

_ATTRIBUTED_USER = """\
Research question: {topic}
{answer_format}
Sub-questions:
{sub_questions}

Facts:
{facts}
Conflicting facts: {conflicts}
"""


def _is_secondary(finding: Any) -> bool:
    """The fact comes from a blog / social / aggregator page, not primary research."""
    from research_swarm.agents.writer_render import _secondary

    return _secondary(finding)


def _select_facts(state: AgentState) -> tuple[list, dict[str, str]]:
    """Writer-eligible findings (not refuted, confidence >= 0.1), sub-question order then
    confidence, at most the run depth's ``max_facts_for_writer``; plus each one's latest verdict."""
    from research_swarm.config import settings

    findings = state.get("findings") or []
    verdicts = _latest_verdicts(state.get("critiques") or [])
    plan = state.get("plan")
    order = {sq.strip().lower(): i for i, sq in enumerate(plan.sub_questions)} if plan else {}
    eligible = [
        f for f in findings
        if verdicts.get(_field(f, "id", "")) != CritiqueVerdict.refuted.value
        and _field(f, "confidence", 0) >= 0.1
        and _field(f, "relevance", "unknown") != "off_topic"
    ]
    eligible.sort(key=lambda f: (
        order.get(_field(f, "sub_question", "").strip().lower(), len(order)),
        _is_secondary(f),                     # primary sources first within a sub-question
        -float(_field(f, "confidence", 0.5)),
    ))
    query = state.get("query")
    cap = settings.for_depth("max_facts_for_writer", getattr(query, "depth", None))
    if cap > 0:
        eligible = eligible[:cap]
    return eligible, verdicts


async def run_attributed_writer(state: AgentState, llm: BaseChatModel) -> FinalReport:
    """Claim-level draft -> deterministic render; falls back to the free-form writer if the
    structured draft cannot be parsed or nothing survives rendering."""
    import asyncio

    from research_swarm.agents._utils import recover_from_parse_failure
    from research_swarm.agents.question import parse_question
    from research_swarm.agents.verdict import decide_verdict
    from research_swarm.agents.writer_render import (
        WriterDraft,
        WriterDraftWithAnalysis,
        deterministic_report,
        ground_free_form_report,
        render_report,
    )

    session_id = state.get("session_id")
    query = state.get("query")
    plan = state.get("plan")
    topic = query.topic if query else ""
    audience = query.audience if query else "general"
    spec = parse_question(topic)
    frame = getattr(plan, "frame", None) if plan else None
    scope = frame.key_constraint if frame is not None and frame.has_constraint else ""
    sub_qs = list(plan.sub_questions) if plan else []
    facts, verdicts = _select_facts(state)
    if not facts:
        return await run_writer(state, llm)                # the legacy empty-report path

    async def grounded_fallback() -> FinalReport:
        """The free-form writer has no per-sentence citation discipline; ground its numbers
        against the same source text the facts came from, same as the attributed path, and
        fall back further to a no-LLM report if nothing grounded survives."""
        report = await run_writer(state, llm)
        grounded, stats = ground_free_form_report(
            report, facts, spec.content or topic, sub_questions=sub_qs,
        )
        trace_event(session_id, "writer.fallback_grounded", "note", **stats)
        if stats["empty"]:
            return deterministic_report(facts, spec.content or topic, plan, audience=audience)
        return grounded
    answer_format = ""
    if spec.instruction:
        answer_format += f"Answer format: {spec.instruction}\n"
    if spec.labels:
        answer_format += f"Allowed answers: {', '.join(spec.labels)}\n"

    sq_index = {sq.strip().lower(): i for i, sq in enumerate(sub_qs, 1)}
    lines = []
    fact_labels = []
    for n, f in enumerate(facts, 1):
        label = "partial" if verdicts.get(_field(f, "id", "")) == CritiqueVerdict.weak.value \
            else "supported"
        relevance = _field(f, "relevance", "unknown")
        if scope and relevance in ("direct", "background"):
            label += f", {relevance}"
        if _is_secondary(f):
            label += ", secondary source"
        fact_labels.append(f"({label})")
        q = sq_index.get(_field(f, "sub_question", "").strip().lower())
        ev = (_field(f, "evidence", []) or [None])[0]
        snippet = (_field(ev, "snippet", "") if ev else "")[:500].replace("\n", " ")
        title = _field(ev, "title", "") if ev else ""
        lines.append(
            f"F{n} [{'Q' + str(q) if q else 'Q?'}] ({label}) {_field(f, 'claim', '')}\n"
            f"   Source: {title}\n   Evidence: «{snippet}»"
        )
    number_of = {_field(f, "id", ""): n for n, f in enumerate(facts, 1)}
    pairs = [
        f"F{number_of[a]} vs F{number_of[b]}"
        for a, b in (state.get("fact_conflicts") or [])
        if a in number_of and b in number_of
    ]
    human_feedback = (
        state.get("writer_instructions") or state.get("human_feedback") or "None provided."
    )
    from research_swarm.config import settings

    draft_model = WriterDraftWithAnalysis if settings.writer_reasoning_section else WriterDraft
    scope_rules = _SCOPE_RULES.format(scope=scope) if scope else ""
    if frame is not None and frame.define_terms:
        scope_rules += _PROOF_RULES.format(
            terms="; ".join(frame.define_terms),
            criterion=frame.proof_criterion or "a source stating it is met, for the whole method",
        )
    if settings.writer_reasoning_section:
        scope_rules += _ANALYSIS_RULE
    system = SystemMessage(content=_ATTRIBUTED_SYSTEM.format(
        audience=audience, structure_guidance=structure_guidance(audience),
        human_feedback=human_feedback, scope_rules=scope_rules,
    ) + schema_output_instruction(draft_model))
    user = HumanMessage(content=_ATTRIBUTED_USER.format(
        topic=spec.content or topic,
        answer_format=answer_format,
        sub_questions="\n".join(f"[Q{i}] {q}" for i, q in enumerate(sub_qs, 1)) or "(none)",
        facts="\n".join(lines),
        conflicts="; ".join(pairs) or "none",
    ))

    structured = llm.with_structured_output(draft_model)

    async def write() -> Any:
        if settings.writer_mode == "sectioned":
            from research_swarm.agents.writer_sections import SectionedContext, write_sectioned

            ctx = SectionedContext(
                question=spec.content or topic, answer_format=answer_format,
                sub_questions=sub_qs, facts=facts, labels=fact_labels,
                conflicts="; ".join(pairs) or "none", audience=audience,
                structure_guidance=structure_guidance(audience), scope_rules=scope_rules,
                human_feedback=human_feedback,
                analysis_enabled=settings.writer_reasoning_section, session_id=session_id,
                compare_items=list(frame.compare_items) if frame is not None else [],
            )
            try:
                sectioned = await write_sectioned(ctx, llm)
            except Exception as exc:  # noqa: BLE001 -- the single-call draft is the fallback
                logger.error("Sectioned writer failed (%s) -- using one call.", exc)
                sectioned = None
            if sectioned is not None:
                return sectioned
            trace_event(session_id, "writer.fallback", "note", reason="sectioned_outline")
        try:
            return await ainvoke_with_retry(structured, [system, user], agent="writer",
                                            session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            return exc

    # The claim verdict (claim-check questions only) runs alongside the draft.
    draft, verdict = await asyncio.gather(
        write(), decide_verdict(spec, facts, llm, session_id=session_id),
    )
    if isinstance(draft, Exception):
        exc = draft
        draft = recover_from_parse_failure(exc, draft_model)
        if draft is None:
            logger.error("Attributed writer failed (%s) -- using the free-form writer.", exc)
            trace_event(session_id, "writer.fallback", "note", reason="attributed_parse",
                        error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return await grounded_fallback()

    report, stats = render_report(
        draft, facts, spec.content or topic, plan, verdict=verdict, labels=spec.labels,
        audience=audience, frame=frame, sub_questions=sub_qs,
        analysis_enabled=settings.writer_reasoning_section,
    )
    if stats["no_direct_answer"]:
        trace_event(session_id, "writer.no_direct_answer", "note", scope=scope)
    if draft.stance == "insufficient" and any(
        verdicts.get(_field(f, "id", "")) == CritiqueVerdict.supported.value for f in facts
    ):
        trace_event(session_id, "writer.overabstain", "note", n_facts=len(facts))
    trace_event(session_id, "writer.render", "note", stance=draft.stance, **stats)
    if stats["empty"]:
        trace_event(session_id, "writer.fallback", "note", reason="empty_render")
        return await grounded_fallback()
    return report
