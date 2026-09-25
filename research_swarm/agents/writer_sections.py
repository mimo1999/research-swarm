"""Section-by-section writer: outline -> one call per section, in order -> a final review pass.

One call for the whole report asks a model to plan, write and self-check ~30 facts at once; a
2B writer used 5 of 23 verified facts and misread the question. Splitting the job:

  1. **Outline** (one call): the direct answer, stance, and which facts (F#) go in which
     section. Report types with fixed sections (``writer_render.REPORT_SECTIONS``: academic,
     technical) use exactly those; others (general) take the outline's headings. Code then gives
     each fact to ONE evidence section and places any direct fact the outline left out.
  2. **Sections** (one call each, ONE AT A TIME): evidence sections first, then interpretive
     ones (Discussion / Conclusion), then the overview (Abstract) -- each seeing every section
     written before it, so it builds on them instead of repeating them. Evidence sections see
     their own facts with full evidence and source (title + URL); the others cite the facts the
     earlier sections used.
  3. **Review** (one call): sees every drafted sentence (S#) next to the evidence of the facts it
     cites, and flags unsupported / overclaiming / mis-cited / contradictory / off-topic /
     duplicate sentences with a fix (or deletion); it then writes the final direct answer,
     stance and summary so they agree with the corrected body.

The result is an ordinary ``WriterDraft`` rendered by ``writer_render.render_report``, so every
code-side check (citations by first use, ungrounded numbers, scope overclaims, the strict-bar
guard, evidence gaps) still runs after the review. Returns None when the outline fails, and the
caller falls back to the single-call draft.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from research_swarm.agents._utils import (
    _field,
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)
from research_swarm.agents.writer_render import (
    _OVERVIEW_RE,
    REPORT_SECTIONS,
    ComparisonTable,
    DraftSection,
    DraftSentence,
    SectionSpec,
    WriterDraft,
    WriterDraftWithAnalysis,
    _canonicalize_heading,
    _finding_text,
    is_direct,
    strong_terms,
)
from research_swarm.runtime.trace import trace_event

logger = logging.getLogger(__name__)

OUTLINE_EVIDENCE_CHARS = 300
SECTION_EVIDENCE_CHARS = 900
REVIEW_EVIDENCE_CHARS = 500
MAX_SECTIONS = 8
# An evidence section holds at most max(this, an even share of the facts) -- see _rebalance.
MAX_FACTS_PER_SECTION = 12
# A section call's output is cut to this many sentences: given 36 facts, one looped out 214.
MAX_SECTION_SENTENCES = 10


# --------------------------------------------------------------------------- schemas

class SectionPlan(BaseModel):
    heading: str
    purpose: str = Field(default="", description="One sentence: what this section establishes")
    facts: list[int] = Field(default_factory=list, description="F# numbers this section uses")


class ReportOutline(BaseModel):
    title: str
    direct_answer: str = Field(..., description="1-2 sentences answering the research question")
    answer_facts: list[int] = Field(default_factory=list)
    stance: Literal["answered", "partial", "insufficient"]
    sections: list[SectionPlan] = Field(default_factory=list)
    limitations: str = ""


class SectionText(BaseModel):
    sentences: list[DraftSentence] = Field(default_factory=list)


class SentenceFix(BaseModel):
    sentence: int = Field(..., description="The S# of the flagged sentence")
    problem: Literal["unsupported", "overclaim", "wrong_facts", "contradiction", "off_topic",
                     "duplicate"]
    fix: str = Field(default="", description="The corrected sentence, or empty to delete it")
    facts: list[int] = Field(default_factory=list, description="F# numbers for the fix")


class ClaimCheck(BaseModel):
    """The review's ruling on one sentence marked as a strong claim."""
    sentence: int = Field(..., description="The S# of a sentence listed under 'Strong claims'")
    level: Literal["component", "output", "behavior", "none"] = Field(
        ..., description="What the cited evidence actually establishes: an exact step inside a "
                         "method (component), equal outputs vs the reference (output), equal "
                         "system behaviour (behavior), or nothing on this point (none)",
    )
    supported: bool = Field(..., description="Does the evidence support the sentence AS WORDED?")
    fix: str = Field(default="", description="If not supported: the sentence reworded to what the "
                                             "evidence shows, or empty to delete it")


class ReviewPass(BaseModel):
    # Answer fields come FIRST and are optional: a reviewer once looped on `fixes` (106 entries
    # for ~15 sentences), never reached the answer, and the whole review failed to parse. Now
    # the answer is written before the list, and a review cut short still parses.
    direct_answer: str = Field(default="", description="Final 1-2 sentence answer, consistent "
                                                        "with the corrected body")
    answer_facts: list[int] = Field(default_factory=list)
    stance: Literal["answered", "partial", "insufficient"] | None = None
    summary: list[DraftSentence] = Field(default_factory=list, description="2-4 sentences")
    limitations: str = ""
    claim_checks: list[ClaimCheck] = Field(
        default_factory=list,
        description="One entry for EVERY sentence listed under 'Strong claims'",
    )
    fixes: list[SentenceFix] = Field(
        default_factory=list,
        description="Only sentences with a problem, each S# at most once",
    )

    @field_validator("summary", mode="before")
    @classmethod
    def _plain_summary_lines(cls, value: Any) -> Any:
        """Accept summary lines given as plain strings (nemotron did, and the whole review failed
        to parse); they get no citations, so the render keeps them only as short transitions."""
        if isinstance(value, list):
            return [{"text": v, "facts": []} if isinstance(v, str) else v for v in value]
        return value


class ReviewPassWithAnalysis(ReviewPass):
    analysis: list[str] = Field(
        default_factory=list,
        description="Optional: 1-4 sentences of reasoning from general principles for a "
                    "conceptual question. No citations, no numbers.",
    )


# --------------------------------------------------------------------------- prompts

_OUTLINE_SYSTEM = """\
You plan a research report from numbered, verified facts (F#). Do not write the sections yet.
Decide:
- direct_answer: at most 2 sentences and 50 words that answer the research question AS ASKED.
  If the question asks for a specific format (a label, yes/no, a name, a number, a list), give
  exactly that format first. Keep detailed figures for the body. answer_facts: the F# numbers
  the answer rests on; prefer primary research over facts marked (secondary source).
- stance: insufficient ONLY if no fact addresses the question; otherwise answered or partial.
- sections: {sections_rule}
- limitations: what the facts do not establish.
{scope_rules}
Report structure for this audience:
{structure_guidance}

Audience: {audience}. Human feedback: {human_feedback}
"""

_SECTION_SYSTEM = """\
You write ONE section of a research report, using only the facts given for this section.
Every factual sentence must list the F# numbers it states in `facts` (only numbers shown below).
Do not put [n] markers in the text; citations are added automatically from `facts`.
The sections already written are shown: never repeat what they say, and build on them.
Rules:
{kind_rule}
- Serve the section's purpose; do not restate the overall answer.
- Cite at most 3 facts per sentence: the ones that state it.
- Never copy source markup (placeholders such as "T01", LaTeX commands); if a fact only gives
  a value as a placeholder, leave the value out.
- Attribute results to their source by the paper or system name as it appears in the Source
  line ("DroidSpeak shows ...", "a study of ... reports"). Never add author names that are not
  in the evidence text.
- Prefer primary research. Cite a fact marked (secondary source) -- a blog, social post or
  aggregator page -- only when no primary fact states the same thing.
- Write for the reader: never mention "facts", "F#", "the provided evidence" or this prompt.
- Copy numbers, names and conditions exactly. Never compute new numbers.
- Facts marked (partial) need hedging ("suggests", "in one study").
- If facts conflict, present both sides with their numbers.
- Do not claim more than the evidence text says: an approximate or partial result is not an
  exact or general one.
{scope_rules}
Audience: {audience}.
"""

# What each kind of section (writer_render.SectionSpec.kind) is for.
_KIND_RULES = {
    "evidence": (
        "- An evidence section: 3-8 sentences presenting the facts given for it. Add nothing that "
        "an earlier section already states."
    ),
    "synthesis": (
        "- An interpretive section, written after the evidence sections: 3-6 sentences on what\n"
        "  those results mean for the research question -- the conditions under which the answer\n"
        "  holds, where sources agree or disagree, limitations, open questions. Refer to a result\n"
        "  briefly (citing its F#) instead of restating it. Add no new results."
    ),
    "overview": (
        "- The overview, written after the rest of the report: 3-5 sentences in fresh words on\n"
        "  the question, the kind of evidence found and the main findings, each citing the F#\n"
        "  behind it. Do not copy sentences from the sections."
    ),
}

_REVIEW_SYSTEM = """\
You are the final reviewer of a research report draft. Each draft sentence S# lists the facts
F# it cites; the evidence text of every fact is given.
FIRST write the final direct_answer (at most 2 sentences and 50 words, consistent with the
evidence; if the question sets a strict bar such as "exact" or "lossless", say yes only when a
fact states it is met; figures belong in the body), its answer_facts (prefer primary research
over facts marked (secondary source)), the stance, a 2-4 sentence summary (each with its facts,
not repeating body sentences) and the limitations.
NEXT, for EVERY sentence listed under "Strong claims", add a `claim_checks` entry: the level its
cited evidence actually reaches (component = an exact step inside a method; output = equal
outputs versus the reference; behavior = equal system behaviour; none), whether the evidence
supports the sentence as worded, and if not, a `fix` rewording it to what the evidence shows
(or empty to delete it). A step that is exact inside a method does not make the method exact;
accuracy, variance explained and "negligible loss" are approximate results.
THEN check every sentence against the evidence of the facts it cites, and list in `fixes` only
the sentences that are (each S# at most once):
  unsupported   - its cited evidence does not state it
  overclaim     - it states more certainty or scope than the evidence (e.g. "exact" or
                  "lossless" when the evidence shows an approximate result)
  wrong_facts   - the claim is supported, but by other F# than the ones cited
  contradiction - it conflicts with another sentence or with the answer
  off_topic     - it does not help answer the research question
  duplicate     - it repeats another sentence, including one in a DIFFERENT section
  wrong_facts also covers citing facts that do not state the sentence (e.g. a list of 8 sources
  where 2 support it) and sentences containing source markup such as "T01--T02".
For each flagged sentence give `fix`: the corrected sentence (using only the evidence) with its
`facts`, or an empty fix to delete it. Do not list sentences that are fine. Write for the
reader: never mention "facts", "F#" or "the provided evidence".
{scope_rules}{analysis_rule}"""

_REVIEW_ANALYSIS_RULE = """\
- `analysis` (optional): for a conceptual or theoretical question, 1-4 sentences of reasoning
  from general principles, with no citations and no numbers; leave it empty otherwise.
"""


# --------------------------------------------------------------------------- helpers

@dataclass
class SectionedContext:
    """Everything the sectioned writer needs, prepared by ``writer.run_attributed_writer``."""
    question: str
    answer_format: str
    sub_questions: list[str]
    facts: list
    labels: list[str]            # "(supported, direct)" etc., per fact
    conflicts: str
    audience: str
    structure_guidance: str
    scope_rules: str
    human_feedback: str
    analysis_enabled: bool
    session_id: str | None
    compare_items: list[str] = field(default_factory=list)   # question frame; table rows


_COMPARISON_SYSTEM = """\
You build the comparison table of a research report. Rows: exactly these items, in this order:
{items}
Columns: choose 3-4 short headings that answer the research question for every item -- for
example what is transferred or changed, what is recomputed, whether the question's strict
requirement is established, and the key evidence.
Each cell: at most 12 words. Cite in `facts` the F# numbers that state it. Write
"not established" when no fact addresses the cell. A short definitional cell (what the item does
by definition, no numbers) may be uncited. Never invent results, and never call a result exact,
lossless or guaranteed unless a cited fact says so.
{scope_rules}"""


def _fact_block(ctx: SectionedContext, nums: list[int], evidence_chars: int,
                with_url: bool = False) -> str:
    sq_index = {sq.strip().lower(): i for i, sq in enumerate(ctx.sub_questions, 1)}
    lines = []
    for n in nums:
        f = ctx.facts[n - 1]
        q = sq_index.get(_field(f, "sub_question", "").strip().lower())
        ev = (_field(f, "evidence", []) or [None])[0]
        snippet = (_field(ev, "snippet", "") if ev else "")[:evidence_chars].replace("\n", " ")
        source = _field(ev, "title", "") if ev else ""
        if with_url and ev is not None and _field(ev, "url", ""):
            source += f" <{_field(ev, 'url', '')}>"
        lines.append(
            f"F{n} [{'Q' + str(q) if q else 'Q?'}] {ctx.labels[n - 1]} {_field(f, 'claim', '')}\n"
            f"   Source: {source}\n   Evidence: «{snippet}»"
        )
    return "\n".join(lines)


def _valid(nums: list[int], n_facts: int) -> list[int]:
    return [n for n in dict.fromkeys(nums) if 1 <= n <= n_facts]


@dataclass
class _Planned:
    """One section to write: a fixed SectionSpec or an outline heading, with its facts."""
    heading: str
    purpose: str
    kind: str                         # evidence | synthesis | overview
    facts: list[int] = field(default_factory=list)


def _sections_rule(specs: tuple[SectionSpec, ...] | None) -> str:
    if specs:
        lines = ["use exactly these sections. Give facts only to the evidence sections, each fact "
                 "to ONE section; the interpretive and overview sections are written from the "
                 "evidence sections and take no facts of their own:"]
        lines += [f"  {i}. {s.heading} ({s.kind}): {s.purpose}" for i, s in enumerate(specs, 1)]
        return "\n".join(lines)
    return (
        "follow the report structure below. For each: heading, a one-sentence purpose, and the "
        "F# numbers it will use. Put every fact that answers the question in at least one "
        "section, and give each fact to ONE section. Sections must not overlap. Omit a section "
        f"that no fact supports. At most {MAX_SECTIONS} sections."
    )


def _plan_sections(outline: ReportOutline, ctx: SectionedContext,
                   specs: tuple[SectionSpec, ...] | None) -> tuple[list[_Planned], int]:
    """The sections to write, in reading order, and how many leftover facts were placed.

    With fixed specs the outline only distributes facts: headings it wrote are mapped onto the
    specs, and only evidence sections keep facts. Either way each fact belongs to ONE evidence
    section (an outline once gave the abstract AND the discussion all 20 facts; both repeated the
    experiments), and every direct fact the outline left out goes to the evidence section already
    holding most facts of its sub-question.
    """
    n = len(ctx.facts)
    if specs:
        planned = [_Planned(s.heading, s.purpose, s.kind) for s in specs]
        by_heading = {p.heading: p for p in planned}
        template = [s.heading for s in specs]
        for sec in outline.sections:
            target = by_heading.get(_canonicalize_heading(sec.heading.strip(), template))
            if target is not None:
                target.facts += _valid(sec.facts, n)
    else:
        planned = [
            _Planned(sec.heading.strip(), sec.purpose,
                     "overview" if _OVERVIEW_RE.search(sec.heading) else "evidence",
                     _valid(sec.facts, n))
            for sec in outline.sections[:MAX_SECTIONS] if sec.heading.strip()
        ]
        if not any(p.kind == "evidence" for p in planned):
            planned.append(_Planned("Findings", "What the evidence shows.", "evidence"))

    used: set[int] = set()
    for p in planned:
        if p.kind == "evidence":
            p.facts = [m for m in dict.fromkeys(p.facts) if m not in used]
            used.update(p.facts)
        else:
            p.facts = []              # written from what the evidence sections cite

    evidence = [p for p in planned if p.kind == "evidence"]
    placed = 0
    for m in range(1, n + 1):
        f = ctx.facts[m - 1]
        if m in used or not is_direct(f):
            continue
        sq = _field(f, "sub_question", "").strip().lower()

        def overlap(p: _Planned, sq: str = sq) -> int:
            return sum(1 for k in p.facts
                       if _field(ctx.facts[k - 1], "sub_question", "").strip().lower() == sq)

        max(evidence, key=overlap).facts.append(m)
        placed += 1
    _rebalance(evidence, ctx)
    return planned, placed


def _rebalance(evidence: list[_Planned], ctx: SectionedContext) -> int:
    """Spread facts across the evidence sections: no section above
    max(MAX_FACTS_PER_SECTION, an even share). An outline once gave all 36 facts to "Previous
    Work" and none to "Experiments"; the one section call then looped out 214 sentences.
    Whole sub-question groups move (smallest first) so each section stays coherent. Returns how
    many facts moved."""
    total = sum(len(p.facts) for p in evidence)
    if len(evidence) < 2 or not total:
        return 0
    limit = max(MAX_FACTS_PER_SECTION, -(-total // len(evidence)))
    moved = 0
    for src in evidence:
        while len(src.facts) > limit:
            dst = min((p for p in evidence if p is not src), key=lambda p: len(p.facts))
            room = limit - len(dst.facts)
            if room <= 0:
                break
            groups: dict[str, list[int]] = {}
            for m in src.facts:
                key = _field(ctx.facts[m - 1], "sub_question", "").strip().lower()
                groups.setdefault(key, []).append(m)
            smallest = min(groups.values(), key=len)
            take = smallest[:min(room, len(src.facts) - limit, len(smallest))] \
                if len(groups) == 1 else smallest[:room]
            if not take:
                break
            src.facts = [m for m in src.facts if m not in take]
            dst.facts += take
            moved += len(take)
    return moved


def _prior_block(prior: list[DraftSection]) -> str:
    if not prior:
        return "(none yet)"
    lines = []
    for sec in prior:
        lines.append(f"## {sec.heading}")
        lines += [f"{s.text} [{', '.join(f'F{n}' for n in s.facts)}]" if s.facts else s.text
                  for s in sec.sentences]
    return "\n".join(lines)


async def _write_section(ctx: SectionedContext, outline: ReportOutline, sec: _Planned,
                         facts: list[int], prior: list[DraftSection],
                         llm: BaseChatModel) -> DraftSection:
    """Write one section, seeing every section written before it."""
    system = _SECTION_SYSTEM.format(kind_rule=_KIND_RULES[sec.kind], scope_rules=ctx.scope_rules,
                                    audience=ctx.audience)
    evidence_chars = SECTION_EVIDENCE_CHARS if sec.kind == "evidence" else REVIEW_EVIDENCE_CHARS
    user = (
        f"Research question: {ctx.question}\n"
        f"The report's answer (for coherence; do not restate it): {outline.direct_answer}\n\n"
        f"Sections already written:\n{_prior_block(prior)}\n\n"
        f"Section to write now: {sec.heading}\nPurpose: {sec.purpose or '(see heading)'}\n\n"
        f"Facts for this section:\n{_fact_block(ctx, facts, evidence_chars, True)}\n"
    )
    out = await _call(llm, SectionText, system, user, ctx.session_id)
    allowed = set(facts)
    raw = [s for s in (out.sentences if out else []) if s.text.strip()]
    sentences: list[DraftSentence] = []
    seen_text: set[str] = set()
    for s in raw:                     # a looping model repeats itself: keep the first of each
        key = " ".join(s.text.lower().split())
        if key in seen_text:
            continue
        seen_text.add(key)
        sentences.append(DraftSentence(text=s.text, facts=[n for n in s.facts if n in allowed]))
        if len(sentences) >= MAX_SECTION_SENTENCES:
            break
    trace_event(ctx.session_id, "writer.section", "note", heading=sec.heading[:80],
                kind=sec.kind, n_facts=len(facts), n_prior=len(prior), n_raw=len(raw),
                n_sentences=len(sentences), failed=out is None)
    return DraftSection(heading=sec.heading, sentences=sentences)


async def _write_all(ctx: SectionedContext, outline: ReportOutline, planned: list[_Planned],
                     llm: BaseChatModel) -> list[DraftSection]:
    """Write the sections one at a time -- evidence first, then synthesis, then the overview --
    each seeing everything written so far; return them in reading order.

    Sequential on purpose: sections written concurrently could not see each other, and
    repeated each other (a whole section once came out word for word twice). Same number of
    calls as writing them concurrently, so no extra compute -- only a few seconds of latency on
    the writer's cloud model.
    """
    rank = {"evidence": 0, "synthesis": 1, "overview": 2}
    order = sorted(range(len(planned)), key=lambda i: (rank[planned[i].kind], i))
    written: dict[int, DraftSection] = {}
    for i in order:
        sec = planned[i]
        prior = [written[j] for j in order if j in written]
        if sec.kind == "evidence":
            facts = sec.facts
        else:
            facts = sorted({n for d in prior for s in d.sentences for n in s.facts})
        if not facts:
            continue
        drafted = await _write_section(ctx, outline, sec, facts, prior, llm)
        if drafted.sentences:
            written[i] = drafted
    return [written[i] for i in range(len(planned)) if i in written]


async def _call(llm: BaseChatModel, schema: type[BaseModel], system: str, user: str,
                session_id: str | None) -> Any:
    """One structured call; a recovered parse on failure; None if nothing usable came back."""
    structured = llm.with_structured_output(schema)
    try:
        out = await ainvoke_with_retry(
            structured, [SystemMessage(content=system + schema_output_instruction(schema)),
                         HumanMessage(content=user)],
            agent="writer", session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        out = recover_from_parse_failure(exc, schema)
        if out is None:
            logger.warning("Sectioned writer: %s call failed (%s)", schema.__name__, exc)
            return None
    return out if isinstance(out, schema) else None


# --------------------------------------------------------------------------- stages

async def _outline(ctx: SectionedContext, llm: BaseChatModel,
                   specs: tuple[SectionSpec, ...] | None) -> ReportOutline | None:
    system = _OUTLINE_SYSTEM.format(
        sections_rule=_sections_rule(specs), scope_rules=ctx.scope_rules,
        structure_guidance=ctx.structure_guidance, audience=ctx.audience,
        human_feedback=ctx.human_feedback,
    )
    user = (
        f"Research question: {ctx.question}\n{ctx.answer_format}"
        f"Sub-questions:\n"
        + ("\n".join(f"[Q{i}] {q}" for i, q in enumerate(ctx.sub_questions, 1)) or "(none)")
        + "\n\nFacts:\n"
        + _fact_block(ctx, list(range(1, len(ctx.facts) + 1)), OUTLINE_EVIDENCE_CHARS)
        + f"\nConflicting facts: {ctx.conflicts}\n"
    )
    outline = await _call(llm, ReportOutline, system, user, ctx.session_id)
    if outline is None:
        return None
    outline.answer_facts = _valid(outline.answer_facts, len(ctx.facts))
    return outline


def _backed_by_cited(sentence: DraftSentence, ctx: SectionedContext) -> bool:
    """Code fallback for an unruled strong claim: every strong term in it must appear in the
    text of one of its own cited facts."""
    cited = " ".join(_finding_text(ctx.facts[n - 1]) for n in sentence.facts
                     if 1 <= n <= len(ctx.facts)).lower()
    return bool(cited) and all(
        re.search(rf"\b{re.escape(t.split()[0][:5])}", cited)     # "exactly" ~ "exact"
        for t in strong_terms(sentence.text)
    )


async def _review(ctx: SectionedContext, outline: ReportOutline, sections: list[DraftSection],
                  llm: BaseChatModel) -> tuple[list[DraftSection], ReviewPass | None]:
    """The final critic pass. Besides free-form fixes it must rule on every sentence with
    strong-claim wording ("exact", "lossless", "guarantees", "only", ...): the level its evidence
    reaches and whether it is supported as worded. A strong claim it leaves unruled -- or every
    strong claim, if the call fails -- falls back to a code rule: kept only if its own cited
    facts contain the same strong wording."""
    schema = ReviewPassWithAnalysis if ctx.analysis_enabled else ReviewPass
    index: dict[int, tuple[int, int]] = {}
    body_lines = []
    strong: list[int] = []
    s_no = 0
    for i, sec in enumerate(sections):
        body_lines.append(f"## {sec.heading}")
        for j, s in enumerate(sec.sentences):
            s_no += 1
            index[s_no] = (i, j)
            if strong_terms(s.text):
                strong.append(s_no)
            body_lines.append(f"S{s_no} [{', '.join(f'F{n}' for n in s.facts) or 'no facts'}] "
                              f"{s.text}")
    cited = sorted({n for sec in sections for s in sec.sentences for n in s.facts}
                   | set(outline.answer_facts)) or list(range(1, len(ctx.facts) + 1))
    system = _REVIEW_SYSTEM.format(
        scope_rules=ctx.scope_rules,
        analysis_rule=_REVIEW_ANALYSIS_RULE if ctx.analysis_enabled else "",
    )
    user = (
        f"Research question: {ctx.question}\n{ctx.answer_format}"
        f"Draft answer: {outline.direct_answer} "
        f"[{', '.join(f'F{n}' for n in outline.answer_facts) or 'no facts'}]\n\n"
        "Draft body:\n" + ("\n".join(body_lines) or "(no sections)")
        + "\n\nStrong claims (each needs a claim_checks entry): "
        + (", ".join(f"S{k}" for k in strong) or "none")
        + "\n\nFacts cited (with evidence):\n"
        + _fact_block(ctx, cited, REVIEW_EVIDENCE_CHARS) + "\n"
    )
    review = await _call(llm, schema, system, user, ctx.session_id)

    n = len(ctx.facts)
    fixed = [DraftSection(heading=s.heading, sentences=list(s.sentences)) for s in sections]
    counts: dict[str, int] = {}
    deleted = 0

    def put(k: int, text: str, facts: list[int]) -> None:
        nonlocal deleted
        i, j = index[k]
        fixed[i].sentences[j] = DraftSentence(text=text.strip(), facts=_valid(facts, n))
        if not text.strip():
            deleted += 1                      # an empty sentence is dropped by the render

    if review is not None:
        seen_fix: set[int] = set()
        for fx in review.fixes:
            if fx.sentence not in index or fx.sentence in seen_fix:   # a looping reviewer repeats
                continue
            seen_fix.add(fx.sentence)
            counts[fx.problem] = counts.get(fx.problem, 0) + 1
            put(fx.sentence, fx.fix, fx.facts)

    # Strong claims: the review's ruling, else the code rule.
    rulings = {c.sentence: c for c in (review.claim_checks if review else [])
               if c.sentence in index}
    claim_counts = {"ruled_supported": 0, "ruled_unsupported": 0, "code_kept": 0, "code_dropped": 0}
    for k in strong:
        i, j = index[k]
        current = fixed[i].sentences[j]
        if not current.text.strip():
            continue                          # already deleted by a fix
        ruling = rulings.get(k)
        if ruling is not None and ruling.supported:
            claim_counts["ruled_supported"] += 1
        elif ruling is not None:
            claim_counts["ruled_unsupported"] += 1
            put(k, ruling.fix, current.facts)
        elif _backed_by_cited(current, ctx):
            claim_counts["code_kept"] += 1
        else:
            claim_counts["code_dropped"] += 1
            put(k, "", [])

    if review is None:
        trace_event(ctx.session_id, "writer.review", "note", failed=True,
                    n_strong=len(strong), **claim_counts)
        return fixed, None
    review.answer_facts = _valid(review.answer_facts, n)
    review.summary = [DraftSentence(text=s.text, facts=_valid(s.facts, n))
                      for s in review.summary if s.text.strip()]
    trace_event(ctx.session_id, "writer.review", "note", n_sentences=s_no,
                n_fixed=sum(counts.values()), n_deleted=deleted, n_strong=len(strong),
                **claim_counts, **counts)
    return fixed, review


async def _comparison(ctx: SectionedContext, sections: list[DraftSection],
                      llm: BaseChatModel) -> ComparisonTable | None:
    """One call: a row per item the question asks to distinguish (question frame's
    compare_items), consistent with the sections already written. Cells are checked and
    citations assembled in code (writer_render.render_report). None if the call fails."""
    system = _COMPARISON_SYSTEM.format(
        items="\n".join(f"  - {item}" for item in ctx.compare_items), scope_rules=ctx.scope_rules,
    )
    user = (
        f"Research question: {ctx.question}\n\n"
        f"Sections already written:\n{_prior_block(sections)}\n\n"
        f"Facts:\n{_fact_block(ctx, list(range(1, len(ctx.facts) + 1)), OUTLINE_EVIDENCE_CHARS)}\n"
    )
    table = await _call(llm, ComparisonTable, system, user, ctx.session_id)
    if table is not None:
        wanted = {i.strip().lower() for i in ctx.compare_items}
        table.rows = [r for r in table.rows if r.item.strip().lower() in wanted] or table.rows
        n = len(ctx.facts)
        for row in table.rows:
            for cell in row.cells:
                cell.facts = _valid(cell.facts, n)
    trace_event(ctx.session_id, "writer.comparison", "note", failed=table is None,
                n_rows=len(table.rows) if table else 0,
                columns=table.columns if table else [])
    return table


async def write_sectioned(ctx: SectionedContext, llm: BaseChatModel) -> WriterDraft | None:
    """Outline -> sections (one at a time, each seeing the ones before) -> review, assembled into
    a ``WriterDraft``; None if the outline call failed (the caller then uses the single-call
    draft). Report types with fixed sections (writer_render.REPORT_SECTIONS) are written in
    exactly those sections."""
    audience = ctx.audience.strip().lower()
    specs = REPORT_SECTIONS.get(audience)
    outline = await _outline(ctx, llm, specs)
    if outline is None:
        trace_event(ctx.session_id, "writer.outline", "note", failed=True)
        return None
    planned, placed = _plan_sections(outline, ctx, specs)
    trace_event(ctx.session_id, "writer.outline", "note", n_sections=len(planned),
                headings=[f"{p.heading[:50]} ({p.kind}, {len(p.facts)})" for p in planned],
                leftovers_placed=placed, fixed=bool(specs))

    # The executive audience renders no sections: skip writing them, the review writes the
    # one-paragraph summary.
    sections = [] if audience == "executive" else await _write_all(ctx, outline, planned, llm)
    table = (await _comparison(ctx, sections, llm)
             if ctx.compare_items and audience != "executive" else None)
    sections, review = await _review(ctx, outline, sections, llm)

    draft_cls = WriterDraftWithAnalysis if ctx.analysis_enabled else WriterDraft
    fields: dict[str, Any] = {
        "title": outline.title,
        "direct_answer": (review.direct_answer if review else "") or outline.direct_answer,
        "answer_facts": (review.answer_facts if review else []) or outline.answer_facts,
        "stance": (review.stance if review else None) or outline.stance,
        "summary": review.summary if review else [],
        "sections": sections,
        "limitations": (review.limitations if review else "") or outline.limitations,
        "comparison": table,
    }
    if ctx.analysis_enabled:
        fields["analysis"] = getattr(review, "analysis", []) if review else []
    return draft_cls(**fields)
