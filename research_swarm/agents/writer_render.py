"""Attributed writer: claim-level draft schema and the code that renders it into a FinalReport.

The model writes sentences and names the fact numbers (F#) each one states; it never writes
``[n]`` markers or the reference list. This module then, with no LLM call:

  * numbers references by first use and appends ``[a, b]`` markers from the sentence's facts,
  * drops a sentence whose numbers are not in its own facts' evidence (invented / computed),
  * drops an uncited sentence unless it is a short number-free transition,
  * puts the direct answer first in ``exec_summary``,
  * lists only the sources that were actually cited.

The result is an ordinary ``FinalReport`` (the UI, API and ``eval/claims.py`` read it unchanged).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from research_swarm.agents._utils import _field
from research_swarm.agents.expansion import mentions_any, subject_stems
from research_swarm.agents.papers import paper_key
from research_swarm.agents.text import terms
from research_swarm.eval.numbers import ungrounded_numbers
from research_swarm.schemas import FinalReport, ReportSection, Source

# Strong-claim wording: the review must rule on sentences using it (writer_sections._review) and
# an uncited table cell may not use it. The language in which an exact sub-step ("the inversion
# is exact") became "lossless behavior is guaranteed", and an unsupported "only within a family".
_STRONG_RE = re.compile(
    r"\b(?:lossless(?:ly)?|exact(?:ly)?|equivalen\w*|guarantee\w*|prove[sn]?|proof|"
    r"identical(?:ly)?|never|always|impossible|"
    r"only\s+(?:when|if|within|under|in|for|between)|all\s+(?:models|llms|cases|architectures))\b",
    re.IGNORECASE,
)


def strong_terms(text: str) -> list[str]:
    """The strong-claim wording in *text* (lower-cased, de-duplicated)."""
    return list(dict.fromkeys(m.group(0).lower() for m in _STRONG_RE.finditer(text or "")))


MAX_TRANSITION_WORDS = 25
MAX_ANSWER_SENTENCES = 2
# A sentence citing more facts than this keeps only the ones that actually match its wording: a
# summary sentence once cited 8 sources, two of them unrelated cache-eviction papers.
MAX_FACTS_PER_SENTENCE = 3
# Near-duplicate sentences (content-word Jaccard at or above this) are dropped after the first:
# two section calls given overlapping facts once wrote the same six sentences word for word.
DUPLICATE_JACCARD = 0.8
# Unrendered source markup copied from arXiv HTML ("retain T01--T02 of standalone accuracy"),
# LaTeX commands, template braces.
_MARKUP_RE = re.compile(
    r"\b[A-Z]\d{2,3}\s*[-–]{1,2}\s*[A-Z]\d{2,3}\b"   # "T01--T02" placeholders
    r"|\\[A-Za-z]{2,}|\{\{|\}\}"                      # LaTeX commands, template braces
    r"|[_^]\{"                                        # LaTeX sub/superscripts: n_{kv}^s
)
# Overview sections (their duplicates yield to the detailed sections).
_OVERVIEW_RE = re.compile(r"\b(abstract|summary|overview|introduction)\b", re.IGNORECASE)

@dataclass(frozen=True)
class SectionSpec:
    """One fixed section of a report type.

    kind:
      evidence  -- presents the facts assigned to it;
      synthesis -- interprets what the evidence sections say (written after them, cites their
                   facts, adds no new results);
      overview  -- summarises the finished report (written last, rendered where it stands).
    """
    heading: str
    purpose: str
    kind: Literal["evidence", "synthesis", "overview"]


# The sections each report type is written in, in reading order. The sectioned writer
# (writer_sections.py) plans and writes exactly these; the single-call writer is prompted with
# them (writer.py::structure_guidance); render_report canonicalizes drafted headings onto them.
REPORT_SECTIONS: dict[str, tuple[SectionSpec, ...]] = {
    "academic": (
        SectionSpec("Abstract", "The question, the kind of evidence found and the main findings, "
                    "in 3-5 sentences.", "overview"),
        SectionSpec("Previous Work", "What the rest builds on: definitions, the mechanisms "
                    "involved and earlier approaches.", "evidence"),
        SectionSpec("Experiments", "The specific studies the sources report (run by their "
                    "authors, not by us): method, setup, scale and measured results with exact "
                    "numbers, attributed to each source.", "evidence"),
        SectionSpec("Discussion", "What the results mean for the question: the conditions under "
                    "which the answer holds, where sources agree or disagree, limitations and open "
                    "questions.", "synthesis"),
    ),
    "technical": (
        SectionSpec("Use Case", "Where this question matters in practice and what a practitioner "
                    "needs from the answer.", "evidence"),
        SectionSpec("Problem Statement", "The precise technical problem and what makes it hard.",
                    "evidence"),
        SectionSpec("Proposed Solutions", "The approaches found in the literature: how each "
                    "works and its measured results.", "evidence"),
        SectionSpec("Conclusion", "Which approach fits which situation, the bottom line, and "
                    "the caveats.", "synthesis"),
    ),
}
_SECTION_TEMPLATES: dict[str, list[str]] = {
    audience: [s.heading for s in specs] for audience, specs in REPORT_SECTIONS.items()
}
# The executive audience gets no sections at all -- "a one-minute summary" collapses to a
# single paragraph, so any sections the model drafted anyway are dropped rather than rendered.
EXECUTIVE_MAX_SUMMARY_SENTENCES = 6


# A line that is only "key:value" / "key=value" (JSON-ish residue such as "next_agent:dispatch"
# that a small planner appended to its strategy text).
_RESIDUE_LINE_RE = re.compile(r"^\s*[\w\-]+\s*[:=]\s*[\w\-\"']*\s*$")


def _methodology(plan: Any) -> str:
    """The plan's strategy, minus planner residue lines."""
    text = _field(plan, "strategy", "") if plan else ""
    return "\n".join(ln for ln in (text or "").splitlines()
                     if not _RESIDUE_LINE_RE.match(ln)).strip()


def _canonicalize_heading(heading: str, template: list[str]) -> str:
    """*heading* mapped onto the matching entry of *template* by keyword, else left as-is.

    A small model won't always spell "Proposed Solutions" back verbatim; matching on the
    heading's first significant word (or the full canonical phrase) keeps the report's
    section labels consistent with the requested structure without forcing content into
    a slot the model didn't intend, and without erroring on a heading that matches nothing.
    """
    h = heading.lower()
    for canon in template:
        if canon.lower() in h or canon.split()[0].lower() in h:
            return canon
    return heading


class DraftSentence(BaseModel):
    text: str = Field(..., description="One sentence, no citation markers")
    facts: list[int] = Field(default_factory=list, description="F# numbers this sentence states")


class DraftSection(BaseModel):
    heading: str
    sentences: list[DraftSentence] = Field(default_factory=list)


class TableCell(BaseModel):
    text: str = Field(..., description="At most 12 words, or 'not established'")
    facts: list[int] = Field(default_factory=list, description="F# numbers that state this cell")


class TableRow(BaseModel):
    item: str = Field(..., description="The item, as named in the question")
    cells: list[TableCell] = Field(default_factory=list, description="One per column, in order")


class ComparisonTable(BaseModel):
    """A comparison of the items a question asks to distinguish (question frame's
    compare_items), written by writer_sections._comparison and rendered in code."""
    columns: list[str] = Field(default_factory=list,
                               description="3-4 short attribute headings (not the item column)")
    rows: list[TableRow] = Field(default_factory=list)


class WriterDraft(BaseModel):
    title: str
    direct_answer: str = Field(..., description="1-2 sentences answering the research question")
    answer_facts: list[int] = Field(default_factory=list)
    stance: Literal["answered", "partial", "insufficient"]
    summary: list[DraftSentence] = Field(default_factory=list, description="2-4 sentences")
    sections: list[DraftSection] = Field(default_factory=list)
    limitations: str = ""
    # Set by the sectioned writer's comparison call, never by a model writing this schema.
    comparison: SkipJsonSchema[ComparisonTable | None] = None


class WriterDraftWithAnalysis(WriterDraft):
    """The draft schema when ``settings.writer_reasoning_section`` is on (kept a separate model so
    the field is not even offered to the writer when the section is off)."""
    analysis: list[str] = Field(
        default_factory=list,
        description="Optional: 1-4 sentences of reasoning from general principles for a "
                    "conceptual question. No citations, no numbers. Shown as reasoning, not "
                    "evidence.",
    )


ANALYSIS_HEADING = "Analysis (reasoning, not from sources)"
COMPARISON_HEADING = "Comparison"
MAX_TABLE_COLUMNS = 4
MAX_TABLE_ROWS = 8
MAX_UNCITED_CELL_WORDS = 6
_NOT_ESTABLISHED = {"not established", "unknown", "n/a", "na", "-", "—", "none", "not reported"}
_YES_RE = re.compile(r"^\s*yes\b[\s,.:;!-]*", re.IGNORECASE)
# Clause boundaries for affirmed_terms, and the words that negate a clause.
_CLAUSE_SPLIT_RE = re.compile(r"[,;:.!?()—–]|\s-\s|\b(?:but|except|unless|although|"
                              r"though|whereas|while)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|cannot|can't|cant|isn't|aren't|doesn't|don't|won't|fails?|failed|"
    r"neither|nor|unable|impossible|lacks?|without guarantee|rather than)\b|n't\b",
    re.IGNORECASE,
)


def affirmed_terms(text: str, terms_: list[str]) -> list[str]:
    """The strict qualifiers *text* asserts: mentioned in some clause that does not negate them.
    "enabling exact equivalence under specific conditions" asserts it; "exact equivalence is not
    achievable" and "cannot be transferred without loss" do not."""
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(text or "") if c and c.strip()]
    return [t for t in terms_ if any(
        mentions_any(c, [t]) and not _NEGATION_RE.search(c) for c in clauses
    )]
MAX_ANALYSIS_WORDS = 40
MAX_ANALYSIS_SENTENCES = 4


def _secondary(finding: Any) -> bool:
    """The fact's source is a blog / social / aggregator page rather than primary research."""
    from research_swarm.agents.papers import is_secondary_source

    ev = (_field(finding, "evidence", []) or [None])[0]
    return is_secondary_source(_field(ev, "url", "") if ev else "")


def is_direct(finding: Any) -> bool:
    """A fact that answers the question as asked ("unknown" counts: no frame labelled it)."""
    return _field(finding, "relevance", "unknown") not in ("background", "off_topic")


def evidence_gaps(facts: list, sub_questions: list[str] | tuple[str, ...]) -> list[str]:
    """Sub-questions with no direct fact among *facts* (already writer-eligible)."""
    answered = {
        _field(f, "sub_question", "").strip().lower() for f in facts if is_direct(f)
    }
    return [sq for sq in sub_questions if sq.strip().lower() not in answered]


def with_gaps(limitations: str, gaps: list[str]) -> str:
    """*limitations* with one "Not answered by the retrieved evidence" line per gap first."""
    if not gaps:
        return limitations
    lines = " ".join(f"Not answered by the retrieved evidence: {sq.rstrip('.?')}." for sq in gaps)
    return f"{lines}\n\n{limitations}".strip() if limitations else lines


def _with_marker(text: str, refs: list[int]) -> str:
    """Append `` [a, b]`` to *text*, before its final sentence punctuation; a sentence without
    one gets a full stop (joined sentences otherwise run together: "... Qin et al. [3] When")."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    if not refs:
        return text
    marker = " [" + ", ".join(str(r) for r in refs) + "]"
    return text[:-1] + marker + text[-1]


# "(Qin et al.)", "(Smith and Lee, 2024)": an author attribution the model may invent -- one run
# credited four different papers to "Qin et al.". Kept only when the name is in the cited evidence.
_ATTRIBUTION_RE = re.compile(
    r"\s*\(([A-Z][A-Za-z'\-]+)(?:\s+(?:et al\.?|and|&)\s*[A-Za-z'\-]*)?,?\s*(?:\d{4})?\)"
)
# The writer's internal vocabulary leaking into the report ("not detailed in the provided facts").
_META_RE = re.compile(
    r"\b(?:provided|given|supplied|available|listed)\s+(?:facts|evidence|sources)\b"
    r"|\bthe facts\b|\bF\d+\b",
    re.IGNORECASE,
)


def _finding_text(finding: Any) -> str:
    """Claim plus evidence snippets: everything a sentence based on this fact may quote."""
    parts = [_field(finding, "claim", "")]
    for e in _field(finding, "evidence", []) or []:
        parts.append(_field(e, "snippet", ""))
    return " ".join(parts)


_VERDICT_SENTENCE = {
    "support": "the verified facts directly support the claim.",
    "contradict": "the verified facts directly contradict the claim.",
    "insufficient": "none of the verified facts directly tests the claim.",
}


def _names_label(text: str, label: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", text) is not None


def _norm_sentence(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def apply_verdict(
    draft: WriterDraft, verdict: Any, labels: tuple[str, ...],
) -> tuple[str, list[int], bool]:
    """(direct answer, answer facts, replaced?) with the code-decided verdict label first.

    The draft's own answer is kept when it names only the decided label (then the label is
    guaranteed to lead); otherwise it is replaced by a fixed sentence, so a report can never
    open with one verdict and argue another.
    """
    if not verdict.label:                       # insufficient evidence, no label to lead with
        return (
            "The verified facts do not directly test the claim, so it can be neither "
            "confirmed nor refuted.",
            [], True,
        )
    other = [lab for lab in labels if lab != verdict.label]
    text = draft.direct_answer.strip()
    facts = list(verdict.deciding_facts) or (
        [] if verdict.role == "insufficient" else list(draft.answer_facts)
    )
    if text and not any(_names_label(text, lab) for lab in other):
        if not text.startswith(verdict.label):
            text = f"{verdict.label}. {text}"
        return text, facts, False
    return f"{verdict.label}: {_VERDICT_SENTENCE[verdict.role]}", facts, True


def render_report(
    draft: WriterDraft, facts: list, topic: str, plan: Any = None,
    verdict: Any = None, labels: tuple[str, ...] = (), audience: str = "general",
    frame: Any = None, sub_questions: list[str] | tuple[str, ...] = (),
    analysis_enabled: bool = False,
) -> tuple[FinalReport, dict[str, Any]]:
    """Render *draft* against *facts* (``F1`` is ``facts[0]``) -> (report, stats).

    With a *verdict* (``agents/verdict.py``) the answer leads with its label and any sentence
    naming a different label is dropped. ``stats["empty"]`` is True when nothing outside the
    direct answer survived; the caller should then fall back to the legacy writer.

    *audience* shapes the result: "executive" drops any drafted sections and caps the summary
    at ``EXECUTIVE_MAX_SUMMARY_SENTENCES`` (a one-paragraph report); "academic"/"technical"
    canonicalize section headings onto their fixed template (``_SECTION_TEMPLATES``); anything
    else (e.g. "general") renders the model's own headings unchanged.

    With a question *frame* (agents/expansion.py) the report may not claim the question's scope
    on evidence that doesn't reach it: answer facts must be direct; with no direct fact at all
    the answer opens by saying so; a cited sentence naming the scope (or a term to define) is
    dropped unless one of its own facts mentions it. Sub-questions with no direct fact are listed
    in ``limitations``. *analysis_enabled* renders the draft's optional reasoning section.
    """
    stats: dict[str, Any] = {
        "invalid_fact_refs": 0, "dropped_ungrounded_number": 0, "dropped_uncited": 0,
        "kept_transitions": 0, "kept_cited": 0, "answer_ungrounded_number": 0,
        "dropped_conflicting_verdict": 0, "dropped_duplicate": 0, "answer_replaced": False,
        "executive_summary_truncated": False, "dropped_scope_overclaim": 0,
        "no_direct_answer": False, "gap_sub_questions": 0, "analysis_sentences": 0,
        "kept_cited_direct": 0, "strict_bar_unmet": [], "dropped_markup": 0,
        "trimmed_citations": 0, "merged_mirror_refs": 0, "stripped_attribution": 0,
        "dropped_meta": 0, "dropped_secondary_citations": 0, "answer_truncated": False,
        "answer_sentences_removed": 0, "table_rows": 0, "table_cells_unbacked": 0,
    }
    audience_key = (audience or "general").strip().lower()
    is_executive = audience_key == "executive"
    heading_template = _SECTION_TEMPLATES.get(audience_key)
    # every label conflicts when the verdict has none (insufficient, no label for it)
    conflicting = [lab for lab in labels if verdict is not None and lab != verdict.label]
    has_scope = frame is not None and getattr(frame, "has_constraint", False)
    # (phrase, words to ignore): scope phrasings ignore the question's general-subject words;
    # terms to define ("lossless") are themselves question words, so nothing is ignored for them.
    subject = subject_stems(frame) if has_scope else set()
    scope_phrases: list[tuple[str, set[str]]] = (
        [(p, subject) for p in frame.scope_phrases()] + [(t, set()) for t in frame.define_terms]
        if has_scope else []
    )
    n_scope = len(frame.scope_phrases()) if has_scope else 0   # the rest are define_terms

    def valid(nums: list[int]) -> list[int]:
        good = [n for n in dict.fromkeys(nums) if 1 <= n <= len(facts)]
        stats["invalid_fact_refs"] += len(set(nums)) - len(good)
        return good

    def overclaims_scope(text: str, nums: list[int]) -> bool:
        """The sentence claims the question's scope or a strict qualifier that none of its own
        facts mention -- e.g. "bottlenecks for lossless migration stem from ..." citing a
        scheduling-only fact. The scope counts as mentioned in ANY of its phrasings: a sentence
        saying "across different LLMs" is backed by a fact saying "cross-model" (requiring the
        same phrasing dropped four good sentences from one report). A strict qualifier only
        counts when the sentence ASSERTS it (affirmed_terms): "the paper does not prove the
        transfer is lossless" is the correct audit finding, and dropping it as an overclaim once
        removed 8 of 13 sentences from an audit report."""
        cited = " ".join(_finding_text(facts[n - 1]) for n in nums)
        scope = [p for p, _ignore in scope_phrases[:n_scope]]
        if scope and mentions_any(text, scope, subject) and not mentions_any(cited, scope, subject):
            return True
        strict = [term for term, _ignore in scope_phrases[n_scope:]]
        return any(not mentions_any(cited, [term]) for term in affirmed_terms(text, strict))

    seen_terms: list[set[str]] = []           # kept sentences so far, sections then summary

    def duplicate(text: str) -> bool:
        words = terms(text)
        return bool(words) and any(
            len(words & prev) / len(words | prev) >= DUPLICATE_JACCARD for prev in seen_terms
        )

    def trim(text: str, nums: list[int]) -> list[int]:
        """At most MAX_FACTS_PER_SENTENCE citations: the facts sharing most words with it."""
        words = terms(text)
        overlap = {n: len(words & terms(_finding_text(facts[n - 1]))) for n in nums}
        ranked = sorted(nums, key=lambda n: -overlap[n])
        chosen = set([n for n in ranked if overlap[n]][:MAX_FACTS_PER_SENTENCE]
                     or ranked[:MAX_FACTS_PER_SENTENCE])
        return [n for n in nums if n in chosen]

    def strip_attributions(text: str, nums: list[int]) -> str:
        """Remove "(Name et al.)" attributions whose name is not in the cited facts' evidence."""
        evidence = " ".join(_finding_text(facts[n - 1]) for n in nums).lower()

        def repl(m: re.Match) -> str:
            if m.group(1).lower() in evidence:
                return m.group(0)
            stats["stripped_attribution"] += 1
            return ""

        return _ATTRIBUTION_RE.sub(repl, text).strip()

    def primary_only(nums: list[int]) -> list[int]:
        """Drop blog / social / aggregator citations when the sentence also cites a primary
        source for it (a report asked to cite primary research listed a YouTube video and two
        blogs beside the papers saying the same thing)."""
        primary = [n for n in nums if not _secondary(facts[n - 1])]
        if primary and len(primary) < len(nums):
            stats["dropped_secondary_citations"] += len(nums) - len(primary)
            return primary
        return nums

    def keep(sentence: DraftSentence) -> tuple[str, bool, list[int]]:
        """(text, keep?, valid fact numbers) for one summary / section sentence; the text may
        be cleaned (an invented author attribution removed)."""
        text = sentence.text.strip()
        if not text:
            return text, False, []
        if _MARKUP_RE.search(text):
            stats["dropped_markup"] += 1
            return text, False, []
        if _META_RE.search(text):
            stats["dropped_meta"] += 1
            return text, False, []
        if duplicate(text):
            stats["dropped_duplicate"] += 1
            return text, False, []
        if any(_names_label(text, lab) for lab in conflicting):
            stats["dropped_conflicting_verdict"] += 1
            return text, False, []
        nums = primary_only(valid(sentence.facts))
        if len(nums) > MAX_FACTS_PER_SENTENCE:
            nums = trim(text, nums)
            stats["trimmed_citations"] += 1
        text = strip_attributions(text, nums)
        if nums:
            allowed = " ".join(_finding_text(facts[n - 1]) for n in nums) + " " + topic
            if ungrounded_numbers(text, allowed):
                stats["dropped_ungrounded_number"] += 1
                return text, False, nums
            if scope_phrases and overclaims_scope(text, nums):
                stats["dropped_scope_overclaim"] += 1
                return text, False, nums
            stats["kept_cited"] += 1
            if all(is_direct(facts[n - 1]) for n in nums):
                stats["kept_cited_direct"] += 1
            seen_terms.append(terms(text))
            return text, True, nums
        if re.search(r"\d", text) or len(text.split()) > MAX_TRANSITION_WORDS:
            stats["dropped_uncited"] += 1
            return text, False, []
        stats["kept_transitions"] += 1
        seen_terms.append(terms(text))
        return text, True, []

    direct_answer = draft.direct_answer
    answer_facts = draft.answer_facts
    if verdict is not None:
        direct_answer, answer_facts, stats["answer_replaced"] = apply_verdict(
            draft, verdict, labels)
    answer_nums = primary_only([n for n in valid(answer_facts) if is_direct(facts[n - 1])])
    # The headline answer is at most MAX_ANSWER_SENTENCES sentences: an unreviewed outline
    # answer once ran to four sentences and eight citations.
    parts = [p for p in _SENTENCE_SPLIT_RE.split(direct_answer.strip()) if p.strip()]
    if len(parts) > MAX_ANSWER_SENTENCES:
        direct_answer = " ".join(parts[:MAX_ANSWER_SENTENCES])
        stats["answer_truncated"] = True
    strict_terms = list(frame.define_terms) if has_scope else []
    if strict_terms and verdict is None:
        # Claiming a strict bar ("losslessly", "exact equivalence") needs an answer fact that
        # states it. Two ways an answer claims it: a leading "yes" (a live run answered "Yes"
        # from facts about approximate transfer), or a clause asserting the qualifier without
        # negating it ("No, ... except where the models match, enabling exact equivalence", when
        # the evidence showed 73-98% accuracy retention).
        says_yes = bool(_YES_RE.match(direct_answer))
        claimed = strict_terms if says_yes else affirmed_terms(direct_answer, strict_terms)
        unmet = [t for t in claimed if not any(
            mentions_any(_finding_text(facts[n - 1]), [t]) for n in answer_nums)]
        if unmet:
            # Remove the answer's sentences that assert the unmet qualifier, rather than only
            # prefixing a warning: a prefixed answer still went on to claim transfer "without
            # loss ... under specific conditions", contradicting its own first line. What is
            # left (or, if nothing is, the answer facts' own claims) follows the verdict.
            stats["strict_bar_unmet"] = unmet
            rest = _YES_RE.sub("", direct_answer, count=1).lstrip() if says_yes \
                else direct_answer.strip()
            parts = [p for p in _SENTENCE_SPLIT_RE.split(rest) if p.strip()]
            kept = [p for p in parts if not affirmed_terms(p, unmet)]
            stats["answer_sentences_removed"] = len(parts) - len(kept)
            if not kept:
                kept = [_field(facts[n - 1], "claim", "").strip() for n in answer_nums[:2]]
            body = " ".join(k for k in kept if k)
            direct_answer = (
                f"Not established: no retrieved source shows the "
                f"{', '.join(repr(t) for t in unmet)} requirement is met."
                + (f" {body[:1].upper()}{body[1:]}" if body else "")
            )
    if has_scope and verdict is None and not any(is_direct(f) for f in facts):
        # Nothing retrieved answers the question within its scope: say so first, rather than
        # presenting adjacent evidence as the answer. (A claim verdict already leads with its
        # own label, including the insufficient one.)
        stats["no_direct_answer"] = True
        direct_answer = (
            f"No retrieved source directly addresses {frame.key_constraint}; the evidence below "
            f"covers related topics only. {direct_answer.strip()}"
        ).strip()
    if answer_nums:
        allowed = " ".join(_finding_text(facts[n - 1]) for n in answer_nums) + " " + topic
        if ungrounded_numbers(direct_answer, allowed):
            stats["answer_ungrounded_number"] += 1           # kept (it is the headline); counted

    # Executive reports are a single paragraph -- whatever the model drafted into `sections`
    # is dropped rather than rendered, instead of asking the model to leave them empty (a
    # small model still writes them anyway).
    # Sentences are checked body sections first, overview sections (abstract / summary) after:
    # duplicate detection keeps the first occurrence, and a repeated result belongs in the
    # detailed section -- checking in reading order once emptied an Experiments section whose
    # sentences the Abstract had already restated.
    drafted = [] if is_executive else list(draft.sections)
    order = sorted(range(len(drafted)), key=lambda i: bool(_OVERVIEW_RE.search(drafted[i].heading)))
    checked: dict[int, list[tuple[str, bool, list[int]]]] = {
        i: [keep(s) for s in drafted[i].sentences] for i in order
    }
    sections = [
        (
            _canonicalize_heading(sec.heading.strip(), heading_template)
            if heading_template else sec.heading.strip(),
            checked[i],
        )
        for i, sec in enumerate(drafted)
    ]
    # The summary is the overview: a summary sentence repeated verbatim in a section is dropped
    # from the summary (113/300 reports repeated at least half of their summary).
    in_sections = {_norm_sentence(t) for _h, sents in sections for t, ok, _n in sents if ok}
    summary = []
    for s in draft.summary:
        if _norm_sentence(s.text) in in_sections:
            stats["dropped_duplicate"] += 1
            continue
        summary.append(keep(s))

    # Reference numbers follow first use in the FINAL text: answer, summary, sections.
    ref_of_fact: dict[int, int] = {}
    references: list[Source] = []
    ref_of_url: dict[str, int] = {}

    def refs_for(nums: list[int]) -> list[int]:
        out: list[int] = []
        for n in nums:
            if n not in ref_of_fact:
                evidence = _field(facts[n - 1], "evidence", []) or []
                if not evidence:
                    continue
                src = evidence[0]
                url = _field(src, "url", "")
                # One reference per paper: arxiv.org / alphaxiv / emergentmind copies of the same
                # arXiv id share a number, listed under the arxiv.org link when one is cited.
                key = paper_key({"url": url}) or url
                source = src if isinstance(src, Source) else Source(**src)
                if key not in ref_of_url:
                    references.append(source)
                    ref_of_url[key] = len(references)
                elif url != (listed := references[ref_of_url[key] - 1]).url:
                    stats["merged_mirror_refs"] += 1
                    if "arxiv.org" in url and "arxiv.org" not in listed.url:
                        references[ref_of_url[key] - 1] = source
                ref_of_fact[n] = ref_of_url[key]
            out.append(ref_of_fact[n])
        return sorted(set(out))

    answer = _with_marker(direct_answer, refs_for(answer_nums))
    summary_out = [_with_marker(t, refs_for(nums)) for t, ok, nums in summary if ok]
    if is_executive and len(summary_out) > EXECUTIVE_MAX_SUMMARY_SENTENCES:
        summary_out = summary_out[:EXECUTIVE_MAX_SUMMARY_SENTENCES]
        stats["executive_summary_truncated"] = True
    body_sections: list[ReportSection] = []
    for heading, sents in sections:
        rendered: list[str] = []
        cited: set[int] = set()
        for t, ok, nums in sents:
            if not ok:
                continue
            refs = refs_for(nums)
            cited.update(refs)
            rendered.append(_with_marker(t, refs))
        if rendered:
            body_sections.append(ReportSection(
                heading=heading or "Findings", body_md=" ".join(rendered),
                citations=sorted(cited),
            ))

    # Comparison table (question frame's compare_items): rendered and checked in code. A cited
    # cell must not add numbers its facts don't state; an uncited cell may only be a short,
    # number-free, definitional phrase without strong-claim wording -- otherwise it reads
    # "not established".
    table = None if is_executive else draft.comparison
    if table is not None and table.rows and table.columns:
        columns = [c.strip() for c in table.columns if c.strip()][:MAX_TABLE_COLUMNS]

        def cell_md(cell: TableCell | None) -> tuple[str, list[int]]:
            text = " ".join((cell.text if cell else "").replace("|", "/").split())
            nums = primary_only(valid(cell.facts)) if cell else []
            if _META_RE.search(text):
                # the writer's own vocabulary ("F1, F2, F3 ...") listed as a cell's content
                stats["table_cells_unbacked"] += 1
                return "not established", []
            if nums:
                allowed = " ".join(_finding_text(facts[n - 1]) for n in nums) + " " + topic
                if not ungrounded_numbers(text, allowed):
                    refs = refs_for(nums)
                    return (f"{text} [{', '.join(map(str, refs))}]" if refs else text), refs
            if (not text or text.lower().rstrip(".") in _NOT_ESTABLISHED
                    or re.search(r"\d", text) or len(text.split()) > MAX_UNCITED_CELL_WORDS
                    or strong_terms(text)):
                if text and text.lower().rstrip(".") not in _NOT_ESTABLISHED:
                    stats["table_cells_unbacked"] += 1
                return "not established", []
            return text, []

        lines = ["| Item | " + " | ".join(columns) + " |",
                 "|" + "---|" * (len(columns) + 1)]
        cited: set[int] = set()
        for row in table.rows[:MAX_TABLE_ROWS]:
            cells = []
            for k in range(len(columns)):
                md, refs = cell_md(row.cells[k] if k < len(row.cells) else None)
                cells.append(md)
                cited.update(refs)
            lines.append(f"| {' '.join(row.item.replace('|', '/').split())} | "
                         + " | ".join(cells) + " |")
        synthesis = {s.heading for s in REPORT_SECTIONS.get(audience_key, ())
                     if s.kind == "synthesis"}
        at = next((i for i, s in enumerate(body_sections) if s.heading in synthesis),
                  len(body_sections))
        body_sections.insert(at, ReportSection(
            heading=COMPARISON_HEADING, body_md="\n".join(lines), citations=sorted(cited),
        ))
        stats["table_rows"] = min(len(table.rows), MAX_TABLE_ROWS)

    # Optional labelled reasoning: uncited, number-free, short -- never evidence.
    if analysis_enabled and not is_executive:
        kept_analysis = [
            s.strip() for s in getattr(draft, "analysis", []) or []
            if s.strip() and not re.search(r"\d|\[", s) and len(s.split()) <= MAX_ANALYSIS_WORDS
        ][:MAX_ANALYSIS_SENTENCES]
        stats["analysis_sentences"] = len(kept_analysis)
        if kept_analysis:
            body_sections.append(ReportSection(
                heading=ANALYSIS_HEADING, body_md=" ".join(kept_analysis), citations=[],
            ))

    gaps = evidence_gaps(facts, sub_questions)
    stats["gap_sub_questions"] = len(gaps)

    exec_summary = f"**Answer:** {answer}"
    if summary_out:
        exec_summary += "\n\n" + " ".join(summary_out)
    stats["empty"] = not summary_out and not any(
        s.heading != ANALYSIS_HEADING for s in body_sections
    )
    stats["n_sentences"] = len(summary_out) + sum(
        1 for _h, sents in sections for _t, ok, _n in sents if ok
    )
    stats["n_references"] = len(references)

    title = draft.title.strip() or f"Research Report: {topic}"
    if any(_names_label(title, lab) for lab in conflicting):
        title = f"Claim check: {topic}"                  # never headline a different verdict
    report = FinalReport(
        title=title,
        exec_summary=exec_summary,
        sections=body_sections,
        references=references,
        methodology=_methodology(plan),
        limitations=with_gaps(draft.limitations, gaps),
    )
    return report, stats


# --------------------------------------------------------------------------- #
# Safety net for the free-form fallback writer (run_writer): it has no per-sentence
# citation discipline, so on a weak model it is the one path that can state an invented or
# wrong number with nothing to catch it. These two functions give it the same guarantee the
# attributed path has, without another LLM call.
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_ungrounded_sentences(text: str, allowed: str) -> tuple[str, int]:
    """*text* with any sentence whose numbers are not in *allowed* removed; (text, n_dropped)."""
    if not text.strip():
        return text, 0
    kept, dropped = [], 0
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        if sentence.strip() and ungrounded_numbers(sentence, allowed):
            dropped += 1
            continue
        if sentence.strip():
            kept.append(sentence)
    return " ".join(kept), dropped


def ground_free_form_report(
    report: FinalReport, facts: list, topic: str,
    sub_questions: list[str] | tuple[str, ...] = (),
) -> tuple[FinalReport, dict]:
    """Strip numbers the free-form writer invented from *report*, using the same source text the
    facts were extracted from. A section left empty by stripping is dropped. Sub-questions with
    no direct fact are listed in ``limitations``, as ``render_report`` does.

    ``stats["empty"]`` is True when nothing survives; the caller should then use
    ``deterministic_report`` instead of showing a report with no real content.
    """
    allowed = topic + " " + " ".join(_finding_text(f) for f in facts)
    summary, n1 = _strip_ungrounded_sentences(report.exec_summary, allowed)
    sections: list[ReportSection] = []
    n_dropped = n1
    for sec in report.sections:
        body, n = _strip_ungrounded_sentences(sec.body_md, allowed)
        n_dropped += n
        if body.strip():
            sections.append(sec.model_copy(update={"body_md": body}))
    gaps = evidence_gaps(facts, sub_questions)
    stats = {"dropped_ungrounded_number": n_dropped, "empty": not summary.strip() and not sections,
             "gap_sub_questions": len(gaps)}
    grounded = report.model_copy(update={
        "exec_summary": summary, "sections": sections,
        "limitations": with_gaps(report.limitations, gaps),
    })
    return grounded, stats


def deterministic_report(
    facts: list, topic: str, plan: Any = None, audience: str = "general",
) -> FinalReport:
    """A report built with no LLM call: each fact's own (already-grounded) claim text, verbatim,
    one section per sub-question, citations assembled the same way ``render_report`` does. The
    last-resort fallback when even the free-form writer's output has nothing grounded left.

    For the "executive" audience there is no per-sub-question breakdown -- every fact's claim
    is joined into the single summary paragraph instead, matching what ``render_report`` does
    for a real draft.
    """
    references: list[Source] = []
    ref_of_url: dict[str, int] = {}

    def ref_for(n: int) -> int | None:
        ev = (_field(facts[n - 1], "evidence", []) or [None])[0]
        url = _field(ev, "url", "") if ev else ""
        if not url:
            return None
        if url not in ref_of_url:
            references.append(ev if isinstance(ev, Source) else Source(**ev))
            ref_of_url[url] = len(references)
        return ref_of_url[url]

    lead = _field(facts[0], "claim", "").strip()
    limitations = with_gaps(
        "Generated from extracted facts directly; the writer's own draft could not be used.",
        evidence_gaps(facts, list(_field(plan, "sub_questions", []) or []) if plan else []),
    )
    is_executive = (audience or "general").strip().lower() == "executive"
    if is_executive:
        nums = list(range(1, min(len(facts), EXECUTIVE_MAX_SUMMARY_SENTENCES) + 1))
        lines = []
        for n in nums:
            ref = ref_for(n)
            claim = _field(facts[n - 1], "claim", "").strip()
            lines.append(f"{claim} [{ref}]" if ref else claim)
        exec_summary = (
            f"**Answer:** {' '.join(lines)}" if lines else "Insufficient evidence was gathered."
        )
        return FinalReport(
            title=f"Research Report: {topic}",
            exec_summary=exec_summary, sections=[], references=references,
            methodology=_methodology(plan),
            limitations=limitations,
        )

    by_sq: dict[str, list[int]] = {}
    for n in range(1, len(facts) + 1):
        by_sq.setdefault(_field(facts[n - 1], "sub_question", "") or topic, []).append(n)

    sections = []
    for sq, nums in by_sq.items():
        lines = []
        for n in nums:
            ref = ref_for(n)
            claim = _field(facts[n - 1], "claim", "").strip()
            lines.append(f"{claim} [{ref}]" if ref else claim)
        sections.append(ReportSection(heading=sq, body_md=" ".join(lines),
                                      citations=sorted({ref_for(n) for n in nums if ref_for(n)})))
    return FinalReport(
        title=f"Research Report: {topic}",
        exec_summary=f"**Answer:** {lead}" if lead else "Insufficient evidence was gathered.",
        sections=sections, references=references,
        methodology=_methodology(plan),
        limitations=limitations,
    )
