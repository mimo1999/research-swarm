"""Claim-level evaluation of a generated report against the documents it was written from.

The unit is a sentence (like ALCE): the report is split into sentences, each with the ``[n]``
citation markers it carries, and one LLM judge -- which must NOT be the model that wrote the
report -- rules on every sentence against the supplied documents:

  checkworthy     does it assert a specific verifiable fact/number/finding? (transitions,
                  section intros, hedges and opinions don't count)
  cited_support   do the documents this sentence CITES support it?   yes / partial / no
  corpus_support  does ANY supplied document support it?            yes / partial / no
  contradicted    does a supplied document say the opposite?

From those, per report:

  faithfulness           checkworthy sentences supported by the documents / checkworthy
  citation_recall        checkworthy sentences whose OWN citations support them / checkworthy (ALCE)
  citation_correctness   cited sentences whose citations support them / cited sentences
  citation_completeness  document-supported checkworthy sentences that carry a citation / those
  contradictions         sentences a document contradicts (the critical-error signal)
  dangling_citation_rate citations pointing at no supplied document / all citations (no LLM)

Closed corpora (the smoke benchmark) make "supported" well defined: the supplied documents ARE
the evidence, so no external fact-checking is needed. Public API::

    claims = split_claims(report_dict)
    verdicts, dangling = await judge_claims(claims, ref_urls, documents, judge_llm)
    metrics = claim_metrics(claims, verdicts, dangling)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_swarm.agents._utils import (
    ainvoke_with_retry,
    recover_from_parse_failure,
    schema_output_instruction,
)

logger = logging.getLogger(__name__)

MIN_CLAIM_WORDS = 4
DEFAULT_CHUNK = 12
MAX_DOC_CHARS = 3000

_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[\d)")
_ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "vs.", "fig.", "approx.", "no.", "dr.", "cf.")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

@dataclass
class Claim:
    idx: int
    text: str                                   # sentence with citation markers removed
    citations: list[int] = field(default_factory=list)      # 1-based reference numbers
    section: str = ""


def parse_citations(marker_body: str) -> list[int]:
    """"1, 2" -> [1, 2]; "3-5" -> [3, 4, 5]; tolerates en dashes and spaces."""
    out: list[int] = []
    for part in re.split(r"\s*,\s*", marker_body.strip()):
        m = re.fullmatch(r"(\d+)\s*[\-–]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.extend(range(lo, hi + 1) if 0 < hi - lo < 50 else [lo, hi])
        elif part.isdigit():
            out.append(int(part))
    return out


def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)      # bullet / numbered-list marker
    line = re.sub(r"(\*\*|__|`)", "", line)                    # emphasis / code marks
    return line.strip()


# "patients. [4] Next" -> "patients [4]. Next": a marker written after the full stop belongs to
# the sentence it follows, and moving it in front of the stop makes the split unambiguous.
_TRAILING_MARKER_RE = re.compile(
    r"([.!?])((?:\s*\[\d+(?:\s*[,\-–]\s*\d+)*\])+)"
)


def _sentences(text: str) -> list[str]:
    """Split one paragraph into sentences without breaking on abbreviations or trailing [n]."""
    text = _TRAILING_MARKER_RE.sub(r"\2\1", text)
    pieces = [p for p in _SPLIT_RE.split(text) if p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and (
            merged[-1].lower().endswith(_ABBREVIATIONS) or piece[:1].islower()
        ):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


# The writer's optional reasoning section (writer_render.ANALYSIS_HEADING) starts with this.
ANALYSIS_HEADING_PREFIX = "Analysis (reasoning"


def split_claims(report: dict[str, Any]) -> list[Claim]:
    """Sentences of the executive summary and every section body, with their citations.

    *report* is ``FinalReport.model_dump()``. Headings, table rows and fragments shorter than
    ``MIN_CLAIM_WORDS`` words are dropped.
    """
    blocks: list[tuple[str, str]] = [("summary", report.get("exec_summary", "") or "")]
    for sec in report.get("sections") or []:
        heading = sec.get("heading", "") or ""
        if heading.startswith(ANALYSIS_HEADING_PREFIX):
            continue          # labelled reasoning, not claims from sources: not scored
        blocks.append((heading, sec.get("body_md", "") or ""))

    claims: list[Claim] = []
    for section, body in blocks:
        for raw in body.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("|"):
                continue
            for sentence in _sentences(_clean_line(stripped)):
                cites: list[int] = []
                for m in _MARKER_RE.finditer(sentence):
                    cites.extend(parse_citations(m.group(1)))
                text = re.sub(r"\s+", " ", _MARKER_RE.sub(" ", sentence)).strip(" ")
                text = re.sub(r"\s+([.,;:!?])", r"\1", text)
                if len(text.split()) < MIN_CLAIM_WORDS:
                    continue
                claims.append(Claim(
                    idx=len(claims) + 1, text=text,
                    citations=sorted(dict.fromkeys(cites)), section=section,
                ))
    return claims


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #

Support = Literal["yes", "partial", "no"]


class ClaimVerdict(BaseModel):
    claim: int = Field(..., description="The claim's number, exactly as listed (C<number>)")
    checkworthy: bool = Field(
        ..., description="True only if the sentence asserts a specific verifiable fact, number, "
                         "finding or comparison",
    )
    cited_support: Support = Field(
        ..., description="Do the documents THIS sentence cites (its `cites:`) support it? "
                         "yes = fully, partial = only part of it, no = not at all",
    )
    corpus_support: Support = Field(
        ..., description="Does ANY supplied document support it, cited or not?",
    )
    contradicted: bool = Field(
        default=False, description="True if a supplied document states the opposite",
    )


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict] = Field(
        default_factory=list, description="One verdict per claim listed, in order",
    )


_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. You are given SOURCE DOCUMENTS and numbered CLAIMS "
    "taken from a report someone else wrote from those documents. Judge every claim using ONLY "
    "the documents -- never your own knowledge.\n\n"
    "For each claim decide:\n"
    "  checkworthy    true only if it asserts a specific verifiable fact, number, finding or "
    "comparison. Transitions, section introductions, hedges, recommendations and opinions are "
    "NOT checkworthy. Neither are statements about what the sources or research do or do not "
    "contain (e.g. the findings do not mention X, no information was found): an honest "
    "admission of missing evidence is not a factual claim.\n"
    "  cited_support  whether the documents listed in the claim's `cites:` support it -- yes "
    "(fully entailed), partial (only part of the claim is supported), no. Judge these documents "
    "alone; a different document supporting the claim does not count. If `cites:` is "
    "none, answer no.\n"
    "  corpus_support whether ANY of the documents supports it (yes / partial / no).\n"
    "  contradicted   true if a document states the opposite of the claim.\n\n"
    "Be precise about numbers, names and dates: a claim with a wrong figure is not supported. "
    "Return exactly one verdict per claim, in order, using the claim's number."
    + schema_output_instruction(ClaimVerdicts)
)


def _doc_block(documents: list[dict[str, Any]], max_chars: int) -> str:
    return "\n\n".join(
        f"[D{i}] {d.get('title', '')}\n{str(d.get('text', ''))[:max_chars]}"
        for i, d in enumerate(documents, 1)
    )


def _resolve(
    claim: Claim, ref_urls: list[str], doc_index: dict[str, int],
) -> tuple[list[int], int]:
    """(document numbers the claim's citations resolve to, number of dangling citations)."""
    docs, dangling = [], 0
    for n in claim.citations:
        url = ref_urls[n - 1] if 1 <= n <= len(ref_urls) else None
        if url is not None and url in doc_index:
            docs.append(doc_index[url])
        else:
            dangling += 1
    return sorted(dict.fromkeys(docs)), dangling


async def judge_claims(
    claims: list[Claim],
    ref_urls: list[str],
    documents: list[dict[str, Any]],
    llm: BaseChatModel,
    *,
    chunk_size: int = DEFAULT_CHUNK,
    max_doc_chars: int = MAX_DOC_CHARS,
    session_id: str | None = None,
) -> tuple[dict[int, ClaimVerdict | None], dict[int, tuple[int, int]]]:
    """Judge every claim; never raises.

    *ref_urls* is the report's reference list in citation order (``[n]`` -> ``ref_urls[n-1]``);
    *documents* the supplied evidence (``url``, ``title``, ``text``). Returns
    ``(verdicts, dangling)``: verdicts by claim idx (``None`` where the judge failed or
    skipped it), and per claim ``(dangling citations, total citations)`` -- citations that
    resolve to no supplied document are counted here without asking the judge.

    Sentences with no citation are forced to ``cited_support="no"`` and sentences whose
    citations all dangle likewise, regardless of what the judge says.
    """
    doc_index = {d["url"]: i for i, d in enumerate(documents, 1)}
    resolved = {c.idx: _resolve(c, ref_urls, doc_index) for c in claims}
    dangling = {c.idx: (resolved[c.idx][1], len(c.citations)) for c in claims}
    verdicts: dict[int, ClaimVerdict | None] = {c.idx: None for c in claims}
    if not claims:
        return verdicts, dangling

    docs_text = _doc_block(documents, max_doc_chars)
    structured = llm.with_structured_output(ClaimVerdicts)
    for start in range(0, len(claims), chunk_size):
        chunk = claims[start:start + chunk_size]
        listing = "\n".join(
            f"C{c.idx}: {c.text}\n    cites: "
            + (", ".join(f"D{d}" for d in resolved[c.idx][0]) or "none")
            for c in chunk
        )
        msg = HumanMessage(
            content=f"SOURCE DOCUMENTS:\n{docs_text}\n\nCLAIMS ({len(chunk)}):\n{listing}"
        )
        try:
            result: ClaimVerdicts = await ainvoke_with_retry(
                structured, [SystemMessage(content=_SYSTEM_PROMPT), msg],
                agent="claim_judge", session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Claim judge failed on %d claims: %s", len(chunk), exc)
            result = recover_from_parse_failure(exc, ClaimVerdicts)  # type: ignore[assignment]
            if result is None:
                continue                                    # chunk stays None (judge error)
        wanted = {c.idx for c in chunk}
        for v in result.verdicts:
            if v.claim in wanted and verdicts[v.claim] is None:
                verdicts[v.claim] = v

    for c in claims:                                         # deterministic overrides
        v = verdicts[c.idx]
        if v is not None and not resolved[c.idx][0]:
            verdicts[c.idx] = v.model_copy(update={"cited_support": "no"})
    return verdicts, dangling


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def claim_metrics(
    claims: list[Claim],
    verdicts: dict[int, ClaimVerdict | None],
    dangling: dict[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Per-report metrics from judged claims (rates are None when their denominator is 0)."""
    scored = [(c, verdicts[c.idx]) for c in claims if verdicts.get(c.idx) is not None]
    cw = [(c, v) for c, v in scored if v.checkworthy]
    cited = [(c, v) for c, v in cw if c.citations]
    supported = [(c, v) for c, v in cw if v.corpus_support == "yes"]
    n_dangling = sum(d for d, _ in (dangling or {}).values())
    n_citations = sum(t for _, t in (dangling or {}).values())
    return {
        "n_claims": len(claims),
        "n_judged": len(scored),
        "n_checkworthy": len(cw),
        "judge_error_rate": _rate(len(claims) - len(scored), len(claims)),
        "faithfulness": _rate(len(supported), len(cw)),
        "faithfulness_soft": (
            round(sum(1.0 if v.corpus_support == "yes" else 0.5 if v.corpus_support == "partial"
                      else 0.0 for _, v in cw) / len(cw), 4) if cw else None
        ),
        "unsupported_rate": _rate(sum(v.corpus_support == "no" for _, v in cw), len(cw)),
        "citation_recall": _rate(sum(v.cited_support == "yes" for _, v in cw), len(cw)),
        "citation_correctness": _rate(sum(v.cited_support == "yes" for _, v in cited), len(cited)),
        "citation_completeness": _rate(sum(1 for c, _ in supported if c.citations), len(supported)),
        "n_contradicted": sum(v.contradicted for _, v in scored),
        "dangling_citation_rate": _rate(n_dangling, n_citations),
        "n_citations": n_citations,
    }
