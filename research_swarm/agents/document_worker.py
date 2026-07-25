"""Document worker -- single-shot full-document claim extraction.

Unlike ``run_worker`` (a ReAct tool loop over live web search), a document
worker already has everything it needs: the full text of one ingested
document (or one size-bounded slice of it, for oversized documents -- see
``_split_into_parts``). No tool calls, no ``WorkerRole``, just one
structured-output extraction call against the plan's sub-questions.

Returned ``Finding.sub_question`` values are snapped to the exact string
from the provided sub-questions list, so they slot into the same
merge-by-id reducer, rework loop, and critic that web-research findings
already use -- no separate handling needed downstream.
"""
from __future__ import annotations

import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core import Document as _LIDocument
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel, Field

from research_swarm.agents._utils import recover_from_parse_failure, schema_output_instruction
from research_swarm.schemas import Finding, Source
from research_swarm.schemas.source import SourceType

logger = logging.getLogger(__name__)

# Rough per-call character budget for the "standard" tier, leaving headroom
# for the sub-questions list, system prompt, and structured-output schema
# instructions on top of the document text itself. ~4 chars/token.
MAX_DOC_CHARS = 40_000


class DocumentFindingItem(BaseModel):
    sub_question: str = Field(
        ..., description="Must exactly match one of the provided sub-questions, verbatim"
    )
    claim: str = Field(
        ...,
        description="Concise claim answering that sub-question, grounded only in the document text",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    quote: str = Field(default="", description="A short supporting excerpt from the document text")


class DocumentExtraction(BaseModel):
    items: list[DocumentFindingItem] = Field(
        default_factory=list,
        description=(
            "One item per sub-question this text can help answer. Omit sub-questions "
            "this text doesn't address -- do not force a claim."
        ),
    )


_SYSTEM_PROMPT = """\
You are an expert research assistant extracting claims from a single source document.

You will be given the full text of one document (or a slice of one, if it was too long \
for a single pass) and a list of research sub-questions. For each sub-question this text \
actually helps answer, extract one concise, factual claim grounded ONLY in this text -- \
do not use outside knowledge. If the text doesn't address a sub-question, omit it \
entirely; do not force a claim.

Preserve exact numbers, dates, percentages, and named entities from the text -- do not \
paraphrase a quantitative result into a qualitative gist.\
"""

_JSON_SUFFIX = schema_output_instruction(DocumentExtraction)


def _split_into_parts(text: str, max_chars: int = MAX_DOC_CHARS) -> list[str]:
    """Split *text* into <= max_chars parts at sentence boundaries.

    Reuses LlamaIndex's SentenceSplitter purely as a boundary-finder -- the
    same tool already used for chunking in rag/ingestion.py -- not for
    embedding. Consecutive sentence-level chunks are packed greedily into
    parts so each stays under max_chars, instead of a raw character cut
    that could split mid-sentence. Returns [text] unchanged when it already
    fits in one part.
    """
    if len(text) <= max_chars:
        return [text]

    # chunk_size is measured in tokens; ~4 chars/token is a rough estimate
    # good enough for finding sentence boundaries (not for embedding).
    splitter = SentenceSplitter(chunk_size=max(max_chars // 4, 64), chunk_overlap=0)
    sentence_chunks = [
        n.get_content() for n in splitter.get_nodes_from_documents([_LIDocument(text=text)])
    ]

    parts: list[str] = []
    current = ""
    for chunk in sentence_chunks:
        if current and len(current) + len(chunk) > max_chars:
            parts.append(current)
            current = chunk
        else:
            current = f"{current}\n{chunk}" if current else chunk
    if current:
        parts.append(current)
    return parts or [text]


async def run_document_worker(
    document: dict,
    part_index: int,
    part_total: int,
    part_text: str,
    sub_questions: list[str],
    llm: BaseChatModel,
) -> list[Finding]:
    """Extract claims from one document (or one slice of an oversized one).

    Returns one Finding per sub-question this text actually addresses.
    Sub-question values the model returns are matched case/whitespace-
    insensitively against *sub_questions* and snapped to the canonical
    string; any item that doesn't match a real sub-question is dropped
    rather than silently kept, so it can't slip past the dispatch/rework
    matching downstream (which relies on exact string identity).
    """
    if not sub_questions:
        return []

    sub_q_lookup = {sq.strip().lower(): sq for sq in sub_questions}
    sub_q_block = "\n".join(f"- {sq}" for sq in sub_questions)
    part_note = f" (part {part_index + 1} of {part_total})" if part_total > 1 else ""
    title = document.get("title") or document.get("url", "untitled")

    user_msg = HumanMessage(
        content=(
            f"Document{part_note}: {title}\n\n"
            f"Sub-questions:\n{sub_q_block}\n\n"
            f"Document text:\n{part_text}"
        )
    )

    structured_llm = llm.with_structured_output(DocumentExtraction)
    try:
        result: DocumentExtraction = await structured_llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT + _JSON_SUFFIX), user_msg]
        )
    except Exception as exc:
        logger.warning("Document worker failed for %r%s: %s", document.get("url"), part_note, exc)
        recovered = recover_from_parse_failure(exc, DocumentExtraction)
        if recovered is None:
            return []
        result = recovered

    findings: list[Finding] = []
    for item in result.items:
        canonical_sq = sub_q_lookup.get(item.sub_question.strip().lower())
        if canonical_sq is None:
            logger.warning(
                "Document worker returned sub_question %r not in the plan -- dropping.",
                item.sub_question,
            )
            continue
        findings.append(Finding(
            id=str(uuid.uuid4()),
            claim=item.claim,
            confidence=item.confidence,
            sub_question=canonical_sq,
            evidence=[Source(
                url=document.get("url", ""),
                title=document.get("title", ""),
                snippet=item.quote or part_text[:2000],
                source_type=document.get("source_type", SourceType.pdf),
            )],
        ))
    return findings
