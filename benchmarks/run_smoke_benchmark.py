"""Run a reproducible smoke benchmark over public QA / claim-verification datasets.

Closed-corpus setup: each task's supplied evidence documents are handed to the graph
as ``ingested_documents`` (the same path user-uploaded PDFs take -- one full-text
extraction call per document, no chunking, no embeddings, no vector store), the
web/literature scout is switched off, and workers may only consult the task's own
corpus. So the score measures the swarm's reading, reasoning and writing over a known
evidence set, not its search.

Every metric below is computed without an LLM (see ``compute_task_metrics``), so a run's
scoring is deterministic and free. Per task:

  correctness   answer_score, evidence_answerable (its ceiling), normalized_answer_score
  trust         number_grounding (numbers in the report that appear in the corpus),
                abstention on unanswerable tasks / over-abstention on answerable ones
  citations     support_doc_recall / support_doc_precision against gold documents
  localization  finding_recall vs answer_score (loss after extraction), false_refute and
                writer_drop rates (answer-bearing findings the verifier/writer discarded)
  efficiency    seconds, per-stage wall time, LLM calls, tokens, reasoning share, LLM errors

plus the independent LLM judge's ``judge_overall`` / verdict when the graph ran it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import mean_or_none as _mean_or_none  # noqa: E402
from bench_common import percentile as _percentile  # noqa: E402
from bench_common import rate as _rate  # noqa: E402
from bench_common import trace_metrics as _trace_metrics  # noqa: E402

from research_swarm.agents._utils import _latest_verdicts
from research_swarm.agents.base import get_agent_llm
from research_swarm.config import settings
from research_swarm.eval.numbers import (  # noqa: F401  (re-exported)
    MIN_COUNTED_INT,
    extract_numbers,
    number_grounding,
)
from research_swarm.graph.builder import build_graph, get_thread_config
from research_swarm.runtime.budget import clear_budget
from research_swarm.schemas.query import ResearchDepth, ResearchQuery

SEED = 42
DATA_ROOT = Path("data/benchmarks")
RESULTS_ROOT = Path("data/benchmark_results")


@dataclass
class BenchmarkTask:
    id: str
    dataset: str
    prompt: str
    evidence: list[dict[str, str]]
    expected: list[str]
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_counts(n: int, weights: tuple[int, ...]) -> list[int]:
    """Split *n* across buckets in proportion to *weights* (largest-remainder; sums to *n*)."""
    total = sum(weights)
    raw = [n * w / total for w in weights]
    counts = [int(r) for r in raw]
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


# n == 8 is the original 24-task sample; keep its 3/3/2 mix so those samples are unchanged.
_LEGACY_N = 8


def _mix(n: int) -> tuple[int, int, int]:
    return (3, 3, 2) if n == _LEGACY_N else (1, 1, 1)


def _sample_alce(rng: random.Random, n: int = _LEGACY_N) -> list[BenchmarkTask]:
    root = DATA_ROOT / "alce" / "ALCE-data"
    asqa, qampari, eli5 = _split_counts(n, _mix(n))
    specs = [
        ("asqa", "asqa_eval_gtr_top100_reranked_oracle.json", asqa),
        ("qampari", "qampari_eval_gtr_top100_reranked_oracle.json", qampari),
        ("eli5", "eli5_eval_bm25_top100_reranked_oracle.json", eli5),
    ]
    tasks: list[BenchmarkTask] = []
    for subset, filename, count in specs:
        rows = json.loads((root / filename).read_text(encoding="utf-8"))
        for index in rng.sample(range(len(rows)), count):
            row = rows[index]
            docs = row.get("docs") or row.get("wikipages") or []
            evidence = [
                {
                    "title": doc.get("title", f"{subset} document {i + 1}"),
                    "text": doc.get("text") or doc.get("summary") or "",
                }
                for i, doc in enumerate(docs[:5])
                if doc.get("text") or doc.get("summary")
            ]
            if subset == "asqa":
                question = row.get("ambiguous_question") or row.get("question")
                qa_pairs = row.get("qa_pairs", [])
                expected = [
                    answer
                    for pair in qa_pairs
                    for answer in pair.get("short_answers", [])[:1]
                ]
            elif subset == "qampari":
                question = row["question"]
                expected = [answers[0] for answers in row.get("answers", []) if answers]
            else:
                question = row["question"]
                expected = row.get("claims", [])
            tasks.append(
                BenchmarkTask(
                    id=f"alce-{subset}-{index}",
                    dataset=f"alce/{subset}",
                    prompt=(
                        f"Using only the supplied benchmark corpus, answer this question: "
                        f"{question}"
                    ),
                    evidence=evidence,
                    expected=expected,
                    metadata={"source_index": index},
                )
            )
    return tasks


def _sample_hotpot(rng: random.Random, n: int = _LEGACY_N) -> list[BenchmarkTask]:
    table = pq.read_table(DATA_ROOT / "hotpotqa" / "validation-0000.parquet")
    rows = table.to_pylist()
    tasks: list[BenchmarkTask] = []
    counts = _split_counts(n, (1, 1))
    for question_type, count in zip(("bridge", "comparison"), counts, strict=True):
        candidates = [row for row in rows if row["type"] == question_type]
        for row in rng.sample(candidates, count):
            context = row["context"]
            evidence = [
                {"title": title, "text": " ".join(sentences)}
                for title, sentences in zip(
                    context["title"], context["sentences"], strict=True
                )
            ]
            tasks.append(
                BenchmarkTask(
                    id=f"hotpotqa-{row['id']}",
                    dataset=f"hotpotqa/{question_type}",
                    prompt=(
                        "Using only the supplied benchmark corpus, answer the following "
                        f"multi-hop question clearly: {row['question']}"
                    ),
                    evidence=evidence,
                    expected=[row["answer"]],
                    metadata={
                        "level": row["level"], "type": question_type,
                        # The 2 gold supporting documents among the 10 (8 are distractors).
                        "gold_docs": sorted(set(row["supporting_facts"]["title"])),
                    },
                )
            )
    return tasks


def _scifact_label(row: dict[str, Any]) -> str:
    if not row["evidence"]:
        return "NOT_ENOUGH_INFO"
    first_group = next(iter(row["evidence"].values()))
    label = first_group[0]["label"]
    return {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT"}[label]


def _sample_scifact(rng: random.Random, n: int = _LEGACY_N) -> list[BenchmarkTask]:
    root = DATA_ROOT / "scifact" / "data"
    claims = _read_jsonl(root / "claims_dev.jsonl")
    corpus = {str(row["doc_id"]): row for row in _read_jsonl(root / "corpus.jsonl")}
    by_label: dict[str, list[dict[str, Any]]] = {
        label: [row for row in claims if _scifact_label(row) == label]
        for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
    }
    requested = dict(zip(
        ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"), _split_counts(n, _mix(n)), strict=True,
    ))
    tasks: list[BenchmarkTask] = []
    for label, count in requested.items():
        for row in rng.sample(by_label[label], min(count, len(by_label[label]))):
            evidence = []
            for doc_id in row.get("cited_doc_ids", [])[:5]:
                doc = corpus.get(str(doc_id))
                if doc:
                    evidence.append(
                        {"title": doc["title"], "text": " ".join(doc["abstract"])}
                    )
            tasks.append(
                BenchmarkTask(
                    id=f"scifact-{row['id']}",
                    dataset="scifact",
                    prompt=(
                        "Using only the supplied scientific abstracts, classify the claim "
                        "as exactly SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO, then explain "
                        f"the verdict: {row['claim']}"
                    ),
                    evidence=evidence,
                    expected=[label],
                    metadata={
                        "label": label, "claim_id": row["id"],
                        "gold_docs": [doc["title"] for doc in evidence],
                    },
                )
            )
    return tasks


def build_tasks(
    limit: int | None = None, datasets: list[str] | None = None, n_per_dataset: int = _LEGACY_N,
) -> list[BenchmarkTask]:
    """The fixed seed-42 sample: *n_per_dataset* tasks from each of ALCE, HotpotQA and SciFact
    (the default 8 is the original 24-task set). *datasets* keeps tasks whose dataset starts
    with any of the given prefixes (e.g. ``["hotpotqa", "scifact"]``); *limit* is applied
    after (None = no cap)."""
    rng = random.Random(SEED)
    tasks = (
        _sample_alce(rng, n_per_dataset) + _sample_hotpot(rng, n_per_dataset)
        + _sample_scifact(rng, n_per_dataset)
    )
    if datasets:
        tasks = [t for t in tasks if any(t.dataset.startswith(d) for d in datasets)]
    return tasks[:limit]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _answer_score(text: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = _normalize(text)
    hits = sum(_normalize(item) in normalized for item in expected if item.strip())
    return hits / len(expected)


_VERDICT_RE = re.compile(
    r"\b(NOT[\s_-]+ENOUGH[\s_-]+INFO(?:RMATION)?|SUPPORT(?:S|ED)?|CONTRADICT(?:S|ED)?)\b",
    re.IGNORECASE,
)


def _scifact_verdict(text: str) -> str | None:
    """The first SUPPORT / CONTRADICT / NOT_ENOUGH_INFO verdict named in *text*."""
    match = _VERDICT_RE.search(text)
    if match is None:
        return None
    word = match.group(1).upper()
    if word.startswith("NOT"):
        return "NOT_ENOUGH_INFO"
    return "SUPPORT" if word.startswith("SUPPORT") else "CONTRADICT"


def _task_score(task: BenchmarkTask, text: str) -> float:
    """Answer score for one task.

    SciFact is a classification task, so it is scored on the verdict the report
    names first (1.0 exact match, else 0.0) -- substring-matching the label made
    every SUPPORT/CONTRADICT task score 0 and every NOT_ENOUGH_INFO task score 1
    regardless of the answer. Everything else: fraction of expected answers found.
    """
    if task.dataset == "scifact":
        return float(_scifact_verdict(text) == task.expected[0])
    return _answer_score(text, task.expected)


# --------------------------------------------------------------------------- #
# Deterministic metrics
# --------------------------------------------------------------------------- #

_ABSTAIN_RE = re.compile(
    r"\b(?:insufficient (?:evidence|information)"
    r"|not enough (?:evidence|information|info)"
    r"|(?:does|do|did) not (?:provide|contain|state|mention|specify|address|include)"
    r"|(?:is|are) not (?:stated|specified|mentioned|provided|addressed)"
    r"|cannot be determined|can(?:not|'t) (?:be )?(?:determined|answered|concluded)"
    r"|unable to (?:determine|answer)|not possible to (?:determine|answer)"
    r"|no (?:direct )?(?:evidence|information|data)"
    r"|NOT_ENOUGH_INFO)\b",
    re.IGNORECASE,
)


def _is_unanswerable(task: BenchmarkTask, evidence_answerable: float | None) -> bool:
    """Tasks the supplied corpus cannot answer: expected answers absent from it (quoted-answer
    datasets) or SciFact's NOT_ENOUGH_INFO."""
    if task.dataset == "scifact":
        return task.expected[0] == "NOT_ENOUGH_INFO"
    return evidence_answerable == 0


def _abstained(task: BenchmarkTask, exec_summary: str, text: str) -> bool:
    """Did the report decline to answer? SciFact: it named NOT_ENOUGH_INFO. Otherwise the
    executive summary (the report's top-line conclusion, not its caveats) says the
    information is missing. A keyword heuristic -- see _ABSTAIN_RE."""
    if task.dataset == "scifact":
        return _scifact_verdict(text) == "NOT_ENOUGH_INFO"
    return bool(_ABSTAIN_RE.search(exec_summary))


def _cited_titles(report_refs: list[str], documents: list[dict[str, str]]) -> set[str]:
    by_url = {d["url"]: d["title"] for d in documents}
    return {by_url[u] for u in report_refs if u in by_url}


def compute_task_metrics(
    task: BenchmarkTask,
    documents: list[dict[str, str]],
    report: dict[str, Any],
    findings: list[dict[str, Any]],
    critiques: list[dict[str, Any]],
) -> dict[str, Any]:
    """All deterministic, LLM-free metrics for one finished task.

    *report* is ``FinalReport.model_dump()``; *findings* / *critiques* are the graph's final
    state as dicts with ``id, claim, confidence, evidence_urls`` and ``finding_id, verdict``
    (see ``_findings_payload``). Pure function of its inputs so it can be re-run on saved results.
    """
    sections = report.get("sections") or []
    exec_summary = report.get("exec_summary", "")
    text = " ".join([report.get("title", ""), exec_summary, *(sec["body_md"] for sec in sections)])
    corpus = " ".join(d["text"] for d in documents)
    is_scifact = task.dataset == "scifact"

    # --- correctness --------------------------------------------------------------------
    answer_score = _task_score(task, text)
    ceiling = None if is_scifact else _answer_score(corpus, task.expected)
    m: dict[str, Any] = {
        "answer_score": answer_score,
        "evidence_answerable": ceiling,
        "normalized_answer_score": (
            round(min(answer_score / ceiling, 1.0), 4) if ceiling else None
        ),
    }

    # --- trust: numbers must come from the evidence; abstain when it cannot answer ---------
    m["number_grounding"] = number_grounding(text, corpus, task.prompt)
    unanswerable = _is_unanswerable(task, ceiling)
    m["unanswerable"] = unanswerable
    m["abstained"] = _abstained(task, exec_summary, text)

    # --- citations vs gold documents --------------------------------------------------------
    gold = set(task.metadata.get("gold_docs") or [])
    cited = _cited_titles([r["url"] for r in report.get("references") or []], documents)
    if gold:
        hit = len(gold & cited)
        m["support_doc_recall"] = round(hit / len(gold), 4)
        m["support_doc_precision"] = round(hit / len(cited), 4) if cited else 0.0
    else:
        m["support_doc_recall"] = m["support_doc_precision"] = None

    # --- localization: where does the answer get lost? ---------------------------------------
    if is_scifact:
        m.update(finding_recall=None, synthesis_loss=None, answer_findings=None,
                 false_refuted=None, writer_dropped=None)
    else:
        finding_recall = _answer_score(" ".join(f["claim"] for f in findings), task.expected)
        verdicts = _latest_verdicts(critiques)
        bearing = [f for f in findings if _answer_score(f["claim"], task.expected) > 0]
        m.update(
            finding_recall=finding_recall,
            # >0: the answer was in the findings but did not survive into the report
            synthesis_loss=round(finding_recall - answer_score, 4),
            answer_findings=len(bearing),
            # answer-bearing findings the verifier called weak/refuted (hedged or dropped)
            false_refuted=sum(verdicts.get(f["id"]) in ("weak", "refuted") for f in bearing),
            # ... and those the writer never sees (its rule: refuted, or confidence < 0.1)
            writer_dropped=sum(
                verdicts.get(f["id"]) == "refuted" or f.get("confidence", 1.0) < 0.1
                for f in bearing
            ),
        )
    return m


def _findings_payload(state: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Final-state findings and critiques as plain dicts (also saved with each result)."""
    findings = [
        {
            "id": f.id, "sub_question": f.sub_question, "claim": f.claim,
            "confidence": f.confidence, "evidence_urls": [e.url for e in f.evidence],
        }
        for f in state.get("findings") or []
    ]
    critiques = [
        {"finding_id": c.finding_id, "verdict": getattr(c.verdict, "value", str(c.verdict))}
        for c in state.get("critiques") or []
    ]
    return findings, critiques


def _report_text(report: Any) -> str:
    return " ".join(
        [
            report.title,
            report.exec_summary,
            *(section.body_md for section in report.sections or []),
        ]
    )


def _task_documents(task: BenchmarkTask) -> list[dict[str, str]]:
    """The task's evidence as ``ingested_documents`` (what an uploaded PDF becomes)."""
    return [
        {
            "url": f"benchmark://{task.id}/{index}",
            "title": doc["title"],
            "text": doc["text"],
            "source_type": "pdf",
        }
        for index, doc in enumerate(task.evidence)
        if doc.get("text", "").strip()
    ]


# The task whose corpus the closed-corpus tool below may read. A ContextVar (not a
# global) so concurrent tasks each see their own corpus: asyncio tasks, LangGraph
# node tasks and asyncio.to_thread all propagate the ambient context.
_CURRENT_DOCS: ContextVar[list[dict[str, str]]] = ContextVar("smoke_current_docs", default=[])
_TERM_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _terms(text: str) -> set[str]:
    return set(_TERM_RE.findall(text.lower()))


async def _corpus_gap_sources(sub_question: str, query: str, session_id: str) -> list[dict]:
    """Closed-corpus stand-in for gap fill's web search: the task's own documents that share the
    most terms with the sub-question (no network)."""
    want = _terms(f"{sub_question} {query}")
    docs = sorted(
        _CURRENT_DOCS.get(),
        key=lambda d: -len(want & _terms(f"{d['title']} {d['text']}")),
    )
    return [
        {"url": d["url"], "title": d["title"], "text": d["text"][:4000], "source_type": "pdf",
         "credibility_score": 0.8}
        for d in docs[:3]
    ]

def _initial_state(
    task: BenchmarkTask,
    session_id: str,
    model_provider: str = "ollama",
    model_name: str = "minimax-m2.5:cloud",
) -> dict[str, Any]:
    return {
        "messages": [],
        "query": ResearchQuery(
            topic=task.prompt,
            depth=ResearchDepth.shallow,
            max_sources=5,
            audience="technical",
        ),
        "plan": None,
        "findings": [],
        "critiques": [],
        "draft_report": None,
        "final_report": None,
        "human_feedback": None,
        "writer_instructions": None,
        "iteration_count": 0,
        "next_agent": None,
        "session_id": session_id,
        "model_provider": model_provider,
        "model_name": model_name,
        "schema_version": 2,
        "research_rounds": 0,
        "pre_dispatch_finding_ids": [],
        "active_sub_question": None,
        "ingested_documents": _task_documents(task),
    }


async def run_task(
    task: BenchmarkTask,
    timeout: float,
    model_provider: str = "ollama",
    model_name: str = "minimax-m2.5:cloud",
    worker_model: str | None = None,
    llm_concurrency: int = 2,
) -> dict[str, Any]:
    session_id = f"bench-{task.id}-{uuid.uuid4().hex[:6]}"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "task_id": task.id,
        "dataset": task.dataset,
        "status": "error",
        "model": model_name,
        "worker_model": worker_model or model_name,
    }
    try:
        documents = _task_documents(task)
        _CURRENT_DOCS.set(documents)
        graph = build_graph(interrupt_before_writer=False)
        # Caps parallel node tasks. A HotpotQA task fans out one document worker per document
        # (10), and an uncapped burst gets 'too many concurrent requests' 429s from the provider;
        # a failed document worker silently yields no findings, quietly degrading the evidence.
        config = {**get_thread_config(session_id), "max_concurrency": llm_concurrency}

        async def execute() -> Any:
            async for _ in graph.astream(
                _initial_state(task, session_id, model_provider, model_name),
                config,
                stream_mode="updates",
            ):
                pass
            return (await graph.aget_state(config)).values

        state = await asyncio.wait_for(execute(), timeout=timeout)
        report = state.get("final_report")
        if report is None:
            raise RuntimeError("Graph completed without a final report")
        report_dict = report.model_dump(mode="json")
        findings, critiques = _findings_payload(state)
        result.update(
            {
                "status": "ok",
                "documents": len(documents),
                **compute_task_metrics(task, documents, report_dict, findings, critiques),
                "judge_overall": (
                    round(report.llm_judge.overall, 3) if report.llm_judge else None
                ),
                "judge_verdict": (
                    report.llm_judge.verdict.value if report.llm_judge else None
                ),
                "grounded": bool(report.references),
                "findings": len(findings),
                "references": len(report.references or []),
                "sections": len(report.sections or []),
                "research_rounds": state.get("research_rounds"),
                "report": report_dict,
                "finding_details": findings,
                "critique_details": critiques,
            }
        )
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Exceeded {timeout:.0f}s task timeout"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        clear_budget(session_id)
        result["seconds"] = round(time.perf_counter() - started, 3)
        result.update(_trace_metrics(session_id))
        if result["status"] == "ok" and not result.get("llm_calls"):
            # Every LLM call failed (quota / outage) and the stage fallbacks still produced a
            # report; that is not a measurement of the pipeline, so it must not count as ok.
            result["status"] = "llm_unavailable"
            result["error"] = "no LLM call succeeded (provider quota or outage?)"
    return result


def _by_dataset(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in sorted({r["dataset"] for r in results}):
        rows = [r for r in results if r["dataset"] == dataset]
        ok = [r for r in rows if r["status"] == "ok"]

        def col(key: str) -> list[float]:
            return [r[key] for r in ok if r.get(key) is not None]

        out[dataset] = {
            "tasks": len(rows),
            "successful": len(ok),
            "mean_answer_score": _mean_or_none(col("answer_score")),
            "mean_normalized_answer_score": _mean_or_none(col("normalized_answer_score")),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 3),
        }
    return out


def summarize(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    ok = [r for r in results if r["status"] == "ok"]

    def col(key: str, rows: list[dict[str, Any]] | None = None) -> list[float]:
        return [r[key] for r in (ok if rows is None else rows) if r.get(key) is not None]

    nums = [r["number_grounding"] for r in ok if r.get("number_grounding")]
    answerable = [r for r in ok if not r.get("unanswerable")]
    unanswerable = [r for r in ok if r.get("unanswerable")]
    bearing = sum(r.get("answer_findings") or 0 for r in ok)
    stages: dict[str, list[float]] = {}
    for r in ok:
        for stage, secs in (r.get("stage_wall_s") or {}).items():
            stages.setdefault(stage, []).append(secs)
    total_tokens = [
        r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in ok if r.get("llm_calls")
    ]
    reasoning = sum(r.get("reasoning_chars", 0) for r in ok)
    out_chars = sum(r.get("out_chars", 0) for r in ok)
    seconds = [r["seconds"] for r in results]

    return {
        "seed": SEED,
        "model": results[0].get("model", "") if results else "",
        "worker_model": results[0].get("worker_model", "") if results else "",
        "tasks": len(results),
        "successful": len(ok),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "elapsed_seconds": round(elapsed, 3),
        "correctness": {
            "mean_answer_score": _mean_or_none(col("answer_score")),
            "mean_evidence_answerable": _mean_or_none(col("evidence_answerable")),
            "mean_normalized_answer_score": _mean_or_none(col("normalized_answer_score")),
        },
        "trust": {
            # micro-average: every number in every report counts once
            "number_grounding_rate": _rate(
                sum(n["grounded"] for n in nums), sum(n["numbers"] for n in nums),
            ),
            "reports_with_ungrounded_numbers": sum(1 for n in nums if n["grounded"] < n["numbers"]),
            "abstention_rate_on_unanswerable": _rate(
                sum(bool(r["abstained"]) for r in unanswerable), len(unanswerable),
            ),
            "over_abstention_rate_on_answerable": _rate(
                sum(bool(r["abstained"]) for r in answerable), len(answerable),
            ),
            "unanswerable_tasks": len(unanswerable),
            "grounded_rate": _mean_or_none([float(r["grounded"]) for r in ok]),
        },
        "citations": {
            "mean_support_doc_recall": _mean_or_none(col("support_doc_recall")),
            "mean_support_doc_precision": _mean_or_none(col("support_doc_precision")),
        },
        "localization": {
            "mean_finding_recall": _mean_or_none(col("finding_recall")),
            # >0 means answers found during extraction were lost before the report
            "mean_synthesis_loss": _mean_or_none(col("synthesis_loss")),
            "answer_bearing_findings": bearing,
            "false_refute_rate": _rate(sum(r.get("false_refuted") or 0 for r in ok), bearing),
            "writer_drop_rate": _rate(sum(r.get("writer_dropped") or 0 for r in ok), bearing),
        },
        "efficiency": {
            "success_rate": _rate(len(ok), len(results)),
            "seconds_p50": _percentile(seconds, 50),
            "seconds_p95": _percentile(seconds, 95),
            "mean_llm_calls": _mean_or_none(col("llm_calls")),
            "mean_tokens": _mean_or_none(total_tokens),
            "reasoning_share_of_output_chars": _rate(reasoning, reasoning + out_chars),
            # reports built from zero findings (e.g. the plan never happened): they distort every
            # other metric, so a run with a non-trivial rate here should be re-run, not averaged
            "empty_report_rate": _rate(sum(1 for r in ok if not r.get("findings")), len(ok)),
            "mean_llm_errors": _mean_or_none(col("llm_errors")),
            "max_peak_llm_concurrency": max(col("peak_llm_concurrency"), default=None),
            "total_llm_retries": int(sum(col("llm_retries"))),
            "mean_slot_wait_s": _mean_or_none(col("slot_wait_s")),
            "tasks_with_fallbacks": sum(1 for r in ok if r.get("stage_fallbacks")),
            "mean_research_rounds": _mean_or_none(col("research_rounds")),
            "mean_stage_wall_s": {
                k: round(statistics.mean(v), 2) for k, v in sorted(stages.items())
            },
            "answer_score_per_100k_tokens": (
                _rate(
                    (_mean_or_none(col("answer_score")) or 0) * 100_000,
                    statistics.mean(total_tokens),
                ) if total_tokens else None
            ),
        },
        "judge": {
            "mean_overall": _mean_or_none(col("judge_overall")),
            "verdicts": dict(Counter(r["judge_verdict"] for r in ok if r.get("judge_verdict"))),
        },
        "by_dataset": _by_dataset(results),
    }


def _write_summary(
    path: Path, results: list[dict[str, Any]], elapsed: float,
    overrides: dict[str, Any] | None = None,
) -> None:
    summary = summarize(results, elapsed)
    if overrides:
        summary["overrides"] = overrides
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_override(text: str) -> tuple[str, Any]:
    """``KEY=VALUE`` -> (key, value); VALUE is JSON when it parses (true, 3, "x"), else a string."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise SystemExit(f"--set expects KEY=VALUE, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def apply_overrides(items: list[str] | None) -> dict[str, Any]:
    """Apply ``--set`` overrides to the settings singleton; an unknown key is an error so a
    typo cannot silently leave the experiment running the default."""
    applied: dict[str, Any] = {}
    for item in items or []:
        key, value = parse_override(item)
        if not hasattr(settings, key):
            raise SystemExit(f"--set: unknown setting {key!r}")
        setattr(settings, key, value)
        applied[key] = value
    return applied


async def main(args: argparse.Namespace) -> None:
    model_provider = "ollama"
    model_name = args.model
    # --worker-model lets the "standard" (worker) tier run a different, cheaper
    # model than the rest of the pipeline -- omit it for the old uniform-model
    # comparison mode (every tier on --model).
    worker_model = args.worker_model or model_name
    settings.default_model_provider = model_provider
    settings.default_model_name = model_name
    settings.tier_fast_provider = model_provider
    settings.tier_fast_model = model_name
    settings.tier_standard_provider = model_provider
    settings.tier_standard_model = worker_model
    settings.tier_thorough_provider = model_provider
    settings.tier_thorough_model = model_name
    settings.max_sources = 5
    settings.max_llm_calls = 12

    # Closed corpus: no web/literature scout, and gap fill (a sub-question the document pass left
    # thin) may only read the task's own documents.
    import research_swarm.graph.nodes as nodes

    settings.enable_fetch_pass = False
    nodes._get_gap_fill_sources = _corpus_gap_sources

    # get_tiered_llm now auto-picks a cheaper model for the "standard" (worker)
    # tier in production, which defeats a uniform-model comparison run. Replace
    # it with an explicit mapping instead: every tier uses --model, except
    # "standard" which uses --worker-model (same as --model when omitted, i.e.
    # true uniform mode).
    def _uniform_tiered_llm(tier, temperature=0.0, provider_override=None):
        model = worker_model if tier == "standard" else model_name
        return get_agent_llm(provider=model_provider, model=model, temperature=temperature)

    nodes.get_tiered_llm = _uniform_tiered_llm

    print(f"MODEL  {model_provider}/{model_name}  (worker tier: {worker_model})", flush=True)
    overrides = apply_overrides(args.set)
    if args.task_file:
        tasks = [BenchmarkTask(**row) for row in json.loads(
            Path(args.task_file).read_text(encoding="utf-8"))][: args.limit]
    else:
        tasks = build_tasks(
            args.limit, args.datasets.split(",") if args.datasets else None, args.n_per_dataset,
        )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = RESULTS_ROOT / f"smoke-{run_id}-tasks.json"
    results_path = RESULTS_ROOT / f"smoke-{run_id}-results.jsonl"
    summary_path = RESULTS_ROOT / f"smoke-{run_id}-summary.json"
    manifest_path.write_text(
        json.dumps([asdict(task) for task in tasks], indent=2), encoding="utf-8"
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []
    write_lock = asyncio.Lock()
    started = time.perf_counter()

    async def guarded(task: BenchmarkTask, mp: str, mn: str, wm: str) -> None:
        async with semaphore:
            print(f"START {task.id}", flush=True)
            result = await run_task(
                task, args.timeout, mp, mn, worker_model=wm,
                llm_concurrency=args.llm_concurrency,
            )
            async with write_lock:
                results.append(result)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
            print(
                f"DONE  {task.id} status={result['status']} "
                f"seconds={result['seconds']}",
                flush=True,
            )

    await asyncio.gather(*(
        guarded(task, model_provider, model_name, worker_model) for task in tasks
    ))
    elapsed = time.perf_counter() - started
    _write_summary(summary_path, results, elapsed, overrides)
    print(f"RESULTS {results_path}")
    print(f"SUMMARY {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap on tasks (default: all)")
    parser.add_argument("--n-per-dataset", type=int, default=_LEGACY_N,
                        help="tasks per dataset family (8 = the original 24-task sample)")
    parser.add_argument("--datasets", type=str, default="",
                        help="comma-separated dataset prefixes to run, e.g. hotpotqa,scifact")
    parser.add_argument("--task-file", type=str, default="",
                        help="JSON list of tasks (a smoke-*-tasks.json), not the seeded sample")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a settings field for this run (repeatable), e.g. "
                             "--set min_grounded_facts=2 --set llm_judge_enabled=false")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="tasks run in parallel")
    parser.add_argument("--llm-concurrency", type=int, default=2,
                        help="parallel graph nodes (LLM calls) within one task")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model", type=str, default="gemma4:e2b",
                        help="Ollama model name (e.g. gemma4:e2b, nemotron-3-nano:30b-cloud)")
    parser.add_argument("--worker-model", type=str, default=None,
                        help="Model for the 'standard' (worker) tier only -- "
                             "supervisor/verifier/writer stay on --model. "
                             "Omit for uniform mode (every tier on --model).")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
