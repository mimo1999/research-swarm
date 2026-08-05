"""Run a reproducible two-hour smoke benchmark over four public datasets."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from research_swarm.agents.base import get_agent_llm
from research_swarm.config import settings
from research_swarm.eval.faithfulness import score_report
from research_swarm.graph.builder import build_graph, get_thread_config
from research_swarm.rag.indexes import get_embed_model
from research_swarm.rag.ingestion import IngestionPipeline
from research_swarm.runtime.budget import clear_budget
from research_swarm.schemas.query import ResearchDepth, ResearchQuery
from research_swarm.tools.retriever_tool import build_retriever_tool

SEED = 42
DEFAULT_LIMIT = 24
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


def _sample_alce(rng: random.Random) -> list[BenchmarkTask]:
    root = DATA_ROOT / "alce" / "ALCE-data"
    specs = [
        ("asqa", "asqa_eval_gtr_top100_reranked_oracle.json", 3),
        ("qampari", "qampari_eval_gtr_top100_reranked_oracle.json", 3),
        ("eli5", "eli5_eval_bm25_top100_reranked_oracle.json", 2),
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


def _sample_hotpot(rng: random.Random) -> list[BenchmarkTask]:
    table = pq.read_table(DATA_ROOT / "hotpotqa" / "validation-0000.parquet")
    rows = table.to_pylist()
    tasks: list[BenchmarkTask] = []
    for question_type in ("bridge", "comparison"):
        candidates = [row for row in rows if row["type"] == question_type]
        for row in rng.sample(candidates, 4):
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
                    metadata={"level": row["level"], "type": question_type},
                )
            )
    return tasks


def _scifact_label(row: dict[str, Any]) -> str:
    if not row["evidence"]:
        return "NOT_ENOUGH_INFO"
    first_group = next(iter(row["evidence"].values()))
    label = first_group[0]["label"]
    return {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT"}[label]


def _sample_scifact(rng: random.Random) -> list[BenchmarkTask]:
    root = DATA_ROOT / "scifact" / "data"
    claims = _read_jsonl(root / "claims_dev.jsonl")
    corpus = {str(row["doc_id"]): row for row in _read_jsonl(root / "corpus.jsonl")}
    by_label: dict[str, list[dict[str, Any]]] = {
        label: [row for row in claims if _scifact_label(row) == label]
        for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
    }
    requested = {"SUPPORT": 3, "CONTRADICT": 3, "NOT_ENOUGH_INFO": 2}
    tasks: list[BenchmarkTask] = []
    for label, count in requested.items():
        for row in rng.sample(by_label[label], count):
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
                    metadata={"label": label, "claim_id": row["id"]},
                )
            )
    return tasks


def build_tasks(limit: int = DEFAULT_LIMIT) -> list[BenchmarkTask]:
    rng = random.Random(SEED)
    tasks = _sample_alce(rng) + _sample_hotpot(rng) + _sample_scifact(rng)
    return tasks[:limit]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _answer_score(text: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = _normalize(text)
    hits = sum(_normalize(item) in normalized for item in expected if item.strip())
    return hits / len(expected)


def _report_text(report: Any) -> str:
    return " ".join(
        [
            report.title,
            report.exec_summary,
            *(section.body_md for section in report.sections or []),
        ]
    )


def _ingest_task(task: BenchmarkTask, session_id: str) -> int:
    pipeline = IngestionPipeline(session_id)
    embed_model = get_embed_model()
    chunks = 0
    for index, doc in enumerate(task.evidence):
        chunks += pipeline.ingest_text(
            doc["text"],
            {
                "url": f"benchmark://{task.id}/{index}",
                "title": doc["title"],
                "source_type": "retriever",
                "credibility_score": 1.0,
            },
            embed_model,
        )
    return chunks


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
        "active_worker_role": None,
    }


async def run_task(
    task: BenchmarkTask,
    timeout: float,
    model_provider: str = "ollama",
    model_name: str = "minimax-m2.5:cloud",
    worker_model: str | None = None,
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
        chunks = await asyncio.to_thread(_ingest_task, task, session_id)
        graph = build_graph(interrupt_before_writer=False)
        config = get_thread_config(session_id)

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
        text = _report_text(report)
        result.update(
            {
                "status": "ok",
                "chunks": chunks,
                "answer_score": _answer_score(text, task.expected),
                "faithfulness": (
                    score_report(report, report.references or [])
                    if report.references
                    else 0.0
                ),
                "grounded": bool(report.references),
                "findings": len(state.get("findings") or []),
                "references": len(report.references or []),
                "sections": len(report.sections or []),
                "report": report.model_dump(mode="json"),
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
    return result


def _write_summary(path: Path, results: list[dict[str, Any]], elapsed: float) -> None:
    successful = [row for row in results if row["status"] == "ok"]
    summary = {
        "seed": SEED,
        "model": results[0].get("model", "") if results else "",
        "worker_model": results[0].get("worker_model", "") if results else "",
        "tasks": len(results),
        "successful": len(successful),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "elapsed_seconds": round(elapsed, 3),
        "median_task_seconds": (
            round(statistics.median(row["seconds"] for row in results), 3)
            if results
            else 0
        ),
        "mean_answer_score": (
            round(statistics.mean(row["answer_score"] for row in successful), 4)
            if successful
            else 0
        ),
        "mean_faithfulness": (
            round(statistics.mean(row["faithfulness"] for row in successful), 4)
            if successful
            else 0
        ),
        "grounded_rate": (
            round(statistics.mean(float(row["grounded"]) for row in successful), 4)
            if successful
            else 0
        ),
        "by_dataset": {},
    }
    for dataset in sorted({row["dataset"] for row in results}):
        rows = [row for row in results if row["dataset"] == dataset]
        ok = [row for row in rows if row["status"] == "ok"]
        summary["by_dataset"][dataset] = {
            "tasks": len(rows),
            "successful": len(ok),
            "mean_answer_score": (
                round(statistics.mean(row["answer_score"] for row in ok), 4)
                if ok
                else 0
            ),
            "mean_seconds": round(statistics.mean(row["seconds"] for row in rows), 3),
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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

    # Keep benchmark retrieval deterministic and avoid an extra LLM inside LlamaIndex.
    #
    # query_engines.probe_ollama() caches its result for only 60s. The old
    # one-time `_ollama_probe_cache = (False, ...)` assignment expired mid-run
    # on any benchmark run longer than 60s (this one takes minutes), after
    # which a live probe found Ollama reachable and later tasks silently
    # switched to LlamaIndex's SubQuestionQueryEngine routing path -- which
    # returns zero evidence, corrupting grounded_rate/faithfulness for
    # whichever tasks happened to run after the flip. Patch the function
    # itself so it stays disabled for the entire run, not just 60s of it.
    import research_swarm.graph.nodes as nodes
    import research_swarm.rag.query_engines as query_engines

    query_engines.probe_ollama = lambda: False
    nodes._get_researcher_tools = lambda max_sources=None, session_id=None: [
        build_retriever_tool(max_sources=max_sources, session_id=session_id)
    ]

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
    tasks = build_tasks(args.limit)
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
            result = await run_task(task, args.timeout, mp, mn, worker_model=wm)
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
    _write_summary(summary_path, results, elapsed)
    print(f"RESULTS {results_path}")
    print(f"SUMMARY {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud",
                        help="Ollama model name (e.g. minimax-m2.5:cloud)")
    parser.add_argument("--worker-model", type=str, default=None,
                        help="Model for the 'standard' (worker) tier only -- "
                             "supervisor/critic/fact-checker/writer stay on --model. "
                             "Omit for uniform mode (every tier on --model).")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
