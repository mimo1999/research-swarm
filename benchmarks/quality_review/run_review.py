# ruff: noqa: E501
"""Independent quality-review harness: run the swarm on a fixed question set
and capture full traces, final reports and summary metrics.

Usage (from repo root, inside the poetry env):
    python benchmarks/quality_review/run_review.py --depth standard --out data/quality_review/run1
    python benchmarks/quality_review/run_review.py --only q1,q2

Uses the local Ollama daemon (http://localhost:11434, proxying -cloud models) with
the product's own default model tiers, so it exercises the shipped
configuration rather than a tuned one. One trace file per question lands in
<out>/traces/, one report JSON in <out>/reports/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_swarm.config import settings  # noqa: E402

QUESTIONS = {
    "q1": "Do GLP-1 receptor agonists slow disease progression in Parkinson's disease?",
    "q2": "How does FlashAttention reduce memory use and speed up transformer attention compared with standard attention?",
    "q3": "RAG versus fine-tuning for adding domain knowledge to LLMs: what are the accuracy and cost trade-offs?",
    "q4": "What is the current state of solid-state battery commercialization for electric vehicles?",
    "q5": "Does intermittent fasting outperform continuous calorie restriction for weight loss and metabolic health?",
    "q6": "What are the measured energy and carbon costs of training and serving large language models?",
    "q7": "How do Apache Kafka and Apache Pulsar differ in architecture and performance for large-scale event streaming?",
    "q8": "What do clinical trials show about CRISPR-based gene editing (exa-cel/Casgevy) for sickle cell disease?",
    # Constraint-qualified questions: a report on the general subject (KV caching, creatine,
    # LoRA) misses the point. The first is the incident behind agents/expansion.py.
    "q9": "Can we losslessly migrate KV cache from one LLM to another?",
    "q10": "Does creatine supplementation improve cognition in vegetarians?",
    "q11": "Do LoRA adapters trained on one base model transfer to a different base model?",
}


def _configure_models(model: str, ollama_url: str) -> None:
    settings.ollama_base_url = ollama_url
    settings.ollama_deployment = "cloud"
    settings.default_model_provider = "ollama"
    settings.default_model_name = model
    for tier in ("fast", "standard", "thorough"):
        setattr(settings, f"tier_{tier}_provider", "ollama")
        setattr(settings, f"tier_{tier}_model", model)
    settings.tier_standard_model_cloud = model


async def run_one(qid: str, topic: str, depth: str, out: Path, max_concurrency: int = 2) -> dict:

    from research_swarm.graph.builder import build_graph, get_thread_config
    from research_swarm.runtime.budget import clear_budget, get_budget
    from research_swarm.runtime.trace import reset_session, trace_event
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery

    session_id = f"review-{qid}-{uuid.uuid4().hex[:6]}"
    reset_session(session_id)
    query = ResearchQuery(
        topic=topic, depth=ResearchDepth(depth), max_sources=settings.max_sources, audience="technical",
    )
    state = {
        "messages": [], "query": query, "plan": None, "findings": [], "critiques": [],
        "draft_report": None, "final_report": None, "human_feedback": None,
        "writer_instructions": None, "iteration_count": 0, "next_agent": None,
        "session_id": session_id, "model_provider": settings.default_model_provider,
        "model_name": settings.default_model_name, "schema_version": 2, "research_rounds": 0,
        "pre_dispatch_finding_ids": [], "active_sub_question": None,
    }
    graph = build_graph(interrupt_before_writer=False)
    # max_concurrency caps parallel node tasks (Send fan-out): Ollama Cloud returned
    # 429 "timed out waiting for a concurrent request slot" with 4 uncapped workers.
    config = {**get_thread_config(session_id), "recursion_limit": 150, "max_concurrency": max_concurrency}

    trace_event(session_id, "harness", "note", text=f"START {qid}: {topic}", depth=depth)
    t0 = time.perf_counter()
    error = None
    try:
        async for _ in graph.astream(state, config, stream_mode="updates"):
            pass
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    trace_event(session_id, "harness", "note", text=f"END {qid}", dur=round(wall, 2), error=error)

    snap = await graph.aget_state(config)
    values = snap.values or {}
    report = values.get("final_report")
    findings = values.get("findings") or []
    critiques = values.get("critiques") or []

    research, review = get_budget(session_id, pool="research"), get_budget(session_id, pool="review")
    result = {
        "qid": qid, "topic": topic, "depth": depth, "session_id": session_id,
        "wall_s": round(wall, 1), "max_concurrency": max_concurrency, "error": error,
        "llm_calls_research": research.used, "llm_calls_review": review.used,
        "tokens_in": research.input_tokens + review.input_tokens,
        "tokens_out": research.output_tokens + review.output_tokens,
        "n_findings": len(findings), "n_critiques": len(critiques),
        "rounds": values.get("research_rounds"),
    }
    if report is not None:
        refs = report.references or []
        result["n_sections"] = len(report.sections or [])
        result["n_references"] = len(refs)
        result["report_words"] = len(
            " ".join([report.exec_summary or ""] + [s.body_md for s in (report.sections or [])]).split()
        )
        if getattr(report, "llm_judge", None) is not None:
            result["llm_judge"] = json.loads(report.llm_judge.model_dump_json())
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / f"{qid}.json").write_text(
            json.dumps(
                {
                    "result": result,
                    "plan": json.loads(values["plan"].model_dump_json()) if values.get("plan") else None,
                    "findings": [json.loads(f.model_dump_json()) for f in findings],
                    "critiques": [json.loads(c.model_dump_json()) for c in critiques],
                    "report": json.loads(report.model_dump_json()),
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    clear_budget(session_id)
    return result


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="standard", choices=["shallow", "standard", "deep"])
    ap.add_argument("--model", default="nemotron-3-nano:30b-cloud")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="local daemon (proxies -cloud models; the product's normal setup)")
    ap.add_argument("--only", default="", help="comma-separated question ids, e.g. q1,q3")
    ap.add_argument("--max-concurrency", type=int, default=2)
    ap.add_argument("--out", default="data/quality_review/run1")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings.data_dir = out  # traces land in <out>/traces, checkpoints unused (MemorySaver)
    _configure_models(args.model, args.ollama_url)

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    tl = logging.getLogger("research_swarm.trace")
    tl.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "trace.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    tl.addHandler(fh)

    ids = [i for i in (args.only.split(",") if args.only else QUESTIONS) if i in QUESTIONS]
    results = []
    for qid in ids:
        print(f"=== {qid}: {QUESTIONS[qid]}", flush=True)
        res = await run_one(qid, QUESTIONS[qid], args.depth, out, args.max_concurrency)
        print(json.dumps(res, ensure_ascii=False)[:600], flush=True)
        results.append(res)
        (out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
