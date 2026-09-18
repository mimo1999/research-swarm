"""Aggregate one session's trace (see ``trace.py``) into per-stage time / cost / signal stats.

Used by ``benchmarks/quality_review/analyze_traces.py`` (CLI over a run directory) and by
``benchmarks/run_smoke_benchmark.py`` (per-task efficiency metrics).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _stage_of_llm(agent: str) -> str:
    if agent.endswith("/summarizer"):
        return "worker.summarizer"
    for p in (
        "paper_scout", "paper_worker", "worker", "document_worker", "supervisor",
        "verifier", "gap_fill", "writer", "judge",
    ):
        if agent.startswith(p):
            return p
    return agent


def _union(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _fallback_counts(events: list[dict]) -> dict[str, int]:
    """How often each stage degraded to its fallback (``<stage>.fallback`` note events)."""
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if str(e["agent"]).endswith(".fallback"):
            counts[e["agent"]] += 1
    return dict(counts)


def _peak_concurrency(events: list[dict]) -> int:
    """Max number of LLM requests in flight at once (llm_start .. llm_end/llm_error)."""
    edges: list[tuple[float, int]] = []
    for e in events:
        if e["ev"] == "llm_start":
            edges.append((e["t"], 1))
        elif e["ev"] in ("llm_end", "llm_error"):
            edges.append((e["t"], -1))
    # at equal timestamps process ends first, so back-to-back calls don't count as overlapping
    edges.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, delta in edges:
        cur += delta
        peak = max(peak, cur)
    return peak


def analyze(path: Path) -> dict:
    """Stats for one ``<session>.jsonl`` trace: stage wall/busy seconds, LLM calls and tokens,
    tool timings and quality signals (verifier verdicts, fallbacks, LLM errors)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    ev = [json.loads(line) for line in lines if line.strip()]
    total = max((e["t"] for e in ev), default=0.0)

    node_iv: dict[str, list[tuple[float, float]]] = defaultdict(list)
    node_busy: dict[str, float] = defaultdict(float)
    node_n: dict[str, int] = defaultdict(int)
    for e in ev:
        if e["ev"] == "node_end" and e.get("dur") is not None:
            node_iv[e["agent"]].append((e["t"] - e["dur"], e["t"]))
            node_busy[e["agent"]] += e["dur"]
            node_n[e["agent"]] += 1

    llm: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for e in ev:
        if e["ev"] == "llm_end":
            st = _stage_of_llm(e["agent"])
            d = llm[st]
            d["calls"] += 1
            d["llm_s"] += e.get("dur") or 0
            d["in_tokens"] += e.get("in_tokens") or 0
            d["out_tokens"] += e.get("out_tokens") or 0
            d["reasoning_chars"] += e.get("reasoning_chars") or 0
            d["out_chars"] += e.get("out_chars") or 0
            d["max_call_s"] = max(d["max_call_s"], e.get("dur") or 0)
        elif e["ev"] == "llm_error":
            llm[_stage_of_llm(e["agent"])]["errors"] += 1

    steps: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for e in ev:
        if e["ev"] == "step" and e.get("dur") is not None:
            key = f"{e['agent']}:{e.get('name', '')}"
            steps[key]["n"] += 1
            steps[key]["dur"] += e["dur"]
            steps[key]["chunks"] += e.get("chunks_embedded", 0) or 0
        if e["ev"] == "tool":
            key = f"tool:{e.get('tool')}"
            steps[key]["n"] += 1
            steps[key]["dur"] += e.get("dur") or 0
            steps[key]["results"] += e.get("n_results") or 0
            if e.get("error"):
                steps[key]["errors"] += 1
            if not e.get("n_results"):
                steps[key]["empty"] += 1

    quality = {
        "verdicts": defaultdict(int), "worker_conf": [],
        "worker_n_evidence": [],
        "incomplete_claims": 0, "synthesis_failures": 0, "llm_errors": 0,
    }
    for e in ev:
        if e["agent"] == "verifier.verdict":
            quality["verdicts"][e.get("verdict")] += 1
        elif e["agent"] in ("worker.finding", "paper_worker.finding"):
            quality["worker_conf"].append(e.get("confidence"))
            quality["worker_n_evidence"].append(e.get("n_evidence"))
            if str(e.get("text", "")).startswith("[Research incomplete"):
                quality["incomplete_claims"] += 1
        elif e["agent"] == "worker.synthesis":
            quality["synthesis_failures"] += 1
        elif e["ev"] == "llm_error":
            quality["llm_errors"] += 1
    quality["verdicts"] = dict(quality["verdicts"])

    # Question-scope signals (agents/expansion.py): did the run know the question's constraint,
    # did it enforce it, and does the report rest on facts that address it?
    frame = next((e for e in ev if e["agent"] == "expansion.frame"), None)
    render = next((e for e in reversed(ev) if e["agent"] == "writer.render"), {})
    relevance: dict[str, int] = defaultdict(int)
    for e in ev:
        if e["agent"] == "verifier.verdict" and e.get("relevance"):
            relevance[e["relevance"]] += 1
    cited = render.get("kept_cited") or 0
    quality["scope"] = {
        "key_constraint": frame.get("key_constraint", "") if frame else None,
        "scope_enforced": sum(1 for e in ev if e["agent"] == "supervisor.scope_enforced"),
        "coverage_scope_miss": sum(1 for e in ev if e["agent"] == "coverage.scope_miss"),
        "fact_relevance": dict(relevance),
        "direct_cited_share": round((render.get("kept_cited_direct") or 0) / cited, 3)
        if cited else None,
        "no_direct_answer": bool(render.get("no_direct_answer")),
        "gap_sub_questions": render.get("gap_sub_questions", 0),
        "dropped_scope_overclaim": render.get("dropped_scope_overclaim", 0),
    }

    return {
        "total_s": round(total, 1),
        "stage_wall_s": {k: round(_union(v), 1) for k, v in node_iv.items()},
        "stage_busy_s": {k: round(v, 1) for k, v in node_busy.items()},
        "stage_invocations": dict(node_n),
        "llm": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in llm.items()},
        "steps": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in steps.items()},
        "quality": quality,
        "peak_llm_concurrency": _peak_concurrency(ev),
        # transient-error retries (ainvoke_with_retry) and the total time spent queued for a slot
        "llm_retries": sum(1 for e in ev if e["ev"] == "note" and "retry" in e),
        "slot_wait_s": round(sum(e.get("slot_wait_s", 0) for e in ev if e["ev"] == "note"), 1),
        "fallbacks": _fallback_counts(ev),
    }
