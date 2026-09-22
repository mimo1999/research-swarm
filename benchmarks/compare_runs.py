"""Paired comparison of two benchmark runs on the tasks they share.

    python benchmarks/compare_runs.py --baseline data/benchmark_results/smoke-A \
        --candidate data/benchmark_results/smoke-B [--json out.json]

A run prefix is the path without ``-results.jsonl``; claim-level metrics are read from
``<prefix>-claims.jsonl`` when it exists (``score_claims.py`` output). Only tasks that finished
``ok`` in BOTH runs are compared. Each row is the mean in each run and the paired-bootstrap
difference (candidate minus baseline) with its 95% CI; a row is BETTER / WORSE when the CI
excludes zero in the good / bad direction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import fmt_ci, mean_or_none, paired_bootstrap_diff  # noqa: E402


def _ratio(num: Any, den: Any) -> float | None:
    return None if not den else float(num or 0) / float(den)


def _claim(key: str):
    return lambda res, cl: ((cl or {}).get("claim_metrics") or {}).get(key)


def _over_abstention(res: dict[str, Any], cl: Any) -> float | None:
    return None if res.get("unanswerable") else float(bool(res.get("abstained")))


def _critical(res: dict[str, Any], cl: Any) -> float | None:
    return None if not cl else float(bool(cl.get("critical_error")))


# name -> (higher_is_better, extractor(result_row, claims_row) -> float | None)
METRICS: dict[str, tuple[bool, Any]] = {
    "answer_score": (True, lambda r, c: r.get("answer_score")),
    "finding_recall": (True, lambda r, c: r.get("finding_recall")),
    "false_refute_rate": (
        False, lambda r, c: _ratio(r.get("false_refuted"), r.get("answer_findings")),
    ),
    "over_abstention": (False, _over_abstention),
    "number_grounding": (True, lambda r, c: (r.get("number_grounding") or {}).get("rate")),
    "faithfulness": (True, _claim("faithfulness")),
    "citation_recall": (True, _claim("citation_recall")),
    "citation_correctness": (True, _claim("citation_correctness")),
    "citation_completeness": (True, _claim("citation_completeness")),
    "critical_error": (False, _critical),
    "seconds": (False, lambda r, c: r.get("seconds")),
    "llm_calls": (False, lambda r, c: r.get("llm_calls")),
    "llm_retries": (False, lambda r, c: r.get("llm_retries")),
}


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_run(prefix: str, only: str = "") -> dict[str, dict[str, Any]]:
    """task_id -> {"res": result row, "claims": claims row | None} for ``ok`` tasks; *only*
    keeps task ids starting with that prefix (e.g. ``scifact``)."""
    base = Path(prefix)
    claims = {r["task_id"]: r for r in _read(Path(f"{base}-claims.jsonl"))}
    return {
        r["task_id"]: {"res": r, "claims": claims.get(r["task_id"])}
        for r in _read(Path(f"{base}-results.jsonl"))
        if r.get("status") == "ok" and r["task_id"].startswith(only)
    }


def compare(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(candidate))
    rows: list[dict[str, Any]] = []
    for name, (higher_better, fn) in METRICS.items():
        a: list[float] = []                    # candidate and baseline, paired, both non-None
        b: list[float] = []
        for tid in shared:
            va = fn(candidate[tid]["res"], candidate[tid]["claims"])
            vb = fn(baseline[tid]["res"], baseline[tid]["claims"])
            if va is not None and vb is not None:
                a.append(float(va))
                b.append(float(vb))
        if not a:
            continue
        diff = paired_bootstrap_diff(a, b)
        verdict = ""
        if diff and diff[1] > 0:
            verdict = "BETTER" if higher_better else "WORSE"
        elif diff and diff[2] < 0:
            verdict = "WORSE" if higher_better else "BETTER"
        rows.append({
            "metric": name, "n": len(a), "baseline": mean_or_none(b),
            "candidate": mean_or_none(a), "diff": diff, "verdict": verdict,
        })
    return {"tasks": len(shared), "rows": rows}


def render(result: dict[str, Any]) -> str:
    lines = [
        f"Paired comparison on {result['tasks']} shared ok tasks (candidate - baseline)", "",
        "| Metric | n | Baseline | Candidate | Diff [95% CI] | |", "|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(f"| {r['metric']} | {r['n']} | {r['baseline']} | {r['candidate']} | "
                     f"{fmt_ci(r['diff'])} | {r['verdict']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="", help="task-id prefix, e.g. scifact / hotpotqa / alce")
    args = ap.parse_args()
    result = compare(load_run(args.baseline, args.only), load_run(args.candidate, args.only))
    print(render(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
