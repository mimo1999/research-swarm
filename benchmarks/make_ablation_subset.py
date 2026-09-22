"""Build the fixed ablation task subset (default: 30 ok tasks per dataset family, seed 42).

    python benchmarks/make_ablation_subset.py

Every pipeline change is measured on the same tasks so runs can be compared pairwise
(``compare_runs.py``). Only tasks that finished ``ok`` in the source run are eligible, so a
task that timed out for provider reasons cannot bias the subset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import RESULTS_ROOT, SEED  # noqa: E402

DEFAULT_PREFIX = RESULTS_ROOT / "smoke-20260925-110129"
DEFAULT_OUT = RESULTS_ROOT / "ablation-90-tasks.json"


def family(dataset: str) -> str:
    return dataset.split("/")[0]


def build_subset(
    tasks: list[dict[str, Any]], results: list[dict[str, Any]], per_family: int = 30,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    ok = {r["task_id"] for r in results if r.get("status") == "ok"}
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        if task["id"] in ok:
            groups.setdefault(family(task["dataset"]), []).append(task)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for fam in sorted(groups):
        pool = sorted(groups[fam], key=lambda t: t["id"])
        rng.shuffle(pool)
        chosen += pool[:per_family]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_PREFIX), help="run prefix (no -tasks.json)")
    ap.add_argument("--per-family", type=int, default=30)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    prefix = Path(args.source)
    tasks = json.loads(Path(f"{prefix}-tasks.json").read_text(encoding="utf-8"))
    results = [json.loads(line) for line in
               Path(f"{prefix}-results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    subset = build_subset(tasks, results, args.per_family)
    Path(args.out).write_text(json.dumps(subset, indent=2), encoding="utf-8")
    print(f"wrote {len(subset)} tasks to {args.out}")


if __name__ == "__main__":
    main()
