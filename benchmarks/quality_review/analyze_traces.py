# ruff: noqa: E501
"""Aggregate swarm traces into a per-stage time / cost / quality-signal breakdown.

Usage:
    python benchmarks/quality_review/analyze_traces.py data/quality_review/run1
Writes <run>/analysis.json and prints a markdown summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research_swarm.runtime.trace_stats import analyze  # noqa: E402


def main() -> None:
    run = Path(sys.argv[1])
    out = {}
    for f in sorted((run / "traces").glob("*.jsonl")):
        qid = f.stem.split("-")[1]
        out[qid] = analyze(f)
    (run / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    stages = ["supervisor", "paper_scout", "paper_worker", "worker", "collect", "verifier", "writer"]
    print("| q | total s | " + " | ".join(stages) + " |")
    print("|" + "---|" * (len(stages) + 2))
    for qid, a in out.items():
        cells = [f"{a['stage_wall_s'].get(s, 0):.0f}" for s in stages]
        print(f"| {qid} | {a['total_s']:.0f} | " + " | ".join(cells) + " |")

    print("\n| q | key constraint | scope enforced | coverage misses | direct-cited share | no direct answer | gaps |")
    print("|---|---|---|---|---|---|---|")
    for qid, a in out.items():
        s = a["quality"].get("scope", {})
        print(f"| {qid} | {s.get('key_constraint') or '-'} | {s.get('scope_enforced', 0)} | "
              f"{s.get('coverage_scope_miss', 0)} | {s.get('direct_cited_share')} | "
              f"{s.get('no_direct_answer')} | {s.get('gap_sub_questions', 0)} |")


if __name__ == "__main__":
    main()
