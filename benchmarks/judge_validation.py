"""Validate the claim judge against hand labels.

The claim-level columns of Table A are only as good as the judge. This exports a stratified
sample of judged sentences for a person to label, then reports agreement and Cohen's kappa.

    python benchmarks/judge_validation.py export --claims <smoke-run>-claims.jsonl
    # open data/benchmark_results/judge_validation.csv, fill human_cited_support (and optionally
    # human_corpus_support) with yes / partial / no, using ONLY the excerpts in the row
    python benchmarks/judge_validation.py score --csv data/benchmark_results/judge_validation.csv

``score`` writes ``judge_validation.json``, which Table A picks up to replace its
"judge not validated" warning with the measured kappa.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import RESULTS_ROOT, SEED  # noqa: E402

LABELS = ("yes", "partial", "no")
CSV_PATH = RESULTS_ROOT / "judge_validation.csv"
JSON_PATH = RESULTS_ROOT / "judge_validation.json"
COLUMNS = [
    "id", "task_id", "dataset", "sentence", "cited_docs", "all_docs",
    "judge_checkworthy", "judge_cited_support", "judge_corpus_support",
    "human_cited_support", "human_corpus_support",
]


def _family(dataset: str) -> str:
    return dataset.split("/")[0]


# --------------------------------------------------------------------------- #
# Agreement statistics
# --------------------------------------------------------------------------- #

def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Unweighted Cohen's kappa between two label lists; None if undefined (no data, or both
    raters constant and identical so chance agreement is 1)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def confusion(judge: list[str], human: list[str]) -> dict[str, dict[str, int]]:
    """{human label: {judge label: count}}."""
    table = {h: {j: 0 for j in LABELS} for h in LABELS}
    for j, h in zip(judge, human):
        if h in table and j in table[h]:
            table[h][j] += 1
    return table


def agreement_report(judge: list[str], human: list[str]) -> dict[str, Any]:
    pairs = [(j, h) for j, h in zip(judge, human) if j in LABELS and h in LABELS]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None, "kappa_binary": None, "confusion": {}}
    jj, hh = [p[0] for p in pairs], [p[1] for p in pairs]
    binary = lambda xs: ["yes" if x == "yes" else "no" for x in xs]  # noqa: E731
    return {
        "n": len(pairs),
        "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4),
        "kappa": _round(cohen_kappa(jj, hh)),
        "kappa_binary": _round(cohen_kappa(binary(jj), binary(hh))),   # supported vs not
        "confusion": confusion(jj, hh),
    }


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


# --------------------------------------------------------------------------- #
# Export / score
# --------------------------------------------------------------------------- #

def _excerpt(docs: list[dict[str, str]], numbers: list[int], chars: int = 700) -> str:
    return "\n---\n".join(
        f"[D{n}] {docs[n - 1]['title']}: {docs[n - 1]['text'][:chars]}"
        for n in numbers if 1 <= n <= len(docs)
    ) or "(no resolvable citation)"


def sample_rows(claim_rows: list[dict[str, Any]], n: int, seed: int = SEED) -> list[dict[str, str]]:
    """*n* judged checkworthy sentences, spread evenly over (dataset family, judge cited verdict)
    so the sample isn't all easy 'yes' cases; deterministic under *seed*."""
    strata: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in claim_rows:
        for claim in row["claims"]:
            v = claim.get("verdict")
            if not v or not v["checkworthy"]:
                continue
            docs = row["documents"]
            strata.setdefault((_family(row["dataset"]), v["cited_support"]), []).append({
                "task_id": row["task_id"], "dataset": row["dataset"], "sentence": claim["text"],
                "cited_docs": _excerpt(docs, claim["cite_docs"]),
                "all_docs": _excerpt(docs, list(range(1, len(docs) + 1)), chars=300),
                "judge_checkworthy": "yes", "judge_cited_support": v["cited_support"],
                "judge_corpus_support": v["corpus_support"],
            })
    rng = random.Random(seed)
    for pool in strata.values():
        rng.shuffle(pool)
    chosen: list[dict[str, str]] = []
    keys = sorted(strata)
    while len(chosen) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(chosen) < n:
                chosen.append(strata[k].pop())
    rng.shuffle(chosen)
    return [{"id": str(i + 1), **row, "human_cited_support": "", "human_corpus_support": ""}
            for i, row in enumerate(chosen)]


def cmd_export(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in Path(args.claims).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    sample = sample_rows(rows, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sample)
    print(f"wrote {len(sample)} sentences to {out}\nFill human_cited_support (yes/partial/no): "
          "does the CITED document(s) alone support the sentence? Then run: "
          "judge_validation.py score")


def cmd_score(args: argparse.Namespace) -> None:
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labelled = [r for r in rows if r["human_cited_support"].strip().lower() in LABELS]
    if not labelled:
        raise SystemExit("No rows have a human_cited_support label (yes / partial / no) yet.")
    result: dict[str, Any] = {
        "n": len(labelled),
        "cited_support": agreement_report(
            [r["judge_cited_support"] for r in labelled],
            [r["human_cited_support"].strip().lower() for r in labelled],
        ),
    }
    both = [r for r in labelled if r["human_corpus_support"].strip().lower() in LABELS]
    if both:
        result["corpus_support"] = agreement_report(
            [r["judge_corpus_support"] for r in both],
            [r["human_corpus_support"].strip().lower() for r in both],
        )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--claims", required=True, help="smoke-*-claims.jsonl from score_claims.py")
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--seed", type=int, default=SEED)
    e.add_argument("--out", default=str(CSV_PATH))
    s = sub.add_parser("score")
    s.add_argument("--csv", default=str(CSV_PATH))
    s.add_argument("--out", default=str(JSON_PATH))
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    (cmd_export if a.cmd == "export" else cmd_score)(a)
