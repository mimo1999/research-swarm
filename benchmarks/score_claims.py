"""Claim-level scoring of a smoke-benchmark run: faithfulness, citation quality, critical errors.

Generation and judging are decoupled: ``run_smoke_benchmark.py`` writes each report, this script
judges them afterwards (so the judge can be swapped or re-run without regenerating).

    python benchmarks/score_claims.py --results data/benchmark_results/smoke-<ts>-results.jsonl
    python benchmarks/score_claims.py --results ... --summarize-only --update-readme

The judge must not be the model that wrote the reports (refused unless --allow-same-model).
Resumable: tasks already in the output ``-claims.jsonl`` are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_smoke_benchmark as smoke  # noqa: E402
from bench_common import (  # noqa: E402
    RESULTS_ROOT,
    ROOT,
    bootstrap_ci,
    md_table,
    percentile,
    update_block,
)

DEFAULT_JUDGE = "gemma4:e2b"
CLAIM_METRICS = (
    "faithfulness", "faithfulness_soft", "citation_recall", "citation_correctness",
    "citation_completeness", "unsupported_rate", "dangling_citation_rate", "judge_error_rate",
)
EXCERPT_CHARS = 1500


def family(dataset: str) -> str:
    """"alce/asqa" -> "alce"."""
    return dataset.split("/")[0]


# --------------------------------------------------------------------------- #
# Loading and guards
# --------------------------------------------------------------------------- #

def load_run(results_path: Path, tasks_path: Path | None = None):
    """[(BenchmarkTask, result row)] for every finished task with a report."""
    tasks_path = tasks_path or Path(str(results_path).replace("-results.jsonl", "-tasks.json"))
    tasks = {t["id"]: smoke.BenchmarkTask(**t)
             for t in json.loads(tasks_path.read_text(encoding="utf-8"))}
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("status") == "ok" and result.get("report") and result["task_id"] in tasks:
            rows.append((tasks[result["task_id"]], result))
    return rows


def generator_models(results: list[dict]) -> set[str]:
    return {m for r in results for m in (r.get("model"), r.get("worker_model")) if m}


def check_independent(judge_model: str, results: list[dict], allow_same: bool = False) -> None:
    """Refuse to let a model grade reports it wrote."""
    if judge_model in generator_models(results) and not allow_same:
        raise SystemExit(
            f"Judge model {judge_model!r} also generated these reports "
            f"({sorted(generator_models(results))}). Pick a different --judge-model, or pass "
            "--allow-same-model to accept a self-grading judge."
        )


def build_judge(model: str):
    """Deterministic, thinking-off judge on the local Ollama daemon, under the concurrency cap."""
    from research_swarm.agents.base import get_agent_llm, without_thinking
    from research_swarm.runtime.limits import set_llm_context

    llm = without_thinking(get_agent_llm("ollama", model, temperature=0.0), 4096)
    set_llm_context("ollama", "claim-judge")
    return llm


# --------------------------------------------------------------------------- #
# Per-task scoring
# --------------------------------------------------------------------------- #

def is_critical_error(task: Any, report_text: str, metrics: dict[str, Any]) -> tuple[bool, bool]:
    """(critical_error, verdict_flip). A critical error is a report a reader would be misled by:
    a SciFact verdict that flips SUPPORT <-> CONTRADICT, or any sentence a document contradicts.
    (Missing the answer or abstaining is a different failure and is not counted here.)"""
    flip = False
    if task.dataset == "scifact":
        verdict = smoke._scifact_verdict(report_text)
        expected = task.expected[0]
        flip = (verdict in ("SUPPORT", "CONTRADICT") and expected in ("SUPPORT", "CONTRADICT")
                and verdict != expected)
    return bool(flip or metrics["n_contradicted"] > 0), flip


async def score_task(task: Any, result: dict[str, Any], llm: Any) -> dict[str, Any]:
    from research_swarm.eval.claims import claim_metrics, judge_claims, split_claims

    documents = smoke._task_documents(task)
    report = result["report"]
    claims = split_claims(report)
    ref_urls = [r["url"] for r in report.get("references") or []]
    doc_numbers = {d["url"]: i for i, d in enumerate(documents, 1)}
    verdicts, dangling = await judge_claims(claims, ref_urls, documents, llm)
    metrics = claim_metrics(claims, verdicts, dangling)
    text = " ".join([report.get("title", ""), report.get("exec_summary", ""),
                     *(s["body_md"] for s in report.get("sections") or [])])
    critical, flip = is_critical_error(task, text, metrics)
    return {
        "task_id": task.id, "dataset": task.dataset, "claim_metrics": metrics,
        "critical_error": critical, "verdict_flip": flip,
        # kept so a person can audit (and label) the judge: the sentence, its verdict and the
        # excerpts of the documents it cites
        "documents": [{"title": d["title"], "text": d["text"][:EXCERPT_CHARS]} for d in documents],
        "claims": [
            {
                "idx": c.idx, "text": c.text, "citations": c.citations, "section": c.section,
                "cite_docs": sorted({
                    doc_numbers[ref_urls[n - 1]] for n in c.citations
                    if 1 <= n <= len(ref_urls) and ref_urls[n - 1] in doc_numbers
                }),
                "verdict": verdicts[c.idx].model_dump() if verdicts[c.idx] else None,
            }
            for c in claims
        ],
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _ci_of(values: list[float]):
    return bootstrap_ci([v for v in values if v is not None])


def summarize_claims(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per dataset family and overall: means with 95% bootstrap CIs over tasks."""
    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"tasks": len(rs)}
        for m in CLAIM_METRICS:
            out[m] = _ci_of([r["claim_metrics"].get(m) for r in rs])
        out["critical_error_rate"] = _ci_of([float(r["critical_error"]) for r in rs])
        out["verdict_flip_rate"] = _ci_of([float(r["verdict_flip"]) for r in rs])
        out["tasks_with_contradiction"] = _ci_of(
            [float(r["claim_metrics"]["n_contradicted"] > 0) for r in rs],
        )
        out["mean_checkworthy_claims"] = (
            round(sum(r["claim_metrics"]["n_checkworthy"] for r in rs) / len(rs), 2) if rs else None
        )
        return out

    summary: dict[str, Any] = {"overall": block(rows), "by_family": {}}
    for fam in sorted({family(r["dataset"]) for r in rows}):
        summary["by_family"][fam] = block([r for r in rows if family(r["dataset"]) == fam])
    return summary


def _fmt(ci, digits: int = 2, pct: bool = False) -> str:
    if ci is None:
        return "-"
    mean, lo, hi = ci
    scale = 100 if pct else 1
    unit = "%" if pct else ""
    return f"{mean * scale:.{digits}f}{unit} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def _vals(rs: list[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in rs if r.get(key) is not None]


def render_table_a(
    results: list[dict[str, Any]], claim_rows: list[dict[str, Any]], judge_model: str,
    validation: dict[str, Any] | None = None,
) -> str:
    """Markdown for README block TABLE-A: per dataset family, correctness + truth quality."""
    claims_by_task = {r["task_id"]: r for r in claim_rows}
    ok = [r for r in results if r.get("status") == "ok"]
    fams = sorted({family(r["dataset"]) for r in ok})
    rows = []
    for fam in [*fams, "all"]:
        rs = [r for r in ok if fam == "all" or family(r["dataset"]) == fam]
        cs = [claims_by_task[r["task_id"]] for r in rs if r["task_id"] in claims_by_task]
        un = [r for r in rs if r.get("unanswerable")]
        an = [r for r in rs if not r.get("unanswerable")]
        mets = [c["claim_metrics"] for c in cs]
        rows.append([
            fam, len(rs),
            _fmt(_ci_of(_vals(rs, "answer_score"))),
            _fmt(_ci_of(_vals(rs, "normalized_answer_score"))),
            _fmt(_ci_of([float(c["critical_error"]) for c in cs]), 2, pct=True),
            _fmt(_ci_of([m["faithfulness"] for m in mets]), 2),
            _fmt(_ci_of([m["citation_recall"] for m in mets]), 2),
            _fmt(_ci_of([m["citation_correctness"] for m in mets]), 2),
            _fmt(_ci_of([m["citation_completeness"] for m in mets]), 2),
            _fmt(_ci_of([m["dangling_citation_rate"] for m in mets]), 2),
            f"{sum(bool(r.get('abstained')) for r in un)}/{len(un)}",
            f"{sum(bool(r.get('abstained')) for r in an)}/{len(an)}",
            _fmt(_ci_of(_vals(rs, "support_doc_recall")), 2),
            percentile([r["seconds"] for r in rs], 50),
        ])
    validated = (
        f"Judge agreement with hand labels: kappa {validation['cited_support']['kappa']:.2f} on "
        f"{validation['n']} sentences."
        if validation else "**The judge is not yet validated against hand labels**"
        " (`benchmarks/judge_validation.py`); read faithfulness and citation columns accordingly."
    )
    return "\n".join([
        "### Table A - closed-corpus answer quality and truth (current pipeline)",
        "",
        f"Mean over tasks, 95% bootstrap CI in brackets. Claim-level columns are judged by "
        f"`{judge_model}` (a different model from the generator). {validated}",
        "",
        md_table(
            ["Dataset", "Tasks", "Answer", "Answer / ceiling", "Critical errors", "Faithfulness",
             "Cite recall", "Cite correctness", "Cite completeness", "Dangling cites",
             "Abstained: unanswerable", "Abstained: answerable", "Gold-doc recall", "Time p50 (s)"],
            rows,
        ),
        "",
        "Answer = fraction of expected answers in the report (SciFact: verdict match). "
        "Answer / ceiling divides by what the supplied documents can answer at all. Critical "
        "errors = reports with a SciFact SUPPORT<->CONTRADICT flip or a sentence a document "
        "contradicts. Abstained columns are counts of tasks where the report declined to answer "
        "(unanswerable: should be high; answerable: should be low).",
    ])


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def _read_scored(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        r["task_id"]: r
        for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    }


def _load_validation() -> dict[str, Any] | None:
    path = RESULTS_ROOT / "judge_validation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


async def main(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    pairs = load_run(results_path, Path(args.tasks) if args.tasks else None)
    if args.limit:
        pairs = pairs[: args.limit]
    out_path = Path(args.out) if args.out else Path(
        str(results_path).replace("-results.jsonl", "-claims.jsonl")
    )
    check_independent(args.judge_model, [r for _, r in pairs], args.allow_same_model)
    scored = _read_scored(out_path)
    print(f"{len(pairs)} finished tasks, {len(scored)} already judged -> {out_path.name}",
          flush=True)

    if not args.summarize_only:
        llm = build_judge(args.judge_model)
        todo = [(t, r) for t, r in pairs if t.id not in scored]
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()

        async def one(i: int, task: Any, result: dict[str, Any]) -> None:
            async with sem:
                row = await score_task(task, result, llm)
                row["judge_model"] = args.judge_model
                async with lock:
                    scored[task.id] = row
                    with out_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                m = row["claim_metrics"]
                print(f"[{i}/{len(todo)}] {task.id} claims={m['n_claims']} "
                      f"faith={m['faithfulness']} cite_corr={m['citation_correctness']} "
                      f"critical={row['critical_error']}", flush=True)

        await asyncio.gather(*(one(i, t, r) for i, (t, r) in enumerate(todo, 1)))

    rows = [scored[t.id] for t, _ in pairs if t.id in scored]
    summary = summarize_claims(rows)
    summary["judge_model"] = args.judge_model
    summary_path = Path(str(out_path).replace("-claims.jsonl", "-claims-summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = render_table_a([r for _, r in pairs], rows, args.judge_model, _load_validation())
    print("\n" + table + f"\n\nSUMMARY {summary_path}")
    if args.update_readme:
        update_block(ROOT / "benchmarks" / "README.md", "TABLE-A", table)
        print("README table block updated")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="smoke-*-results.jsonl")
    ap.add_argument("--tasks", default=None, help="smoke-*-tasks.json (default: next to results)")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--allow-same-model", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--update-readme", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
