"""Score the pipeline's located evidence against SciFact's gold rationale sentences.

SciFact marks, for every labelled claim, which sentences of which abstract carry the evidence.
The pipeline keeps, for every fact, the exact source text it rests on (``Finding.quote``: the
model's quote as located in the source, or the matched passage). This maps each fact's quote
back to sentence indices of its abstract and compares them with the gold rationales -- a
deterministic check of the grounding step, with no LLM judge. It is SciFact's sentence-selection
metric without the label condition (facts are not labelled SUPPORT / CONTRADICT themselves).

Usage:
    python benchmarks/score_rationales.py --results data/benchmark_results/smoke-<ts>-results.jsonl

Reports micro precision / recall / F1 over (abstract, sentence) pairs with a task-level bootstrap
CI, for three fact sets -- every extracted fact, facts whose quote was located verbatim, and
facts the verifier kept for the writer -- plus two reference points on the same tasks: selecting
every sentence of the supplied abstracts (recall 1) and selecting as many sentences as the
pipeline did, at random (the expected score of a selector that ignores the claim).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_common import SEED, _read_jsonl, md_table  # noqa: E402
from research_swarm.agents.grounding import locate_quote  # noqa: E402

DATA_ROOT = Path("data/benchmarks/scifact/data")
MAX_DOCS = 5            # the benchmark supplies a claim's first five cited abstracts
MIN_OVERLAP = 10        # characters of a sentence a quote must cover to select it

Pair = tuple[int, int]  # (abstract index in the task, sentence index)


def sentence_spans(sentences: list[str]) -> list[tuple[int, int]]:
    """Character spans of *sentences* inside ``" ".join(sentences)`` (the benchmark's text)."""
    spans, pos = [], 0
    for sent in sentences:
        spans.append((pos, pos + len(sent)))
        pos += len(sent) + 1
    return spans


def quote_sentences(quote: str, sentences: list[str]) -> set[int] | None:
    """Indices of the sentences *quote* covers (at least MIN_OVERLAP characters, or the whole
    sentence if shorter); None when the quote cannot be located in the abstract at all."""
    text = " ".join(sentences)
    start = text.find(quote)
    if start >= 0:
        end = start + len(quote)
    else:
        match = locate_quote(quote, text)
        if match is None:
            return None
        start, end = match.start, match.end
    picked = set()
    for i, (s, e) in enumerate(sentence_spans(sentences)):
        overlap = min(end, e) - max(start, s)
        if overlap >= min(MIN_OVERLAP, e - s) and overlap > 0:
            picked.add(i)
    return picked


def gold_pairs(claim: dict[str, Any], doc_ids: list[str]) -> set[Pair]:
    """Gold rationale (abstract index, sentence) pairs for the task's supplied abstracts."""
    gold: set[Pair] = set()
    for doc_id, groups in (claim.get("evidence") or {}).items():
        if str(doc_id) not in doc_ids:
            continue
        idx = doc_ids.index(str(doc_id))
        for group in groups:
            gold.update((idx, int(s)) for s in group.get("sentences", []))
    return gold


def _latest_verdicts(critiques: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in critiques or []:
        out[str(c.get("finding_id"))] = str(c.get("verdict"))
    return out


def fact_sets(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The facts of one result, by fact set: extracted / verbatim / kept."""
    facts = [f for f in result.get("finding_details") or [] if f.get("quote")]
    verdicts = _latest_verdicts(result.get("critique_details") or [])
    return {
        "extracted": facts,
        "verbatim": [f for f in facts if f.get("grounding") == "quote"],
        "kept": [f for f in facts if verdicts.get(str(f.get("id"))) != "refuted"
                 and float(f.get("confidence", 0) or 0) >= 0.1],
    }


def predicted_pairs(facts: list[dict[str, Any]], task_id: str,
                    abstracts: list[list[str]]) -> tuple[set[Pair], int]:
    """(selected (abstract, sentence) pairs, number of quotes that could not be mapped)."""
    pairs: set[Pair] = set()
    unmapped = 0
    prefix = f"benchmark://{task_id}/"
    for fact in facts:
        url = next((u for u in fact.get("evidence_urls") or [] if u.startswith(prefix)), None)
        if url is None:
            unmapped += 1
            continue
        idx = int(url[len(prefix):])
        if idx >= len(abstracts):
            unmapped += 1
            continue
        sents = quote_sentences(fact["quote"], abstracts[idx])
        if sents is None:
            unmapped += 1
            continue
        pairs.update((idx, s) for s in sents)
    return pairs, unmapped


def prf(tp: float, fp: float, fn: float) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f}


def micro_ci(rows: list[tuple[float, float, float]], n: int = 2000,
             seed: int = SEED) -> dict[str, tuple[float, float] | None]:
    """Task-level percentile bootstrap CI of micro precision / recall / F1."""
    if not rows:
        return {"precision": None, "recall": None, "f1": None}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        pick = rng.choices(rows, k=len(rows))
        m = prf(*(sum(r[i] for r in pick) for i in range(3)))
        for key, value in m.items():
            if value is not None:
                samples[key].append(value)
    out: dict[str, tuple[float, float] | None] = {}
    for key, vals in samples.items():
        vals.sort()
        out[key] = (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1,
                    int(0.975 * len(vals)))]) if vals else None
    return out


def score(results: list[dict[str, Any]], claims: dict[int, dict[str, Any]],
          corpus: dict[str, dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Per fact set and reference point: micro P/R/F1 with CIs, task hit rate, unmapped quotes."""
    rows: dict[str, list[tuple[float, float, float]]] = {}
    hits: dict[str, list[bool]] = {}
    unmapped: dict[str, int] = {}
    nei_selected: dict[str, list[int]] = {}
    tasks = 0
    task_ids: list[str] = []
    for result in results:
        task_id = str(result.get("task_id", ""))
        if not task_id.startswith("scifact-") or result.get("status") not in (None, "ok"):
            continue
        claim = claims.get(int(task_id.split("-", 1)[1]))
        if claim is None:
            continue
        doc_ids = [str(d) for d in claim.get("cited_doc_ids", [])[:MAX_DOCS]
                   if str(d) in corpus]
        abstracts = [corpus[d]["abstract"] for d in doc_ids]
        gold = gold_pairs(claim, doc_ids)
        n_sentences = sum(len(a) for a in abstracts)
        tasks += 1
        task_ids.append(task_id)
        for name, facts in fact_sets(result).items():
            pred, missing = predicted_pairs(facts, task_id, abstracts)
            tp = len(pred & gold)
            rows.setdefault(name, []).append((tp, len(pred) - tp, len(gold) - tp))
            unmapped[name] = unmapped.get(name, 0) + missing
            if gold:
                hits.setdefault(name, []).append(tp > 0)
            else:
                nei_selected.setdefault(name, []).append(len(pred))
            if name == "extracted":
                # Reference points on the same task.
                rows.setdefault("all sentences", []).append(
                    (len(gold), n_sentences - len(gold), 0))
                k = len(pred)
                exp_tp = k * len(gold) / n_sentences if n_sentences else 0.0
                rows.setdefault("random, same count", []).append(
                    (exp_tp, k - exp_tp, len(gold) - exp_tp))
    summary: dict[str, Any] = {"tasks": tasks, "sets": {}, "rows": rows, "task_ids": task_ids}
    for name, task_rows in rows.items():
        totals = [sum(r[i] for r in task_rows) for i in range(3)]
        entry: dict[str, Any] = {**prf(*totals), "ci": micro_ci(task_rows, n_boot),
                                 "tp": totals[0], "fp": totals[1], "fn": totals[2]}
        if name in hits:
            entry["task_hit_rate"] = sum(hits[name]) / len(hits[name]) if hits[name] else None
            entry["unmapped_quotes"] = unmapped.get(name, 0)
            nei = nei_selected.get(name, [])
            entry["nei_sentences_per_task"] = sum(nei) / len(nei) if nei else None
        summary["sets"][name] = entry
    return summary


def paired_diff(a: dict[str, Any], b: dict[str, Any], name: str, n: int = 2000,
                seed: int = SEED) -> dict[str, tuple[float, float, float] | None]:
    """Candidate *a* minus baseline *b* on the tasks both scored: (diff, lo, hi) of micro
    precision / recall / F1, resampling task pairs."""
    ia = dict(zip(a["task_ids"], a["rows"].get(name, []), strict=False))
    ib = dict(zip(b["task_ids"], b["rows"].get(name, []), strict=False))
    common = sorted(set(ia) & set(ib))
    if not common:
        return {"precision": None, "recall": None, "f1": None}

    def metrics(ids: list[str], rows: dict[str, tuple[float, float, float]]) -> dict:
        return prf(*(sum(rows[t][i] for t in ids) for i in range(3)))

    point = {k: (metrics(common, ia)[k], metrics(common, ib)[k]) for k in ("precision", "recall",
                                                                            "f1")}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n):
        pick = rng.choices(common, k=len(common))
        ma, mb = metrics(pick, ia), metrics(pick, ib)
        for k in draws:
            if ma[k] is not None and mb[k] is not None:
                draws[k].append(ma[k] - mb[k])
    out: dict[str, tuple[float, float, float] | None] = {}
    for k, (va, vb) in point.items():
        d = sorted(draws[k])
        out[k] = ((va - vb, d[int(0.025 * len(d))], d[min(len(d) - 1, int(0.975 * len(d)))])
                  if va is not None and vb is not None and d else None)
    return out


def _fmt(value: float | None, ci: tuple[float, float] | None = None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def markdown(summary: dict[str, Any]) -> str:
    rows = []
    for name, e in summary["sets"].items():
        rows.append([
            name, _fmt(e["precision"], e["ci"]["precision"]), _fmt(e["recall"], e["ci"]["recall"]),
            _fmt(e["f1"], e["ci"]["f1"]), _fmt(e.get("task_hit_rate")),
            e.get("unmapped_quotes", "-"), _fmt(e.get("nei_sentences_per_task")),
        ])
    return (f"SciFact rationale selection over {summary['tasks']} tasks (micro over "
            "(abstract, sentence) pairs; 95% task-bootstrap CI)\n\n"
            + md_table(["Fact set", "Precision", "Recall", "F1", "Task hit rate",
                        "Unmapped quotes", "Sentences picked on NEI claims"], rows))


def load_claims(root: Path) -> dict[int, dict[str, Any]]:
    claims: dict[int, dict[str, Any]] = {}
    for split in ("claims_dev.jsonl", "claims_train.jsonl"):
        path = root / split
        if path.exists():
            claims.update({int(r["id"]): r for r in _read_jsonl(path)})
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="a results file for the same tasks: report paired differences")
    args = parser.parse_args()
    results = [r for p in args.results for r in _read_jsonl(p)]
    corpus = {str(r["doc_id"]): r for r in _read_jsonl(args.data_root / "corpus.jsonl")}
    claims = load_claims(args.data_root)
    summary = score(results, claims, corpus, args.boot)
    print(markdown(summary))
    if args.baseline:
        base = score(_read_jsonl(args.baseline), claims, corpus, args.boot)
        rows = []
        for name in ("extracted", "verbatim", "kept"):
            d = paired_diff(summary, base, name, args.boot)
            rows.append([name] + [
                "-" if d[k] is None else f"{d[k][0]:+.3f} [{d[k][1]:+.3f}, {d[k][2]:+.3f}]"
                for k in ("precision", "recall", "f1")])
        print(f"\nPaired difference vs {args.baseline.name} (candidate minus baseline, 95% CI "
              "over task pairs)\n\n" + md_table(["Fact set", "Precision", "Recall", "F1"], rows))
        summary["paired_vs"] = str(args.baseline)
    summary.pop("rows", None)
    out = args.results[0].with_name(args.results[0].stem.replace("-results", "")
                                    + "-rationales.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {out}")


if __name__ == "__main__":
    main()
