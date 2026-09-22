"""Claim-matcher: quantify the gap between expected and generated findings.

Matches each expected finding (ground-truth JSON, e.g. a manually-compiled
PubMed reference set) against the research swarm's generated findings, then
scores each matched pair on three independent axes so a single high
similarity score can't hide a contradiction or a missing number:

  1. Coverage   -- embedding similarity (BGE small,
                   faithfulness elsewhere in this project). Answers "did the
                   swarm say anything about this fact at all?"
  2. Consistency -- NLI cross-encoder entailment/neutral/contradiction.
                   Answers "if it did, did it get the *direction* right?"
                   This is the check embedding similarity alone cannot do:
                   "positive result" and "no significant benefit" can be
                   textually close but are opposite claims. Uses a domain-tuned
                   biomedical NLI model by default (PubMedBERT fine-tuned
                   MNLI -> MedNLI) rather than a general-purpose one, since a
                   general MNLI/SNLI-trained model has no biomedical priors and
                   tends to call true clinical paraphrases "neutral" rather
                   than "entailment" -- pass ``--nli-model general`` to compare
                   against the original cross-encoder/nli-deberta-v3-base.
                   The domain model has its own failure mode, guarded against
                   separately: it calls epistemic hedges ("the report does not
                   detail/discuss X") an "entailment" of specific claims about
                   X, with >99% confidence, regardless of premise length --
                   see ``_HEDGE_RE`` / ``_nli_label_guarded``, which detects
                   this pattern and falls back to the general model only in
                   that specific case.
  3. Precision   -- numeric-tolerance match against any `quantitative` block
                   on the expected finding. Answers "did it get the *numbers*
                   right, or just the gist?"

By default this scores the *final report text* (``report.sections``) --
what a reader actually receives -- not the intermediate worker findings.
A finding can be produced, survive the critic, and still be dropped, hedged,
or reworded by the writer; scoring the findings list instead would credit
content the reader never sees. Pass ``--source findings`` to score the
pre-writer findings instead, e.g. to diagnose whether content is lost at the
worker/critic stage or at the writer stage.

Usage:
    poetry run python benchmarks/manual_comparison/claim_matcher.py \\
        --expected benchmarks/manual_comparison/glp1_parkinsons_findings.json \\
        --generated benchmarks/manual_comparison/swarm_output.json \\
        --source report \\
        --out benchmarks/manual_comparison/claim_match_report.json

Both embedding and NLI models run locally on CPU (no API calls, no cost).
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Below this cosine similarity, two claims are treated as being about
# different things entirely -- not "the swarm got it wrong", but "the swarm
# never addressed this at all" (or, from the generated side, "this doesn't
# correspond to anything in the ground truth", i.e. possible scope drift).
MATCH_THRESHOLD = 0.55

# NLI cross-encoder must be at least this confident to override "neutral".
NLI_CONFIDENCE_THRESHOLD = 0.5

# Fraction of an expected finding's quantitative values that must appear
# (within tolerance) in the matched generated claim to count as "precise"
# rather than "vague".
NUMERIC_RECALL_THRESHOLD = 0.5

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


# ---------------------------------------------------------------------------
# Models (lazy-loaded, cached)
# ---------------------------------------------------------------------------

class _Embedder:
    """Tiny wrapper exposing get_text_embedding() over sentence-transformers.

    The swarm itself no longer embeds anything; this offline eval script still
    does, so it depends on ``sentence-transformers`` directly (install it
    separately: ``pip install "sentence-transformers>=4,<5"``).
    """

    def __init__(self, name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(name)

    def get_text_embedding(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


@functools.lru_cache(maxsize=1)
def _get_embed_model() -> _Embedder:
    return _Embedder()


# General-purpose NLI: trained on everyday sentence pairs (MNLI/SNLI-style).
# Conservative for domain paraphrase -- "GLP-1 exerts neuroprotective effects
# via PI3K/Akt" vs. a differently-worded specific claim about the same
# mechanism often lands as "neutral" rather than "entailment" here, because
# the model has no biomedical vocabulary/relationship priors to draw on.
GENERAL_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Domain-tuned: PubMedBERT (pretrained on PubMed abstracts/full text -- the
# same domain our sources come from) fine-tuned MNLI -> MedNLI. Same 3-label
# scheme (contradiction/entailment/neutral) as the general model, so it's a
# drop-in swap, not a different interface.
DOMAIN_NLI_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"


@functools.lru_cache(maxsize=4)
def _get_nli_model(model_name: str):
    """Load a CrossEncoder NLI model. Downloads to the HF cache on first use;
    cached thereafter. Falls back to None (consistency checks skipped, not
    silently faked) if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI cross-encoder %s ...", model_name)
        model = CrossEncoder(model_name)
        logger.info("NLI model loaded. Label order: %s", model.config.id2label)
        return model
    except Exception as exc:
        logger.warning(
            "Could not load NLI model %s (%s) -- consistency checks will be skipped.",
            model_name, exc,
        )
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ---------------------------------------------------------------------------
# Data loading / normalisation
# ---------------------------------------------------------------------------

def _load_expected(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": f["id"],
            "claim": f["claim"],
            "quantitative": f.get("quantitative"),
            "source": f.get("source", {}),
        }
        for f in data["findings"]
    ]


def _load_generated_report(path: Path) -> list[dict[str, Any]]:
    """Score against the actual report the reader sees (report.sections).

    This is the default and the one that matters: a finding can be produced by
    a worker, survive the critic, and still never reach the final report (the
    writer drops/reshapes content independently), or reach it reworded in a
    way that loses detail. Scoring the findings list instead of this would
    silently credit content the reader never actually gets.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    sections = data.get("report", {}).get("sections", [])
    return [
        {
            "id": s.get("heading", f"section-{i}"),
            "claim": s.get("body_md", ""),
            "citations": s.get("citations", []),
        }
        for i, s in enumerate(sections)
        if s.get("body_md", "").strip()
    ]


def _load_generated_findings(path: Path) -> list[dict[str, Any]]:
    """Score against the intermediate worker findings, before the writer/critic
    stage. Useful for diagnosing *where* content is lost (worker synthesis vs.
    critic rejection vs. writer omission) -- not what a reader actually sees.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": f.get("sub_question", f"generated-{i}"),
            "claim": f["claim"],
            "confidence": f.get("confidence"),
            "n_evidence": f.get("n_evidence"),
        }
        for i, f in enumerate(data["findings"])
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> set[float]:
    return {round(float(m), 2) for m in _NUMBER_RE.findall(text)}


def _numeric_recall(
    expected_quant: dict, generated_claim: str,
) -> tuple[float, list[float], list[float]]:
    """Return (recall, found, missing) for the expected finding's key numbers."""
    expected_numbers: set[float] = set()
    for key, val in expected_quant.items():
        if isinstance(val, (int, float)):
            expected_numbers.add(round(float(val), 2))
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, (int, float)):
                    expected_numbers.add(round(float(v), 2))
    if not expected_numbers:
        return 1.0, [], []

    generated_numbers = _extract_numbers(generated_claim)
    # Tolerance match: within 0.05 absolute, or within 2% relative for larger values.
    found, missing = [], []
    for exp_n in expected_numbers:
        hit = any(
            abs(exp_n - gen_n) <= max(0.05, abs(exp_n) * 0.02)
            for gen_n in generated_numbers
        )
        (found if hit else missing).append(exp_n)

    recall = len(found) / len(expected_numbers)
    return recall, found, missing


# Detects epistemic/reporting hedges ("the report does not detail/discuss/
# address X") as distinct from direct ontological negation ("X does not
# happen"). Empirically isolated: PubMedBERT-MNLI-MedNLI handles direct
# negation correctly at any length tested (up to 30+ words, high or low
# vocabulary overlap with the hypothesis), but calls a hedge like "does not
# explicitly detail the mechanism" an "entailment" of a specific claim about
# that very mechanism, with >99% confidence, regardless of length. A plain
# length or lexical-overlap threshold does not fire on this case (and would
# also wrongly veto long-but-correct direct negations/entailments) -- the
# hedge verb is the actual signal, not length.
_HEDGE_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|don'?t|doesn'?t)\s+(?:\w+\s+){0,2}"
    r"(?:detail|discuss|address|cover|mention|specify|elaborate|state)\w*\b"
    r"|\bnot\s+(?:explicitly\s+)?"
    r"(?:detailed|discussed|addressed|covered|mentioned|specified|available)\b"
    r"|\bno\s+(?:information|data|details|evidence)\s+"
    r"(?:is\s+|was\s+)?(?:provided|available|given|found)\b",
    re.IGNORECASE,
)


def _is_hedge_text(text: str) -> bool:
    return bool(_HEDGE_RE.search(text))


def _nli_label(model, premise: str, hypothesis: str) -> tuple[str, float]:
    """Return (label, score) for whether *premise* entails *hypothesis*.

    premise  = the generated (swarm) claim
    hypothesis = the expected (ground-truth) claim
    entailment  -> generated claim confirms the expected fact
    contradiction -> generated claim states the opposite
    neutral -> generated claim doesn't clearly confirm or deny it
    """
    if model is None:
        return "unknown", 0.0
    # apply_softmax=True: predict() returns raw logits by default, which are
    # not confidence scores (can be >1 or <0) -- softmax turns them into a
    # proper probability distribution over {contradiction, entailment, neutral}.
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
    label_idx = int(scores.argmax())
    label = model.config.id2label[label_idx].lower()
    confidence = float(scores[label_idx])
    if confidence < NLI_CONFIDENCE_THRESHOLD:
        return "neutral", confidence
    return label, confidence


def _nli_label_guarded(
    primary_model, primary_name: str, premise: str, hypothesis: str,
) -> tuple[str, float, bool]:
    """_nli_label(), with the hedge-phrase guard applied to the domain model.

    If the domain model calls "entailment" on premise text that contains an
    epistemic hedge ("does not detail/discuss/address ..."), that specific
    result is untrusted (see _HEDGE_RE) and the general-purpose model's
    verdict is used instead. Only entailment is guarded -- the domain model's
    contradiction/neutral calls on hedge text weren't observed to misfire.

    Returns (label, confidence, guard_triggered) -- the third value is
    recorded per-row so a reader can see exactly when/how often the fallback
    fired, rather than it happening silently.
    """
    label, confidence = _nli_label(primary_model, premise, hypothesis)
    if primary_name == DOMAIN_NLI_MODEL and label == "entailment" and _is_hedge_text(premise):
        fallback_model = _get_nli_model(GENERAL_NLI_MODEL)
        fb_label, fb_confidence = _nli_label(fallback_model, premise, hypothesis)
        return fb_label, fb_confidence, True
    return label, confidence, False


def match_findings(
    expected: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    threshold: float = MATCH_THRESHOLD,
    nli_model_name: str = DOMAIN_NLI_MODEL,
) -> dict[str, Any]:
    embed_model = _get_embed_model()
    nli_model = _get_nli_model(nli_model_name)

    exp_embs = [embed_model.get_text_embedding(f["claim"]) for f in expected]
    gen_embs = [embed_model.get_text_embedding(f["claim"]) for f in generated]

    sim_matrix = [[_cosine(e, g) for g in gen_embs] for e in exp_embs]

    per_expected: list[dict[str, Any]] = []
    matched_generated_idx: set[int] = set()

    for i, exp in enumerate(expected):
        sims = sim_matrix[i]
        best_idx = max(range(len(sims)), key=lambda j: sims[j]) if sims else None
        best_sim = sims[best_idx] if best_idx is not None else 0.0

        row: dict[str, Any] = {
            "expected_id": exp["id"],
            "expected_claim": exp["claim"],
            "matched_generated_id": None,
            "matched_generated_claim": None,
            "similarity": round(best_sim, 4),
            "nli_label": None,
            "nli_confidence": None,
            "nli_hedge_guard_triggered": False,
            "numeric_recall": None,
            "numeric_found": [],
            "numeric_missing": [],
            "verdict": "missing",
        }

        if best_idx is not None and best_sim >= threshold:
            gen = generated[best_idx]
            matched_generated_idx.add(best_idx)
            row["matched_generated_id"] = gen["id"]
            row["matched_generated_claim"] = gen["claim"]

            label, confidence, guard_triggered = _nli_label_guarded(
                nli_model, nli_model_name, gen["claim"], exp["claim"]
            )
            row["nli_label"] = label
            row["nli_confidence"] = round(confidence, 4)
            row["nli_hedge_guard_triggered"] = guard_triggered

            if exp["quantitative"]:
                recall, found, missing = _numeric_recall(exp["quantitative"], gen["claim"])
                row["numeric_recall"] = round(recall, 4)
                row["numeric_found"] = found
                row["numeric_missing"] = missing

            if label == "contradiction":
                row["verdict"] = "contradicted"
            elif exp["quantitative"] and row["numeric_recall"] is not None \
                    and row["numeric_recall"] < NUMERIC_RECALL_THRESHOLD:
                row["verdict"] = "covered_but_vague"
            elif label != "entailment":
                # High embedding similarity only means "topically related" --
                # e.g. a section that explicitly says "we don't have this
                # info" is still about the right topic, so it can out-score
                # every other candidate on cosine similarity alone. Only an
                # NLI "entailment" verdict means the text actually confirms
                # the expected claim; "neutral" here means the match is
                # real but unconfirmed, not a false positive to wave through.
                row["verdict"] = "covered_but_unconfirmed"
            else:
                row["verdict"] = "covered_and_consistent"

        per_expected.append(row)

    unmatched_generated = [
        {"id": generated[j]["id"], "claim": generated[j]["claim"]}
        for j in range(len(generated))
        if j not in matched_generated_idx
    ]

    return {"per_expected": per_expected, "unmatched_generated": unmatched_generated}


def summarize(result: dict[str, Any], n_generated: int) -> dict[str, Any]:
    rows = result["per_expected"]
    n = len(rows)
    verdict_counts = {v: 0 for v in (
        "missing", "contradicted", "covered_but_vague",
        "covered_but_unconfirmed", "covered_and_consistent",
    )}
    for r in rows:
        verdict_counts[r["verdict"]] += 1

    covered = n - verdict_counts["missing"]
    numeric_scores = [r["numeric_recall"] for r in rows if r["numeric_recall"] is not None]

    return {
        "n_expected": n,
        "n_generated": n_generated,
        "coverage_recall": round(covered / n, 4) if n else 0.0,
        "contradiction_rate": round(verdict_counts["contradicted"] / n, 4) if n else 0.0,
        "consistency_rate_of_covered": (
            round(verdict_counts["covered_and_consistent"] / covered, 4) if covered else None
        ),
        "numeric_precision_mean": (
            round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else None
        ),
        "verdict_counts": verdict_counts,
        "n_hedge_guard_triggered": sum(1 for r in rows if r["nli_hedge_guard_triggered"]),
        "n_unmatched_generated": len(result["unmatched_generated"]),
        "generated_precision": (
            round((n_generated - len(result["unmatched_generated"])) / n_generated, 4)
            if n_generated else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(result: dict[str, Any], summary: dict[str, Any]) -> None:
    print("\n=== Per-expected-finding results ===")
    for r in result["per_expected"]:
        print(f"\n[{r['expected_id']}] verdict={r['verdict']} sim={r['similarity']}")
        print(f"  expected : {r['expected_claim'][:110]}")
        if r["matched_generated_claim"]:
            print(f"  matched  : {r['matched_generated_claim'][:110]}")
            guard_note = " [hedge guard: fell back to general model]" \
                if r["nli_hedge_guard_triggered"] else ""
            print(f"  nli      : {r['nli_label']} ({r['nli_confidence']}){guard_note}")
            if r["numeric_recall"] is not None:
                print(f"  numbers  : recall={r['numeric_recall']} "
                      f"found={r['numeric_found']} missing={r['numeric_missing']}")
        else:
            print("  matched  : (none above threshold)")

    if result["unmatched_generated"]:
        print("\n=== Generated findings with no expected match (possible scope drift) ===")
        for g in result["unmatched_generated"]:
            print(f"  [{g['id']}] {g['claim'][:110]}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--expected", type=Path, default=here / "glp1_parkinsons_findings.json")
    parser.add_argument("--generated", type=Path, default=here / "swarm_output.json")
    parser.add_argument(
        "--source", choices=["report", "findings"], default="report",
        help="'report' (default) scores report.sections -- what the reader sees. "
             "'findings' scores the pre-writer worker findings instead.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument(
        "--nli-model", choices=["domain", "general"], default="domain",
        help="'domain' (default) uses PubMedBERT-MNLI-MedNLI, tuned on biomedical "
             "claim pairs. 'general' uses the original MNLI/SNLI-trained cross-encoder, "
             "kept for comparison -- it's a stricter, less domain-aware judge.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_path = args.out or here / f"claim_match_report_{args.source}_{args.nli_model}.json"
    nli_model_name = DOMAIN_NLI_MODEL if args.nli_model == "domain" else GENERAL_NLI_MODEL

    expected = _load_expected(args.expected)
    generated = (
        _load_generated_report(args.generated) if args.source == "report"
        else _load_generated_findings(args.generated)
    )

    result = match_findings(
        expected, generated, threshold=args.threshold, nli_model_name=nli_model_name,
    )
    summary = {
        "source": args.source, "nli_model": nli_model_name,
        **summarize(result, n_generated=len(generated)),
    }

    _print_report(result, summary)

    out_path.write_text(
        json.dumps({"summary": summary, **result}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
