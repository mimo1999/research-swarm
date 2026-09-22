"""RAGAs evaluation: are the swarm's findings actually grounded in the source
material its own workers retrieved?

This is a different question from claim_matcher.py's job. claim_matcher.py
asks "does the report match a known-correct external ground truth?" -- it
needs the glp1_parkinsons_findings.json reference set. This script asks "is
each finding's claim actually supported by the evidence *the swarm itself*
retrieved?" -- it needs no external ground truth at all, just the finding and
its own cited sources. A finding can score well here and still be wrong
about the world (if its sources were bad or irrelevant); it can score badly
here despite being correct (if the claim says more than its sources show).
The two tools answer different questions and are meant to be read together.

Metrics (both from ragas.metrics.collections, the "modern" v2 API in
ragas>=0.4 -- classic ragas.metrics.Faithfulness etc. do not exist in this
version, everything moved under .collections and requires an instructor-
wrapped LLM client, not a bare LangChain model):

  1. Faithfulness -- decomposes the claim into atomic statements, checks each
     against the retrieved source snippets via NLI. Answers "did the swarm
     say anything its sources don't actually support?"
  2. ContextPrecisionWithoutReference -- checks whether each retrieved source
     was actually useful for answering the sub-question. Answers "how much
     of what got retrieved was noise?" -- a low score here means the worker's
     tool calls pulled in a lot of irrelevant material even if the final
     claim itself stayed faithful to the *relevant* subset.

Requires a live LLM as judge -- this project's real provider is Ollama, used
here via its OpenAI-compatible endpoint (http://localhost:11434/v1) since
ragas's modern API is built on `instructor`, which speaks to OpenAI-shaped
clients rather than arbitrary LangChain models directly. Pass --provider
anthropic/openai with a real key in .env to use those instead.

Install once: pip install ragas (not a core project dependency -- same
treatment as pyarrow in run_smoke_benchmark.py, kept out of pyproject.toml
since it's a heavy benchmark-only tool, not a runtime dependency).

Usage:
    poetry run python benchmarks/manual_comparison/ragas_eval.py \\
        --generated benchmarks/manual_comparison/swarm_output.json \\
        --provider ollama --model gemma4:31b-cloud \\
        --out benchmarks/manual_comparison/ragas_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (findings + their own cited source snippets -- no external
# ground truth needed, unlike claim_matcher.py)
# ---------------------------------------------------------------------------

def _load_findings_with_contexts(path: Path) -> list[dict[str, Any]]:
    """Return each finding paired with the snippet text of its own cited
    sources, looked up from report.references by URL."""
    data = json.loads(path.read_text(encoding="utf-8"))
    url_to_snippet = {
        r["url"]: r.get("snippet", "")
        for r in data.get("report", {}).get("references", [])
    }

    items = []
    for f in data["findings"]:
        contexts = [
            url_to_snippet[u] for u in f.get("evidence_urls", [])
            if url_to_snippet.get(u, "").strip()
        ]
        items.append({
            "sub_question": f.get("sub_question", ""),
            "claim": f["claim"],
            "contexts": contexts,
            "n_evidence_urls": len(f.get("evidence_urls", [])),
            "n_contexts_with_snippet": len(contexts),
        })
    return items


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _build_llm(provider: str, model: str):
    """Return a ragas InstructorBaseRagasLLM for the given provider.

    ollama: uses Ollama's OpenAI-compatible endpoint (no real API key needed
    -- Ollama ignores it, but the OpenAI client requires a non-empty string).
    anthropic / openai: uses a real key from research_swarm.config.settings.
    """
    from ragas.llms.base import llm_factory

    from research_swarm.config import settings

    if provider == "ollama":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",  # unused by Ollama, but the client requires *something*
        )
        return llm_factory(model, client=client)

    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        return llm_factory(model, provider="anthropic", client=client)

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return llm_factory(model, client=client)

    raise ValueError(f"Unsupported provider: {provider!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_findings(items: list[dict[str, Any]], llm) -> list[dict[str, Any]]:
    from ragas.metrics.collections import ContextPrecisionWithoutReference, Faithfulness

    faithfulness = Faithfulness(llm=llm)
    context_precision = ContextPrecisionWithoutReference(llm=llm)

    results = []
    for item in items:
        row: dict[str, Any] = {
            "sub_question": item["sub_question"],
            "claim": item["claim"][:200],
            "n_evidence_urls": item["n_evidence_urls"],
            "n_contexts_with_snippet": item["n_contexts_with_snippet"],
            "faithfulness": None,
            "context_precision": None,
            "error": None,
        }
        if not item["contexts"]:
            row["error"] = "no source snippets available -- skipped"
            results.append(row)
            continue

        try:
            faith_result = await faithfulness.ascore(
                user_input=item["sub_question"],
                response=item["claim"],
                retrieved_contexts=item["contexts"],
            )
            row["faithfulness"] = round(float(faith_result.value), 4)
        except Exception as exc:
            logger.warning("Faithfulness scoring failed for %r: %s", item["sub_question"][:60], exc)
            row["error"] = f"faithfulness: {type(exc).__name__}: {exc}"

        try:
            prec_result = await context_precision.ascore(
                user_input=item["sub_question"],
                response=item["claim"],
                retrieved_contexts=item["contexts"],
            )
            row["context_precision"] = round(float(prec_result.value), 4)
        except Exception as exc:
            logger.warning(
                "ContextPrecision scoring failed for %r: %s", item["sub_question"][:60], exc,
            )
            row["error"] = (row["error"] + "; " if row["error"] else "") + \
                f"context_precision: {type(exc).__name__}: {exc}"

        results.append(row)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    prec_scores = [r["context_precision"] for r in results if r["context_precision"] is not None]
    n_errors = sum(1 for r in results if r["error"])
    return {
        "n_findings": len(results),
        "n_scored": len(faith_scores),
        "n_errors_or_skipped": n_errors,
        "mean_faithfulness": (
            round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None
        ),
        "mean_context_precision": (
            round(sum(prec_scores) / len(prec_scores), 4) if prec_scores else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("\n=== Per-finding results ===")
    for r in results:
        print(f"\n[{r['sub_question'][:80]}]")
        print(f"  claim: {r['claim']}")
        print(
            f"  sources: {r['n_contexts_with_snippet']}/{r['n_evidence_urls']} "
            "had usable snippet text"
        )
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  faithfulness={r['faithfulness']}  context_precision={r['context_precision']}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


async def _main_async(args: argparse.Namespace) -> None:
    items = _load_findings_with_contexts(args.generated)
    llm = _build_llm(args.provider, args.model)
    results = await evaluate_findings(items, llm)
    summary = summarize(results)
    _print_report(results, summary)

    args.out.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--generated", type=Path, default=here / "swarm_output.json")
    parser.add_argument("--out", type=Path, default=here / "ragas_report.json")
    parser.add_argument(
        "--provider", choices=["ollama", "anthropic", "openai"], default="ollama",
        help="'ollama' (default) uses this project's real provider via its "
             "OpenAI-compatible endpoint. Requires Ollama running locally.",
    )
    parser.add_argument(
        "--model", default="gemma4:31b-cloud",
        help="Model name for the chosen provider (e.g. gemma4:31b-cloud for ollama, "
             "claude-haiku-4-5-20251001 for anthropic, gpt-5-nano for openai).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
