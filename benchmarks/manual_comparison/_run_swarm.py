"""Run the research swarm on the same topic as the manual PubMed comparison.

Cloud-only (gemma4:31b-cloud for every tier) to avoid local model inference
load, given a prior local-model benchmark run overheated the machine.
"""
import asyncio
import json
import logging
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from research_swarm.config import settings

settings.default_model_provider = "ollama"
settings.default_model_name = "gemma4:31b-cloud"
settings.tier_fast_provider = "ollama"
settings.tier_fast_model = "gemma4:31b-cloud"
settings.tier_standard_provider = "ollama"
settings.tier_standard_model = "gemma4:31b-cloud"
settings.tier_thorough_provider = "ollama"
settings.tier_thorough_model = "gemma4:31b-cloud"
settings.max_sources = 6
settings.max_llm_calls = 40

import research_swarm.graph.nodes as nodes
from research_swarm.agents.base import get_agent_llm


def _uniform_tiered_llm(tier, temperature=0.0, provider_override=None):
    return get_agent_llm(provider="ollama", model="gemma4:31b-cloud", temperature=temperature)


nodes.get_tiered_llm = _uniform_tiered_llm

from research_swarm.graph.builder import build_graph, get_thread_config
from research_swarm.runtime.budget import clear_budget
from research_swarm.schemas.query import ResearchDepth, ResearchQuery

TOPIC = "GLP-1 receptor agonists as a neuroprotective, disease-modifying therapy in Parkinson's disease"

session_id = "manual-cmp-" + uuid.uuid4().hex[:8]

state_in = {
    "messages": [],
    "query": ResearchQuery(
        topic=TOPIC,
        depth=ResearchDepth.standard,
        max_sources=6,
        audience="technical",
    ),
    "plan": None, "findings": [], "critiques": [],
    "draft_report": None, "final_report": None,
    "human_feedback": None, "writer_instructions": None,
    "iteration_count": 0, "next_agent": None,
    "session_id": session_id,
    "model_provider": "ollama", "model_name": "gemma4:31b-cloud",
    "schema_version": 2, "research_rounds": 0,
    "pre_dispatch_finding_ids": [], "active_sub_question": None,
}


async def run():
    graph = build_graph(interrupt_before_writer=False)
    config = get_thread_config(session_id)
    started = time.perf_counter()

    async for chunk in graph.astream(state_in, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            msgs = update.get("messages", []) if isinstance(update, dict) else []
            for m in msgs:
                print(f"[{node_name}] {m.content}", flush=True)

    elapsed = time.perf_counter() - started
    final_state = (await graph.aget_state(config)).values
    clear_budget(session_id)

    report = final_state.get("final_report")
    findings = final_state.get("findings") or []
    critiques = final_state.get("critiques") or []

    out = {
        "topic": TOPIC,
        "session_id": session_id,
        "model": "gemma4:31b-cloud",
        "depth": "standard",
        "elapsed_seconds": round(elapsed, 2),
        "research_rounds": final_state.get("research_rounds"),
        "n_findings": len(findings),
        "n_critiques": len(critiques),
        "findings": [
            {
                "sub_question": f.sub_question,
                "claim": f.claim,
                "confidence": f.confidence,
                "n_evidence": len(f.evidence),
                "evidence_urls": [e.url for e in f.evidence],
            }
            for f in findings
        ],
        "critiques": [
            {"finding_id": c.finding_id, "verdict": c.verdict.value if hasattr(c.verdict, "value") else str(c.verdict),
             "reasoning": c.reasoning}
            for c in critiques
        ],
        "report": report.model_dump(mode="json") if report else None,
    }

    with open("benchmarks/manual_comparison/swarm_output.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nDONE in {elapsed:.1f}s -- {len(findings)} findings, "
          f"{len(report.references) if report else 0} references, "
          f"report.title={report.title if report else None}", flush=True)


asyncio.run(run())
