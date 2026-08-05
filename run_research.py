"""
Programmatic research runner — no Streamlit required.

Usage:
    poetry run python run_research.py "transformer attention mechanisms"
"""
from __future__ import annotations

import asyncio
import sys
import textwrap
import time
import uuid

from research_swarm.config import settings
from research_swarm.eval.faithfulness import FAITHFULNESS_THRESHOLD, score_report
from research_swarm.graph.builder import build_graph, get_thread_config
from research_swarm.schemas.query import ResearchDepth, ResearchQuery

# ── ANSI colours ──────────────────────────────────────────────────────────────
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    colour = _GREEN if score >= 0.6 else _YELLOW if score >= 0.3 else _RED
    return colour + "#" * filled + _DIM + "." * (width - filled) + _RESET


def _wrap(text: str, indent: int = 4, width: int = 88) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def _safe(text: str) -> str:
    """Strip non-ASCII characters that Windows cp1252 console can't display."""
    return text.encode("ascii", errors="replace").decode("ascii").replace(b"?".decode(), "?")


async def run(topic: str, depth: str = "shallow", model_name: str | None = None) -> None:
    if model_name:
        settings.default_model_name = model_name
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    query = ResearchQuery(
        topic=topic,
        depth=ResearchDepth(depth),
        max_sources=settings.max_sources,
        audience="technical",
    )

    initial_state = {
        "messages": [],
        "query": query,
        "plan": None,
        "findings": [],
        "critiques": [],
        "draft_report": None,
        "final_report": None,
        "human_feedback": None,
        "writer_instructions": None,
        "iteration_count": 0,
        "next_agent": None,
        "session_id": session_id,
        "model_provider": settings.default_model_provider,
        "model_name": settings.default_model_name,
        "schema_version": 2,
        "research_rounds": 0,
        "pre_dispatch_finding_ids": [],
        "active_sub_question": None,
        "active_worker_role": None,
    }

    graph = build_graph(interrupt_before_writer=False)
    config = get_thread_config(session_id)

    SEP  = "-" * 60
    SEP2 = "=" * 60
    print(f"\n{_BOLD}{SEP}{_RESET}")
    print(f"{_BOLD}  Research Swarm  {_RESET}  {_CYAN}{topic}{_RESET}")
    print(f"  depth={depth}  model={settings.default_model_provider}/{settings.default_model_name}")
    print(f"{_BOLD}{SEP}{_RESET}\n")

    node_times: dict[str, float] = {}
    t0 = time.perf_counter()
    last_node_start = t0

    async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
        elapsed = time.perf_counter() - t0
        for node_name, update in chunk.items():
            dt = time.perf_counter() - last_node_start
            node_times[node_name] = node_times.get(node_name, 0) + dt
            last_node_start = time.perf_counter()

            # Print per-node summary
            msgs = update.get("messages", [])
            summary = msgs[-1].content if msgs else ""
            findings_n = len(update.get("findings", []))
            critiques_n = len(update.get("critiques", []))

            extras = []
            if findings_n:
                extras.append(f"{findings_n} finding(s)")
            if critiques_n:
                extras.append(f"{critiques_n} critique(s)")
            if update.get("final_report"):
                extras.append("[report]")
            if update.get("next_agent"):
                extras.append(f"-> {update['next_agent']}")
            extra_str = "  " + "  ".join(extras) if extras else ""

            print(f"  {_CYAN}[{elapsed:5.1f}s]{_RESET} {_BOLD}{node_name:<16}{_RESET}{extra_str}")
            if summary:
                # Trim to 100 chars, strip bracketed prefix like [Writer]
                short = summary.lstrip()
                if short.startswith("["):
                    short = short.split("]", 1)[-1].strip()
                print(f"          {_DIM}{_safe(short[:100])}{_RESET}")

    total = time.perf_counter() - t0

    # ── Load final state ───────────────────────────────────────────────────────
    snap        = await graph.aget_state(config)
    state       = snap.values
    report      = state.get("final_report")
    findings    = state.get("findings") or []
    critiques   = state.get("critiques") or []
    plan        = state.get("plan")

    print(f"\n{_BOLD}{SEP}{_RESET}")
    print(f"{_BOLD}  Timing  ({total:.1f}s total){_RESET}")
    for node, t in sorted(node_times.items(), key=lambda x: -x[1]):
        print(f"    {node:<20} {t:5.1f}s")

    if report is None:
        print(f"\n{_RED}No final report was produced.{_RESET}")
        return

    # ── Quality metrics ────────────────────────────────────────────────────────
    references = report.references or []
    faith_score = score_report(report, references)

    # Confidence stats
    confs = [
        (f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.0))
        for f in findings
    ]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    min_conf  = min(confs) if confs else 0.0

    # Verdict counts
    verdict_counts: dict[str, int] = {}
    for c in critiques:
        v = (c.verdict if hasattr(c, "verdict") else c.get("verdict", "?"))
        v_str = str(v.value if hasattr(v, "value") else v)
        verdict_counts[v_str] = verdict_counts.get(v_str, 0) + 1

    print(f"\n{_BOLD}{SEP}{_RESET}")
    print(f"{_BOLD}  Quality Scorecard{_RESET}")
    faith_ok = faith_score >= FAITHFULNESS_THRESHOLD
    faith_label = "OK" if faith_ok else "below threshold"
    print(f"  Faithfulness   {_bar(faith_score)}  {faith_score:.2f}  {faith_label}")
    print(f"  Mean confidence{_bar(mean_conf)}  {mean_conf:.2f}")
    print(f"  Min confidence {_bar(min_conf)}   {min_conf:.2f}")
    if verdict_counts:
        vstr = "  ".join(f"{k}:{v}" for k, v in sorted(verdict_counts.items()))
        print(f"  Verdicts       {vstr}")
    print(f"  Findings       {len(findings)}")
    print(f"  References     {len(references)}")
    print(f"  Sections       {len(report.sections or [])}")
    if plan:
        sq = plan.sub_questions if hasattr(plan, "sub_questions") else plan.get("sub_questions", [])
        print(f"  Sub-questions  {len(sq)}")
        workers = plan.assignments if hasattr(plan, "assignments") else plan.get("assignments", [])
        if workers:
            roles = [
                (a.worker_role if hasattr(a, "worker_role") else a.get("worker_role", "general"))
                for a in workers
            ]
            print(f"  Worker roles   {', '.join(str(r) for r in roles)}")

    # ── Full Report ────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{SEP2}{_RESET}")
    print(f"{_BOLD}  REPORT: {report.title}{_RESET}")
    print(f"{_BOLD}{SEP2}{_RESET}")

    print(f"\n{_BOLD}  Executive Summary{_RESET}")
    for line in report.exec_summary.splitlines():
        print(_wrap(_safe(line.strip())))

    for section in (report.sections or []):
        print(f"\n{_BOLD}  {_safe(section.heading)}{_RESET}")
        for line in section.body_md.splitlines():
            stripped = line.strip()
            if stripped:
                print(_wrap(_safe(stripped)))
        if section.citations:
            cite_str = ", ".join(f"[{c}]" for c in section.citations)
            print(f"    {_DIM}Citations: {cite_str}{_RESET}")

    if report.methodology:
        print(f"\n{_BOLD}  Methodology{_RESET}")
        print(_wrap(_safe(report.methodology)))

    if report.limitations:
        print(f"\n{_BOLD}  Limitations{_RESET}")
        print(_wrap(_safe(report.limitations)))

    if references:
        print(f"\n{_BOLD}  References{_RESET}")
        for i, src in enumerate(references[:10], 1):
            url   = src.url   if hasattr(src, "url")   else src.get("url", "?")
            title = src.title if hasattr(src, "title") else src.get("title", "")
            stype = src.source_type if hasattr(src, "source_type") else src.get("source_type", "")
            score = (src.credibility_score if hasattr(src, "credibility_score")
                     else src.get("credibility_score", 0.0))
            label = f"{_safe(title[:55]):<55}" if title else f"{_safe(url[:55]):<55}"
            print(f"    [{i}] {label}  cred={score:.2f}  {_DIM}{stype}{_RESET}")

    print(f"\n{_BOLD}{SEP}{_RESET}")
    passed = faith_ok and mean_conf >= 0.4 and len(findings) >= 1
    verdict = "PASS" if passed else "NEEDS IMPROVEMENT"
    colour = _GREEN if verdict == "PASS" else _RED
    print(f"  Overall: {colour}{_BOLD}{verdict}{_RESET}"
          f"  (faith={faith_score:.2f}, conf={mean_conf:.2f})")
    print(f"{_BOLD}{SEP}{_RESET}\n")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "transformer attention mechanisms"
    # No model_name override -- uses settings.default_model_name so this stays
    # in sync with the configured default instead of hardcoding a model string
    # that can silently drift out of sync with what's actually pulled locally
    # (this line previously hardcoded "gemma4:e2b", which is no longer a valid
    # local Ollama tag and 404'd on every call).
    asyncio.run(run(topic))
