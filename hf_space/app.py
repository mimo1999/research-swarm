"""Multi-Agent Research Swarm — Gradio Space entry point.

A custom gr.Blocks UI (not gr.ChatInterface) that mirrors the shape of the
Next.js frontend built in frontend/: a progressive-disclosure settings form,
a live segmented agent-trace timeline, a real human-in-the-loop review panel,
and a structured, cited report view. Drives the same research_swarm LangGraph
backend directly in-process (no separate FastAPI layer needed here — a
Gradio event handler IS the server, running on its own persistent asyncio
loop, so this doesn't need the background-event-loop workaround app.py's
Streamlit version needs for the same checkpointer).

Deployment: this file plus requirements.txt/README.md here are staged copies
for a Hugging Face Space. They get copied to the Space repo root ALONGSIDE
the real research_swarm/ package (not duplicated here) -- see the README in
this directory for the exact steps.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid

import gradio as gr

from research_swarm.agents._utils import _field
from research_swarm.config import settings
from research_swarm.graph.builder import build_graph, get_thread_config, make_async_checkpointer
from research_swarm.runtime.budget import clear_budget
from research_swarm.runtime.session_ctx import (
    SessionCredentials,
    bind_session,
    session_scope,
    unbind_session,
)
from research_swarm.schemas import ResearchQuery

try:
    import spaces
except ImportError:  # not on ZeroGPU hardware -- see _zerogpu_placeholder below
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(fn):
            return fn


@spaces.GPU
def _zerogpu_placeholder() -> bool:
    """Satisfies ZeroGPU's startup check ("No @spaces.GPU function detected
    during startup"). This app's LLM calls go to Ollama Cloud and its
    embedding model is CPU-sized -- it never actually needs a GPU. ZeroGPU
    is required only because it's (at time of writing) the only hardware
    tier available on this Space's plan; if that changes, this function and
    the `spaces` import above can be deleted along with switching the
    Space's hardware to cpu-basic.
    """
    return True


logger = logging.getLogger(__name__)

# This deployment pays for Ollama Cloud (OLLAMA_API_KEY, a Space secret --
# see settings.ollama_api_key) and reaches it *directly*, with no local
# `ollama serve` process in the container: confirmed live that
# https://ollama.com mirrors the local daemon's API surface (GET /api/tags
# is public, POST /api/chat returns a clean 401 without a valid bearer
# token) rather than requiring the `ollama login`-session-mediated proxying
# every other path in this codebase assumes. Anthropic/OpenAI are NOT
# funded by this deployment -- there's no server key for them at all, so a
# visitor who wants those must type in their own (see the password fields
# below, threaded through session_ctx per-request, never settings).
_SPACE_DEFAULT_PROVIDER = "ollama"
_SPACE_DEFAULT_MODEL = "nemotron-3-nano:30b-cloud"
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

_PROVIDER_MODEL_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5-nano",
    "ollama": _SPACE_DEFAULT_MODEL,
}


# ---------------------------------------------------------------------------
# Graph / checkpointer — lazily built on Gradio's own event loop
# ---------------------------------------------------------------------------
# Streamlit needs a dedicated background event loop because asyncio.run()
# makes a NEW loop every script rerun, and AsyncSqliteSaver's internal
# asyncio.Lock objects are bound to whichever loop first awaited them.
# Gradio's async event handlers are awaited directly on the app's own
# long-lived server loop, so building the checkpointer inside the first real
# handler call (instead of at import time, before any loop is running) is
# enough to avoid that mismatch -- no background thread needed here.

_checkpointer = None
_checkpointer_lock = asyncio.Lock()
_graph_cache: dict[bool, object] = {}
_run_semaphore: asyncio.Semaphore | None = None


async def _get_graph(hitl: bool):
    global _checkpointer, _run_semaphore
    if _checkpointer is None:
        async with _checkpointer_lock:
            if _checkpointer is None:
                _checkpointer = await make_async_checkpointer()
                if settings.space_mode:
                    _run_semaphore = asyncio.Semaphore(settings.space_max_concurrent_runs)
    if hitl not in _graph_cache:
        _graph_cache[hitl] = build_graph(checkpointer=_checkpointer, interrupt_before_writer=hitl)
    return _graph_cache[hitl]


_pruned_once = False


def _prune_sessions_once() -> None:
    """Mirror app.py's once-per-process session pruning (space_mode only)."""
    global _pruned_once
    if _pruned_once or not settings.space_mode:
        return
    _pruned_once = True
    from research_swarm.persistence.sessions import prune_expired_sessions
    try:
        prune_expired_sessions(settings.space_retention_seconds, settings.space_max_sessions)
    except Exception:
        logger.exception("Session pruning failed")


# ---------------------------------------------------------------------------
# Document ingestion — same shape as app.py's _ingest_documents, adapted for
# gr.File's plain filepath list instead of Streamlit's UploadedFile objects.
# ---------------------------------------------------------------------------

def _ingest_documents(file_paths: list[str] | None, extra_urls: list[str]) -> list[dict]:
    if not file_paths and not extra_urls:
        return []

    from research_swarm.tools.pdf_loader import load_pdf
    from research_swarm.tools.url_fetcher import fetch_url

    documents: list[dict] = []

    for path in file_paths or []:
        try:
            result = load_pdf.invoke({"file_path": path})
            text = "\n\n".join(c["text"] for c in result.get("chunks", []) if c.get("text"))
            if text:
                documents.append({
                    "url": result.get("url", path),
                    "title": result.get("title", ""),
                    "text": text,
                    "source_type": "pdf",
                })
        except Exception:
            logger.exception("Failed to read uploaded PDF %s", path)

    for url in extra_urls or []:
        try:
            result = fetch_url.invoke({"url": url, "max_chars": 20000})
            snippet = result.get("snippet", "")
            if snippet.startswith("["):
                continue  # fetch error placeholder
            documents.append({
                "url": result.get("url", url),
                "title": result.get("title", ""),
                "text": snippet,
                "source_type": "web",
            })
        except Exception:
            logger.exception("Failed to fetch URL %s", url)

    return documents


# ---------------------------------------------------------------------------
# Trace timeline rendering — ports lib/research/nodeConfig.ts +
# traceSegments.ts + TraceView.tsx from the Next.js frontend to plain HTML
# strings. "stop" nodes (real research output) get a lettered badge + card;
# "connector" nodes (deterministic routing) get a one-line entry. Consecutive
# same-node stop entries merge into one card, exactly like the frontend, so
# a parallel worker_node fan-out reads as one "Research" step, not N.
# ---------------------------------------------------------------------------

_NODE_CONFIG = {
    "supervisor":           {"label": "Planning", "kind": "stop", "letter": "P"},
    "document_pass_node":   {"label": "Documents", "kind": "connector", "letter": ""},
    "document_worker_node": {"label": "Document extraction", "kind": "stop", "letter": "D"},
    "dispatch_node":        {"label": "Dispatch", "kind": "connector", "letter": ""},
    "worker_node":          {"label": "Research", "kind": "stop", "letter": "R"},
    "researcher":           {"label": "Research", "kind": "stop", "letter": "R"},
    "collect_node":         {"label": "Collect", "kind": "connector", "letter": ""},
    "critic":               {"label": "Critic review", "kind": "stop", "letter": "C"},
    "fact_checker":         {"label": "Fact-check", "kind": "stop", "letter": "F"},
    "writer":                {"label": "Report", "kind": "stop", "letter": "W"},
}
_DEFAULT_NODE_CONFIG = {"label": "Agent", "kind": "stop", "letter": "A"}


def _node_config(node: str) -> dict:
    return _NODE_CONFIG.get(node, _DEFAULT_NODE_CONFIG)


def _segment_kind(node: str, update: dict) -> str:
    if node == "supervisor":
        return "stop" if (update or {}).get("plan") is not None else "connector"
    return _node_config(node)["kind"]


def _connector_label(node: str, update: dict) -> str:
    update = update or {}
    if node == "dispatch_node":
        ids = update.get("pre_dispatch_finding_ids") or []
        n = len(ids) if isinstance(ids, list) else 0
        return f"Dispatching next round · {n} finding(s) so far" if n else "Dispatching research workers"
    if node == "collect_node":
        round_ = update.get("research_rounds", "?")
        nxt = "moving to review" if update.get("next_agent") == "critic" else "another research pass"
        return f"Round {round_} complete → {nxt}"
    if node == "document_pass_node":
        return "Preparing document extraction"
    if node == "supervisor":
        return f"Routing to {update.get('next_agent') or 'next agent'}"
    return _node_config(node)["label"]


def _build_segments(trace: list[tuple[str, dict]]) -> list[dict]:
    segments: list[dict] = []
    for node, update in trace:
        kind = _segment_kind(node, update)
        last = segments[-1] if segments else None
        if kind == "stop" and last and last["kind"] == "stop" and last["node"] == node:
            last["entries"].append(update)
        else:
            segments.append({"node": node, "kind": kind, "entries": [update]})
    return segments


def _esc(text) -> str:
    text = "" if text is None else str(text)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


_VERDICT_META = {
    "supported": ("Supported", "success"),
    "weak": ("Weak", "warning"),
    "refuted": ("Refuted", "danger"),
    "approve": ("Approved", "success"),
    "revise": ("Needs revision", "warning"),
    "reject": ("Rejected", "danger"),
    "pending": ("Pending", "muted"),
}


def _verdict_badge(verdict: str) -> str:
    label, kind = _VERDICT_META.get(verdict, _VERDICT_META["pending"])
    return f'<span class="rs-badge rs-badge-{kind}">{_esc(label)}</span>'


def _stop_body_html(node: str, entries: list[dict]) -> str:
    if node == "supervisor":
        plan = (entries[0] or {}).get("plan")
        sub_qs = list(_field(plan, "sub_questions", []) or []) if plan else []
        if not sub_qs:
            return '<p class="rs-muted">Planning the research approach…</p>'
        items = "".join(f"<li>{_esc(q)}</li>" for q in sub_qs)
        return (
            f"<details><summary>Research plan ({len(sub_qs)} sub-questions)</summary>"
            f"<ol>{items}</ol></details>"
        )

    if node in ("worker_node", "document_worker_node", "researcher"):
        findings = [f for e in entries for f in ((e or {}).get("findings") or [])]
        count_line = f"<strong>{len(findings)}</strong> finding(s) produced"
        if len(entries) > 1:
            count_line += f" across {len(entries)} worker(s)"
        if not findings:
            return f'<p class="rs-muted">{count_line}</p>'
        rows = "".join(
            f'<div class="rs-finding">'
            f'<p class="rs-finding-q">{_esc(_field(f, "sub_question"))}</p>'
            f'<p>{_esc(_field(f, "claim"))}</p>'
            f'<p class="rs-muted rs-small">confidence: {float(_field(f, "confidence", 0)):.2f} · '
            f'{len(_field(f, "evidence", []) or [])} source(s)</p>'
            f"</div>"
            for f in findings
        )
        return (
            f'<p class="rs-muted">{count_line}</p>'
            f"<details><summary>View findings</summary>{rows}</details>"
        )

    if node == "critic":
        critiques = [c for e in entries for c in ((e or {}).get("critiques") or [])]
        if not critiques:
            return '<p class="rs-muted">0 critique(s) produced</p>'
        rows = "".join(
            f'<div class="rs-finding">'
            f'<p>{_verdict_badge(str(_field(c, "verdict", "")))} '
            f'<code class="rs-small">{_esc((_field(c, "finding_id") or "")[:8])}</code></p>'
            f'<p>{_esc(_field(c, "reasoning"))}</p>'
            f"</div>"
            for c in critiques
        )
        return (
            f'<p class="rs-muted"><strong>{len(critiques)}</strong> critique(s) produced</p>'
            f"<details><summary>View critiques</summary>{rows}</details>"
        )

    if node == "fact_checker":
        findings = [f for e in entries for f in ((e or {}).get("findings") or [])]
        if not findings:
            return '<p class="rs-muted">0 finding(s) fact-checked</p>'
        rows = "".join(
            f'<div class="rs-finding">'
            f'<p class="rs-finding-q">{_esc(_field(f, "sub_question"))}</p>'
            f'<p class="rs-muted rs-small">{float(_field(f, "confidence", 0)):.2f} confidence</p>'
            f'<div class="rs-bar"><div class="rs-bar-fill" style="width:{float(_field(f, "confidence", 0)) * 100:.0f}%"></div></div>'
            f"</div>"
            for f in findings
        )
        return (
            f'<p class="rs-muted"><strong>{len(findings)}</strong> finding(s) fact-checked</p>'
            f"<details><summary>Updated confidence scores</summary>{rows}</details>"
        )

    if node == "writer":
        report = (entries[-1] or {}).get("final_report")
        if report is not None:
            title = _field(report, "title", "")
            return f'<p class="rs-success">Report complete: <strong>{_esc(title)}</strong></p>'
        return '<p class="rs-muted">Writing report…</p>'

    return f"<details><summary>Raw update</summary><pre>{_esc(entries)}</pre></details>"


def render_trace_html(trace: list[tuple[str, dict]], live: bool) -> str:
    segments = _build_segments(trace)
    rows = []
    for seg in segments:
        cfg = _node_config(seg["node"])
        if seg["kind"] == "connector":
            label = _connector_label(seg["node"], seg["entries"][-1])
            rows.append(
                f'<div class="rs-row rs-connector"><span class="rs-dot"></span>'
                f'<span class="rs-connector-text">{_esc(label)}</span></div>'
            )
        else:
            rows.append(
                f'<div class="rs-row rs-stop">'
                f'<span class="rs-marker">{_esc(cfg["letter"])}</span>'
                f'<div class="rs-card"><p class="rs-card-title">{_esc(cfg["label"])}</p>'
                f'{_stop_body_html(seg["node"], seg["entries"])}</div></div>'
            )
    if live:
        rows.append(
            '<div class="rs-row rs-connector"><span class="rs-dot rs-spin"></span>'
            '<span class="rs-connector-text rs-muted">Waiting for the next agent…</span></div>'
        )
    body = "".join(rows) or '<p class="rs-muted">Starting…</p>'
    return f'<div class="rs-trace"><div class="rs-line"></div><div class="rs-rows">{body}</div></div>'


def render_hitl_html(findings: list, critiques: list) -> str:
    verdict_by_finding = {}
    for c in critiques or []:
        fid = _field(c, "finding_id")
        if fid:
            verdict_by_finding[fid] = str(_field(c, "verdict", ""))

    if not findings:
        rows = '<p class="rs-muted">No findings were produced.</p>'
    else:
        rows = "".join(
            f'<div class="rs-finding">'
            f'<p>{_verdict_badge(verdict_by_finding.get(_field(f, "id"), "pending"))} '
            f'<strong>{_esc(_field(f, "sub_question"))}</strong></p>'
            f'<p>{_esc(_field(f, "claim"))}</p>'
            f'<p class="rs-muted rs-small">confidence: {float(_field(f, "confidence", 0)):.2f}</p>'
            f"</div>"
            for f in findings
        )
    return (
        '<div class="rs-panel">'
        '<p class="rs-panel-title">Human review required</p>'
        '<p class="rs-muted">The graph has paused before writing. Review the findings below and choose how to proceed.</p>'
        f'<p><strong>Findings ({len(findings)})</strong></p>{rows}'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Report rendering — ports ReportView.tsx's citation-footnote approach:
# rewrite literal "[N]" markers into real markdown links to a "#ref-N"
# anchor before markdown conversion, then style those anchors with CSS. Out-
# of-range citations are left as plain (unlinked) text instead of a dead link.
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\[(\d+)\](?!\()")


def _linkify_citations(md: str, ref_count: int) -> str:
    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{n}](#ref-{n})" if 1 <= n <= ref_count else m.group(0)
    return _CITE_RE.sub(repl, md or "")


def _markdown_to_html(md: str, ref_count: int) -> str:
    import markdown as md_lib
    return md_lib.markdown(_linkify_citations(md, ref_count), extensions=["tables", "fenced_code"])


def render_report_html(report) -> str:
    if report is None:
        return '<p class="rs-muted">No report was produced.</p>'

    ref_count = len(_field(report, "references", []) or [])
    parts = [f'<h2 class="rs-report-title">{_esc(_field(report, "title"))}</h2>']

    quality = _field(report, "quality_score")
    judge = _field(report, "llm_judge")
    if quality is not None or judge is not None:
        stats = []
        if quality is not None:
            def _pct(v):
                return f"{v * 100:.0f}%" if v is not None else "Not computed"
            stats.append(
                '<div class="rs-stats">'
                f'<div><p class="rs-muted rs-small">Faithfulness</p><p class="rs-stat">{_pct(_field(quality, "faithfulness"))}</p></div>'
                f'<div><p class="rs-muted rs-small">Relevance</p><p class="rs-stat">{_pct(_field(quality, "relevance"))}</p></div>'
                f'<div><p class="rs-muted rs-small">Completeness</p><p class="rs-stat">{_pct(_field(quality, "completeness"))}</p></div>'
                f'<div><p class="rs-muted rs-small">Overall</p><p class="rs-stat">{_pct(_field(quality, "overall"))}</p></div>'
                "</div>"
            )
        if judge is not None:
            verdict = str(_field(judge, "verdict", ""))
            overall = _field(judge, "overall", 0)
            stats.append(
                f'<p><strong>LLM judge</strong> {_verdict_badge(verdict)} '
                f'<span class="rs-muted">{overall:.1f}/5</span></p>'
                '<div class="rs-stats">'
                f'<div><p class="rs-muted rs-small">Coherence</p><p class="rs-stat">{_field(judge, "coherence")}/5</p></div>'
                f'<div><p class="rs-muted rs-small">Relevance</p><p class="rs-stat">{_field(judge, "relevance")}/5</p></div>'
                f'<div><p class="rs-muted rs-small">Completeness</p><p class="rs-stat">{_field(judge, "completeness")}/5</p></div>'
                f'<div><p class="rs-muted rs-small">Citation quality</p><p class="rs-stat">{_field(judge, "citation_quality")}/5</p></div>'
                "</div>"
                f'<details><summary>Judge reasoning</summary><p class="rs-muted">{_esc(_field(judge, "reasoning"))}</p></details>'
            )
        parts.append(f'<div class="rs-quality">{"".join(stats)}</div>')

    parts.append('<h3>Executive summary</h3>')
    parts.append(f'<div class="rs-prose">{_markdown_to_html(_field(report, "exec_summary", ""), ref_count)}</div>')

    for i, section in enumerate(_field(report, "sections", []) or [], start=1):
        heading = _field(section, "heading", "")
        body = _field(section, "body_md", "")
        citations = _field(section, "citations", []) or []
        parts.append(f'<h3>{i}. {_esc(heading)}</h3>')
        parts.append(f'<div class="rs-prose">{_markdown_to_html(body, ref_count)}</div>')
        inline_nums = {int(n) for n in re.findall(r"\[(\d+)\]", body or "")}
        orphaned = [c for c in citations if c not in inline_nums]
        if orphaned:
            marks = " ".join(f'<a class="rs-cite" href="#ref-{n}">{n}</a>' for n in orphaned)
            parts.append(f'<p class="rs-muted rs-small">Also drawn from: {marks}</p>')

    references = _field(report, "references", []) or []
    if references:
        parts.append(f'<h3>References ({len(references)})</h3>')
        rows = []
        for i, ref in enumerate(references, start=1):
            title = _field(ref, "title") or _field(ref, "url", "")
            url = _field(ref, "url", "")
            snippet = _field(ref, "snippet", "")
            source_type = str(_field(ref, "source_type", ""))
            credibility = _field(ref, "credibility_score", 0) or 0
            rows.append(
                f'<div class="rs-ref" id="ref-{i}">'
                f'<span class="rs-muted rs-small">[{i}]</span> '
                f'<a href="{_esc(url)}" target="_blank" rel="noreferrer"><strong>{_esc(title)}</strong></a>'
                + (f'<p class="rs-muted rs-small">{_esc(snippet)}</p>' if snippet else "")
                + f'<p class="rs-muted rs-small">{_esc(source_type)} · {credibility * 100:.0f}% credibility</p>'
                "</div>"
            )
        parts.append("".join(rows))

    methodology = _field(report, "methodology", "")
    limitations = _field(report, "limitations", "")
    if methodology or limitations:
        inner = ""
        if methodology:
            inner += f"<p><strong>Methodology</strong></p><p>{_esc(methodology)}</p>"
        if limitations:
            inner += f"<p><strong>Limitations</strong></p><p>{_esc(limitations)}</p>"
        parts.append(f"<details><summary>Methodology &amp; limitations</summary>{inner}</details>")

    return f'<div class="rs-report">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Graph driving — shared by the initial submit and every resume path, same
# shape as api/runs.py's _drive: normalize a parallel Send() fan-out's
# list/tuple-of-updates into individual trace entries instead of crashing on
# them (see api/runs.py + api/serialize.py for the original bug this fixes).
# ---------------------------------------------------------------------------

def _idle_outputs():
    return (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        gr.update(), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), None, None, [],
    )


async def _drive(graph, input_state, config, trace, session_id):
    """Drive the graph inside this session's credential scope.

    session_scope makes resolve_api_key() etc. (called deep inside every LLM
    factory call under graph.astream) see THIS session's bound credentials --
    the BYOK Anthropic/OpenAI key a visitor typed in, or nothing for Ollama,
    which falls back to settings.ollama_api_key (the Space's own key) since
    no per-visitor key was bound for it. See session_ctx.py.
    """
    sem = _run_semaphore if settings.space_mode else None
    try:
        with session_scope(session_id):
            if sem is not None:
                await sem.acquire()
            try:
                async for chunk in graph.astream(input_state, config, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        updates = node_update if isinstance(node_update, (list, tuple)) else (node_update,)
                        for u in updates:
                            trace.append((node_name, u or {}))
                    yield (
                        gr.update(visible=False),
                        gr.update(value=render_trace_html(trace, live=True), visible=True),
                        gr.update(visible=False), gr.update(),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(), gr.update(), trace,
                    )

                snapshot = await graph.aget_state(config)
                values = snapshot.values or {}
                if snapshot.next:
                    findings = values.get("findings", []) or []
                    critiques = values.get("critiques", []) or []
                    yield (
                        gr.update(visible=False),
                        gr.update(value=render_trace_html(trace, live=False), visible=True),
                        gr.update(visible=True), gr.update(value=render_hitl_html(findings, critiques)),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(), gr.update(), trace,
                    )
                else:
                    report = values.get("final_report")
                    yield (
                        gr.update(visible=False),
                        gr.update(value=render_trace_html(trace, live=False), visible=True),
                        gr.update(visible=False), gr.update(),
                        gr.update(value=render_report_html(report), visible=True),
                        gr.update(visible=True), gr.update(visible=False),
                        gr.update(), gr.update(), trace,
                    )
            finally:
                if sem is not None:
                    sem.release()
    except Exception as exc:
        logger.exception("Research run failed")
        clear_budget(session_id)
        yield (
            gr.update(visible=False),
            gr.update(value=render_trace_html(trace, live=False), visible=bool(trace)),
            gr.update(visible=False), gr.update(),
            gr.update(visible=False), gr.update(visible=True),
            gr.update(value=f"**Research run failed:** {_esc(exc)}", visible=True),
            gr.update(), gr.update(), trace,
        )


def _error_outputs(message: str):
    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(value=message, visible=True),
        None, None, [],
    )


async def start_research(
    topic, audience, provider, model, ollama_deployment, ollama_url,
    anthropic_key, openai_key, depth, max_sources, hitl, files, urls_text,
):
    _prune_sessions_once()
    if not topic or not topic.strip():
        yield _error_outputs("Please enter a research topic.")
        return
    if provider == "anthropic" and not (anthropic_key or "").strip():
        yield _error_outputs("This Space doesn't fund Anthropic usage -- please enter your own Anthropic API key in Advanced options.")
        return
    if provider == "openai" and not (openai_key or "").strip():
        yield _error_outputs("This Space doesn't fund OpenAI usage -- please enter your own OpenAI API key in Advanced options.")
        return

    session_id = str(uuid.uuid4())
    # Per-session credentials, never on the global settings singleton --
    # mutating settings here would let a concurrent visitor's request read
    # this visitor's key. See session_ctx.py's module docstring.
    bind_session(
        session_id,
        SessionCredentials(
            anthropic_api_key=(anthropic_key or "").strip(),
            openai_api_key=(openai_key or "").strip(),
            ollama_base_url=ollama_url if provider == "ollama" else None,
            ollama_deployment=(ollama_deployment or "cloud") if provider == "ollama" else None,
        ),
    )

    query = ResearchQuery(
        topic=topic.strip(), depth=depth, max_sources=int(max_sources), audience=audience,
    )
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    ingested_documents = _ingest_documents(files, urls)

    initial_state = {
        "messages": [], "query": query, "plan": None, "findings": [], "critiques": [],
        "draft_report": None, "final_report": None, "human_feedback": None,
        "writer_instructions": None, "iteration_count": 0, "next_agent": None,
        "session_id": session_id, "model_provider": provider, "model_name": model,
        "schema_version": 1, "ingested_documents": ingested_documents,
    }

    graph = await _get_graph(hitl)
    config = get_thread_config(session_id)
    trace: list[tuple[str, dict]] = []

    yield (
        gr.update(visible=False),
        gr.update(value=render_trace_html(trace, live=True), visible=True),
        gr.update(visible=False), gr.update(),
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        session_id, config, trace,
    )
    async for out in _drive(graph, initial_state, config, trace, session_id):
        yield out


async def approve_and_write(feedback, session_id, config, trace, hitl):
    if config is None:
        return
    graph = await _get_graph(hitl)
    with session_scope(session_id):
        await graph.aupdate_state(config, {"writer_instructions": feedback or "Approved."})
    async for out in _drive(graph, None, config, trace, session_id):
        yield out


async def edit_and_retry(feedback, session_id, config, trace, hitl):
    if config is None:
        return
    graph = await _get_graph(hitl)
    with session_scope(session_id):
        await graph.aupdate_state(config, {
            "human_feedback": feedback or "Please re-research weak findings more thoroughly.",
            "next_agent": None,
        })
    async for out in _drive(graph, None, config, trace, session_id):
        yield out


def discard_or_restart(session_id):
    if session_id:
        clear_budget(session_id)
        unbind_session(session_id)
    return _idle_outputs()


def on_provider_change(provider):
    is_ollama = provider == "ollama"
    return (
        gr.update(value=_PROVIDER_MODEL_DEFAULTS.get(provider, "")),
        gr.update(visible=is_ollama),
        gr.update(visible=is_ollama, value=_OLLAMA_CLOUD_BASE_URL if is_ollama else gr.update()),
        gr.update(visible=provider == "anthropic"),
        gr.update(visible=provider == "openai"),
    )


def on_deployment_change(deployment):
    return gr.update(
        value=_OLLAMA_CLOUD_BASE_URL if deployment == "cloud" else settings.ollama_base_url,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_CSS = """
.rs-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.rs-header .rs-mark { display:flex; align-items:center; justify-content:center; width:40px; height:40px;
  border-radius:10px; background:#2563eb; color:#fff; font-weight:700; font-size:18px; flex-shrink:0; }
.rs-header h1 { margin:0; font-size:1.25rem; }
.rs-header p { margin:2px 0 0; opacity:.7; font-size:.9rem; }
.rs-trace { position:relative; }
.rs-line { display:none; }
.rs-rows { display:flex; flex-direction:column; gap:14px; }
.rs-row { display:flex; align-items:flex-start; gap:12px; }
.rs-connector { align-items:center; }
.rs-dot { width:8px; height:8px; border-radius:50%; background:currentColor; opacity:.35; flex-shrink:0; margin-left:10px; }
.rs-dot.rs-spin { opacity:.6; animation: rs-pulse 1.4s ease-in-out infinite; }
@keyframes rs-pulse { 0%,100% { opacity:.25; } 50% { opacity:.8; } }
.rs-connector-text { font-size:.85rem; opacity:.7; }
.rs-marker { display:flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%;
  background:rgba(127,127,127,.15); font-weight:600; font-size:.8rem; flex-shrink:0; }
.rs-card { flex:1; border:1px solid rgba(127,127,127,.25); border-radius:10px; padding:12px 14px; }
.rs-card-title { font-weight:600; margin:0 0 6px; }
.rs-muted { opacity:.7; }
.rs-small { font-size:.8rem; }
.rs-success { color:#16a34a; }
.rs-finding { border-top:1px solid rgba(127,127,127,.2); padding-top:8px; margin-top:8px; font-size:.9rem; }
.rs-finding:first-child { border-top:none; margin-top:0; padding-top:0; }
.rs-finding-q { font-weight:600; margin:0 0 2px; }
.rs-badge { display:inline-flex; align-items:center; padding:1px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }
.rs-badge-success { background:rgba(22,163,74,.12); color:#16a34a; }
.rs-badge-warning { background:rgba(217,119,6,.12); color:#d97706; }
.rs-badge-danger { background:rgba(220,38,38,.12); color:#dc2626; }
.rs-badge-muted { background:rgba(127,127,127,.15); color:inherit; }
.rs-bar { height:6px; border-radius:999px; background:rgba(127,127,127,.2); overflow:hidden; margin-top:4px; }
.rs-bar-fill { height:100%; background:#2563eb; }
.rs-panel { border:1px solid rgba(127,127,127,.25); border-radius:10px; padding:16px; }
.rs-panel-title { font-weight:600; margin:0 0 4px; }
.rs-report-title { margin-top:0; }
.rs-quality { border:1px solid rgba(127,127,127,.25); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
.rs-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:8px 0; }
.rs-stat { font-weight:700; font-size:1.1rem; margin:0; }
.rs-prose { max-width:70ch; line-height:1.6; }
.rs-prose p { margin:0 0 10px; }
.rs-ref { border-top:1px solid rgba(127,127,127,.2); padding-top:8px; margin-top:8px; font-size:.9rem; }
.rs-ref:first-child { border-top:none; margin-top:0; padding-top:0; }
a.rs-cite { display:inline-flex; align-items:center; justify-content:center; min-width:16px; height:16px;
  padding:0 4px; margin:0 2px; border-radius:4px; background:rgba(127,127,127,.15); font-size:.7rem;
  font-weight:600; text-decoration:none; vertical-align:super; }
.rs-prose a[href^="#ref-"] { display:inline-flex; align-items:center; justify-content:center; min-width:16px; height:16px;
  padding:0 4px; margin:0 2px; border-radius:4px; background:rgba(127,127,127,.15); font-size:.7rem;
  font-weight:600; text-decoration:none; vertical-align:super; }
"""

with gr.Blocks(title="Research Swarm") as demo:
    session_id_state = gr.State(None)
    config_state = gr.State(None)
    trace_state = gr.State([])

    gr.HTML(
        '<div class="rs-header"><span class="rs-mark">RS</span>'
        "<div><h1>Research Swarm</h1>"
        "<p>Autonomous multi-agent research, built on LangGraph and LlamaIndex.</p></div></div>"
    )

    error_md = gr.Markdown(visible=False)

    with gr.Column(visible=True) as form_col:
        topic = gr.Textbox(
            label="Research topic",
            placeholder="e.g. Impact of large language models on drug discovery",
            lines=1,
            max_lines=1,
        )
        audience = gr.Dropdown(
            ["general", "technical", "academic", "executive"], value="technical", label="Audience",
        )
        with gr.Accordion("Advanced options", open=False):
            provider = gr.Radio(
                ["ollama", "anthropic", "openai"], value=_SPACE_DEFAULT_PROVIDER, label="Provider",
                info="Ollama runs on this Space's own account. Anthropic/OpenAI need your own API key.",
            )
            model = gr.Textbox(value=_SPACE_DEFAULT_MODEL, label="Model")
            anthropic_key = gr.Textbox(
                label="Anthropic API key", type="password", placeholder="sk-ant-...", visible=False,
            )
            openai_key = gr.Textbox(
                label="OpenAI API key", type="password", placeholder="sk-...", visible=False,
            )
            ollama_deployment = gr.Radio(
                ["cloud", "local"], value="cloud", label="Ollama deployment",
                visible=_SPACE_DEFAULT_PROVIDER == "ollama",
                info="cloud = Ollama Cloud (needs no daemon here); local = an Ollama server you point the URL at.",
            )
            ollama_url = gr.Textbox(
                value=_OLLAMA_CLOUD_BASE_URL, label="Ollama URL",
                visible=_SPACE_DEFAULT_PROVIDER == "ollama",
            )
            depth = gr.Radio(
                ["shallow", "standard", "deep"], value="shallow", label="Depth",
                info="shallow=1 tool call, standard=3, deep=6 (per sub-question)",
            )
            max_sources = gr.Slider(3, 30, value=settings.max_sources, step=1, label="Max sources")
            hitl = gr.Checkbox(value=True, label="Pause before writing (human-in-the-loop)")
            files = gr.File(label="Documents", file_count="multiple", file_types=[".pdf"], type="filepath")
            urls_text = gr.Textbox(
                label="URLs (one per line)",
                placeholder="https://example.com/paper\nhttps://arxiv.org/abs/...",
                lines=3,
            )
        start_btn = gr.Button("Start research", variant="primary", size="lg")

    trace_html = gr.HTML(visible=False)

    with gr.Column(visible=False) as hitl_col:
        findings_html = gr.HTML()
        feedback = gr.Textbox(
            label="Feedback for the writer (optional)",
            placeholder="e.g. 'Focus more on economic impact. Exclude the speculative claims.'",
        )
        with gr.Row():
            approve_btn = gr.Button("Approve & write", variant="primary")
            edit_btn = gr.Button("Edit & retry")
            discard_btn = gr.Button("Discard", variant="stop")

    report_html = gr.HTML(visible=False)
    restart_btn = gr.Button("Start new research", visible=False)

    _OUTPUTS = [
        form_col, trace_html, hitl_col, findings_html, report_html, restart_btn, error_md,
        session_id_state, config_state, trace_state,
    ]

    provider.change(
        on_provider_change,
        inputs=[provider],
        outputs=[model, ollama_deployment, ollama_url, anthropic_key, openai_key],
    )
    ollama_deployment.change(on_deployment_change, inputs=[ollama_deployment], outputs=[ollama_url])

    start_btn.click(
        start_research,
        inputs=[
            topic, audience, provider, model, ollama_deployment, ollama_url,
            anthropic_key, openai_key, depth, max_sources, hitl, files, urls_text,
        ],
        outputs=_OUTPUTS,
    )
    approve_btn.click(
        approve_and_write,
        inputs=[feedback, session_id_state, config_state, trace_state, hitl],
        outputs=_OUTPUTS,
    )
    edit_btn.click(
        edit_and_retry,
        inputs=[feedback, session_id_state, config_state, trace_state, hitl],
        outputs=_OUTPUTS,
    )
    discard_btn.click(discard_or_restart, inputs=[session_id_state], outputs=_OUTPUTS)
    restart_btn.click(discard_or_restart, inputs=[session_id_state], outputs=_OUTPUTS)


if __name__ == "__main__":
    demo.queue().launch(css=_CSS)
