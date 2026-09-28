"""Multi-Agent Research Swarm — Streamlit entry point.

Run with:  streamlit run app.py
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

# ── Persistent background event loop ─────────────────────────────────────────
# asyncio.run() creates a *new* event loop on every call and destroys it when
# done.  asyncio.Lock objects (inside AsyncSqliteSaver / aiosqlite) are bound
# to the loop they were first awaited in.  Calling asyncio.run() a second time
# produces a different loop → "bound to a different event loop" crash.
#
# Fix: one long-lived loop in a daemon thread.  All coroutines are dispatched
# to it via run_coroutine_threadsafe(), so every asyncio object always sees
# the exact same loop for its entire lifetime.
#
# This MUST be built inside @st.cache_resource, not as a plain module-level statement:
# Streamlit reruns the entire script top-to-bottom on every single interaction (every button
# click), so a bare `asyncio.new_event_loop()` here would create a BRAND NEW loop and leak a
# brand new daemon thread on every rerun. Any long-lived asyncio object created against an
# earlier rerun's loop (the checkpointer's internal lock, for instance -- itself cached via
# @st.cache_resource so it correctly survives reruns) would then be used from a *different*
# loop on the next rerun and crash with exactly the "bound to a different event loop" error
# this comment used to warn about. @st.cache_resource makes this function's body run exactly
# once per process, giving every rerun the identical loop + thread.
@st.cache_resource(show_spinner=False)
def _get_bg_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def _run_bg_loop() -> None:
        # asyncio.set_event_loop() must be called from *within* the thread that will run the
        # loop -- it's thread-local. Without it, this thread has no "current" loop registered,
        # so any internal library code that calls the older asyncio.get_event_loop() (rather
        # than get_running_loop()) from a synchronous context on this thread -- e.g. during
        # garbage collection of an unclosed async generator -- can silently get a *different*
        # implicit loop instead of this one.
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run_bg_loop, daemon=True, name="swarm-async").start()
    return loop


_BG_LOOP: asyncio.AbstractEventLoop = _get_bg_loop()


def _run(coro):
    """Submit *coro* to the shared background loop and block until it finishes."""
    return asyncio.run_coroutine_threadsafe(coro, _BG_LOOP).result()


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Swarm",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from research_swarm.config import settings
from research_swarm.graph.builder import build_graph, get_thread_config, make_async_checkpointer
from research_swarm.graph.rework import request_rework
from research_swarm.runtime.budget import clear_budget
from research_swarm.runtime.langsmith_trace import (
    make_tracer,
    project_name,
    run_url,
    tracing_enabled,
)
from research_swarm.schemas import ResearchQuery
from research_swarm.ui.graph_view import render_graph_diagram
from research_swarm.ui.report_view import render_report
from research_swarm.ui.sessions_view import render_sessions_tab
from research_swarm.ui.sidebar import render_sidebar
from research_swarm.ui.style import badge, inject_css
from research_swarm.ui.trace import render_node_update, render_trace_header


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to checkpoint store…")
def _get_checkpointer():
    """Create (and cache) the AsyncSqliteSaver on the shared background loop.

    Using _run() guarantees the checkpointer's internal asyncio.Lock objects
    are bound to _BG_LOOP, the same loop used for every subsequent graph call.
    """
    return _run(make_async_checkpointer())


def _agent_code_hash() -> str:
    """Hash the mtime of every agent/tool module so the cache busts on code changes."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).parent / "research_swarm"
    h = hashlib.md5()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()[:8]


@st.cache_resource(show_spinner="Loading graph…")
def _get_graph(hitl: bool, _code_hash: str = ""):  # noqa: ARG001
    """Build (and cache) the compiled LangGraph.  One instance per HITL setting.

    _code_hash is derived from agent module mtimes — it busts the cache
    automatically whenever agent or tool code changes, so a server restart
    is no longer needed after edits.
    """
    return build_graph(checkpointer=_get_checkpointer(), interrupt_before_writer=hitl)


@st.cache_resource(show_spinner=False)
def _prune_sessions_once() -> int:
    """Prune expired/excess sessions, once per server process (space_mode only).

    @st.cache_resource makes this run exactly once per process lifetime no
    matter how many times Streamlit reruns the script (every user
    interaction reruns app.py top to bottom) or how many browser sessions
    hit this server — a plain module-level call would otherwise re-scan the
    checkpoint DB on every rerun for no benefit. No-ops (returns 0) unless
    settings.space_mode is enabled, so local/dev runs are unaffected.
    """
    if not settings.space_mode:
        return 0
    from research_swarm.persistence.sessions import prune_expired_sessions
    return prune_expired_sessions(
        settings.space_retention_seconds, settings.space_max_sessions,
    )


# In-process cap on concurrent graph runs. Only meaningful in space_mode: a
# public multi-tenant Space can have several browser sessions hitting one
# server process at once, and each run holds its search results and LLM
# responses in memory -- unbounded concurrency there risks exhausting a
# small Space's RAM and the provider's concurrent-request slots. Harmless
# outside space_mode since a single local user never contends on it. Not
# cached/session-scoped: it must be one
# shared semaphore across the whole process, which is exactly what a plain
# module-level object gives for free (module-level code runs once per
# process on first import, unlike the rest of this script).
_RUN_SEMAPHORE = threading.Semaphore(settings.space_max_concurrent_runs)


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "session_id":    None,       # UUID for the current research run
        "running":       False,      # graph currently streaming
        "interrupted":   False,      # paused at HITL checkpoint
        "agent_trace":   [],         # [(node_name, update_dict), ...]
        "final_report":  None,       # FinalReport | None
        "error_msg":     None,       # last error string
        "langsmith_url": None,       # link to this run's trace, once traced and finished
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Migrate stale provider selection: if the session still has the old
    # hard-coded "anthropic" default but the configured default has changed,
    # push the new default so existing sessions pick it up automatically.
    if (
        st.session_state.get("ui_provider") == "anthropic"
        and settings.default_model_provider != "anthropic"
    ):
        st.session_state["ui_provider"] = settings.default_model_provider
        # Also reset the deployment so the Ollama cloud path activates.
        st.session_state["ui_ollama_deployment"] = settings.ollama_deployment
        st.session_state["ui_model_ollama"] = settings.ollama_cloud_model


def _reset_run() -> None:
    # Release the previous run's budget guard — session IDs are per-run UUIDs,
    # so stale guards would otherwise accumulate for the life of the server.
    old_session = st.session_state.get("session_id")
    if old_session:
        clear_budget(old_session)
    st.session_state.update(
        session_id=str(uuid.uuid4()),
        running=False,
        interrupted=False,
        agent_trace=[],
        final_report=None,
        error_msg=None,
        langsmith_url=None,
    )


# ── Ingestion helper ──────────────────────────────────────────────────────────

def _ingest_documents(uploaded_pdfs: list, extra_urls: list[str]) -> list[dict]:
    """Extract full text from uploaded PDFs and URLs for the document pass.

    Returns a list of {url, title, text, source_type} dicts, stored on
    AgentState.ingested_documents and consumed by document_pass_node /
    document_worker_node -- one full-document extraction call per document,
    no chunking, no embedding, no vector store.

    fetch_url's own max_chars ceiling (20,000 -- its schema maximum) still
    bounds how much of a web page can be retrieved here; PDFs aren't
    similarly capped since load_pdf returns full per-page text for every
    page it reads (up to its own max_pages=50 default).
    """
    if not uploaded_pdfs and not extra_urls:
        return []

    from research_swarm.tools.pdf_loader import load_pdf
    from research_swarm.tools.url_fetcher import fetch_url

    documents: list[dict] = []

    if uploaded_pdfs:
        with st.spinner(f"Reading {len(uploaded_pdfs)} PDF(s)…"):
            for uf in uploaded_pdfs:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    result = load_pdf.invoke({"file_path": tmp_path})
                    text = "\n\n".join(
                        c["text"] for c in result.get("chunks", []) if c.get("text")
                    )
                    if text:
                        documents.append({
                            "url":         result.get("url", tmp_path),
                            "title":       result.get("title", ""),
                            "text":        text,
                            "source_type": "pdf",
                        })
                finally:
                    os.unlink(tmp_path)

    if extra_urls:
        with st.spinner(f"Fetching {len(extra_urls)} URL(s)…"):
            for url in extra_urls:
                result = fetch_url.invoke({"url": url, "max_chars": 20000})
                snippet = result.get("snippet", "")
                if snippet.startswith("["):
                    continue  # fetch error placeholder -- skip
                documents.append({
                    "url":         result.get("url", url),
                    "title":       result.get("title", ""),
                    "text":        snippet,
                    "source_type": "web",
                })

    return documents


# ── Graph streaming helpers ───────────────────────────────────────────────────

def _apply_ui_settings(ui: dict) -> None:
    """Apply settings that cannot yet be threaded through AgentState.

    model_provider and model_name are already in AgentState so
    they are NOT mutated here.  Only Ollama infrastructure config (URL,
    deployment mode) is written to settings so that the LLM factory
    always sees the user's current selection consistently.
    """
    settings.llm_judge_enabled = bool(ui.get("llm_judge"))
    if ui["provider"] == "ollama":
        settings.ollama_deployment = ui.get("ollama_deployment") or "local"
        # In both local and cloud mode the daemon URL is the same (localhost).
        # Cloud mode uses the local daemon which proxies to Ollama's cloud via
        # `ollama login` credentials — no separate URL needed.
        if ui.get("ollama_url"):
            settings.ollama_base_url = ui["ollama_url"]


# ── Graph runs: background jobs that survive Streamlit reruns ─────────────────
#
# Streamlit reruns this whole script on every widget interaction and stops the run in progress
# to do it. The graph used to be driven from inside the script run (a blocking loop that drew
# each update as it arrived), so touching any sidebar control mid-run killed that loop: the
# graph kept going on _BG_LOOP with nobody reading it, and the rerun landed in the "running"
# branch, which drew only "Research is in progress…" -- the diagram and trace vanished and the
# page stayed stuck even after the run ended, because nothing ever cleared `running`.
#
# Now a run is a _RunJob that lives outside any script run: its coroutine is dispatched to
# _BG_LOOP and records every update on the job; the job is kept in a process-wide registry keyed
# by session id. Any script run -- the first one or one triggered by a settings change -- finds
# the job, redraws everything recorded so far and keeps drawing new updates until the job
# finishes. A rerun only interrupts the *drawing*, never the research.

logger = logging.getLogger(__name__)


@dataclass
class _RunJob:
    session_id: str
    graph: Any
    config: dict
    updates: list = field(default_factory=list)     # (node_name, update) in arrival order
    done: bool = False
    interrupted: bool = False                        # paused at the HITL checkpoint
    error: str | None = None
    final_report: Any = None
    langsmith_url: str | None = None
    started: float = field(default_factory=time.monotonic)
    future: Any = None


@st.cache_resource(show_spinner=False)
def _jobs() -> dict[str, _RunJob]:
    """Process-wide registry of runs by session id (survives reruns, like _BG_LOOP)."""
    return {}


def _start_run(graph, input_state, config: dict) -> _RunJob:
    """Start one leg of the graph (a new run, or a resume after HITL) in the background."""
    session_id = str(config.get("configurable", {}).get("thread_id", ""))
    job = _RunJob(session_id=session_id, graph=graph, config=config)
    tracer = make_tracer(session_id)
    run_config = config if tracer is None else {**config, "callbacks": [tracer]}

    async def _produce() -> None:
        # The astream() generator's whole lifetime stays on _BG_LOOP: LangGraph's cleanup on a
        # HITL interrupt calls get_event_loop() and crashes on any other thread.
        acquired = False
        try:
            if settings.space_mode:            # bounds concurrent runs (see _RUN_SEMAPHORE)
                await asyncio.to_thread(_RUN_SEMAPHORE.acquire)
                acquired = True
            async for chunk in graph.astream(input_state, run_config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    if node_name == "__interrupt__":
                        continue               # the HITL pause signal (a tuple), not an update
                    job.updates.append((node_name, node_update))
                    if (node_name == "writer" and isinstance(node_update, dict)
                            and node_update.get("final_report")):
                        job.final_report = node_update["final_report"]
            snapshot = await graph.aget_state(config)
            job.interrupted = bool(snapshot.next)   # non-empty `next` = paused at HITL
            if tracer is not None:
                job.langsmith_url = run_url(tracer)
        except asyncio.CancelledError:
            job.error = "Run cancelled."
            raise
        except Exception as exc:  # noqa: BLE001 -- reported in the UI, not raised on a thread
            logger.exception("Graph run %s failed", session_id)
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            if acquired:
                _RUN_SEMAPHORE.release()
            job.done = True

    job.future = asyncio.run_coroutine_threadsafe(_produce(), _BG_LOOP)
    _jobs()[session_id] = job
    return job


def _finish_run(job: _RunJob) -> None:
    """Move a finished job's results into this browser session and drop the job."""
    st.session_state.agent_trace = list(st.session_state.agent_trace) + list(job.updates)
    if job.final_report is not None:
        st.session_state.final_report = job.final_report
    if job.langsmith_url:
        st.session_state.langsmith_url = job.langsmith_url
    st.session_state.running = False
    st.session_state.interrupted = job.interrupted and not job.error
    if job.error:
        st.session_state.error_msg = f"Graph error: {job.error}"
    _jobs().pop(job.session_id, None)


def _render_running(job: _RunJob) -> None:
    """Draw a run in progress -- everything recorded so far, then each new update as it lands --
    until it finishes, then rerun into the HITL / done view. Safe to re-enter on every rerun."""
    graph = job.graph
    st.markdown("## Research")
    status_ph = st.empty()
    if st.button("Cancel run", key="cancel_run"):
        job.future.cancel()
    render_trace_header()
    diagram_ph = st.empty()
    cards = st.container()

    # Earlier legs of this session (before a HITL resume) are already in agent_trace.
    prior = list(st.session_state.agent_trace)
    done_nodes = {n for n, _ in prior}
    with cards:
        for node_name, update in prior:
            render_node_update(node_name, update)

    shown, last_node, first = 0, None, True
    while True:
        finished = job.done                    # read BEFORE slicing: nothing is missed
        new = job.updates[shown:]
        for node_name, update in new:
            with cards:
                render_node_update(node_name, update)
            done_nodes.add(node_name)
            last_node = node_name
        shown += len(new)
        if new or first:                       # the diagram is an iframe: redraw only on change
            with diagram_ph.container():
                render_graph_diagram(graph, last_node, done_nodes)
            first = False
        if finished:
            break
        # Also the heartbeat that lets Streamlit act on a rerun request promptly: a script run
        # is only interrupted when it next sends something to the browser.
        status_ph.caption(
            f"Running for {time.monotonic() - job.started:.0f}s · last finished stage: "
            f"{last_node or 'starting'} · changing settings won't interrupt this run "
            f"(they apply to the next one)."
        )
        time.sleep(0.5)

    _finish_run(job)
    st.rerun()


def _render_tracing_status() -> None:
    """A small status line for LangSmith tracing: on/off, project, and a link to this run's
    trace once it has one (tracing needs LANGCHAIN_TRACING_V2=true and a LangSmith API key --
    see .env.example)."""
    if not tracing_enabled():
        st.caption("LangSmith tracing: off")
        return
    url = st.session_state.get("langsmith_url")
    if url:
        st.caption(f"LangSmith project **{project_name()}** — [view this run's trace]({url})")
    else:
        st.caption(f"LangSmith tracing: on (project **{project_name()}**)")


# ── HITL panel ────────────────────────────────────────────────────────────────

def _render_hitl_panel(graph, config: dict) -> None:
    """Render the human-review panel and handle Approve / Edit / Reject."""
    st.divider()
    st.markdown("## Human Review Required")
    st.info(
        "The graph has paused before writing. "
        "Review the findings below and choose how to proceed."
    )

    # Show current state (async checkpointer → must call aget_state)
    snapshot  = _run(graph.aget_state(config))
    state_val = snapshot.values if hasattr(snapshot, "values") else {}
    findings  = state_val.get("findings", [])
    critiques = state_val.get("critiques", [])

    _verdict_badge = {
        "supported": ("SUPPORTED", "success"),
        "weak":      ("WEAK", "warning"),
        "refuted":   ("REFUTED", "danger"),
    }
    critique_by_fid = {}
    for c in critiques:
        fid     = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
        verdict = c.verdict    if hasattr(c, "verdict")    else c.get("verdict", "")
        v_str   = verdict.value if hasattr(verdict, "value") else str(verdict)
        critique_by_fid[fid] = v_str

    with st.expander(f"Findings ({len(findings)})", expanded=True):
        for f in findings:
            fid   = f.id    if hasattr(f, "id")    else f.get("id", "")
            claim = f.claim if hasattr(f, "claim") else f.get("claim", "")
            conf  = f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.5)
            sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
            verdict_str = critique_by_fid.get(fid, "pending")
            label, kind = _verdict_badge.get(verdict_str, ("PENDING", "neutral"))
            with st.container(border=True):
                st.markdown(
                    f"{badge(label, kind)} &nbsp; **{sub_q}**", unsafe_allow_html=True,
                )
                st.markdown(claim)
                st.caption(f"confidence: {conf:.2f}")

    # Feedback text box
    feedback = st.text_area(
        "Feedback for the writer (optional)",
        placeholder="e.g. 'Focus more on economic impact. Exclude the speculative claims.'",
        key="hitl_feedback",
    )

    col1, col2, col3 = st.columns(3)

    if col1.button("Approve & Write", type="primary", use_container_width=True):
        _resume_after_hitl(graph, config, feedback or "Approved.")

    if col2.button("Edit & Re-research", use_container_width=True):
        # Empty feedback is fine: the weakly answered sub-questions are re-researched as
        # planned; text in the box also steers the new searches.
        _request_more_research(graph, config, feedback)

    if col3.button("Discard Session", use_container_width=True, type="secondary"):
        st.session_state.running     = False
        st.session_state.interrupted = False
        st.warning("Session discarded. Start a new query to try again.")
        st.rerun()


def _resume_after_hitl(graph, config: dict, feedback: str) -> None:
    """Update state with writer instructions and resume the graph (in the background) to the
    writer; the running view picks it up on the rerun."""
    _run(graph.aupdate_state(config, {"writer_instructions": feedback}))
    st.session_state.interrupted = False
    _start_run(graph, None, config)
    st.session_state.running = True
    st.rerun()


def _request_more_research(graph, config: dict, instructions: str) -> None:
    """Send the paused run back to research for one more round on the weakly answered
    sub-questions (graph/rework.py), steered by *instructions*, then back to this review.

    It used to only set human_feedback and a `running` flag without starting anything: the page
    sat on "in progress" forever, and resuming would have run the writer anyway, since the
    graph pauses *before the writer*."""
    _run(request_rework(graph, config, instructions))
    st.session_state.interrupted = False
    _start_run(graph, None, config)
    st.session_state.running = True
    st.rerun()


# ── Research tab ──────────────────────────────────────────────────────────────

def render_research_tab(ui: dict) -> None:
    graph  = _get_graph(ui["hitl_enabled"], _code_hash=_agent_code_hash())
    config = get_thread_config(st.session_state.session_id or "init")

    # ── Interrupted state: show HITL panel ──
    if st.session_state.interrupted:
        st.markdown("## Research")
        # Redraw the existing trace
        render_trace_header()
        render_graph_diagram(graph, None, {n for n, _ in st.session_state.agent_trace})
        for node_name, update in st.session_state.agent_trace:
            render_node_update(node_name, update)
        _render_tracing_status()
        _render_hitl_panel(graph, config)
        return

    # ── Running state: reattach to the background run and keep drawing it ──
    if st.session_state.running:
        job = _jobs().get(st.session_state.session_id)
        if job is None:
            # Nothing is driving this session (the server restarted mid-run): don't sit on
            # "in progress" forever. The checkpoint keeps what was done.
            st.session_state.running = False
            st.warning(
                "No run is active for this session (for example, the app restarted while it "
                "was running). Its saved progress can be resumed from the **Sessions** tab."
            )
            return
        _render_running(job)
        return

    # ── Done: show success + report link ──
    if st.session_state.final_report:
        st.markdown("## Research")
        st.success(f"Report ready: **{st.session_state.final_report.title}**")
        st.caption("Switch to the **Report** tab to read and download it.")
        _render_tracing_status()
        # Replay trace (collapsed)
        with st.expander("View agent trace", expanded=False):
            render_graph_diagram(graph, None, {n for n, _ in st.session_state.agent_trace})
            for node_name, update in st.session_state.agent_trace:
                render_node_update(node_name, update)
        if st.button("Start new research"):
            _reset_run()
            st.rerun()
        return

    # ── Idle: show the hero landing state + query form ──
    _render_query_form(ui, graph)


def _render_query_form(ui: dict, graph) -> None:
    """Render the idle-state hero heading and query input, then kick off the
    graph on submit.

    Centred in the middle of a 3-column split rather than full-width -- a
    plain full-width form reads as "a settings page waiting to be filled in";
    narrowing it and pairing it with a large heading reads as a deliberate
    landing state, closer to a chat app's empty-conversation screen than a
    web form.
    """
    _left, mid, _right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div class="rs-hero">'
            "<h1>What should we research?</h1>"
            "<p>Give it a question and the swarm plans sub-questions, searches the literature, "
            "verifies every fact against its source, and writes a cited report.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form("research_form", clear_on_submit=False):
            topic = st.text_input(
                "Research topic",
                placeholder="e.g.  Impact of large language models on drug discovery",
                key="ui_topic",
                label_visibility="collapsed",
            )
            col_audience, col_submit = st.columns([3, 1], vertical_alignment="bottom")
            audience = col_audience.selectbox(
                "Audience",
                ["general", "technical", "academic", "executive"],
                index=1,
                key="ui_audience",
            )
            submitted = col_submit.form_submit_button(
                "Start Research",
                type="primary",
                use_container_width=True,
            )

    if not submitted or not topic.strip():
        return

    # Initialise new run
    _reset_run()
    session_id = st.session_state.session_id
    _apply_ui_settings(ui)

    # Build initial state
    query = ResearchQuery(
        topic=topic.strip(),
        depth=ui["depth"],
        audience=audience,
    )
    # Extract full text from user-supplied documents up front -- consumed by
    # document_pass_node (one full-document extraction call per document, no
    # chunking or embedding).
    ingested_documents = _ingest_documents(ui["uploaded_pdfs"], ui["extra_urls"])
    if ingested_documents:
        st.toast(f"Loaded {len(ingested_documents)} document(s) for research.")

    initial_state = {
        "messages":            [],
        "query":               query,
        "plan":                None,
        "findings":            [],
        "critiques":           [],
        "draft_report":        None,
        "final_report":        None,
        "human_feedback":      None,
        "writer_instructions": None,
        "iteration_count":     0,
        "next_agent":          None,
        "session_id":          session_id,
        "model_provider":      ui["provider"],
        "model_name":          ui["model"],
        "schema_version":      1,
        "ingested_documents":  ingested_documents,
    }

    # Run in the background; the running view (render_research_tab) draws it live and
    # reattaches after any rerun, e.g. a sidebar change mid-run.
    _start_run(graph, initial_state, get_thread_config(session_id))
    st.session_state.running = True
    st.rerun()


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    _prune_sessions_once()
    _init_state()

    st.markdown(
        '<div class="rs-header">'
        "<span class=\"rs-header-name\">Multi-Agent Research Swarm</span>"
        "<span class=\"rs-header-tag\">LangGraph · Streamlit</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Sidebar
    ui = render_sidebar()

    # Tabs
    tab_research, tab_report, tab_sessions = st.tabs(
        ["Research", "Report", "Sessions"]
    )

    with tab_research:
        render_research_tab(ui)

    with tab_report:
        if st.session_state.final_report:
            render_report(st.session_state.final_report)
        else:
            st.info("Run a research query on the **Research** tab to generate a report.")

    with tab_sessions:
        def on_resume(thread_id: str) -> None:
            from research_swarm.runtime.migrations import migrate_state
            # Load saved state into session
            st.session_state.session_id  = thread_id
            st.session_state.running     = False
            st.session_state.interrupted = False
            st.session_state.agent_trace = []
            st.session_state.final_report = None
            # Use the cached graph (with AsyncSqliteSaver) to load state
            graph = _get_graph(ui["hitl_enabled"])
            config = get_thread_config(thread_id)
            snap = _run(graph.aget_state(config))
            if snap and snap.values:
                saved = migrate_state(dict(snap.values))  # upgrade v0 checkpoints
                if saved.get("final_report"):
                    st.session_state.final_report = saved["final_report"]
            # Check if it's paused at HITL
            if snap and snap.next:
                st.session_state.interrupted = True
            st.toast(f"Resumed session `{thread_id[:12]}…`")
            st.rerun()

        render_sessions_tab(on_resume)

    # Error banner
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        if st.button("Clear error"):
            st.session_state.error_msg = None
            st.rerun()


if __name__ == "__main__":
    main()
