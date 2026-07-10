"""Multi-Agent Research Swarm — Streamlit entry point.

Run with:  streamlit run app.py
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import uuid

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
_BG_LOOP: asyncio.AbstractEventLoop = asyncio.new_event_loop()
threading.Thread(target=_BG_LOOP.run_forever, daemon=True, name="swarm-async").start()


def _run(coro):
    """Submit *coro* to the shared background loop and block until it finishes."""
    return asyncio.run_coroutine_threadsafe(coro, _BG_LOOP).result()


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Swarm",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from research_swarm.config import settings
from research_swarm.graph.builder import build_graph, get_thread_config, make_async_checkpointer
from research_swarm.rag.indexes import get_embed_model
from research_swarm.runtime.budget import clear_budget
from research_swarm.rag.ingestion import IngestionPipeline
from research_swarm.schemas import ResearchQuery
from research_swarm.ui.report_view import render_report
from research_swarm.ui.sessions_view import render_sessions_tab
from research_swarm.ui.sidebar import render_sidebar
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


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "session_id":    None,       # UUID for the current research run
        "running":       False,      # graph currently streaming
        "interrupted":   False,      # paused at HITL checkpoint
        "agent_trace":   [],         # [(node_name, update_dict), ...]
        "final_report":  None,       # FinalReport | None
        "error_msg":     None,       # last error string
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
    )


# ── Ingestion helper ──────────────────────────────────────────────────────────

def _ingest_documents(session_id: str, uploaded_pdfs: list, extra_urls: list[str]) -> int:
    """Ingest PDFs and URLs into the session RAG index. Returns total chunks."""
    if not uploaded_pdfs and not extra_urls:
        return 0

    pipeline = IngestionPipeline(session_id)
    embed    = get_embed_model()
    total    = 0

    if uploaded_pdfs:
        with st.spinner(f"Ingesting {len(uploaded_pdfs)} PDF(s) into RAG index…"):
            for uf in uploaded_pdfs:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    total += pipeline.ingest_pdf(tmp_path, embed)
                finally:
                    os.unlink(tmp_path)

    if extra_urls:
        with st.spinner(f"Fetching {len(extra_urls)} URL(s)…"):
            for url in extra_urls:
                total += pipeline.ingest_url(url, embed)

    return total


# ── Graph streaming helpers ───────────────────────────────────────────────────

def _apply_ui_settings(ui: dict) -> None:
    """Apply settings that cannot yet be threaded through AgentState.

    model_provider, model_name, and max_sources are already in AgentState so
    they are NOT mutated here.  Only Ollama infrastructure config (URL,
    deployment mode) is written to settings so that the LLM factory and RAG
    query engines always see the user's current selection consistently.
    """
    if ui["provider"] == "ollama":
        settings.ollama_deployment = ui.get("ollama_deployment") or "local"
        # In both local and cloud mode the daemon URL is the same (localhost).
        # Cloud mode uses the local daemon which proxies to Ollama's cloud via
        # `ollama login` credentials — no separate URL needed.
        if ui.get("ollama_url"):
            settings.ollama_base_url = ui["ollama_url"]


def _stream_graph(
    graph,
    input_state,
    config: dict,
    trace_container,
    show_header: bool = True,
) -> bool:
    """Drive graph.astream() on the shared background loop and render updates.

    Collecting chunks on the background loop then rendering them here (in
    Streamlit's thread) keeps asyncio objects on a single loop while keeping
    all st.* calls on the main thread.

    Returns True if the graph was interrupted (HITL pause), False if it ran to END.
    """
    updates: list[tuple[str, dict]] = []
    final_report_holder: list = []

    async def _collect() -> bool:
        async for chunk in graph.astream(input_state, config, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                updates.append((node_name, node_update))
                if node_name == "writer" and node_update.get("final_report"):
                    final_report_holder.append(node_update["final_report"])
        snapshot = await graph.aget_state(config)
        return bool(snapshot.next)  # non-empty `next` means paused at HITL

    interrupted = _run(_collect())

    # Render collected updates in Streamlit's thread (safe for st.* calls)
    with trace_container:
        if show_header:
            render_trace_header()
        for node_name, node_update in updates:
            st.session_state.agent_trace.append((node_name, node_update))
            render_node_update(node_name, node_update)

    if final_report_holder:
        st.session_state.final_report = final_report_holder[-1]

    return interrupted


# ── HITL panel ────────────────────────────────────────────────────────────────

def _render_hitl_panel(graph, config: dict, trace_container) -> None:
    """Render the human-review panel and handle Approve / Edit / Reject."""
    st.divider()
    st.markdown("## 👤 Human Review Required")
    st.info(
        "The graph has paused before writing. "
        "Review the findings below and choose how to proceed."
    )

    # Show current state (async checkpointer → must call aget_state)
    snapshot  = _run(graph.aget_state(config))
    state_val = snapshot.values if hasattr(snapshot, "values") else {}
    findings  = state_val.get("findings", [])
    critiques = state_val.get("critiques", [])

    _verdict_map = {"supported": "✅", "weak": "⚠️", "refuted": "❌"}
    critique_by_fid = {}
    for c in critiques:
        fid     = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
        verdict = c.verdict    if hasattr(c, "verdict")    else c.get("verdict", "")
        v_str   = verdict.value if hasattr(verdict, "value") else str(verdict)
        critique_by_fid[fid] = v_str

    with st.expander(f"📑 Findings ({len(findings)})", expanded=True):
        for f in findings:
            fid   = f.id    if hasattr(f, "id")    else f.get("id", "")
            claim = f.claim if hasattr(f, "claim") else f.get("claim", "")
            conf  = f.confidence if hasattr(f, "confidence") else f.get("confidence", 0.5)
            sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
            verdict_str = critique_by_fid.get(fid, "pending")
            emoji = _verdict_map.get(verdict_str, "❓")
            st.markdown(
                f"{emoji} **{sub_q}**  \n"
                f"{claim}  \n"
                f"*confidence: {conf:.2f}*"
            )
            st.markdown("---")

    # Feedback text box
    feedback = st.text_area(
        "📝 Feedback for the writer (optional)",
        placeholder="e.g. 'Focus more on economic impact. Exclude the speculative claims.'",
        key="hitl_feedback",
    )

    col1, col2, col3 = st.columns(3)

    if col1.button("✅ Approve & Write", type="primary", use_container_width=True):
        _resume_after_hitl(graph, config, trace_container, feedback or "Approved.")

    if col2.button("✏️ Edit & Re-research", use_container_width=True):
        instructions = feedback or "Please re-research weak findings more thoroughly."
        _request_more_research(graph, config, instructions)

    if col3.button("❌ Discard Session", use_container_width=True, type="secondary"):
        st.session_state.running     = False
        st.session_state.interrupted = False
        st.warning("Session discarded. Start a new query to try again.")
        st.rerun()


def _resume_after_hitl(graph, config: dict, trace_container, feedback: str) -> None:
    """Update state with writer instructions and resume the graph to the writer."""
    _run(graph.aupdate_state(config, {"writer_instructions": feedback}))
    st.session_state.interrupted = False
    # show_header=False: the trace header was already rendered when replaying
    interrupted = _stream_graph(graph, None, config, trace_container, show_header=False)
    st.session_state.running     = False
    st.session_state.interrupted = interrupted
    st.rerun()


def _request_more_research(graph, config: dict, instructions: str) -> None:
    """Inject feedback that asks for more research, then re-run from supervisor.

    Sets human_feedback (consumed by researcher) and clears next_agent so the
    supervisor's human_feedback guard fires cleanly on the next supervisor call.
    Does NOT force next_agent="researcher" — the supervisor's deterministic
    routing handles that based on human_feedback being present (fix 2-A / 2-C).
    """
    _run(graph.aupdate_state(config, {
        "human_feedback": instructions,
        "next_agent":     None,
    }))
    st.session_state.interrupted = False
    st.session_state.running     = True
    st.rerun()


# ── Research tab ──────────────────────────────────────────────────────────────

def render_research_tab(ui: dict) -> None:
    st.markdown("## 🔬 Research")

    graph  = _get_graph(ui["hitl_enabled"], _code_hash=_agent_code_hash())
    config = get_thread_config(st.session_state.session_id or "init")

    # ── Interrupted state: show HITL panel ──
    if st.session_state.interrupted:
        # Redraw the existing trace
        trace_ph = st.container()
        with trace_ph:
            render_trace_header()
            for node_name, update in st.session_state.agent_trace:
                render_node_update(node_name, update)
        _render_hitl_panel(graph, config, trace_ph)
        return

    # ── Running state: show spinner ──
    if st.session_state.running:
        st.info("⏳ Research is in progress…")
        return

    # ── Done: show success + report link ──
    if st.session_state.final_report:
        st.success(f"✅ Report ready: **{st.session_state.final_report.title}**")
        st.caption("Switch to the **Report** tab to read and download it.")
        # Replay trace (collapsed)
        with st.expander("📜 View agent trace", expanded=False):
            for node_name, update in st.session_state.agent_trace:
                render_node_update(node_name, update)
        if st.button("🔄 Start new research"):
            _reset_run()
            st.rerun()
        return

    # ── Idle: show the query form ──
    _render_query_form(ui, graph)


def _render_query_form(ui: dict, graph) -> None:
    """Render the query input form and kick off the graph on submit."""
    with st.form("research_form", clear_on_submit=False):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g.  Impact of large language models on drug discovery",
            key="ui_topic",
        )
        audience = st.selectbox(
            "Audience",
            ["general", "technical", "academic", "executive"],
            index=1,
            key="ui_audience",
        )
        submitted = st.form_submit_button(
            "🚀 Start Research",
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
        max_sources=ui["max_sources"],
        audience=audience,
    )
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
    }

    # Ingest user-supplied documents
    chunks_added = _ingest_documents(session_id, ui["uploaded_pdfs"], ui["extra_urls"])
    if chunks_added:
        st.toast(f"✅ Ingested {chunks_added} chunk(s) into RAG index.")

    config = get_thread_config(session_id)
    st.session_state.running = True

    # Stream the graph — render live trace
    trace_container = st.container()
    try:
        interrupted = _stream_graph(graph, initial_state, config, trace_container)
    except Exception as exc:
        st.session_state.running = False
        st.session_state.error_msg = str(exc)
        st.error(f"Graph error: {exc}")
        return

    st.session_state.running     = False
    st.session_state.interrupted = interrupted

    if interrupted:
        st.rerun()   # re-render to show HITL panel
    else:
        st.rerun()   # re-render to show success


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    st.title("🔬 Multi-Agent Research Swarm")
    st.caption("Powered by LangGraph · LlamaIndex · Streamlit")
    st.divider()

    # Sidebar
    ui = render_sidebar()

    # Tabs
    tab_research, tab_report, tab_sessions = st.tabs(
        ["🔍 Research", "📄 Report", "🗂️ Sessions"]
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
        st.error(f"⚠️ {st.session_state.error_msg}")
        if st.button("Clear error"):
            st.session_state.error_msg = None
            st.rerun()


if __name__ == "__main__":
    main()
