"""Agent trace renderer — live per-node status blocks during graph execution."""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

# ── Icon / colour per node ────────────────────────────────────────────────────
_NODE_META: dict[str, dict] = {
    "supervisor":   {"icon": "🧭", "label": "Supervisor",    "colour": "#5c6bc0"},
    "researcher":   {"icon": "🔍", "label": "Researcher",    "colour": "#26a69a"},
    "critic":       {"icon": "🧐", "label": "Critic",        "colour": "#ef6c00"},
    "fact_checker": {"icon": "✅", "label": "Fact-Checker",  "colour": "#66bb6a"},
    "writer":       {"icon": "✍️", "label": "Writer",        "colour": "#ab47bc"},
}
_DEFAULT_META = {"icon": "⚙️", "label": "Agent", "colour": "#78909c"}


def _node_meta(name: str) -> dict:
    return _NODE_META.get(name, _DEFAULT_META)


def render_node_update(node_name: str, update: dict[str, Any]) -> None:
    """Render a single node update inside an st.status block."""
    meta = _node_meta(node_name)

    with st.container():
        # Compact one-line header
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            st.markdown(f"### {meta['icon']}")
        with col2:
            st.markdown(f"**{meta['label']}**")

        # Show meaningful fields from the update
        _render_update_body(node_name, update)
        st.divider()


def _render_update_body(node_name: str, update: dict[str, Any]) -> None:
    """Render key fields from a node state update in a readable way."""
    if node_name == "supervisor":
        _render_supervisor(update)
    elif node_name == "researcher":
        _render_researcher(update)
    elif node_name == "critic":
        _render_critic(update)
    elif node_name == "fact_checker":
        _render_fact_checker(update)
    elif node_name == "writer":
        _render_writer(update)
    else:
        _render_raw(update)


def _render_supervisor(u: dict) -> None:
    next_a = u.get("next_agent", "—")
    iter_n = u.get("iteration_count", "—")
    st.markdown(f"→ routing to **{next_a}** &nbsp;·&nbsp; iteration {iter_n}")
    if plan := u.get("plan"):
        sub_qs = plan.sub_questions if hasattr(plan, "sub_questions") else plan.get("sub_questions", [])
        if sub_qs:
            with st.expander(f"📋 Research plan ({len(sub_qs)} sub-questions)", expanded=False):
                for i, q in enumerate(sub_qs, 1):
                    st.markdown(f"{i}. {q}")
    for msg in u.get("messages", []):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content.startswith("[Supervisor]"):
            st.caption(content)


def _render_researcher(u: dict) -> None:
    findings = u.get("findings", [])
    st.markdown(f"📝 **{len(findings)}** finding(s) produced")
    if findings:
        with st.expander("View findings", expanded=False):
            for f in findings:
                claim  = f.claim       if hasattr(f, "claim")       else f.get("claim", "")
                conf   = f.confidence  if hasattr(f, "confidence")  else f.get("confidence", 0)
                sub_q  = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                n_src  = len(f.evidence if hasattr(f, "evidence") else f.get("evidence", []))
                st.markdown(
                    f"**{sub_q}**  \n"
                    f"{claim}  \n"
                    f"*confidence: {conf:.2f} · {n_src} source(s)*"
                )
                st.markdown("---")


def _render_critic(u: dict) -> None:
    critiques = u.get("critiques", [])
    st.markdown(f"🧐 **{len(critiques)}** critique(s) produced")
    if critiques:
        _VERDICT_EMOJI = {"supported": "✅", "weak": "⚠️", "refuted": "❌"}
        with st.expander("View critiques", expanded=False):
            for c in critiques:
                verdict  = c.verdict   if hasattr(c, "verdict")   else c.get("verdict", "?")
                v_str    = verdict.value if hasattr(verdict, "value") else str(verdict)
                fid      = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
                reason   = c.reasoning  if hasattr(c, "reasoning")  else c.get("reasoning", "")
                emoji    = _VERDICT_EMOJI.get(v_str, "❓")
                st.markdown(f"{emoji} **{v_str.upper()}** · `{fid[:8]}`  \n{reason}")
                st.markdown("---")


def _render_fact_checker(u: dict) -> None:
    updated = u.get("findings", [])
    st.markdown(f"✅ **{len(updated)}** finding(s) fact-checked")
    if updated:
        with st.expander("Updated confidence scores", expanded=False):
            for f in updated:
                conf  = f.confidence  if hasattr(f, "confidence") else f.get("confidence", 0)
                sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                bar_val = int(conf * 100)
                st.markdown(f"**{sub_q}**  \n`{conf:.2f}` confidence")
                st.progress(bar_val)


def _render_writer(u: dict) -> None:
    report = u.get("final_report")
    if report:
        title = report.title if hasattr(report, "title") else report.get("title", "")
        st.success(f"📄 Report complete: **{title}**")
    else:
        st.info("✍️ Writing report…")


def _render_raw(u: dict) -> None:
    with st.expander("Raw update", expanded=False):
        st.json({k: str(v)[:300] for k, v in u.items() if k != "messages"})


def render_trace_header() -> None:
    """Render the 'Agent Trace' section header."""
    st.markdown("### 🕸️ Agent Trace")
    st.caption("Updates stream in real time as each agent completes.")
