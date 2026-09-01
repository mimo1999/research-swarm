"""Agent trace renderer — live per-node status blocks during graph execution."""
from __future__ import annotations

from typing import Any

import streamlit as st

from research_swarm.ui.style import badge, kicker

# ── Label / accent colour per node ────────────────────────────────────────────
_NODE_META: dict[str, dict] = {
    "supervisor":   {"label": "Supervisor",    "colour": "#5c6bc0"},
    "researcher":   {"label": "Researcher",    "colour": "#26a69a"},
    "critic":       {"label": "Critic",        "colour": "#ef6c00"},
    "fact_checker": {"label": "Fact-Checker",  "colour": "#2e7d32"},
    "writer":       {"label": "Writer",        "colour": "#8e24aa"},
}
_DEFAULT_META = {"label": "Agent", "colour": "#64748b"}

_VERDICT_BADGE = {
    "supported": ("SUPPORTED", "success"),
    "weak":      ("WEAK", "warning"),
    "refuted":   ("REFUTED", "danger"),
}


def _node_meta(name: str) -> dict:
    return _NODE_META.get(name, _DEFAULT_META)


def render_node_update(node_name: str, update: dict[str, Any]) -> None:
    """Render a single node update as a bordered card."""
    meta = _node_meta(node_name)
    with st.container(border=True):
        st.markdown(kicker(meta["label"], meta["colour"]), unsafe_allow_html=True)
        _render_update_body(node_name, update)


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
    st.markdown(f"Routing to **{next_a}** &nbsp;·&nbsp; iteration {iter_n}")
    if plan := u.get("plan"):
        sub_qs = (
            plan.sub_questions
            if hasattr(plan, "sub_questions")
            else plan.get("sub_questions", [])
        )
        if sub_qs:
            with st.expander(f"Research plan ({len(sub_qs)} sub-questions)", expanded=False):
                for i, q in enumerate(sub_qs, 1):
                    st.markdown(f"{i}. {q}")
    for msg in u.get("messages", []):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content.startswith("[Supervisor]"):
            st.caption(content)


def _render_researcher(u: dict) -> None:
    findings = u.get("findings", [])
    st.markdown(f"**{len(findings)}** finding(s) produced")
    if findings:
        with st.expander("View findings", expanded=False):
            for f in findings:
                claim  = f.claim       if hasattr(f, "claim")       else f.get("claim", "")
                conf   = f.confidence  if hasattr(f, "confidence")  else f.get("confidence", 0)
                sub_q  = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                n_src  = len(f.evidence if hasattr(f, "evidence") else f.get("evidence", []))
                with st.container(border=True):
                    st.markdown(f"**{sub_q}**")
                    st.markdown(claim)
                    st.caption(f"confidence: {conf:.2f} · {n_src} source(s)")


def _render_critic(u: dict) -> None:
    critiques = u.get("critiques", [])
    st.markdown(f"**{len(critiques)}** critique(s) produced")
    if critiques:
        with st.expander("View critiques", expanded=False):
            for c in critiques:
                verdict  = c.verdict   if hasattr(c, "verdict")   else c.get("verdict", "?")
                v_str    = verdict.value if hasattr(verdict, "value") else str(verdict)
                fid      = c.finding_id if hasattr(c, "finding_id") else c.get("finding_id", "")
                reason   = c.reasoning  if hasattr(c, "reasoning")  else c.get("reasoning", "")
                label, kind = _VERDICT_BADGE.get(v_str, (v_str.upper() or "UNKNOWN", "neutral"))
                with st.container(border=True):
                    st.markdown(
                        f"{badge(label, kind)} &nbsp; `{fid[:8]}`", unsafe_allow_html=True,
                    )
                    st.markdown(reason)


def _render_fact_checker(u: dict) -> None:
    updated = u.get("findings", [])
    st.markdown(f"**{len(updated)}** finding(s) fact-checked")
    if updated:
        with st.expander("Updated confidence scores", expanded=False):
            for f in updated:
                conf  = f.confidence  if hasattr(f, "confidence") else f.get("confidence", 0)
                sub_q = f.sub_question if hasattr(f, "sub_question") else f.get("sub_question", "")
                bar_val = int(conf * 100)
                st.markdown(f"**{sub_q}** &nbsp;·&nbsp; `{conf:.2f}` confidence")
                st.progress(bar_val)


def _render_writer(u: dict) -> None:
    report = u.get("final_report")
    if report:
        title = report.title if hasattr(report, "title") else report.get("title", "")
        st.success(f"Report complete: **{title}**")
    else:
        st.info("Writing report…")


def _render_raw(u: dict) -> None:
    with st.expander("Raw update", expanded=False):
        st.json({k: str(v)[:300] for k, v in u.items() if k != "messages"})


def render_trace_header() -> None:
    """Render the 'Agent Trace' section header."""
    st.markdown("### Agent Trace")
    st.caption("Updates stream in real time as each agent completes.")
