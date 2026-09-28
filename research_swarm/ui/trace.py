"""Agent trace renderer — live per-node status blocks during graph execution."""
from __future__ import annotations

from typing import Any

import streamlit as st

from research_swarm.ui.style import badge, kicker

# ── Label / accent colour per node ────────────────────────────────────────────
_NODE_META: dict[str, dict] = {
    "supervisor":           {"label": "Supervisor",       "colour": "#5c6bc0"},
    "document_worker_node": {"label": "Document reader",  "colour": "#26a69a"},
    "paper_scout_node":     {"label": "Paper scout",      "colour": "#00897b"},
    "paper_worker_node":    {"label": "Paper reader",     "colour": "#26a69a"},
    "worker_node":          {"label": "Gap fill",         "colour": "#26a69a"},
    "packet_node":          {"label": "Evidence packet",  "colour": "#00897b"},
    "verifier":             {"label": "Verifier",         "colour": "#ef6c00"},
    "writer":               {"label": "Writer",           "colour": "#8e24aa"},
}
_FINDING_NODES = ("document_worker_node", "paper_worker_node", "worker_node")
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
    elif node_name in _FINDING_NODES:
        _render_findings(update)
    elif node_name == "packet_node":
        render_packet(update.get("evidence_packet") or {})
    elif node_name == "verifier":
        _render_verifier(update)
    elif node_name == "writer":
        _render_writer(update)
    else:
        _render_raw(update)


def _render_supervisor(u: dict) -> None:
    if next_a := u.get("next_agent"):
        st.markdown(f"Routing to **{next_a}** &nbsp;·&nbsp; iteration "
                    f"{u.get('iteration_count', '—')}")
    if plan := u.get("plan"):
        frame = getattr(plan, "frame", None)
        if frame is not None and (frame.interpretation or frame.key_constraint):
            if frame.interpretation:
                st.caption(f"Understood as: {frame.interpretation}")
            if frame.key_constraint:
                terms = f" · also: {', '.join(frame.constraint_terms)}" \
                    if frame.constraint_terms else ""
                st.markdown(f"Key constraint: **{frame.key_constraint}**{terms}")
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


def _render_findings(u: dict) -> None:
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


def _render_verifier(u: dict) -> None:
    critiques = u.get("critiques", [])
    st.markdown(f"**{len(critiques)}** finding(s) verified")
    if critiques:
        with st.expander("View verdicts", expanded=False):
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
                    st.caption(reason)
    if conflicts := u.get("fact_conflicts"):
        st.warning(f"{len(conflicts)} pair(s) of findings contradict each other.")


def _render_writer(u: dict) -> None:
    report = u.get("final_report")
    if report:
        title = report.title if hasattr(report, "title") else report.get("title", "")
        st.success(f"Report complete: **{title}**")
    else:
        st.info("Writing report…")
    # Packet path: the synthesis returns the packet sentences it cited as the run's findings.
    cited = u.get("findings") or []
    if cited:
        st.caption(f"Rests on {len(cited)} cited source sentence(s), each verified by ID.")
        with st.expander("View cited sentences", expanded=False):
            for f in cited:
                quote = f.quote if hasattr(f, "quote") else f.get("quote", "")
                ev = (f.evidence if hasattr(f, "evidence") else f.get("evidence", [])) or []
                first = ev[0] if ev else None
                title = (first.title if hasattr(first, "title")
                         else first.get("title", "")) if first is not None else ""
                st.markdown(f"> {quote}")
                if title:
                    st.caption(title)


def packet_summary(packet: dict) -> str:
    """One line describing an evidence packet (``EvidencePacket.to_dict()``)."""
    stats = packet.get("stats") or {}
    sentences = packet.get("sentences") or []
    fit = {"whole": "sources fit whole", "scored": "trimmed by code scoring",
           "screened": "trimmed by local screening"}.get(stats.get("fit", ""), "")
    line = (f"**{len(sentences)}** sentence(s) from **{stats.get('sources', 0)}** source(s), "
            f"~{stats.get('kept_tokens', 0)} of {packet.get('budget_tokens', 0)} budget tokens")
    if fit:
        line += f" · {fit}"
    if stats.get("screened_passages"):
        line += (f" ({stats['rejected_passages']} of {stats['screened_passages']} passages "
                 "rejected)")
    if stats.get("duplicates"):
        line += f" · {stats['duplicates']} duplicate(s) dropped"
    return line


def render_packet(packet: dict, expanded: bool = False) -> None:
    """The evidence packet: what the synthesis call reads, grouped by source, with sentence IDs.
    Shared by the trace card and the review pause."""
    if not packet.get("sentences"):
        st.info("The packet is empty: no source text was supplied.")
        return
    st.markdown(packet_summary(packet))
    sources = {s["number"]: s for s in packet.get("sources") or []}
    by_source: dict[int, list[dict]] = {}
    for s in packet["sentences"]:
        by_source.setdefault(s["source"], []).append(s)
    with st.expander(f"View packet ({len(packet['sentences'])} sentences)", expanded=expanded):
        for number, sents in by_source.items():
            src = sources.get(number, {})
            st.markdown(f"**[S{number}] {src.get('title') or src.get('url', '')}**")
            st.markdown("\n".join(f"- `{s['id']}` {s['text']}" for s in sents))


def _render_raw(u: Any) -> None:
    with st.expander("Raw update", expanded=False):
        if isinstance(u, dict):
            st.json({k: str(v)[:300] for k, v in u.items() if k != "messages"})
        else:
            st.json(str(u)[:500])


def render_trace_header() -> None:
    """Render the 'Agent Trace' section header."""
    st.markdown("### Agent Trace")
    st.caption("Updates stream in real time as each agent completes.")
