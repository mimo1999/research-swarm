"""Shared visual styling — CSS injection and a status-badge helper.

Centralised here so every UI module (sidebar, trace, report, sessions) uses
the same badge/card vocabulary instead of each inventing its own emoji or
ad-hoc markdown. Badges render via a single self-contained st.markdown(...,
unsafe_allow_html=True) call each -- never split across multiple calls (an
unclosed <div> spanning several st.markdown/widget calls is a common
Streamlit hack that depends on incidental DOM nesting and breaks easily;
st.container(border=True) is the supported way to get a bordered card, so
that's what callers should reach for instead).
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
    --rs-success: #15803d;
    --rs-success-soft: #f0fdf4;
    --rs-warning: #b45309;
    --rs-warning-soft: #fffbeb;
    --rs-danger: #b91c1c;
    --rs-danger-soft: #fef2f2;
    --rs-neutral: #475569;
    --rs-neutral-soft: #f1f5f9;
    --rs-accent: #2563eb;
    --rs-accent-soft: #eff6ff;
}

.rs-badge {
    display: inline-block;
    padding: 0.14rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    line-height: 1.6;
    white-space: nowrap;
}
.rs-badge-success { background: var(--rs-success-soft); color: var(--rs-success); }
.rs-badge-warning { background: var(--rs-warning-soft); color: var(--rs-warning); }
.rs-badge-danger  { background: var(--rs-danger-soft);  color: var(--rs-danger); }
.rs-badge-neutral { background: var(--rs-neutral-soft); color: var(--rs-neutral); }
.rs-badge-accent  { background: var(--rs-accent-soft);  color: var(--rs-accent); }

.rs-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.2rem;
}

/* Compact top-of-page header bar: name + tagline on one line instead of a
   stacked title/caption pair -- closer to a single status-card row than a
   full page-title block. */
.rs-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding-top: 0.25rem;
}
.rs-header-name {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--rs-accent);
}
.rs-header-tag {
    font-size: 0.85rem;
    color: var(--rs-neutral);
}

/* Idle-state landing heading above the research topic input. */
.rs-hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.rs-hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.rs-hero p {
    color: var(--rs-neutral);
    font-size: 1rem;
    margin: 0;
}

/* Tighten the default gap between stacked elements slightly so a long trace
   or findings list reads as a coherent column of cards, not a page of
   whitespace-separated fragments. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    margin-bottom: 0.5rem;
}
</style>
"""

_BADGE_CLASS = {
    "success": "rs-badge-success",
    "warning": "rs-badge-warning",
    "danger":  "rs-badge-danger",
    "neutral": "rs-badge-neutral",
    "accent":  "rs-badge-accent",
}


def inject_css() -> None:
    """Inject the shared stylesheet. Call once per script run, before any
    badge()/kicker() usage — Streamlit reruns the whole script on every
    interaction, so this is cheap and idempotent, not a one-time setup cost."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    """Return an inline-HTML status badge. Render with unsafe_allow_html=True."""
    cls = _BADGE_CLASS.get(kind, "rs-badge-neutral")
    return f'<span class="rs-badge {cls}">{text}</span>'


def kicker(text: str, colour: str | None = None) -> str:
    """Return a small uppercase label line, optionally tinted. Render with
    unsafe_allow_html=True."""
    style = f' style="color:{colour};"' if colour else ""
    return f'<div class="rs-kicker"{style}>{text}</div>'
