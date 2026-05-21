"""Report renderer — Markdown display, citation footnotes, quality badges, downloads."""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from research_swarm.schemas import FinalReport


# ── Quality badge helpers ─────────────────────────────────────────────────────

def _score_colour(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.5:
        return "🟡"
    return "🔴"


def _badge(label: str, score: float) -> str:
    emoji = _score_colour(score)
    return f"{emoji} **{label}**: {score:.0%}"


# ── Main render function ──────────────────────────────────────────────────────

def render_report(report: FinalReport) -> None:
    """Render a FinalReport in the Streamlit UI."""
    st.markdown(f"# {report.title}")

    # Quality score row (only shown if scores were computed in Phase 5)
    if report.quality_score:
        qs = report.quality_score
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Faithfulness",
            f"{qs.faithfulness:.0%}",
            help="Fraction of claims backed by citations",
        )
        c2.metric(
            "Relevance",
            f"{qs.relevance:.0%}",
            help="Avg cosine similarity to sub-questions",
        )
        c3.metric(
            "Completeness",
            f"{qs.completeness:.0%}",
            help="Fraction of sub-questions addressed",
        )
        c4.metric("Overall", f"{qs.overall:.0%}", delta=None)
        st.divider()

    # Executive summary
    with st.expander("📋 Executive Summary", expanded=True):
        st.markdown(report.exec_summary)

    # Sections
    st.markdown("## Sections")
    for i, section in enumerate(report.sections, 1):
        with st.expander(f"{i}. {section.heading}", expanded=i == 1):
            body = _inline_citations(section.body_md, section.citations, report.references)
            st.markdown(body)

    # References
    if report.references:
        with st.expander(f"📚 References ({len(report.references)})", expanded=False):
            for i, ref in enumerate(report.references, 1):
                title = ref.title if hasattr(ref, "title") else ref.get("title", "")
                url   = ref.url   if hasattr(ref, "url")   else ref.get("url",   "")
                cred = (
                    ref.credibility_score
                    if hasattr(ref, "credibility_score")
                    else ref.get("credibility_score", 0)
                )
                stype = (
                    ref.source_type
                    if hasattr(ref, "source_type")
                    else ref.get("source_type", "")
                )
                label = title or url
                st.markdown(
                    f"**[{i}]** [{label}]({url})  \n"
                    f"*{stype} · credibility: {cred:.0%}*"
                )

    # Methodology + limitations
    if report.methodology or report.limitations:
        with st.expander("🔬 Methodology & Limitations", expanded=False):
            if report.methodology:
                st.markdown("**Methodology**")
                st.markdown(report.methodology)
            if report.limitations:
                st.markdown("**Limitations**")
                st.markdown(report.limitations)

    st.divider()

    # Download buttons
    _render_downloads(report)


def _inline_citations(body: str, citation_nums: list[int], references: list) -> str:
    """Return body text with citation numbers replaced by inline superscripts."""
    # Already-formatted [N] markers are kept; we add tooltips via markdown hover
    return body  # LLM already formats citations as [N]; no further processing needed


def _render_downloads(report: FinalReport) -> None:
    """Render Markdown and HTML download buttons."""
    st.markdown("#### ⬇️ Download Report")
    col1, col2 = st.columns(2)

    md_text  = _report_to_markdown(report)
    html_text = _report_to_html(md_text, report.title)

    col1.download_button(
        label="📄 Download Markdown",
        data=md_text.encode("utf-8"),
        file_name=f"{_safe_filename(report.title)}.md",
        mime="text/markdown",
    )
    col2.download_button(
        label="🌐 Download HTML",
        data=html_text.encode("utf-8"),
        file_name=f"{_safe_filename(report.title)}.html",
        mime="text/html",
    )


# ── Conversion helpers ────────────────────────────────────────────────────────

def _safe_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    return safe.strip()[:60] or "report"


def _report_to_markdown(report: FinalReport) -> str:
    lines = [f"# {report.title}", "", report.exec_summary, ""]

    for section in report.sections:
        lines += [f"## {section.heading}", "", section.body_md, ""]

    if report.references:
        lines += ["## References", ""]
        for i, ref in enumerate(report.references, 1):
            title = ref.title if hasattr(ref, "title") else ref.get("title", "")
            url   = ref.url   if hasattr(ref, "url")   else ref.get("url", "")
            label = title or url
            lines.append(f"[{i}] [{label}]({url})")
        lines.append("")

    if report.methodology:
        lines += ["## Methodology", "", report.methodology, ""]
    if report.limitations:
        lines += ["## Limitations", "", report.limitations, ""]

    return "\n".join(lines)


def _report_to_html(md_text: str, title: str) -> str:
    try:
        import markdown as md_lib
        body_html = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        body_html = f"<pre>{md_text}</pre>"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>{title}</title>
          <style>
            body {{
              font-family: Georgia, serif;
              max-width: 860px;
              margin: 2rem auto;
              padding: 0 1rem;
              line-height: 1.7;
              color: #222;
            }}
            h1 {{ border-bottom: 2px solid #333; padding-bottom: .4rem; }}
            h2 {{ margin-top: 2rem; color: #444; }}
            a  {{ color: #1a73e8; }}
            blockquote {{
              border-left: 4px solid #ccc;
              margin: 0;
              padding-left: 1rem;
              color: #555;
            }}
            code {{ background: #f5f5f5; padding: .1em .3em; border-radius: 3px; }}
            pre  {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; }}
          </style>
        </head>
        <body>
        {body_html}
        </body>
        </html>
    """)
