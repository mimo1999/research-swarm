"""Streamlit sidebar — settings, file uploads, and URL inputs."""
from __future__ import annotations

import streamlit as st
from research_swarm.config import settings
from research_swarm.schemas import ResearchDepth


_ANTHROPIC_MODELS = ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5"]
_OPENAI_MODELS    = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

_DEPTH_LABELS = {
    ResearchDepth.shallow:  "⚡ Shallow  (fast, 2-3 sources)",
    ResearchDepth.standard: "⚖️  Standard (balanced)",
    ResearchDepth.deep:     "🔬 Deep     (thorough, more LLM calls)",
}

_PROVIDER_LABELS = {
    "anthropic": "🟣 Anthropic (Claude)",
    "openai":    "🟢 OpenAI (GPT)",
    "ollama":    "🦙 Ollama (local)",
}


def render_sidebar() -> dict:
    """Render the sidebar and return the current UI settings as a dict.

    Keys: provider, model, depth, max_sources, hitl_enabled,
          uploaded_pdfs, extra_urls
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        # ── LLM provider ──────────────────────────────────────────────
        st.subheader("Model")
        provider = st.radio(
            "Provider",
            ["anthropic", "openai", "ollama"],
            format_func=lambda x: _PROVIDER_LABELS[x],
            horizontal=True,
            key="ui_provider",
        )

        ollama_url: str | None = None   # populated only for ollama provider

        if provider == "anthropic":
            model = st.selectbox("Model name", _ANTHROPIC_MODELS, key="ui_model_anthropic")

        elif provider == "openai":
            model = st.selectbox("Model name", _OPENAI_MODELS, key="ui_model_openai")

        else:  # ollama
            model = st.text_input(
                "Ollama model",
                value=settings.ollama_model,
                key="ui_model_ollama",
                help="Name of the locally-pulled Ollama model, e.g. llama3.2, mistral, gemma2:9b",
            )
            ollama_url = st.text_input(
                "Ollama base URL",
                value=settings.ollama_base_url,
                key="ui_ollama_url",
                help="Address of your Ollama server (default: http://localhost:11434)",
            )
            # Probe Ollama availability and show status
            _render_ollama_status(ollama_url, model)

        st.divider()

        # ── Research settings ─────────────────────────────────────────
        st.subheader("Research")
        depth_key = st.select_slider(
            "Depth",
            options=[ResearchDepth.shallow, ResearchDepth.standard, ResearchDepth.deep],
            value=ResearchDepth.standard,
            format_func=lambda d: _DEPTH_LABELS[d],
            key="ui_depth",
        )
        max_sources = st.slider(
            "Max sources", min_value=3, max_value=30, value=10, step=1,
            key="ui_max_sources",
        )

        st.divider()

        # ── HITL toggle ───────────────────────────────────────────────
        st.subheader("Human-in-the-Loop")
        hitl_enabled = st.toggle(
            "Pause before writing (HITL)",
            value=True,
            key="ui_hitl",
            help="If enabled, the graph pauses after fact-checking so you can review findings before the report is written.",
        )

        st.divider()

        # ── Document ingestion ────────────────────────────────────────
        st.subheader("📎 Documents")
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="ui_pdfs",
        )
        extra_urls_raw = st.text_area(
            "URLs to include (one per line)",
            placeholder="https://example.com/paper\nhttps://arxiv.org/abs/...",
            key="ui_urls",
            height=100,
        )
        extra_urls = [u.strip() for u in extra_urls_raw.splitlines() if u.strip()]

        st.divider()

        # ── Info ──────────────────────────────────────────────────────
        st.caption("🔗 Multi-Agent Research Swarm · Phase 6")

    # Normalise model: for anthropic/openai the selectbox gives the value,
    # for ollama the text_input. All paths set `model` above.
    return {
        "provider":       provider,
        "model":          model,
        "depth":          depth_key,
        "max_sources":    max_sources,
        "hitl_enabled":   hitl_enabled,
        "uploaded_pdfs":  uploaded_pdfs or [],
        "extra_urls":     extra_urls,
        # Only populated for ollama provider; app.py uses it to update settings
        "ollama_url":     ollama_url if provider == "ollama" else None,
    }


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _render_ollama_status(base_url: str, model: str) -> None:
    """Show a small status indicator for the Ollama server."""
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            names = [t.get("name", "") for t in tags]
            if model in names:
                st.success(f"✅ Ollama reachable · **{model}** is available")
            else:
                available = ", ".join(names[:5]) or "none"
                st.warning(
                    f"⚠️ Ollama reachable but **{model}** not found.  \n"
                    f"Pulled models: {available}"
                )
        else:
            st.error(f"❌ Ollama returned HTTP {resp.status_code}")
    except Exception as exc:
        st.error(f"❌ Cannot reach Ollama at `{base_url}` — {exc}")
