"""Streamlit sidebar — settings, file uploads, and URL inputs."""
from __future__ import annotations

import streamlit as st

from research_swarm.config import settings
from research_swarm.schemas import ResearchDepth

_ANTHROPIC_MODELS    = ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5"]
_OPENAI_MODELS       = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
_OLLAMA_CLOUD_MODELS = [
    "minimax-m2.5:cloud",
    "gpt-oss:120b-cloud",
    "qwen3-coder-next:cloud",
    "qwen3.5:cloud",
    "minimax-m2.7:cloud",
]

_DEPTH_LABELS = {
    ResearchDepth.shallow:  "Shallow  (fastest — 1 sub-question, 1 tool turn, no re-research)",
    ResearchDepth.standard: "Standard (balanced — 4 sub-questions, 3 tool turns)",
    ResearchDepth.deep:     "Deep     (thorough — 6 sub-questions, 6 tool turns)",
}

_PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai":    "OpenAI (GPT)",
    "ollama":    "Ollama",
}

_PROVIDERS = ["ollama", "anthropic", "openai"]


def render_sidebar() -> dict:
    """Render the sidebar and return the current UI settings as a dict.

    Keys: provider, model, depth, max_sources, hitl_enabled,
          uploaded_pdfs, extra_urls
    """
    with st.sidebar:
        st.header("Settings")

        # ── LLM provider ──────────────────────────────────────────────
        st.subheader("Model")
        provider = st.selectbox(
            "Provider",
            _PROVIDERS,
            format_func=lambda x: _PROVIDER_LABELS[x],
            key="ui_provider",
        )

        ollama_url: str | None = None   # populated only for ollama provider

        if provider == "anthropic":
            model = st.selectbox("Model name", _ANTHROPIC_MODELS, key="ui_model_anthropic")

        elif provider == "openai":
            model = st.selectbox("Model name", _OPENAI_MODELS, key="ui_model_openai")

        else:  # ollama
            # ── Local vs Cloud deployment ──────────────────────────────
            # Both modes talk to the same local Ollama daemon (localhost:11434).
            # In cloud mode the daemon proxies requests to Ollama's cloud using
            # credentials stored by `ollama login` — no separate URL is needed.
            ollama_deployment = st.selectbox(
                "Deployment",
                ["local", "cloud"],
                format_func=lambda x: "Local" if x == "local" else "Cloud",
                index=0 if settings.ollama_deployment == "local" else 1,
                key="ui_ollama_deployment",
                help=(
                    "Local: run a model you have pulled locally.  "
                    "Cloud: stream a model from Ollama cloud via your logged-in "
                    "account — requires `ollama login` on this machine."
                ),
            )

            if ollama_deployment == "local":
                model = st.text_input(
                    "Model name",
                    value=settings.ollama_model,
                    key="ui_model_ollama",
                    help="Locally pulled model tag, e.g. llama3.2, mistral, gemma2:9b",
                )
                ollama_url = st.text_input(
                    "Ollama URL",
                    value=settings.ollama_base_url,
                    key="ui_ollama_url",
                    help="Address of your local Ollama server (default: http://localhost:11434)",
                )
                _render_ollama_local_status(ollama_url, model)

            else:  # cloud
                _cloud_default = settings.ollama_cloud_model
                _cloud_idx = _OLLAMA_CLOUD_MODELS.index(_cloud_default) \
                    if _cloud_default in _OLLAMA_CLOUD_MODELS else 0
                model = st.selectbox(
                    "Cloud model",
                    _OLLAMA_CLOUD_MODELS,
                    index=_cloud_idx,
                    key="ui_model_ollama",
                    help="Streamed from ollama.com via your logged-in account.",
                )
                # Cloud always routes through the local daemon — URL is not
                # user-configurable in this mode.
                ollama_url = settings.ollama_base_url
                _render_ollama_cloud_status(ollama_url, model)

        st.divider()

        # ── Research settings ─────────────────────────────────────────
        st.subheader("Research")
        depth_key = st.selectbox(
            "Depth",
            options=[ResearchDepth.shallow, ResearchDepth.standard, ResearchDepth.deep],
            format_func=lambda d: _DEPTH_LABELS[d],
            key="ui_depth",
        )

        with st.expander("Advanced", expanded=False):
            max_sources = st.slider(
                "Max sources", min_value=3, max_value=30, value=settings.max_sources, step=1,
                key="ui_max_sources",
            )
            hitl_enabled = st.toggle(
                "Pause before writing (HITL)",
                value=True,
                key="ui_hitl",
                help=(
                    "If enabled, the graph pauses after fact-checking so you can "
                    "review findings before the report is written."
                ),
            )

        st.divider()

        # ── Document ingestion ────────────────────────────────────────
        st.subheader("Documents")
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
        st.caption("Multi-Agent Research Swarm")

    # Normalise model: for anthropic/openai the selectbox gives the value,
    # for ollama the text_input. All paths set `model` above.
    return {
        "provider":          provider,
        "model":             model,
        "depth":             depth_key,
        "max_sources":       max_sources,
        "hitl_enabled":      hitl_enabled,
        "uploaded_pdfs":     uploaded_pdfs or [],
        "extra_urls":        extra_urls,
        # Ollama-specific — only set when provider == "ollama"
        "ollama_url":        ollama_url if provider == "ollama" else None,
        "ollama_deployment": ollama_deployment if provider == "ollama" else None,
    }


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _ollama_get(base_url: str, path: str, timeout: float = 2.0):
    """GET *path* from the Ollama daemon. Returns the parsed JSON or raises."""
    import httpx
    resp = httpx.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _render_ollama_local_status(base_url: str, model: str) -> None:
    """Status indicator for local Ollama — checks daemon reachability and whether
    the requested model has been pulled."""
    try:
        data = _ollama_get(base_url, "/api/tags")
        names = [t.get("name", "") for t in data.get("models", [])]
        if model in names:
            st.success(f"Daemon reachable · **{model}** is pulled and ready")
        else:
            available = ", ".join(names[:5]) or "none"
            st.warning(
                f"Daemon reachable but **{model}** is not pulled.  \n"
                f"Run `ollama pull {model}` to download it.  \n"
                f"Available: {available}"
            )
    except Exception as exc:
        st.error(f"Cannot reach Ollama at `{base_url}` — {exc}")


def _render_ollama_cloud_status(base_url: str, model: str) -> None:
    """Status indicator for Ollama cloud mode.

    Cloud models are served by the **local daemon** using credentials stored
    by ``ollama login``.  The daemon reads auth credentials directly from disk
    — login state is NOT reliably exposed through the HTTP API (``/api/whoami``
    only exists in Ollama ≥ 0.6).  A 404 or error on that endpoint means the
    version is too old to report it, *not* that the user is logged out.

    A model absent from the local pull list is fine in cloud mode — the daemon
    will stream it from ollama.com at inference time when the model tag routes
    there (e.g. the ``:cloud`` suffix).
    """
    # ── Optional: try /api/whoami for daemons that support it (≥ 0.6) ────────
    username: str | None = None
    whoami_unsupported = False
    try:
        who = _ollama_get(base_url, "/api/whoami")
        username = who.get("username") or who.get("name")
    except Exception as exc:
        # 404 → endpoint absent on this daemon version; any other error is
        # also non-fatal — we just can't confirm the username.
        whoami_unsupported = "404" in str(exc) or "Not Found" in str(exc)

    # ── Main check: is the daemon reachable and is the model local? ───────────
    try:
        data  = _ollama_get(base_url, "/api/tags")
        names = [t.get("name", "") for t in data.get("models", [])]

        if username:
            login_line = f"Logged in as **{username}**"
        elif whoami_unsupported:
            # Daemon is too old to expose login via API — but credentials on
            # disk still work, so don't mislead the user with "not logged in".
            login_line = "Login status: verified via `ollama login` on this machine"
        else:
            login_line = "Not logged in — run `ollama login` for cloud models"

        if model in names:
            st.success(
                f"Daemon reachable · **{model}** available locally  \n"
                f"{login_line}"
            )
        else:
            # Model not pulled — will stream from cloud. Only warn if we know
            # for certain the user is not logged in (username is None AND the
            # whoami endpoint worked and returned no user).
            definitely_logged_out = not username and not whoami_unsupported
            if definitely_logged_out:
                st.warning(
                    f"Daemon reachable · **{model}** not pulled locally  \n"
                    f"{login_line}  \n"
                    "Log in first, or pull the model with `ollama pull`."
                )
            else:
                st.info(
                    f"Daemon reachable · **{model}** not pulled locally  \n"
                    f"{login_line}  \n"
                    "Will stream from ollama.com at inference time."
                )
    except Exception as exc:
        st.error(f"Cannot reach Ollama daemon at `{base_url}` — {exc}")
