---
title: Research Swarm
emoji: 🕸️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
short_description: Multi-agent LangGraph research assistant with live trace
---

# Research Swarm

Autonomous multi-agent research: a LangGraph swarm plans, researches, critiques, fact-checks, and
writes a cited report on any topic you give it — with a live view of every agent in the pipeline
and a human-in-the-loop checkpoint before the final write-up.

## Setup

This Space runs on **Ollama Cloud, funded by the Space owner** — the only server-side secret it
needs. Anthropic and OpenAI are deliberately *not* funded by this deployment: a visitor who wants
either must type in their own key in the form's advanced options (sent per-request, bound to their
session only, never written to `research_swarm.config.settings` — see `session_ctx.py`).

Set as a **Space secret** (Settings → Variables and secrets — never commit a real key to the repo):

- `OLLAMA_API_KEY` — an Ollama Cloud API key (from your ollama.com account), used as a bearer
  token against `https://ollama.com` directly. Confirmed live: `https://ollama.com` mirrors the
  local daemon's API (`GET /api/tags` is public, `POST /api/chat` returns a clean
  `401 {"error":"Unauthorized"}` without a valid token) — **no local `ollama serve` process runs
  in this container**, unlike every other path in this codebase (Streamlit, the FastAPI backend),
  which assume a local daemon proxying via `ollama login`. Verify this still holds against your
  own real key before relying on it in production; if Ollama's cloud API surface changes, the
  fallback path (running `ollama serve` inside the container via Docker SDK) is the one this repo
  has actually proven elsewhere.

The form defaults to `ollama` / `cloud` deployment / `nemotron-3-nano:30b-cloud`, and reasoning
mode is on by default (`ollama_reasoning` in `config.py`) — both validated this session to produce
materially better structured-output reliability than the project's original default model.

Recommended Space **variables**:
- `DATA_DIR=/tmp/research_swarm_space` — the container's disk is ephemeral; keep checkpoints there.
- `SPACE_MODE=true` — enables session pruning and a concurrency cap so one Space process handling
  several simultaneous visitors doesn't run out of memory (each run holds embedding models on top
  of its LLM calls). See `research_swarm/config.py`'s `space_*` settings.

## What's different from the chatbot template

This isn't a `gr.ChatInterface` — the underlying app has a settings form, a live multi-agent trace
timeline, a real approve/discard human-review panel, and a structured cited report, none of which
map onto a back-and-forth chat. `app.py` here is a custom `gr.Blocks` layout instead.
