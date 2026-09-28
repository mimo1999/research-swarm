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

Evidence-first research: a LangGraph pipeline frames the question, plans sub-questions, searches
the literature, extracts quoted facts, verifies them, and writes a cited report — with a live view of every agent in the pipeline
and a human-in-the-loop checkpoint before the final write-up.

## Setup

**Models.** The research stages (paper scoring, fact extraction, gap fill, verification) run
**Gemma 4 E2B (`google/gemma-4-E2B-it`) in this process on the Space's ZeroGPU**, through
`transformers` (`research_swarm/agents/hf_local.py`, provider `huggingface`). The planner and
the writer run `gemma4:31b-cloud` on **Ollama Cloud**, called directly at
`https://ollama.com`; no `ollama serve` runs in this container. That is the same split as a local
run: a small model does the research, and a larger model plans and writes.

**ZeroGPU quota.** GPU time is charged to the *visitor's* daily quota: 2 minutes anonymous,
5 minutes for a signed-in free account, 40 minutes PRO. To keep a run inside that:
- Concurrent calls from one run (one scoring or extraction call per sub-question, the verifier's
  batches) are micro-batched into a single `generate()`, so they cost about as much as the longest
  one.
- Each reply is constrained to its stage's JSON schema (lm-format-enforcer), so no GPU time is
  spent on replies that fail to parse.
- The model is loaded to `cuda` at import time, as ZeroGPU requires, and the constraint tables
  (about 5 s) are built at startup rather than inside a billed call.

If a visitor's quota runs out mid-run, the rest of that run's local calls fail fast and the page
says the result is incomplete. Visitors can also pick the `ollama` provider (every research stage
on Ollama Cloud, no GPU quota used) or bring their own Anthropic or OpenAI key.

**Secrets** (Settings → Variables and secrets):
- `OLLAMA_API_KEY`: the Ollama Cloud API key for the planner and writer, and for the `ollama`
  provider option. Anthropic and OpenAI are not funded by this deployment. A visitor who wants
  them enters their own key in Advanced options, bound to their session only (`session_ctx.py`).

**Variables:**
- `DATA_DIR=/tmp/research_swarm_space`: the container's disk is ephemeral.
- `SPACE_MODE=true`: session pruning and a concurrent-run cap (`space_*` settings in
  `research_swarm/config.py`).
- `SPACE_LOCAL_MODEL`: the in-process model, default `google/gemma-4-E2B-it`. Set it empty to
  run every stage on Ollama Cloud; the app then keeps only a placeholder `@spaces.GPU` function
  for ZeroGPU's startup check.
- Optional: `DEPTH_PROFILES` (the per-depth table in `research_swarm/config.py`: 3 / 5 / 7
  sub-questions) to make runs cheaper, and `HF_MAX_BATCH` (8, at least the deepest profile's
  sub-questions, so each stage's calls fit one GPU batch) / `HF_BATCH_WINDOW_S`.

**Deploy.** Copy `hf_space/app.py`, `hf_space/requirements.txt` and this `README.md` to the Space
repo root, next to the `research_swarm/` package, then push.

## What's different from the chatbot template

This isn't a `gr.ChatInterface` — the underlying app has a settings form, a live multi-agent trace
timeline, a real approve/discard human-review panel, and a structured cited report, none of which
map onto a back-and-forth chat. `app.py` here is a custom `gr.Blocks` layout instead.
