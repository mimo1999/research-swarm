# 🔬 Multi-Agent Research Swarm

A production-grade autonomous research system built on **LangGraph**, **LlamaIndex**, and **Streamlit**. Give it a topic; a swarm of five specialised AI agents researches it, critiques the findings, fact-checks every claim, and writes a structured report — all with optional human review before the final draft.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agents](#agents)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [RAG Layer](#rag-layer)
- [Session Persistence](#session-persistence)
- [Running Tests](#running-tests)
- [LLM Providers](#llm-providers)

---

## Overview

The swarm accepts a research query (topic + audience + depth) and autonomously:

1. **Plans** a set of sub-questions to investigate
2. **Researches** each sub-question using web search, arXiv, URL fetching, and an optional RAG index built from documents you upload
3. **Critiques** the findings for quality and evidence strength
4. **Fact-checks** every claim against its cited sources
5. **Writes** a fully-referenced report in Markdown

Each step is a separate async agent node inside a LangGraph `StateGraph`. The Supervisor routes between agents and enforces an iteration cap to prevent infinite loops.

---

## Architecture

```
START
  └─► supervisor ──────────────────────────────────────────────┐
        │                                                       │
        ├─► researcher  ──────────────────────────────────────►─┤
        │       └─ ReAct tool loop (web_search / arxiv /       │
        │              fetch_url / retrieve_from_rag)          │
        ├─► critic      ──────────────────────────────────────►─┤
        ├─► fact_checker ─────────────────────────────────────►─┤
        └─► writer ◄── [optional HITL interrupt here]         │
              └─► END                                          │
                                                               │
              (supervisor loops until conditions are met) ◄───┘
```

**State** (`AgentState`) is a single `TypedDict` threaded through every node. Two custom reducers keep it consistent:

| Field | Reducer | Behaviour |
|-------|---------|-----------|
| `findings` | merge-by-id | Fact-checker overwrites findings in-place by matching `id` |
| `critiques` | append-only | Each critic pass accumulates; existing critiques are never deleted |

---

## Agents

| Agent | File | Responsibility |
|-------|------|---------------|
| **Supervisor** | `agents/supervisor.py` | Reads the full state, decides the next agent to call, produces the research plan on the first pass. Enforces `MAX_ITERATIONS`. |
| **Researcher** | `agents/researcher.py` | Runs a ReAct tool-calling loop (up to 6 turns) for each un-answered sub-question. Synthesises a `Finding` (claim + confidence + evidence) from the gathered messages. Skips sub-questions already marked `supported` by the Critic. |
| **Critic** | `agents/critic.py` | Reviews each un-critiqued `Finding` and returns a `Critique` with a `supported / weak / refuted` verdict. Stamps the real `finding_id` over any hallucinated ID from the LLM. Falls back to `supported` on LLM failure to avoid triggering expensive re-research loops. |
| **Fact-Checker** | `agents/fact_checker.py` | Cross-checks every non-refuted `Finding` against its cited source snippets and adjusts the confidence score. Penalises findings with no evidence to `0.1`. |
| **Writer** | `agents/writer.py` | Synthesises validated findings into a `FinalReport` with sections, references, methodology, and limitations. Falls back to a minimal report if the LLM fails. Optionally incorporates human feedback. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) 0.2+ |
| LLM abstraction | [LangChain](https://github.com/langchain-ai/langchain) 0.3+ |
| LLM providers | Anthropic Claude, OpenAI GPT, Ollama (local) |
| RAG / embeddings | [LlamaIndex](https://github.com/run-llama/llama_index) 0.11+ · HuggingFace `bge-small-en-v1.5` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (per-session, local) |
| Web search | [Tavily](https://tavily.com/) |
| Academic search | [arXiv API](https://arxiv.org/help/api/) |
| UI | [Streamlit](https://streamlit.io/) 1.37+ |
| Checkpointing | `AsyncSqliteSaver` (LangGraph) + `aiosqlite` |
| Data validation | [Pydantic](https://docs.pydantic.dev/) v2 |
| Packaging | [Poetry](https://python-poetry.org/) |
| Tests | pytest · pytest-asyncio (157 tests, fully offline) |

---

## Project Structure

```
swarm_agent_project/
│
├── app.py                        # Streamlit entry point
├── start.bat                     # Windows one-click launcher
├── pyproject.toml                # Poetry dependencies + tool config
├── .env.example                  # API key template (copy → .env)
│
├── research_swarm/
│   ├── config.py                 # Pydantic BaseSettings singleton
│   ├── main.py                   # CLI entry point (optional)
│   │
│   ├── agents/
│   │   ├── base.py               # get_agent_llm() factory (Anthropic/OpenAI/Ollama)
│   │   ├── supervisor.py         # Routing logic + plan generation
│   │   ├── researcher.py         # ReAct tool loop + FindingSynthesis
│   │   ├── critic.py             # Finding quality review
│   │   ├── fact_checker.py       # Source cross-check + confidence scoring
│   │   └── writer.py             # Report generation + reference deduplication
│   │
│   ├── graph/
│   │   ├── builder.py            # build_graph(), make_async_checkpointer()
│   │   ├── nodes.py              # Async node wrappers for each agent
│   │   └── edges.py              # route_from_supervisor() conditional edge
│   │
│   ├── rag/
│   │   ├── _chroma.py            # Shared Chroma path + collection-name helpers
│   │   ├── indexes.py            # get_embed_model() (cached HuggingFace)
│   │   ├── ingestion.py          # IngestionPipeline (PDF / URL / text)
│   │   └── query_engines.py      # VectorStore + SubQuestion engines
│   │
│   ├── tools/
│   │   ├── web_search.py         # Tavily search → list[Source] (singleton client)
│   │   ├── arxiv_tool.py         # arXiv search → list[Source]
│   │   ├── url_fetcher.py        # httpx + BeautifulSoup readability (SSRF-safe)
│   │   ├── pdf_loader.py         # pypdf page chunker
│   │   └── retriever_tool.py     # LlamaIndex query engine wrapper
│   │
│   ├── utils/
│   │   └── security.py           # URL validation, SSRF blocklist, content sanitiser
│   │
│   ├── schemas/
│   │   ├── state.py              # AgentState TypedDict + custom reducers
│   │   ├── query.py              # ResearchQuery, ResearchDepth
│   │   ├── plan.py               # ResearchPlan
│   │   ├── finding.py            # Finding (id · claim · evidence · confidence)
│   │   ├── critique.py           # Critique, CritiqueVerdict
│   │   ├── report.py             # FinalReport, ReportSection, ReportQualityScore
│   │   └── source.py             # Source, SourceType
│   │
│   ├── eval/
│   │   ├── faithfulness.py       # Claim-vs-source faithfulness scorer
│   │   ├── relevance.py          # Sub-question / section relevance scorer
│   │   └── completeness.py       # Coverage completeness scorer
│   │
│   ├── ui/
│   │   ├── sidebar.py            # Provider selector, depth, HITL toggle, uploads
│   │   ├── trace.py              # Live agent trace renderer
│   │   ├── report_view.py        # Report tabs + Markdown/PDF export
│   │   └── sessions_view.py      # Past sessions browser + resume
│   │
│   └── persistence/
│       └── sessions.py           # list_sessions(), get_session_state(), delete_session()
│
├── tests/
│   └── unit/
│       ├── test_schemas.py       # Pydantic schema validation (12 tests)
│       ├── test_tools.py         # Tools layer with mocked network (22 tests)
│       ├── test_graph.py         # Graph nodes, edges, HITL, pipeline (44 tests)
│       ├── test_db.py            # SQLite persistence layer (25 tests)
│       ├── test_rag.py           # RAG ingestion, indexes, query engines (20 tests)
│       └── test_agents.py        # Direct agent function tests (34 tests)
│
└── data/                         # Runtime only — gitignored
    ├── checkpoints/sessions.db   # LangGraph SQLite checkpoint store
    └── sessions/<uuid>/chroma/   # Per-session ChromaDB vector indexes
```

---

## Quick Start

### Prerequisites

- Python 3.11 – 3.14
- [Poetry](https://python-poetry.org/docs/#installation)
- API keys for at least one LLM provider + Tavily (see `.env.example`)

### 1 — Clone and install

```bash
git clone <repo-url>
cd swarm_agent_project
cp .env.example .env          # then edit .env with your API keys
poetry install
```

### 2 — Launch

**Windows** — double-click `start.bat` (checks deps, optionally starts Ollama, opens the browser).

**Any platform:**

```bash
poetry run streamlit run app.py
# → http://localhost:8501
```

---

## Configuration

All settings live in `.env` (loaded by `research_swarm/config.py` via Pydantic `BaseSettings`):

```ini
# LLM providers — fill in whichever you want to use
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key          # required for web search

# Defaults (overridable from the sidebar at runtime)
DEFAULT_MODEL_PROVIDER=anthropic  # anthropic | openai | ollama
DEFAULT_MODEL_NAME=claude-sonnet-4-6
MAX_ITERATIONS=10
MAX_SOURCES=15

# Local Ollama (no API key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Observability (optional)
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
```

---

## Running the App

1. Enter a research topic and select audience + depth in the sidebar.
2. Optionally upload PDFs or paste URLs to build a session-specific RAG index.
3. Click **Start Research** — a live agent trace streams as the swarm works.
4. When complete, switch to the **Report** tab to read, copy, or download.
5. Previous sessions appear in the **Sessions** tab and can be resumed.

---

## Human-in-the-Loop (HITL)

Enable **Human review before writing** in the sidebar. The graph will pause before calling the Writer node:

- The current findings and critic verdicts are displayed.
- You can **Approve & Write** (with optional feedback), **Edit & Re-research** (sends the swarm back to the Researcher), or **Discard Session**.
- Writer-specific feedback travels through a dedicated `writer_instructions` state channel so it never interferes with the Researcher's `human_feedback` channel (which triggers a new research pass).

---

## RAG Layer

When documents are uploaded or URLs are provided before a run, `IngestionPipeline` chunks them and stores embeddings in a **per-session ChromaDB** collection under `data/sessions/<session_id>/`.

Embeddings use `BAAI/bge-small-en-v1.5` (downloaded ~130 MB to `~/.cache/huggingface` on first use — no API key required).

The Researcher calls `retrieve_from_rag` as its first tool to check the local corpus before going to the web.

---

## Session Persistence

Every research run is checkpointed to `data/checkpoints/sessions.db` (SQLite via `AsyncSqliteSaver`). Sessions survive app restarts and can be resumed from the **Sessions** tab. The persistence layer reads directly from the LangGraph checkpoint table schema.

---

## Running Tests

All 157 tests run fully offline — no API keys, no network calls, all LLMs mocked.

```bash
# Full suite
poetry run pytest

# Single file
poetry run pytest tests/unit/test_graph.py -v

# Single test
poetry run pytest tests/unit/test_graph.py::TestHITLInterruptResume -v

# Lint + type-check
poetry run ruff check .
poetry run mypy research_swarm/
```

---

## LLM Providers

Switch providers from the sidebar at runtime, or set defaults in `.env`:

| Provider | Requires | Notes |
|----------|----------|-------|
| `anthropic` | `ANTHROPIC_API_KEY` | Default. Claude Sonnet / Haiku recommended. |
| `openai` | `OPENAI_API_KEY` | GPT-4o / GPT-4o-mini. |
| `ollama` | Ollama running locally | `start.bat` auto-starts `ollama serve`. No API key needed. Slower on CPU. |
