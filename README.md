# 🔬 Multi-Agent Research Swarm

A LangGraph-based autonomous research system. Give it a topic; a swarm of specialised AI agents researches it, critiques the findings, fact-checks every claim, and writes a structured report - with optional human review before the final draft.

Built with **LangGraph 1.2**, **LlamaIndex**, **ChromaDB**, and **Streamlit**.

---

## Architecture

```
START → supervisor (plan + complexity score)
          ↓
        dispatch_node ──► worker_node × N  (parallel via Send)
          ↑                    ↓
          └── re-research  collect_node (stop-signal check)
                                ↓
                          critic → fact_checker → writer → END
```

**Phase 4 additions:** `dispatch_node` fans out to N parallel workers via LangGraph `Send`, one per sub-question. Each worker carries a role (`academic / industry / skeptic / benchmark / general`) chosen by the supervisor. `collect_node` decides whether to stop (marginal-gain threshold or hard round cap) or dispatch another pass. All non-LLM routing is deterministic - the supervisor LLM is called only once for plan creation.

**State** (`AgentState`) threads through every node as a single `TypedDict`. Custom reducers: `findings` merges by id (fact-checker overwrites in-place); `critiques` appends.

---

## Quick Start

**Requirements:** Python 3.11–3.14, [Poetry](https://python-poetry.org/docs/#installation), at least one LLM API key + Tavily for web search.

```bash
git clone <repo-url> && cd swarm_agent_project
cp .env.example .env          # fill in API keys
poetry install
poetry run streamlit run app.py
# → http://localhost:8501
```

Windows: double-click `start.bat` - checks deps, optionally starts Ollama, opens the browser.

---

## Configuration

```ini
# .env - key settings (see .env.example for full list)
ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY / leave blank for Ollama
TAVILY_API_KEY=...             # required for web search
DEFAULT_MODEL_PROVIDER=ollama  # anthropic | openai | ollama
DEFAULT_MODEL_NAME=gemma4:e2b
DEFAULT_DEPTH=shallow          # shallow | standard | deep
MAX_ITERATIONS=1
MAX_SOURCES=3
MAX_LLM_CALLS=25               # hard budget per session
OLLAMA_BASE_URL=http://localhost:11434
```

All settings are overridable from the sidebar at runtime.

---

## UI

![Landing page - sidebar with model selector, depth, HITL toggle, and document upload](docs/screenshots/01_landing.png)

![Report tab - executive summary, sections with inline citations, and reference list](docs/screenshots/06_report_top.png)

1. Enter a topic, select audience + depth, optionally upload PDFs/URLs for RAG.
2. Click **Start Research** - live agent trace streams as the swarm works.
3. If HITL is on, approve findings before the Writer runs.
4. Switch to **Report** to read, copy, or download. Past sessions live in **Sessions**.

---

## Agents

| Agent | Role |
|---|---|
| **Supervisor** | Creates the research plan with sub-questions, complexity score, and worker-role assignments. Only LLM-invoked once per session. |
| **Workers** (×N) | Parallel ReAct tool loops - each researches one sub-question with a role-specific strategy (academic, industry, skeptic, benchmark, or general). |
| **Critic** | Reviews each finding: `supported / weak / refuted`. Weak/refuted findings trigger another dispatch round (up to the round cap). |
| **Fact-Checker** | Cross-checks claims against source snippets; adjusts confidence scores. Evidence-backed findings are floored at 0.15 so a mis-calibrated model can't zero out a claim that has real sources. |
| **Writer** | Synthesises validated findings into a structured report. Runs a faithfulness check (embedding cosine similarity) and rewrites once if score < 0.25. |

---

## Quality & Safety

| Feature | Detail |
|---|---|
| **Budget guard** | Counts actual LLM calls per session; raises `BudgetExceeded` and forces a graceful writer fallback above `MAX_LLM_CALLS`. |
| **Cross-encoder reranker** | `ms-marco-MiniLM-L-6-v2` (CPU, 22 MB) reranks RAG chunks by relevance before returning to the researcher. |
| **Faithfulness check** | BGE embeddings score each report section against its cited snippets; triggers a rewrite if grounding is too low. |
| **SSRF protection** | URL fetcher validates every hop against a private-IP blocklist; fetched content is sanitised for prompt-injection patterns. |
| **Schema migration** | `migrate_state()` upgrades v0/v1 checkpoints to the current schema on resume - no manual DB work needed. |

---

## Retrieval Quality (BEIR)

RAG retrieval evaluated on three [BEIR](https://github.com/beir-cellar/beir) datasets (seed 42, 100 queries each). Metric: **nDCG@10**.

| Dataset | BM25 ¹ | Contriever ¹ | BGE-Large ¹ | **Ours (BGE-small, dense)** | **Ours (+ reranker)** |
|---|---|---|---|---|---|
| SciFact | 0.678 | 0.677 | 0.752 | **0.749** | 0.696 |
| NFCorpus | 0.321 | 0.328 | 0.381 | **0.341** | 0.324 |
| ArguAna | 0.397 | 0.446 | 0.416 | **0.391** | 0.391 |

¹ Published baselines from the [BEIR paper](https://arxiv.org/abs/2104.08663) and [Resources for Brewing BEIR](https://arxiv.org/abs/2306.07471). BGE-Large is `bge-large-en-v1.5`; our embedder is the much smaller `bge-small-en-v1.5` (~130 MB vs ~1.3 GB).

The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) is skipped for queries longer than 8 words - these scientific datasets have long queries so the guard fires often and the dense-only score is the operative result. See [`benchmarks/README.md`](benchmarks/README.md) for the full per-dataset breakdown.

---

## Development

```bash
# Run all 181 tests (fully offline - all LLMs mocked)
poetry run pytest

# Specific test file
poetry run pytest tests/unit/test_graph.py -v

# Golden regression set (3 topics, full pipeline)
poetry run pytest tests/golden/ -v

# Lint / type-check
poetry run ruff check .
poetry run mypy research_swarm/

# Run a live research job without Streamlit
poetry run python run_research.py "your topic here"
```

---

## LLM Providers

| Provider | Requires | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude Haiku (fast tier) / Sonnet (standard) / Opus (thorough). |
| `openai` | `OPENAI_API_KEY` | GPT-4o-mini / GPT-4o. |
| `ollama` | Ollama running locally | No API key. `start.bat` auto-starts `ollama serve`. Use `gemma4:e2b` for stability. |
