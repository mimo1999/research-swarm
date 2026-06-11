# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Streamlit app
streamlit run app.py

# Install dependencies (Python 3.11+, uses Poetry)
poetry install

# Run all tests (80 unit tests, no API keys needed — all LLMs mocked)
poetry run pytest

# Run a single test file
poetry run pytest tests/unit/test_graph.py -x -q

# Run a single test by name
poetry run pytest tests/unit/test_graph.py::test_supervisor_node_routes_to_researcher -x

# Lint
poetry run ruff check .

# Type-check
poetry run mypy research_swarm/
```

Copy `.env.example` → `.env` and fill in API keys before running.

## Architecture

This is a **LangGraph multi-agent system** with a Streamlit frontend. Five async agents cycle through a state graph until a final report is produced.

### Graph flow

```
START → supervisor (router)
          ├─→ researcher  → supervisor
          ├─→ critic      → supervisor
          ├─→ fact_checker→ supervisor
          └─→ writer      → END
                ↑
           (interrupt_before if HITL enabled)
```

`graph/builder.py` compiles the `StateGraph`. `graph/edges.py::route_from_supervisor` reads `state["next_agent"]` to pick the next node. All nodes are `async def` in `graph/nodes.py` — the graph **must** be driven via `graph.astream()` (not `graph.stream()`).

### AgentState (`schemas/state.py`)

Single `TypedDict` threaded through all nodes. Two custom reducers:
- `findings`: **merge-by-id** — the fact-checker overwrites findings by matching `id`, avoiding duplicates.
- `critiques`: **append-only** — each critic pass accumulates.

### Agent LLM (`agents/base.py`)

`get_agent_llm()` returns a LangChain `BaseChatModel`. Three providers:
- `"anthropic"` → `ChatAnthropic`
- `"openai"` → `ChatOpenAI`
- `"ollama"` → `ChatOllama` (local, from `langchain-ollama`)

Provider/model are stored in `settings.default_model_provider` / `settings.default_model_name` and pushed from the sidebar on each research run via `_apply_ui_settings()` in `app.py`.

### RAG layer (`rag/`)

All computation is local — no cloud required:
- **Embeddings**: `HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")`, cached via `lru_cache`. Downloads ~130 MB to `~/.cache/huggingface` on first use.
- **Vector store**: ChromaDB, persisted per-session at `data/sessions/{session_id}/chroma/`.
- **Query engine** (`rag/query_engines.py`): if Ollama is reachable → `SubQuestionQueryEngine(RouterQueryEngine(vector, summary))`; otherwise → plain vector engine with `response_mode="no_text"`.
- **Ingestion** (`rag/ingestion.py`): `IngestionPipeline` handles PDFs, URLs, and raw text. Called from `app.py` before the graph starts.

### Persistence

`SqliteSaver` checkpoint at `data/checkpoints/sessions.db`. `persistence/sessions.py` reads directly from that table to list/delete sessions in the UI. `get_session_state()` re-builds the graph to load a snapshot.

### Streamlit app (`app.py`)

- `nest_asyncio.apply()` at startup lets `asyncio.run()` work inside Streamlit's event loop.
- `@st.cache_resource` caches one compiled graph per HITL setting (True/False).
- `_stream_graph()` drives `graph.astream()` via `asyncio.run()`, rendering each node update live.
- HITL: `interrupt_before=["writer"]` at compile time. After stream ends, `snapshot.next` non-empty means paused. Resume via `graph.update_state(config, {"human_feedback": ...})` then re-call `_stream_graph(graph, None, config, ...)`.

### Key config (`config.py`)

`Settings` is a Pydantic `BaseSettings` singleton. Mutable at runtime — `_apply_ui_settings()` in `app.py` overwrites `default_model_provider`, `default_model_name`, `max_sources`, and (for Ollama) `ollama_base_url` before each graph run.

## Testing

All 80 tests are offline. LLM calls are mocked with `AsyncMock`:
```python
mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=expected_output)
```

Node patchability: `graph/builder.py` imports nodes as `import research_swarm.graph.nodes as _nodes` and registers them as `_nodes.supervisor_node` — evaluated at `build_graph()` call time, so `unittest.mock.patch("research_swarm.graph.nodes.supervisor_node", ...)` works correctly before calling `build_graph()`.

`pytest.ini_options` sets `asyncio_mode = "auto"` so all `async def test_*` functions run without decorators.
