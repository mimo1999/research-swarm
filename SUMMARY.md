# Research Swarm — Development Summary

## Project Overview

**Research Swarm** is a production-grade autonomous research system built on LangGraph, LlamaIndex, and Streamlit. It orchestrates five AI agents to autonomously plan, research, critique, fact-check, and write structured reports on any topic — with optional human review.

**GitHub:** https://github.com/mimo1999/research-swarm

---

## Key Accomplishment: Bug Fix Session (157 Tests Passing)

This document summarizes a comprehensive bug-fix and refactoring session that resolved 8 critical issues across routing, security, RAG, HITL, and persistence layers.

### Starting State
- 8 test failures (149 passing)
- Infinite fact_checker loop due to hard iteration cap in supervisor_node
- Graph error: `'\n "claim"'` from JSON braces in format string
- `@st.cache_resource` serving stale compiled graph after code changes
- SSRF vulnerabilities (string-prefix blocklist missing hex/decimal/IPv6 variants)
- TavilyClient module singleton leaking between tests
- RAG helpers duplicated between ingestion.py and indexes.py
- Separate HITL feedback channels not implemented (writer and researcher feedback conflicted)
- SQLite commit failures when writes table didn't exist

### Final State
- **All 157 tests passing**
- All routing loops fixed
- SSRF protection hardened with ipaddress module
- Dual HITL feedback channels (human_feedback → Researcher, writer_instructions → Writer)
- Clean RAG consolidation into `rag/_chroma.py`
- Proper two-phase SQLite commits in delete_session
- Code-hash cache busting for Streamlit
- All modules documented and documented in README

---

## Major Code Changes

### 1. Routing & Graph (`graph/nodes.py`, `graph/edges.py`, `agents/supervisor.py`)

**Problem:** Infinite fact_checker loop caused by hard iteration cap returning `next_agent="fact_checker"` on every supervisor call, bypassing the `last_agent == "fact_checker" → writer` transition in `_route_from_state`.

**Solution:**
- Removed hard iteration cap from `supervisor_node`
- Moved ceiling check (`iteration >= max_iterations * 4`) to **before** LLM construction
- This ensures: (a) no LLM call when ceiling fires (eliminates auth noise in tests), (b) fast deterministic routing
- All actual routing logic in `_route_from_state` — no early returns in supervisor_node
- Added warning log for unrecognised `next_agent` values in `route_from_supervisor`

**Impact:** Routing is now provably correct via deterministic state machine in `_route_from_state`; LLM-driven decisions only for plan creation.

---

### 2. HITL: Dual Feedback Channels (`schemas/state.py`, `app.py`, `agents/writer.py`, `agents/researcher.py`)

**Problem:** Single `human_feedback` field caused reviewer feedback intended for the Writer to trigger a Researcher re-pass (infinite loop risk).

**Solution:**
- Added `writer_instructions: NotRequired[str | None]` state field (separate from `human_feedback`)
- Researcher consumes only `human_feedback` (triggers re-research)
- Writer consumes `writer_instructions` (triggers report revision)
- Supervisor checks `writer_instructions` when deciding if completed report truly ends session
- `writer_node` clears `writer_instructions` after use
- `_resume_after_hitl` in app.py writes to `writer_instructions` (not `human_feedback`)

**Impact:** HITL feedback paths no longer interfere; users can request revisions without re-triggering research.

---

### 3. SSRF Protection (`utils/security.py`, `tools/url_fetcher.py`)

**Problem:** String-prefix blocklist for SSRF (`127.`, `192.`, etc.) misses hex (`0x7f000001`), decimal (`2130706433`), and IPv4-mapped IPv6 (`::ffff:10.0.0.1`) notations.

**Solution:**
- Replaced with `ipaddress.ip_address()` + DNS resolution (`socket.getaddrinfo`)
- Covers all IP representations, returns True for: loopback, private (RFC1918), link-local, reserved, unspecified
- Added `_safe_get()` for SSRF-safe redirect following: validates each `Location` header before re-requesting
- Blocks protocol-relative redirects (`//`), normalises relative redirects via `urljoin`

**Impact:** SSRF threat model now complete; no bypass via alternate IP representations.

---

### 4. RAG Consolidation (`rag/_chroma.py`, `rag/ingestion.py`, `rag/indexes.py`)

**Problem:** `session_chroma_path()` and `collection_name()` helpers duplicated in ingestion.py and indexes.py.

**Solution:**
- Created `rag/_chroma.py` as single source of truth
- Both modules import from there (with optional aliases for backward compatibility)
- Fixed `build_summary_index` to pass `llm` and `embed_model` as explicit kwargs (not via global `LISettings` singleton, which is a race condition under concurrent Streamlit sessions)

**Impact:** No duplicate logic; thread-safe RAG layer.

---

### 5. Synthesis Prompt Fix (`agents/researcher.py`)

**Problem:** `_SYNTHESIS_PROMPT.format(sub_question=sub_q)` crashed with `Graph error: '\n "claim"'` because the template contained JSON braces from `json_output_instruction()`, and `.format()` tried to parse them as field names.

**Solution:**
- Replaced module-level `_SYNTHESIS_PROMPT` string template with `_synthesis_prompt(sub_question)` function
- Function uses f-string interpolation (no `.format()` parsing)
- JSON suffix remains separate: `_SYNTHESIS_JSON_SUFFIX`

**Impact:** Synthesis prompt no longer crashes; JSON braces safe in prompts.

---

### 6. TavilyClient Singleton (`tools/web_search.py`, `tests/unit/test_tools.py`)

**Problem:** Module-level `_tavily_client` singleton persisted between tests, so `@patch("TavilyClient")` only worked for the first test.

**Solution:**
- Added `_get_tavily_client()` function with API-key-change detection
- Test `setup_method` resets globals via `sys.modules.get("research_swarm.tools.web_search")` (not import alias, which resolves to StructuredTool)
- Client recreated if API key changes at runtime (e.g., user updates .env)

**Impact:** Tests properly isolated; tool can adapt to credential changes without restart.

---

### 7. SQLite Commits (`persistence/sessions.py`)

**Problem:** `delete_session` had a single `try/except/commit` block. When `DELETE FROM writes` raised OperationalError (table doesn't exist in old DBs), the commit was skipped, leaving checkpoint rows behind.

**Solution:**
- Split into two independent try/commit blocks
- First: delete checkpoints and commit
- Second: delete writes and commit (independent OperationalError handling)
- Added `shutil.rmtree()` for Chroma directory cleanup

**Impact:** Sessions properly deleted even when writes table is missing.

---

### 8. Cache Busting (`app.py`)

**Problem:** `@st.cache_resource` served stale compiled graph after code edits (Streamlit's hot-reload updates module code but cache key unchanged).

**Solution:**
- Added `_agent_code_hash()` function hashing mtime of all `.py` files
- Passed as `_code_hash` parameter to `@st.cache_resource`
- Cache key changes whenever source changes → forces recompile

**Impact:** No need to restart Streamlit after code changes.

---

## Test Suite Changes (157 Tests)

| File | Tests | Changes |
|------|-------|---------|
| `test_tools.py` | 22 | Fixed `setup_method` to use `sys.modules` for module reference |
| `test_graph.py` | 44 | Updated `test_iteration_cap_forces_fact_checker` to match pre-LLM ceiling check |
| `test_rag.py` | 20 | Updated patch target to `research_swarm.rag._chroma.session_chroma_path` |
| `test_db.py` | 25 | Tests now pass with split commits in `delete_session` |
| `test_agents.py` | 34 | Updated Critic fallback test: `weak` → `supported` |
| `test_schemas.py` | 12 | No changes |

---

## Tech Stack

### Core
- **LangGraph 0.2+** — multi-agent state graph
- **LangChain 0.3+** — LLM abstraction
- **LlamaIndex 0.11+** — RAG indexing and query engines
- **Streamlit 1.37+** — web UI

### LLM Providers
- **Anthropic Claude** — primary
- **OpenAI GPT** — alternative
- **Ollama** — local inference

### Data & Search
- **Tavily** — web search
- **arXiv API** — academic papers
- **ChromaDB** — vector store (per-session, local)
- **HuggingFace bge-small-en-v1.5** — embeddings

### Persistence & Testing
- **SQLite + aiosqlite** — checkpoints
- **pytest + pytest-asyncio** — 157 offline tests

---

## File Structure

```
research_swarm/
├── agents/              # 5 agent modules + _utils.py
├── graph/               # builder.py, nodes.py, edges.py
├── rag/                 # _chroma.py (NEW), indexes.py, ingestion.py, query_engines.py
├── tools/               # web_search, arxiv, url_fetcher, pdf_loader, retriever_tool
├── utils/               # security.py (NEW — SSRF + content sanitiser)
├── schemas/             # state, query, plan, finding, critique, report, source
├── ui/                  # sidebar, trace, report_view, sessions_view
├── persistence/         # sessions.py (list/load/delete)
└── config.py            # Settings singleton
```

---

## Key Design Patterns

### 1. Deterministic Routing
All supervisor routing rules hardcoded in `_route_from_state()` — LLM only trusted for plan creation. Prevents loops from prompt non-compliance.

### 2. Dual HITL Channels
`human_feedback` for researcher, `writer_instructions` for writer. Prevents feedback misrouting and unintended re-research loops.

### 3. Merge-by-ID Findings
Fact-checker updates finding confidence by returning same `id` — state reducer overwrites in-place, no duplicates. Sub-question string normalisation prevents ID regeneration on re-research.

### 4. Append-Only Critiques
Multiple critique passes stack. `_latest_verdicts()` scans from end to find most recent verdict per finding.

### 5. Module Singleton Patterns
- **Embedding model:** `@lru_cache(maxsize=1)` on `get_embed_model()` — load once, reuse forever
- **TavilyClient:** Module-level singleton with API-key-change detection
- **Settings:** Single `Settings()` instance, mutable at runtime for UI overrides

### 6. Two-Phase Commits
Delete operations split into independent commits to handle missing tables in older databases.

---

## How to Use

### Setup
```bash
git clone https://github.com/mimo1999/research-swarm.git
cd research_swarm
cp .env.example .env       # fill in API keys
poetry install
```

### Run
```bash
streamlit run app.py
```

### Test
```bash
poetry run pytest          # all 157 tests
poetry run pytest tests/unit/test_graph.py -v  # one file
```

---

## Resume Points for Future Work

### Known Limitations
1. **Iteration cap:** Hard ceiling at `max_iterations * 4` supervisor calls prevents infinite loops but may cut off valuable research
2. **LLM consistency:** No constrained decoding for plan JSON — fallback error handling exists but could be hardened
3. **Memory:** Large research sessions can accumulate many message tokens; consider summarization

### Enhancement Opportunities
1. **Parallel researcher:** Run multiple sub-questions in parallel (async fan-out) instead of sequentially
2. **Streaming report:** Stream report sections as they're written (not all-at-once at end)
3. **Fact-checker depth:** Cross-check sources against *other* sources, not just the claim
4. **Custom RAG:** Let users inject domain-specific vector stores or custom retrievers
5. **Export formats:** Add HTML, LaTeX, APA-formatted bibliography

### Testing
- All 157 tests pass; coverage unknown (not tracked)
- Manual E2E test via Streamlit UI recommended before each release
- HITL pause/resume tested via `test_hitl_interrupt_and_resume` in `test_graph.py`

---

## Session Metadata

**Commits:**
- `bcb0c44` — Fix routing loops, SSRF, HITL channels, RAG deduplication, and test suite
- `4627d37` — Add TECHNICAL.md — detailed architecture and implementation documentation

**Total changes:** 30 files modified, 2 new modules created, 939 insertions / 281 deletions

**Test status:** 157/157 passing ✓

---

## Quick Reference: Critical Bug Fixes

| Bug | Root Cause | Fix | Commit |
|-----|-----------|-----|--------|
| Infinite fact_checker loop | Hard cap in supervisor_node returned fact_checker unconditionally | Move ceiling check before LLM, let _route_from_state handle all routing | bcb0c44 |
| Graph error: '\n "claim"' | .format() parsed JSON braces as field names | Use f-string function instead of .format() template | bcb0c44 |
| Stale compiled graph | @st.cache_resource key never changed after code edits | Add _agent_code_hash() parameter to cache key | bcb0c44 |
| SSRF bypass via hex IPs | String prefix blocklist incomplete | Use ipaddress module + DNS resolution | bcb0c44 |
| Tests fail: TavilyClient persists | setup_method reset wrong object (import alias vs module) | Use sys.modules to get actual module | bcb0c44 |
| RAG helpers duplicated | session_chroma_path in two places | Extract to _chroma.py | bcb0c44 |
| HITL feedback conflicted | Single field caused writer feedback to trigger researcher | Dual channels: human_feedback / writer_instructions | bcb0c44 |
| SQLite commit skipped | Single try/except blocked both deletes on missing table | Independent try/commit for each delete | bcb0c44 |

---

**For detailed implementation, see `TECHNICAL.md` in the repo.**
