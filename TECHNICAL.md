# Research Swarm — Technical Documentation

> A deep-dive into the architecture, design decisions, and implementation details of the multi-agent research system.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture](#3-system-architecture)
4. [Agent State Machine](#4-agent-state-machine)
5. [Agent Modules](#5-agent-modules)
   - [Supervisor](#51-supervisor)
   - [Researcher](#52-researcher)
   - [Critic](#53-critic)
   - [Fact-Checker](#54-fact-checker)
   - [Writer](#55-writer)
6. [Graph Layer](#6-graph-layer)
7. [Tools Layer](#7-tools-layer)
8. [RAG Layer](#8-rag-layer)
9. [Schemas & Data Models](#9-schemas--data-models)
10. [Persistence Layer](#10-persistence-layer)
11. [Security Module](#11-security-module)
12. [Configuration](#12-configuration)
13. [Streamlit UI](#13-streamlit-ui)
14. [Human-in-the-Loop (HITL)](#14-human-in-the-loop-hitl)
15. [Test Suite](#15-test-suite)
16. [Data Flow — End to End](#16-data-flow--end-to-end)

---

## 1. Problem Statement

Manual research is slow, siloed, and inconsistent. A researcher must:

- Query multiple sources (web, academic papers, internal documents)
- Assess source credibility
- Cross-check claims for contradictions
- Synthesise evidence into coherent findings
- Write a structured report

This process can take hours or days for a thorough topic. It also suffers from human bias — a researcher might stop searching too early, fail to verify a claim, or overlook contradictory evidence.

**Research Swarm** solves this by delegating the entire pipeline to a coordinated team of specialised AI agents that work autonomously, check each other's work, and produce a fully-referenced report — in minutes.

---

## 2. Solution Overview

The system orchestrates five AI agents in a directed graph loop:

| Step | Who | What |
|------|-----|-------|
| 1 | **Supervisor** | Analyses the query, decomposes it into sub-questions, routes to agents |
| 2 | **Researcher** | Searches the web, arXiv, and a local RAG index; produces evidence-backed findings |
| 3 | **Critic** | Evaluates each finding: `supported`, `weak`, or `refuted` |
| 4 | **Researcher** *(again)* | Re-researches weak/refuted findings (loop, up to `max_iterations`) |
| 5 | **Fact-Checker** | Cross-checks all claims against their cited source snippets; adjusts confidence |
| 6 | **Writer** | Synthesises validated findings into a structured Markdown report |

Every step is checkpointed to SQLite so sessions survive restarts and can be resumed. An optional **Human-in-the-Loop** interrupt lets a user review findings and provide feedback before the Writer runs.

---

## 3. System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Sidebar  │  │  Agent Trace │  │ Report View  │  │  Sessions  │  │
│  └──────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ graph.astream()
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph StateGraph                            │
│                                                                      │
│   START ──► supervisor ◄────────────────────────────────────────┐   │
│                │                                                 │   │
│       ┌────────┼────────────────────────────────┐               │   │
│       ▼        ▼        ▼                        ▼               │   │
│  researcher  critic  fact_checker              writer ──► END    │   │
│       │        │        │                        ▲               │   │
│       └────────┴────────┴────────────────────────┘               │   │
│                         (all return to supervisor)                │   │
│                                                                      │
│  Checkpointer: AsyncSqliteSaver ──► data/checkpoints/sessions.db     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │  Tools   │  │   RAG    │  │ LLM APIs │
          │ web/arxiv│  │ ChromaDB │  │Anthropic/│
          │ url/pdf  │  │ LlamaIdx │  │OpenAI/   │
          └──────────┘  └──────────┘  │Ollama    │
                                      └──────────┘
```

### Request Lifecycle

```
User enters query
      │
      ▼
app.py builds initial AgentState
      │
      ▼
graph.astream() drives async node loop
      │
      ├─► Supervisor calls LLM → creates ResearchPlan
      │
      ├─► Researcher runs ReAct tool loop (up to 6 turns/sub-question)
      │       └─► Sources ingested into ChromaDB via IngestionPipeline
      │
      ├─► Critic reviews findings → assigns verdicts
      │
      ├─► [Loop back to Researcher if weak/refuted findings remain]
      │
      ├─► Fact-Checker cross-checks claims → adjusts confidence
      │
      ├─► [HITL interrupt — user can review/reject/provide feedback]
      │
      └─► Writer produces FinalReport → streamed to UI
```

---

## 4. Agent State Machine

All agents share a single `AgentState` TypedDict. It is the only thing passed between nodes — no global mutable state, no shared objects.

### AgentState Fields

```python
class AgentState(TypedDict):
    # Conversation history — built-in add_messages reducer (append-only)
    messages:            Annotated[list, add_messages]

    # Core objects
    query:               ResearchQuery | None
    plan:                ResearchPlan  | None

    # Custom reducers:
    findings:            Annotated[list[Finding], _merge_findings]  # merge-by-id
    critiques:           Annotated[list[Critique], _add_list]       # append-only

    # Reports
    draft_report:        FinalReport | None
    final_report:        FinalReport | None

    # HITL channels (separate to avoid cross-agent interference)
    human_feedback:      str | None   # → consumed by Researcher
    writer_instructions: str | None   # → consumed by Writer

    # Routing
    iteration_count:     int
    next_agent:          AgentName | None

    # Identity
    session_id:          str
    model_provider:      str          # NotRequired
    model_name:          str          # NotRequired
```

### Custom Reducers

LangGraph applies reducers when merging partial state updates from nodes:

**`_merge_findings` (merge-by-id)**
```
existing = [Finding(id="A", confidence=0.4), Finding(id="B", confidence=0.7)]
new      = [Finding(id="A", confidence=0.9)]   # fact-checker updated A
result   = [Finding(id="A", confidence=0.9), Finding(id="B", confidence=0.7)]
```
This allows the Fact-Checker to overwrite a finding's confidence in-place without duplicating it.

**`_add_list` (append-only)**
```
existing = [Critique(finding_id="A", verdict="weak")]
new      = [Critique(finding_id="A", verdict="supported")]
result   = [Critique(...weak), Critique(...supported)]
```
Multiple critique passes accumulate. The agent utilities use `_latest_verdicts()` to find the most recent verdict per finding by scanning from the end.

### Routing State Machine

```
                    ┌─────────────────────────────┐
                    │         supervisor           │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     plan == None          findings present         final_report
              │            critiques present         present
              ▼                    │                    │
      [LLM: create plan]           │             ──► END
      next_agent=researcher        │
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
      last==researcher     weak/refuted exist    last==fact_checker
              ▼            & iter ≤ max               ▼
           critic               │                   writer
                                ▼
                           researcher (re-research)
                                │
                         (all supported OR
                          iter > max_iterations)
                                ▼
                          fact_checker
```

---

## 5. Agent Modules

### 5.1 Supervisor

**File:** `research_swarm/agents/supervisor.py`

**Role:** Central intelligence — reads the full research state and decides which agent acts next.

**Two-path design:**

The Supervisor has a **deterministic fast-path** and an **LLM slow-path**:

```
run_supervisor(state, llm)
      │
      ▼
_route_from_state(state)
      │
      ├── Returns SupervisorDecision? ──► return immediately (no LLM call)
      │         (ceiling hit, report done,
      │          feedback pending, no findings,
      │          last_agent routing, verdict checks)
      │
      └── Returns None? ──► call LLM to create initial ResearchPlan
```

**Deterministic routing rules (in priority order):**

| Condition | Route |
|-----------|-------|
| `iteration_count >= max_iterations * 4` | `fact_checker` (hard ceiling) |
| `final_report` exists AND no `writer_instructions` pending | `end` |
| `human_feedback` set AND last agent ≠ researcher | `researcher` |
| `plan is None` | `None` → LLM creates plan |
| `findings` is empty | `researcher` |
| `last_agent == researcher` | `critic` |
| Unreviewed findings exist | `critic` |
| `last_agent == fact_checker` | `writer` |
| Weak/refuted findings AND `iteration <= max_iterations` | `researcher` |
| Otherwise | `fact_checker` |

**Why this matters:** Keeping routing logic deterministic prevents the LLM from creating infinite loops by returning unexpected `next_agent` values. The LLM is only trusted for *plan creation*.

**Hard ceiling check in `supervisor_node`:**

The ceiling check (`iteration >= max_iterations * 4`) fires **before** the LLM object is even constructed, so it never causes authentication noise in tests or logs when no API key is present.

---

### 5.2 Researcher

**File:** `research_swarm/agents/researcher.py`

**Role:** Answers each sub-question by calling tools in a ReAct loop, then synthesises a `Finding`.

**Implementation:**

```
run_researcher(state, llm, tools)
      │
      ├── Determine which sub-questions need research:
      │     - Not yet researched
      │     - Previously marked weak/refuted by Critic
      │     - (Skip sub-questions already marked supported)
      │
      └── For each target sub-question:
            │
            ├── _run_tool_loop()  ← up to MAX_TOOL_TURNS=6 rounds
            │     │
            │     ├── llm.bind_tools(tools).ainvoke(messages)
            │     ├── Execute each tool_call → append ToolMessage
            │     └── Stop when LLM returns no tool calls
            │
            ├── _extract_sources_from_messages()
            │     └── Parse Source dicts out of ToolMessage JSON
            │
            └── Synthesis pass:
                  llm.with_structured_output(FindingSynthesis).ainvoke(...)
                  → Finding(id, claim, confidence, evidence, sub_question)
```

**Re-research ID preservation:**

When re-researching a weak finding, the Researcher looks up the existing `Finding.id` by matching the `sub_question` string (normalised: lowercased, stripped). This means the fact-checker's updated version overwrites the original via the `_merge_findings` reducer rather than creating a duplicate.

**Tools available to the Researcher:**

| Tool | Source | Notes |
|------|--------|-------|
| `retrieve_from_rag` | Local ChromaDB | Checked first — free, fast |
| `web_search` | Tavily API | Up to 20 results per call |
| `arxiv_search` | arXiv API | Academic papers with PDF links |
| `fetch_url` | httpx + BS4 | Full-text extraction, SSRF-safe |

---

### 5.3 Critic

**File:** `research_swarm/agents/critic.py`

**Role:** Assigns a quality verdict to every un-reviewed Finding.

**Verdicts:**

| Verdict | Meaning |
|---------|---------|
| `supported` | Claim is well-evidenced — skip in next research pass |
| `weak` | Claim needs more or stronger evidence — re-research |
| `refuted` | Evidence contradicts the claim — excluded from report |

**Implementation:**

```python
for finding in to_review:
    critique = await llm.with_structured_output(Critique).ainvoke([
        SystemMessage(_SYSTEM_PROMPT),
        HumanMessage(finding_summary),
    ])
    # Force correct finding_id regardless of LLM hallucination
    critique = critique.model_copy(update={"finding_id": fid})
```

**LLM failure fallback:**

On any exception, the Critic falls back to `CritiqueVerdict.supported` rather than `weak`. This is intentional: an LLM outage should not trigger expensive re-research loops that burn tokens and can cause the session to stall.

**`_latest_verdicts()` utility:**

Since critiques are append-only, the same finding can have multiple critiques across passes. `_latest_verdicts()` scans from the end to get the most recent verdict per `finding_id`:

```python
def _latest_verdicts(critiques) -> dict[str, str]:
    result = {}
    for c in critiques:           # iterate oldest-to-newest
        result[c.finding_id] = c.verdict
    return result                 # last write wins
```

---

### 5.4 Fact-Checker

**File:** `research_swarm/agents/fact_checker.py`

**Role:** Independently verifies each claim against its cited source snippets and adjusts the confidence score.

**Implementation:**

```python
for finding in to_check:  # excludes refuted findings
    if not finding.evidence:
        # No sources → penalise to 0.1
        continue

    result = await llm.with_structured_output(FactCheckResult).ainvoke([
        SystemMessage(_SYSTEM_PROMPT),
        HumanMessage(claim + top_5_sources),
    ])
    # Return updated Finding (same id, new confidence)
    # _merge_findings reducer overwrites the original in state
```

**Why it's separate from the Critic:**

The Critic evaluates *sufficiency* of evidence (is there enough?). The Fact-Checker evaluates *accuracy* (does the evidence actually support the specific claim?). These are different questions and benefit from separate prompts and LLM calls.

**Scoring guide in prompt:**

- `0.0` — Completely unsupported / contradicted by sources
- `0.5` — Partially supported
- `1.0` — Fully corroborated by multiple independent sources

Findings with `confidence < 0.3` are excluded from the Writer's input.

---

### 5.5 Writer

**File:** `research_swarm/agents/writer.py`

**Role:** Synthesises all validated findings into a structured `FinalReport`.

**Implementation:**

```python
# 1. Filter out refuted and low-confidence findings
valid_findings = [f for f in findings
                  if f.id not in refuted_ids and f.confidence >= 0.3]

# 2. Deduplicate sources across all findings
references = _collect_references(valid_findings)
ref_index  = {url: i+1 for i, url in enumerate(references)}

# 3. Format findings with [N] citation markers
findings_text = _format_findings(valid_findings, ref_index)

# 4. Generate structured report
report = await llm.with_structured_output(FinalReport).ainvoke([...])
```

**HITL feedback channel:**

The Writer reads from `writer_instructions` first, falling back to `human_feedback` for backward compatibility with older checkpoints. This separation prevents reviewer feedback intended for the Writer from accidentally triggering a new Researcher pass.

**Fallback report:**

If the LLM structured output fails, the Writer constructs a minimal report from the raw findings rather than crashing the session.

---

## 6. Graph Layer

### `graph/builder.py` — Graph Assembly

```python
sg = StateGraph(AgentState)

# Nodes registered by module reference — patches applied before
# build_graph() are picked up correctly by unittest.mock
sg.add_node("supervisor",   _nodes.supervisor_node)
sg.add_node("researcher",   _nodes.researcher_node)
sg.add_node("critic",       _nodes.critic_node)
sg.add_node("fact_checker", _nodes.fact_checker_node)
sg.add_node("writer",       _nodes.writer_node)

sg.add_edge(START, "supervisor")
sg.add_conditional_edges("supervisor", route_from_supervisor, {...})
sg.add_edge("researcher",   "supervisor")
sg.add_edge("critic",       "supervisor")
sg.add_edge("fact_checker", "supervisor")
sg.add_edge("writer",       END)

graph = sg.compile(
    checkpointer=AsyncSqliteSaver,
    interrupt_before=["writer"],   # HITL
)
```

### `graph/nodes.py` — Async Node Wrappers

Each node function follows the same pattern:

```python
async def researcher_node(state: AgentState) -> dict[str, Any]:
    llm = _get_state_llm(state)        # constructs ChatModel from state settings
    new_findings = await run_researcher(state, llm, tools)
    return {
        "findings":       new_findings,
        "human_feedback": None,         # consume feedback to prevent re-trigger
        "messages":       [AIMessage(...)],
    }
```

### `graph/edges.py` — Conditional Routing

```python
_ROUTING = {
    "researcher":   "researcher",
    "critic":       "critic",
    "fact_checker": "fact_checker",
    "writer":       "writer",
    "human":        "writer",    # HITL: graph pauses before writer
    "end":          END,
}

def route_from_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent")
    if next_agent not in _ROUTING:
        logger.warning("Unrecognised next_agent=%r -- routing to END", next_agent)
    return _ROUTING.get(str(next_agent), END)
```

---

## 7. Tools Layer

### `tools/web_search.py` — Tavily Web Search

```
web_search(query, max_results=5) → list[dict]
```

- Uses a **module-level TavilyClient singleton** to avoid re-constructing the client on every call. The singleton is invalidated and recreated if the API key changes at runtime.
- Applies a **heuristic credibility score** based on domain:

| Domain Pattern | Score |
|----------------|-------|
| `.edu`, `.gov`, `.org`, `arxiv.org`, `nature.com` | 0.85 |
| `reddit.com`, `twitter.com`, `x.com`, `quora.com` | 0.30 |
| Everything else | 0.60 |

- On error, returns a single error Source with `credibility_score=0.0` rather than raising, so the tool loop continues.

---

### `tools/arxiv_tool.py` — arXiv Academic Search

```
arxiv_search(query, max_results=5, fetch_pdf_text=False) → list[dict]
```

- Uses the `arxiv` Python client to search preprints.
- Abstracts are truncated to 600 characters for the snippet field.
- Optional `fetch_pdf_text=True` downloads and extracts the first page of the PDF.
- All arXiv results receive `credibility_score=0.85`.

---

### `tools/url_fetcher.py` — SSRF-Safe URL Fetcher

```
fetch_url(url, max_chars=3000) → dict
```

**Content extraction pipeline:**

```
httpx._safe_get(url)
      │
      ├── validate_url(url) → SSRF check (see security.py)
      ├── follow_redirects=False → manual redirect handling
      │     └── validate each Location header before following
      │
      └── resp.text
            │
            ├── Check Content-Type → PDF? return descriptive note
            │
            └── BeautifulSoup parsing:
                  1. Try <article> tag first
                  2. Fall back to <body>
                  3. Strip nav/header/footer/script/style
                  4. Extract .get_text(), truncate to max_chars
```

**Redirect safety (`_safe_get`):**

```python
def _safe_get(url, _hops=0):
    if _hops > 5:
        raise ValueError("Too many redirects")
    resp = httpx.get(url, follow_redirects=False)
    if resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers["location"]
        if location.startswith("//"):       # block protocol-relative
            raise ValueError(...)
        validate_url(location)              # SSRF check on redirect target
        return _safe_get(location, _hops+1)
    return resp
```

---

### `tools/pdf_loader.py` — PDF Page Chunker

```
load_pdf(file_path, max_pages=50) → dict
```

- Uses `pypdf` to extract text page-by-page.
- Returns a list of `PDFChunk` objects (`{page, text, char_count}`).
- Returns an error dict (not an exception) if the file is missing.

---

### `tools/retriever_tool.py` — RAG Retriever

```
retriever_stub(query, session_id, top_k=5) → list[dict]
make_retriever_tool(engine_factory) → StructuredTool
```

- `retriever_stub` is the default tool used when no vector index exists.
- `make_retriever_tool` wraps a LlamaIndex query engine into a LangChain `StructuredTool`.
- Results include `url`, `title`, `snippet`, `credibility_score` from node metadata and similarity score.

---

## 8. RAG Layer

### Overview

The RAG (Retrieval-Augmented Generation) layer gives the Researcher access to documents you provide (PDFs, URLs) before the graph runs. All computation is local — no cloud embedding service required.

```
User uploads PDF / pastes URL
          │
          ▼
    IngestionPipeline
          │
    ┌─────┴──────────────────┐
    │                        │
    ▼                        ▼
 Text chunking          HuggingFace embeddings
 (512 tokens,           (BAAI/bge-small-en-v1.5)
  50 overlap)                │
                             ▼
                       ChromaDB (per-session)
                       data/sessions/{id}/chroma/
                             │
                             ▼
               VectorStoreIndex (LlamaIndex)
                             │
                    ┌────────┴─────────┐
                    │                  │
             Researcher            Summary Index
             retrieve_from_rag    (when Ollama up)
```

### `rag/_chroma.py` — Shared Helpers

Single source of truth for the ChromaDB path and collection name. Both `ingestion.py` and `indexes.py` import from here to avoid duplication:

```python
def session_chroma_path(session_id: str) -> Path:
    path = settings.sessions_dir / session_id / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path

def collection_name(session_id: str) -> str:
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in session_id)
    return f"rs-{safe}"[:63]   # ChromaDB max name length
```

### `rag/ingestion.py` — IngestionPipeline

```python
pipeline = IngestionPipeline(session_id)

pipeline.ingest_text(text, metadata, embed_model)
pipeline.ingest_source_dict(source_dict, embed_model)
pipeline.ingest_source_dicts(sources, embed_model)
pipeline.ingest_url(url, embed_model)
pipeline.ingest_pdf(file_path, embed_model)
```

**Error snippet filtering:**

Snippets starting with `[` (error messages from `url_fetcher` and `pdf_loader`) are silently skipped — they add noise with no informational value.

**VectorStoreIndex creation:**

```python
VectorStoreIndex.from_documents(
    docs,
    storage_context=StorageContext.from_defaults(vector_store=chroma_store),
    embed_model=embed_model,
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)],
)
```

### `rag/indexes.py` — Index Loaders

```python
@functools.lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    """Load BAAI/bge-small-en-v1.5 once; cache forever."""
    return HuggingFaceEmbedding(model_name=settings.embed_model_name)

def load_vector_index(session_id: str) -> VectorStoreIndex:
    """Connect to existing ChromaDB — no re-embedding."""
    ...

def build_summary_index(session_id, llm, documents=None) -> SummaryIndex:
    """In-memory summary index; llm and embed_model passed explicitly
    to avoid mutating the global LISettings singleton (race condition)."""
    return SummaryIndex.from_documents(docs, llm=llm, embed_model=get_embed_model())
```

### `rag/query_engines.py` — Engine Selection

The query engine adapts to whether Ollama is running:

```
probe_ollama()
      │
      ├── Ollama UP ──► SubQuestionQueryEngine(
      │                     RouterQueryEngine(
      │                         VectorStoreQueryEngine,
      │                         SummaryQueryEngine
      │                     )
      │                 )
      │
      └── Ollama DOWN ──► VectorStoreIndex.as_query_engine(
                              response_mode="no_text"
                          )
                          (returns source nodes only, no synthesis)
```

`probe_ollama()` is a lightweight HTTP GET to `http://localhost:11434` with a short timeout, cached per invocation.

---

## 9. Schemas & Data Models

All schemas use **Pydantic v2**, providing automatic validation, JSON serialisation, and `model_copy()` for immutable updates.

### `ResearchQuery`

```python
class ResearchQuery(BaseModel):
    topic:      str
    audience:   str = "general public"
    depth:      ResearchDepth = ResearchDepth.standard
    max_sources: int = 15
```

`ResearchDepth` is an enum: `quick | standard | deep`.

### `ResearchPlan`

```python
class ResearchPlan(BaseModel):
    sub_questions:  list[str]    # e.g. ["What is X?", "How does Y affect Z?"]
    strategy:       str
    required_tools: list[str]
```

### `Finding`

```python
class Finding(BaseModel):
    id:           str            # UUID, stable across re-research passes
    claim:        str            # One or two sentence factual claim
    evidence:     list[Source]   # Cited sources
    confidence:   float          # 0.0–1.0, updated by Fact-Checker
    sub_question: str            # Which plan sub-question this answers
```

### `Source`

```python
class Source(BaseModel):
    url:               str
    title:             str
    snippet:           str
    source_type:       SourceType     # web | arxiv | pdf | retriever
    credibility_score: float
    retrieved_at:      datetime
```

### `Critique`

```python
class Critique(BaseModel):
    finding_id:        str
    verdict:           CritiqueVerdict    # supported | weak | refuted
    reasoning:         str
    suggested_followup: str
```

### `FinalReport`

```python
class FinalReport(BaseModel):
    title:        str
    exec_summary: str
    sections:     list[ReportSection]
    references:   list[Source]
    methodology:  str
    limitations:  str
    quality:      ReportQualityScore | None
```

---

## 10. Persistence Layer

### `persistence/sessions.py`

The persistence layer reads directly from LangGraph's SQLite checkpoint table — no ORM, no migrations.

**`list_sessions()`**

```sql
SELECT
    thread_id,
    MIN(created_at) AS first_seen,
    MAX(created_at) AS last_seen,
    COUNT(*)        AS steps
FROM checkpoints
GROUP BY thread_id
ORDER BY last_seen DESC
```

Returns a list of `SessionSummary` dataclasses.

**`get_session_state(thread_id)`**

Loads the latest graph snapshot by rebuilding the graph with `AsyncSqliteSaver` and calling `graph.aget_state()`. Handles both event-loop contexts:

- **Streamlit** (loop is running + `nest_asyncio`): uses `run_coroutine_threadsafe()`
- **Regular Python**: uses `asyncio.run()`

**`delete_session(thread_id)`**

```python
# Two independent commits — missing 'writes' table doesn't
# prevent checkpoints from being deleted
try:
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", ...)
    conn.commit()
except OperationalError: pass

try:
    conn.execute("DELETE FROM writes WHERE thread_id = ?", ...)
    conn.commit()
except OperationalError: pass

# Also removes the Chroma vector store from disk
shutil.rmtree(settings.sessions_dir / thread_id)
```

### Checkpoint Schema (LangGraph)

```
Table: checkpoints
  thread_id            TEXT  (session identifier)
  checkpoint_ns        TEXT  (namespace, usually '')
  checkpoint_id        TEXT  (UUID per step)
  parent_checkpoint_id TEXT
  type                 TEXT
  checkpoint           BLOB  (serialised state)
  metadata             BLOB
  created_at           TEXT
```

---

## 11. Security Module

**File:** `research_swarm/utils/security.py`

### SSRF Protection

Server-Side Request Forgery (SSRF) happens when user-controlled URLs cause the server to make requests to internal infrastructure. The Researcher calls `fetch_url` on URLs found in search results — a malicious page could embed `http://169.254.169.254/latest/meta-data/` (AWS metadata service) or `http://127.0.0.1:8080/admin`.

**`_is_private_ip(host)`** uses Python's `ipaddress` module after DNS resolution to catch all representations:

```python
def _is_private_ip(host: str) -> bool:
    # Strip IPv6 brackets e.g. [::1]
    stripped = host.strip("[]")

    # Try direct parse first (catches hex: 0x7f000001,
    # decimal: 2130706433, IPv4-mapped IPv6: ::ffff:10.0.0.1)
    try:
        addr = ipaddress.ip_address(stripped)
        return addr.is_loopback or addr.is_private or ...
    except ValueError:
        pass

    # DNS resolve and check each returned address
    try:
        for info in socket.getaddrinfo(stripped, None):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_loopback or addr.is_private or ...:
                return True
    except OSError:
        pass

    return False
```

Blocked address types: loopback, private (RFC1918), link-local, reserved, unspecified.

**`validate_url(url)`** also enforces `https://` or `http://` scheme only — blocks `file://`, `ftp://`, `data:`, etc.

### Content Sanitisation

**`sanitize_fetched_content(text)`** strips control characters and normalises whitespace from text returned by external tools before it enters the LLM context. Applied to all web search snippets and arXiv abstracts.

---

## 12. Configuration

**File:** `research_swarm/config.py`

A single `Settings` singleton (Pydantic `BaseSettings`) loaded from `.env`:

```python
class Settings(BaseSettings):
    # API keys — stored as SecretStr (masked in logs and repr)
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key:    SecretStr = SecretStr("")
    tavily_api_key:    SecretStr = SecretStr("")

    # Defaults (overridable from the sidebar at runtime)
    default_model_provider: str = "anthropic"
    default_model_name:     str = "claude-sonnet-4-6"
    max_iterations:         int = 10
    max_sources:            int = 15

    # RAG
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    chunk_size:       int = 512
    chunk_overlap:    int = 50

    # Ollama
    ollama_base_url:   str = "http://localhost:11434"
    ollama_model:      str = "gemma4:e2b"
    ollama_deployment: str = "local"   # "local" | "cloud"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"
```

Settings are **mutated at runtime** by `_apply_ui_settings()` in `app.py` each time the user clicks Start Research, so the sidebar provider/model selection overrides the `.env` defaults without requiring a restart.

---

## 13. Streamlit UI

**File:** `app.py`

### Key Design Decisions

**`nest_asyncio.apply()`** at startup: Streamlit runs its own event loop. LangGraph's `AsyncSqliteSaver` requires an async context. `nest_asyncio` patches the loop to allow nested `asyncio.run()` calls — the standard pattern for async-in-sync bridging in Streamlit apps.

**`@st.cache_resource` with code hash:** The compiled graph is cached to avoid recompiling on every interaction. A `_agent_code_hash()` function hashes the modification times of all `.py` files and passes it as a cache key parameter — forcing recompilation when source code changes without requiring a server restart.

```python
@st.cache_resource
def _get_graph(hitl_enabled: bool, _code_hash: str):
    ...
    return asyncio.run(_build())
```

**Live streaming:** `graph.astream()` yields state dicts as each node completes. The UI renders each update immediately:

```python
async for chunk in graph.astream(initial_state, config):
    for node_name, node_output in chunk.items():
        render_node_update(node_name, node_output)
```

### UI Modules

| Module | Responsibility |
|--------|---------------|
| `ui/sidebar.py` | LLM provider selector, depth picker, HITL toggle, PDF/URL upload, ingestion progress |
| `ui/trace.py` | Renders per-node updates as expandable cards in the live trace panel |
| `ui/report_view.py` | Tabbed view: Executive Summary, Sections, References, Quality Score, Markdown/PDF export |
| `ui/sessions_view.py` | Past sessions browser, resume button, delete button |

---

## 14. Human-in-the-Loop (HITL)

HITL is implemented using LangGraph's `interrupt_before` compile option:

```python
graph = sg.compile(
    checkpointer=checkpointer,
    interrupt_before=["writer"],
)
```

When the Supervisor routes to `writer`, the graph **pauses** before executing the node. The checkpoint is saved. `snapshot.next` is non-empty, indicating a pending interrupt.

### HITL State Channels

Two separate state fields prevent cross-agent feedback pollution:

```
human_feedback      → consumed by Researcher (triggers re-research pass)
writer_instructions → consumed by Writer (triggers report revision)
```

If both shared a single field, approving the Writer's output with a comment would also trigger the Researcher to do another pass — an unintended loop.

### HITL Flow

```
graph.astream() ends with snapshot.next == ("writer",)
                │
                ▼
         User sees findings + verdicts
                │
         ┌──────┴────────────────────────┐
         │                               │
  "Approve & Write"              "Edit & Re-research"
  (optional comment)                     │
         │                               ▼
         ▼                    graph.update_state(config, {
graph.update_state(config, {     "next_agent": None,
  "writer_instructions": text,   "human_feedback": feedback,
})                              })
         │                               │
         └──────────────┬────────────────┘
                        ▼
              graph.astream(None, config)  ← resume
```

---

## 15. Test Suite

157 tests, all fully offline. No API keys, no network calls, all LLM invocations are `AsyncMock`.

### Mocking Strategy

```python
# LLM mock pattern used throughout
mock_llm = MagicMock()
mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
    return_value=expected_output
)
```

### Test Files

| File | Tests | Covers |
|------|-------|--------|
| `test_schemas.py` | 12 | Pydantic model validation, reducers, state field behaviour |
| `test_tools.py` | 22 | Web search, arXiv, URL fetcher, PDF loader, RAG retriever — all network mocked |
| `test_graph.py` | 44 | Node functions, routing edges, HITL interrupt/resume, full pipeline |
| `test_db.py` | 25 | SQLite path creation, `AsyncSqliteSaver` round-trip, `list_sessions`, `delete_session` |
| `test_rag.py` | 20 | `IngestionPipeline`, index loading, `probe_ollama`, query engine selection |
| `test_agents.py` | 34 | Individual agent functions with mocked LLMs and state |

### Notable Test Patterns

**TavilyClient singleton isolation:**
```python
def setup_method(self):
    import sys
    ws = sys.modules.get("research_swarm.tools.web_search")
    if ws:
        ws._tavily_client = None
        ws._tavily_key_cache = ""
```
`import research_swarm.tools.web_search as ws` resolves to the `StructuredTool` object (because `__init__.py` shadows the submodule name). `sys.modules` is used to get the actual module.

**Node patchability:**
```python
import research_swarm.graph.nodes as _nodes
_nodes.supervisor_node = fake_supervisor   # patched before build_graph()
graph = build_graph(checkpointer=saver)    # picks up the patch
```

---

## 16. Data Flow — End to End

```
User: "Research the impact of LLMs on scientific publishing"
Audience: "researchers", Depth: "deep"
                    │
                    ▼
         ResearchQuery created
         session_id = uuid4()
                    │
         (optional) PDFs/URLs ingested
         → ChromaDB at data/sessions/{id}/chroma/
                    │
                    ▼
┌──────────── LangGraph astream() ────────────────────────────┐
│                                                              │
│  1. supervisor_node                                          │
│       LLM → ResearchPlan:                                    │
│         sub_questions: [                                     │
│           "How are LLMs currently used in research?",        │
│           "What are the risks of AI-generated papers?",      │
│           "How are publishers responding?",                  │
│         ]                                                    │
│       next_agent = "researcher"                              │
│                                                              │
│  2. researcher_node (×3 sub-questions)                       │
│       retrieve_from_rag → (empty first run)                  │
│       web_search → 5 results per query                       │
│       arxiv_search → 3 papers                                │
│       fetch_url → full text of 2 promising articles          │
│       Synthesis → Finding(claim, confidence=0.7, evidence)   │
│       next_agent = "critic"                                  │
│                                                              │
│  3. critic_node                                              │
│       Finding 1 → supported (strong evidence)                │
│       Finding 2 → weak (only 1 source, needs more)           │
│       Finding 3 → supported                                  │
│       next_agent = "researcher" (Finding 2 is weak)          │
│                                                              │
│  4. researcher_node (re-research Finding 2)                  │
│       Additional search → 3 more sources                     │
│       Finding 2 updated (same id, new confidence=0.8)        │
│       → _merge_findings overwrites in state                  │
│                                                              │
│  5. critic_node                                              │
│       Finding 2 → supported                                  │
│       All findings supported → next_agent = "fact_checker"   │
│                                                              │
│  6. fact_checker_node                                        │
│       Cross-checks each claim against snippets               │
│       Finding 1: 0.7 → 0.85                                  │
│       Finding 2: 0.8 → 0.75                                  │
│       Finding 3: 0.6 → 0.65                                  │
│       next_agent = "writer"                                  │
│                                                              │
│  7. [HITL interrupt — user reviews, clicks Approve]          │
│                                                              │
│  8. writer_node                                              │
│       Deduplicate 11 sources → 8 unique references           │
│       Generate FinalReport with 3 sections + citations       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
         FinalReport displayed in UI
         Downloadable as Markdown
         Session checkpointed to SQLite
         Chroma index persisted for future retrieval
```

---

*Generated from source code — research_swarm v1.0, commit `bcb0c44`*
