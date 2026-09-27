# Research Swarm — Design & Development Journey

**Author:** Maitreya Mohapatra
**Repository:** https://github.com/mimo1999/research-swarm
**Development window:** 21 May 2026 – 10 July 2026 (32 commits, all authored locally — the full history with dates is verifiable via `git log`)

This document walks through how this project was designed and built, in the order it actually happened. Every phase below maps to specific commits in the repository, and every design decision is explained: what was chosen, what the alternatives were, and why the alternative lost. The challenges section describes bugs I actually hit — including three separate root causes behind a single benchmark metric reading zero — because the debugging trail is the strongest evidence of original work.

> **Update (September 2026):** sections 1–8 describe the system as of July 2026. The pipeline has since been rebuilt evidence-first (no embeddings, vector store, critic or fact-checker), and a question frame and sectioned writer were added. Section 9 covers those changes; [TECHNICAL_HANDOFF.md](TECHNICAL_HANDOFF.md) describes the current system.

---

## 1. What the system is (the 60-second version)

You give it a research topic. A **supervisor** agent breaks the topic into sub-questions and assigns each one to a **worker** agent with a specific persona (academic, industry, skeptic, benchmark, or general). The workers run in parallel, using web search, arXiv, URL fetching, and a local document index (RAG) to gather evidence. A **critic** grades every finding as supported / weak / refuted; weak findings trigger another research round. A **fact-checker** cross-references each claim against the actual source snippets and adjusts confidence scores. Finally a **writer** synthesizes a structured, cited report — optionally pausing first so a human can review the findings (human-in-the-loop, "HITL").

The whole thing runs as a **state graph**: agents are nodes, a shared state dictionary flows between them, and routing between nodes is deterministic code, not an LLM guessing what to do next.

```
START → supervisor (plan + complexity score, one LLM call)
          ↓
        dispatch ──► worker × N   (parallel fan-out)
          ↑              ↓
          └─ re-research  collect (marginal-gain stop check)
                          ↓
                critic → fact_checker → writer → END
                                          ↑
                              (pause here if HITL enabled)
```

---

## 2. Design decisions and tool choices

These were made up front, before the first commit, and each one was a deliberate comparison.

### 2.1 LangGraph over CrewAI and AutoGen (orchestration)

The core architectural question for any multi-agent system is: *who decides what happens next?*

- **CrewAI** gives you role-based agents with a sequential or hierarchical process, but the orchestration is largely opaque — you hand control to the framework and hope. Debugging "why did the crew loop forever" is hard because the control flow lives inside the library.
- **AutoGen** is conversation-driven: agents talk to each other and an LLM decides when the conversation ends. That is flexible but non-deterministic — the same input can take wildly different paths, which makes testing and cost control nearly impossible.
- **LangGraph** models the system as an explicit `StateGraph`: nodes, edges, and a typed state object. Routing is a function you write (`graph/edges.py::route_from_supervisor` reads `state["next_agent"]`), so it is unit-testable without any LLM. It also ships two features I knew I needed on day one: **checkpointing** (`SqliteSaver` persists every state transition, enabling session resume) and **`interrupt_before`** (compile-time pause points, which is exactly what HITL requires).

The deciding factor was testability. This project has **181 unit tests that run fully offline** — every LLM is mocked — and that is only possible because the control flow is ordinary Python, not framework magic. In the final architecture the supervisor LLM is called **exactly once per session** (to create the plan); everything after that is a deterministic state machine. That gives predictable cost, reproducible behaviour, and provable termination.

### 2.2 LlamaIndex over LangChain retrieval (RAG layer)

LangChain has retrieval primitives, but LlamaIndex is purpose-built for it: its `IngestionPipeline` handles PDFs, URLs and raw text uniformly, and its `RouterQueryEngine` / `SubQuestionQueryEngine` composition lets one query be decomposed and routed between a vector index and a summary index. Using LangGraph for orchestration and LlamaIndex for retrieval means each library does the one thing it is best at, joined by a thin adapter (`tools/retriever_tool.py` wraps the query engine as a plain LangChain tool the workers can call).

One deliberate constraint: **all RAG computation is local**. Embeddings are `BAAI/bge-small-en-v1.5` running on CPU via HuggingFace — no embedding API calls, no per-query cost, and the system works fully offline with Ollama. BGE-small was chosen over BGE-large because it is ~130 MB vs ~1.3 GB, and (as the benchmarks in §6 later confirmed) it retrieves within ~0.3 points of BGE-large's published nDCG@10 on SciFact.

### 2.3 ChromaDB over FAISS, Qdrant, and Pinecone (vector store)

- **Pinecone / Weaviate cloud**: rejected — the design goal was zero cloud dependencies for retrieval.
- **FAISS**: fastest raw ANN search, but it is just an index — no metadata filtering, no built-in persistence model. I would have had to build the document-store layer myself.
- **Qdrant**: excellent, but runs as a separate server process — an unnecessary operational burden for a per-user desktop tool.
- **ChromaDB**: embedded (runs in-process), persists to disk with one line, and supports metadata filtering. Each research session gets its own collection at `data/sessions/{session_id}/chroma/`, so sessions are isolated and deleting a session is just deleting a folder.

### 2.4 Streamlit over Gradio and a custom React frontend (UI)

The UI needs: a sidebar of settings, live streaming of agent activity, a report renderer, and session management. Streamlit gives all of that in pure Python with `@st.cache_resource` for expensive objects (the compiled graph) and a rerun model that maps naturally onto "resume the graph after human feedback." Gradio is optimised for model demos (input → output), not multi-tab stateful apps. A React frontend would have doubled the codebase for zero research value. The one real cost of Streamlit — its synchronous execution model vs LangGraph's async API — became a genuine engineering challenge, covered in §5.1.

### 2.5 Multi-provider LLM layer (Anthropic / OpenAI / Ollama)

`agents/base.py::get_agent_llm()` returns a LangChain `BaseChatModel` for any of three providers. This wasn't just flexibility for its own sake: it decouples the architecture from any one vendor, lets the whole system run **free and local** on Ollama, and — critically — enabled the model-comparison benchmarks in §6, which produced the project's most interesting empirical finding. Phase 4 added **model tiers**: the supervisor (one hard reasoning call) can use a stronger model than the workers (many cheap tool-calling calls).

### 2.6 Other choices, briefly

- **Poetry** over pip/requirements.txt: lockfile reproducibility (`poetry.lock`) matters when your dependency tree includes LangGraph, LlamaIndex, ChromaDB and sentence-transformers, which have historically conflicting pins.
- **Pydantic everywhere**: `Settings` is a `pydantic-settings` singleton; every LLM output (plans, findings, critiques, reports) is parsed through `with_structured_output` into Pydantic models in `research_swarm/schemas/`. Malformed LLM output fails loudly at the boundary instead of corrupting state.
- **Tavily** for web search: purpose-built for LLM consumption (returns cleaned content, not raw SERP HTML), generous free tier.
- **SQLite** (`SqliteSaver`) for checkpoints: zero-setup, single-file, and LangGraph ships a first-party saver for it.

---

## 3. State design — the part that makes everything else work

The single most important data structure is `AgentState` (`schemas/state.py`), a `TypedDict` threaded through every node. Two custom **reducers** (functions that define how concurrent updates merge) encode the system's semantics:

- **`findings` merges by id.** When the fact-checker re-emits a finding with an adjusted confidence score, it *overwrites* the existing finding with the same `id` instead of appending a duplicate. Without this, every fact-check pass would double the findings list.
- **`critiques` is append-only.** Each critic pass accumulates, preserving the full review history.

This reducer design is also what makes the Phase 4 parallel fan-out safe: when N workers return findings concurrently, LangGraph merges them through the same reducer with no race conditions and no manual locking.

---

## 4. Build order — what was built when, and why in that order

### Phase 1 — Core skeleton (21 May, commits `e489f56`–`aec97cb`)

The initial commit contained the full vertical slice: five agents, the state graph, the RAG pipeline, the Streamlit app, and the first test suite. I built the *thinnest end-to-end path first* — supervisor → researcher → critic → fact-checker → writer, sequentially — so every later change could be validated against a working pipeline. The very same day produced two fix commits (Windows encoding issues, event-loop conflicts, enum-comparison bugs), which is what shipping on Windows with an async framework looks like in practice.

The second-day commit ("deterministic routing") was the first major architectural correction: early on, the supervisor LLM was consulted for *every* routing decision. That was expensive, slow, and occasionally wrong (see §5.2). I moved all routing into a deterministic state machine and restricted the LLM to plan creation only.

### Phase 2 — Hardening (22–24 May, commits `0c85165`–`4627d37`)

A dedicated bug-fix and refactoring pass, documented in detail in [SUMMARY.md](SUMMARY.md): fixed routing loops, hardened SSRF protection (§5.3), split HITL into two feedback channels (§5.4), deduplicated the RAG helpers into `rag/_chroma.py`, and fixed SQLite two-phase commit in session deletion. Ended with TECHNICAL.md documenting the architecture, and the test suite at 157 passing.

### Phase 3 — Quality machinery (4 June, commits `8215b74`–`4b46f77`)

With the pipeline stable, I added the layers that distinguish "demo" from "system":

- **Budget guard** (`runtime/budget.py`): counts *actual* LLM calls per session and raises `BudgetExceeded` past `MAX_LLM_CALLS`, forcing a graceful writer fallback instead of an infinite spend. Agent systems that can loop *must* have a hard cost ceiling.
- **Schema migration layer** (`runtime/migrations.py`): checkpoints written by an old code version are upgraded (v0→v1→v2) on resume. Without this, every schema change would orphan all saved sessions.
- **Cross-encoder reranker** (`rag/reranker.py`): `ms-marco-MiniLM-L-6-v2` (22 MB, CPU) rescores retrieved chunks by query relevance. A bi-encoder retrieves fast but coarse; the cross-encoder reads query and chunk *together* and reorders precisely.
- **Writer faithfulness check** (`eval/faithfulness.py`): after the report is drafted, each section is embedded and compared (cosine similarity) against its cited snippets; if grounding scores below 0.25 the writer rewrites once. This is a cheap, local hallucination tripwire.
- **Golden regression tests** (`tests/golden/`): three full-pipeline runs with fixed mocked LLM outputs, asserting on the final report structure — so refactors can't silently change end-to-end behaviour.

### Phase 4 — Parallel worker swarm (4 June, commits `609eedd`–`755f70a`)

The original design was sequential: one researcher, one sub-question at a time. Phase 4 rebuilt the middle of the graph around **LangGraph's `Send` API**, which fans out one dispatch into N concurrent worker invocations, each with its own payload:

- The supervisor now emits a **plan**: sub-questions, a complexity score, and a **worker role** per sub-question (`academic / industry / skeptic / benchmark / general`). The skeptic role exists deliberately — one worker per round is prompted to look for *disconfirming* evidence, a structural guard against confirmation bias.
- `dispatch_node` sends each sub-question to a `worker_node` in parallel; `collect_node` gathers results and applies a **marginal-gain stop rule**: if a round produced too little new validated information relative to the last, stop dispatching (with a hard round cap as backstop). This replaces "fixed N iterations" with "stop when learning plateaus."
- This was shipped as four commits in dependency order — schemas first, then agents, then graph wiring, then tests + the v1→v2 state migration — so each commit left the suite green.

### Phase 5 — Benchmarking and the debugging campaign (11–14 June, commits `5041a12`–`c7cd921`)

I built two benchmark harnesses: a 24-task smoke benchmark (ALCE, HotpotQA, SciFact — seed 42, fixed manifest, SHA256-pinned datasets) and a BEIR retrieval evaluation (nDCG@10 on SciFact, NFCorpus, ArguAna). The first smoke run reported **grounded_rate = 0%** — the system was allegedly citing no retrieved sources at all — which kicked off the most instructive debugging episode of the project (§5.5, three independent root causes). Results in §6.

### Phase 6 — Production tuning (13 June – 10 July, commits `27435a3`–`ca6a5ec`)

Post-benchmark polish: switched the default local model from `minimax-m2.5:cloud` to `gemma4:31b-cloud` based on the grounding data (§6), cut prompt/output token usage ~40–50% by compacting schema instructions, made the writer populate references programmatically instead of asking the LLM (eliminating hallucinated URLs while saving tokens), and moved synchronous tool calls to `asyncio.to_thread` so parallel workers stop serializing on the event loop.

---

## 5. Challenges — what actually went wrong, and how it was fixed

### 5.1 Streamlit is sync; LangGraph is async

All graph nodes are `async def`, driven via `graph.astream()`. Streamlit runs its own event loop, and calling `asyncio.run()` inside it normally raises `RuntimeError: This event loop is already running`. Fix: `nest_asyncio.apply()` at startup patches the loop to allow re-entrancy, and one function (`_stream_graph`) owns all async driving. Not elegant, but contained — and the CLI runner (`run_research.py`) drives the same graph without the patch, proving the core is clean.

### 5.2 The infinite fact-checker loop

The supervisor had a hard iteration cap that returned `next_agent="fact_checker"` on every call once tripped — bypassing the transition rule "after fact-checker comes writer." Result: fact-checker forever. The fix restructured control flow so *all* routing lives in one deterministic function (`_route_from_state`), the ceiling check happens *before* any LLM construction, and `supervisor_node` has no early-return routing shortcuts. Lesson learned and then generalized: **an LLM should propose content, never control flow.**

### 5.3 SSRF in the URL fetcher

Workers fetch arbitrary URLs, so a malicious or manipulated source could point the fetcher at internal addresses (`http://192.168.1.1/admin`, cloud metadata endpoints). The first blocklist was string-prefix matching — trivially bypassed by decimal (`http://3232235777`), hex, or IPv6-mapped encodings of private IPs. The rewrite (`utils/security.py`) resolves the host, parses it with Python's `ipaddress` module, checks `is_private / is_loopback / is_link_local`, and validates **every redirect hop**, not just the first URL. Fetched content is additionally scanned for prompt-injection patterns before it reaches an agent's context.

### 5.4 HITL feedback triggering the wrong agent

One `human_feedback` field served both "researcher, dig deeper into X" and "writer, change the tone." Writer-directed feedback would route back to research — a loop. Fix: two separate state channels (`human_feedback` → researcher re-pass, `writer_instructions` → writer), each consumed exactly once.

### 5.5 grounded_rate = 0: one symptom, three root causes

The benchmark said no report cited any retrieved source. It took three fixes on three layers before the number moved (all documented in [benchmarks/README.md](benchmarks/README.md)):

1. **The retriever tool couldn't see the session.** The tool read `session_id` from state at call time, but LangGraph tools don't get graph state injected — it silently queried nothing. Fix: pre-bake `session_id` into the tool closure at construction time (commit `9d1756c`).
2. **`Send` payloads dropped fields.** The dispatch fan-out forwarded only the sub-question to each worker — not `session_id` or the original query — so even a correctly-built tool had no session to query (commit `1080e36`).
3. **ToolMessage truncation ate the metadata.** Tool results were serialized to JSON and truncated to a fixed length for the LLM context — and source metadata sat at the *end* of the JSON, so it was silently amputated on every call (commit `8fc436d`).

Each fix was individually correct and individually insufficient. This is the strongest argument in the codebase for end-to-end benchmarks: unit tests passed throughout, because each layer honoured its own (wrong) contract.

### 5.6 A silent cloud fallback in "local" mode

LlamaIndex defaults its internal LLM to OpenAI. Even configured for Ollama, certain query-engine paths would attempt OpenAI calls — a cost leak and a privacy violation in a system advertised as local-capable. Fix: set LlamaIndex's global default to a `MockLLM` at import time (commit `261e205`), so nothing can fall through to a cloud API implicitly.

### 5.7 Model miscalibration and the confidence floor

`minimax-m2.5:cloud` scored *highest* on answer accuracy (0.371) but grounded only 4% of its claims — it was answering from parametric memory and ignoring retrieval entirely. Separately, the fact-checker would sometimes zero out a claim's confidence even when real source evidence was attached. Two fixes: a **confidence floor of 0.15 for evidence-backed findings** (a miscalibrated judge can lower, but not erase, an evidenced claim — commit `c2e886b`), and switching the default model to `gemma4:31b-cloud`, which scores lower on raw answers (0.265) but grounds 75% of claims with 0.627 faithfulness. For a *research* system, "shows its sources" beats "sounds right."

### 5.8 An out-of-distribution reranker

The cross-encoder reranker *hurt* nDCG@10 on SciFact — `ms-marco-MiniLM` was trained on short web queries, and long scientific claims are out of distribution for it. Rather than remove it, I added a guard: reranking is **skipped for queries longer than 8 words**. Verified on BEIR: with the guard, SciFact is unharmed (75/100 queries skip reranking) while NFCorpus (short keyword queries) still gains +1.5% nDCG@10.

---

## 6. Evaluation — how I know it works

Everything in this section is reproducible: fixed seed (42), pinned task manifests, SHA256-checksummed datasets, scripts in `benchmarks/`.

**Retrieval (BEIR, nDCG@10, 100 queries/dataset):**

| Dataset | BM25 | Contriever | BGE-Large | **Ours (BGE-small)** | **Ours + reranker** |
|---|---|---|---|---|---|
| SciFact | 0.678 | 0.677 | 0.752 | **0.749** | 0.746 |
| NFCorpus | 0.321 | 0.328 | 0.381 | 0.341 | **0.356** |
| ArguAna | 0.397 | 0.446 | 0.416 | 0.391 | 0.391 |

The pipeline with a 130 MB CPU embedder lands within noise of published BGE-Large (1.3 GB) numbers on SciFact and beats classic BM25/Contriever baselines.

**End-to-end (24-task smoke benchmark, seed 42):** `gemma4:31b-cloud` — 75% grounded rate, 0.627 faithfulness, 17 s median/task. The full model comparison and the grounding-vs-accuracy trade-off analysis are in [benchmarks/README.md](benchmarks/README.md).

**Software correctness:** 181 unit tests, all offline (LLMs mocked via `AsyncMock`), plus golden regression tests over the full pipeline, `ruff` lint and `mypy` type-checking. A deliberate design detail makes the graph mockable: `graph/builder.py` resolves node functions at `build_graph()` call time (`_nodes.supervisor_node`), so `unittest.mock.patch` on the module attribute works.

---

## 7. Current status

The system is feature-complete and stable at commit `ca6a5ec` (10 July 2026):

- Five-agent pipeline with parallel worker dispatch, deterministic routing, and marginal-gain stopping.
- Three LLM providers; fully local operation (Ollama + local embeddings + ChromaDB) verified.
- HITL review with dual feedback channels; session persistence with checkpoint migration (v0→v2).
- Budget guard, SSRF-hardened fetching, prompt-injection scanning, faithfulness-checked writing.
- 181/181 tests passing; retrieval benchmarked against published BEIR baselines; end-to-end grounding at 75%.
- Two entry points: Streamlit UI (`app.py`) and headless CLI (`run_research.py`).

Known limitations, honestly stated: the answer-accuracy score on open-ended long-form tasks (ALCE/ELI5) is weak — the grounding-first design trades fluent-but-unsupported answers away, and the current default model is conservative. HITL pauses only before the writer, not mid-research. The reranker guard is a heuristic (word count), not a learned domain detector.

## 8. Future scope

1. **A domain-adapted reranker** — replace the 8-word heuristic with a cross-encoder fine-tuned on scientific queries (or a small classifier that picks the reranker per query).
2. **Streaming token-level output in the UI** — currently the trace streams per-node; per-token writer streaming would improve perceived latency.
3. **Mid-research HITL** — `interrupt_before` on `collect_node`, letting a human redirect the swarm between rounds, not only before writing.
4. **Citation-level verification** — extend the fact-checker from finding-level to sentence-level attribution (ALCE-style citation precision/recall as a first-class metric).
5. **Cost-aware planning** — the supervisor's complexity score already exists; use it to choose worker count and model tier dynamically against the token budget.
6. **Multi-session knowledge reuse** — cross-session ChromaDB collections so the swarm can build on prior research instead of starting cold.

---

*Every claim in this document is checkable against the repository: `git log --reverse` for the build order, `SUMMARY.md` for the Phase 2 fix session, `benchmarks/README.md` for raw benchmark data, and the test suite (`poetry run pytest`) for correctness.*

---

## 9. August–September 2026: evidence-first rebuild and scope control

### Phase 7 — Pipeline v2: evidence first

Benchmarks showed the agentic middle of the graph was the weak point. ReAct workers, a snippet summarizer, the critic and the fact-checker used most of the time and lost answers the evidence contained. They were replaced by fixed pipelines in which an LLM call is spent only where language understanding is needed:
- **Packed extraction:** facts carry a verbatim quote, and sources and sub-questions are referred to by number.
- **Grounding in code:** each quote is located in its source.
- **One verifier call per 10 facts,** with enum verdicts applied by a fixed policy table.
- **Attributed writer:** the model tags each sentence with its fact numbers, and code renders the citations.

Paper search moved to a no-embedding scout that scores title and abstract with a light LLM. Embeddings, ChromaDB, LlamaIndex and the reranker were removed. Each change was kept only if it passed a paired, same-task ablation (the table is in `benchmarks/README.md`).

### Phase 8 — Keeping the question's scope

A live run on *"Can we losslessly migrate KV cache from one LLM to another?"* produced a fluent report about KV-cache **compression**. The trace showed the constraint "between different LLMs" surviving in the plan's sub-questions but missing from every search query. After that, no stage could notice the drift. The fixes:

- **Question frame** (`agents/expansion.py`): a probe search of the literal question, then one call. It extracts the key constraint, its phrasings in the literature, confusable topics, strict terms ("lossless") and their proof criterion, and the items the question asks to compare.
- **Enforcement in code at every stage:**
  - the plan's count cap and re-attached constraint;
  - the scorer's scope rule;
  - a coverage gate that counts only on-scope facts;
  - verifier relevance labels;
  - writer checks that drop sentences whose facts don't mention the scope or strict term, and remove answers that assert an unmet strict qualifier.
- **Sectioned writer:** fixed sections per audience, written one at a time with earlier sections in context, then a comparison table and a review pass. The review must rule on every sentence with strong wording.
- **Model split:** research stays on local `gemma4:e2b`. Only the planner and writer use a larger model (`nemotron-3-nano:30b-cloud`), and that is a config setting that can be removed on a bigger GPU.
- **Deep read:** full-text passages of the top arXiv papers, so a paper's specifics come from the paper and not from blogs summarising it.
- **UI and HITL:**
  - runs became background jobs, so changing settings mid-run no longer wipes the trace;
  - a live topology diagram and LangSmith links were added;
  - "Edit & Re-research" was fixed to actually re-enter research.

The report quality moved from answering the wrong question, to a correct "no, only approximately within a model family", to a correct verdict on an audit-style question. One case is still open: a question that refers to "the paper" without naming it. The test suite went from 181 to 254 tests.
