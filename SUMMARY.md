# Research Swarm — Development Summary

## Project Overview

**Research Swarm** is a production-grade autonomous research system built on LangGraph, LlamaIndex, and Streamlit. It orchestrates a swarm of AI agents to autonomously plan, research (live web/arXiv/PubMed search *and* directly-ingested documents), critique, fact-check, and write structured reports on any topic — with optional human review.

**GitHub:** https://github.com/mimo1999/research-swarm

---

## Key Accomplishment: RAG/Retrieval Overhaul and Reliability Hardening (348 Tests Passing)

This document summarizes a session that replaced the retrieval stack, added a document-extraction pipeline that bypasses vector search entirely for uploaded files, split the LLM budget so a research-loop overrun can no longer produce an empty report, and fixed a batch of correctness issues surfaced by a full code review and repeated live test runs.

### Starting State
- 181 tests passing, single shared LLM-call budget
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Uploaded PDFs/URLs chunked, embedded, and retrieved via `retrieve_from_rag` like any other RAG content
- No JSON-repair path for structured-output parse failures
- No document-level extraction path — everything routed through the live web-search worker loop

### Final State
- **All 348 tests passing**
- Reranker swapped to `BAAI/bge-reranker-base`, benchmarked against 4 candidates on BEIR
- Ingested documents extracted directly into findings — no chunk/embed/retrieve step
- Dual budget pools (research / review) so the writer always produces a report
- Supervisor generates an *exact* sub-question count and is required to cover every side of a comparison topic
- JSON-repair recovers structured-output failures caused by unescaped LaTeX backslashes
- A concurrent-write crash in the graph (`InvalidUpdateError` on simultaneous budget-exceeded branches) fixed via a proper reducer
- Critic/fact-checker batched and parallelized; duplicate embedding passes removed

---

## Major Code Changes

### 1. Reranker Model Swap + Comparison (`rag/reranker.py`, `benchmarks/run_beir_reranker_compare.py`)

**Problem:** `ms-marco-MiniLM-L-6-v2` was the only reranker ever evaluated in production.

**Solution:**
- Benchmarked `BAAI/bge-reranker-v2-m3` (failed to load — insufficient RAM), `BAAI/bge-reranker-base`, and `mixedbread-ai/mxbai-rerank-xsmall-v1` against the incumbent on BEIR (SciFact/NFCorpus/ArguAna, seed 42, 100 queries each)
- `bge-reranker-base` never regresses nDCG@10 vs. dense retrieval alone (ms-marco does, on SciFact); ~6x slower on CPU
- `mxbai-rerank-xsmall-v1` wins on quality everywhere it's exercised but runs ~10x slower per pair than `bge-reranker-base` despite a quarter of the parameters — parameter count isn't a reliable proxy for CPU inference cost
- Production stays on `bge-reranker-base`; comparison script refactored to a data-driven `RERANKER_MODELS` list

**Impact:** Reranker choice is now evidence-based and documented (`benchmarks/README.md`), not inherited by default.

---

### 2. Document-Extraction Pipeline Replaces Chunk+Embed+Retrieve for Uploads (`agents/document_worker.py`, `graph/nodes.py`, `schemas/state.py`)

**Problem:** Uploaded PDFs/URLs went through the same chunk→embed→top-k-retrieve path as everything else, even though the corpus is small (3-15 documents/session) and known entirely upfront — retrieval risk (chunk-boundary loss, reranker guard skipping long queries) for no benefit over just reading the whole document.

**Solution:**
- `document_pass_node`/`document_worker_node`: one single-shot structured-output extraction call per document (or per size-bounded slice of an oversized one, split at sentence boundaries via LlamaIndex's `SentenceSplitter` used purely as a boundary-finder)
- Runs once, before round-0 sub-question dispatch; `_research_targets`' round-0 branch now skips any sub-question a document already answered
- Sub-question values are snapped to the plan's canonical strings so results slot into the existing merge-by-id reducer, rework loop, and critic — no new state machinery
- Deterministic finding IDs (`uuid5` keyed on `document_url + sub_question`) so two parts of a split document answering the same sub-question collide into one finding instead of producing duplicates

**Impact:** No retrieval-quality risk for uploaded content; `retrieve_from_rag` now exclusively serves evidence discovered by earlier rounds of live research (persisted by `collect_node`, dedup-aware).

---

### 3. Dual LLM-Call Budget Pools (`runtime/budget.py`, `graph/nodes.py`)

**Problem:** A single shared budget meant a worker-loop overrun (multiple rounds × multiple tool turns) could exhaust the budget before critic/fact-checker/writer ever ran — observed live: 49/40 calls used, and the session produced a **completely empty report** despite the worker loop having gathered five good findings.

**Solution:**
- Split into a **research** pool (supervisor, document workers, dispatch/worker loop) and a **review** pool (critic, fact-checker, writer, judge)
- `writer_node` is deliberately never budget-gated — only the *optional* LLM-judge sub-call is
- `next_agent` gained a reducer (`_last_value`) so ≥2 parallel Send-fanned branches independently hitting an exhausted pool in the same graph step no longer crash with `InvalidUpdateError`

**Impact:** A research-loop overrun now degrades gracefully — the writer still produces a real report from whatever findings exist, instead of an empty placeholder.

---

### 4. Supervisor Sub-Question Generation Fix (`agents/supervisor.py`)

**Problem:** The prompt said "generate AT MOST N sub-questions" — no pressure to use the budget. Observed live: for *"key differences between LoRA and QLoRA"*, the model generated exactly **one** sub-question, about LoRA only, silently dropping QLoRA from the plan entirely.

**Solution:**
- "Generate EXACTLY N sub-questions" — removes the model's discretion over count
- Explicit instruction that a comparison topic ("X vs Y") must generate sub-questions covering each side individually *and* their direct comparison, never collapsing to one side

**Impact:** Reproduced and fixed — the same query now reliably dispatches sub-questions covering both techniques.

---

### 5. JSON-Repair for Unescaped-Backslash Parse Failures (`agents/_utils.py`)

**Problem:** A worker synthesizing a claim with LaTeX notation (`\in`, `\mathbb{R}`, `\text{...}`) would break `json.loads` with `Invalid \escape` — observed repeatedly in live runs, always falling back to a `[Research incomplete]` placeholder and discarding real, already-generated content.

**Solution:**
- `_recover_from_bad_escapes` reads `OutputParserException.llm_output` (the raw text LangChain's parser attaches to the exception), strips markdown fences, and doubles any backslash *not* already followed by `\` or `"`
- Deliberately does **not** treat `\n`/`\t`/`\r`/`\b`/`\f`/`\u` as pre-escaped: common LaTeX commands (`\text`, `\theta`, `\times`, `\frac`) start with exactly those letters, and treating them as valid escapes would silently corrupt the content instead of fixing it
- Wired into every structured-output call site: workers, document workers, critic, fact-checker, LLM judge

**Impact:** Verified live — a synthesis call that previously produced a placeholder now recovers the full real claim.

---

### 6. Critic/Fact-Checker Batching, Parallelization, and Correctness Fixes (`agents/critic.py`, `agents/fact_checker.py`, `graph/nodes.py`)

**Problem:** One LLM call per finding instead of per batch; independent batches awaited sequentially instead of concurrently; `rework_counts` incremented for *any* sub-question lacking a finding, including the mandatory round-0→round-1 loop that fires before the critic ever runs — silently consuming a finding's rework budget before any critique existed.

**Solution:**
- Batched review (`judge_batch_size` findings per call, shared sources deduped by URL within a batch), batches run via `asyncio.gather`
- New `_weak_or_refuted_sub_questions` helper — `rework_counts` now increments only for sub-questions the critic actually flagged, not every findingless one

**Impact:** Faster review pass; rework budget now means what its name says.

---

### 7. Report Quality-Score and Faithfulness-Check Fixes (`schemas/report.py`, `agents/writer.py`)

**Problem:** `ReportQualityScore.relevance`/`completeness` defaulted to `0.0` — indistinguishable from "computed and genuinely zero" — and once the writer started always attaching a `quality_score`, the UI showed a misleading "Relevance: 0%, Completeness: 0%" on every report. Separately, `_attach_quality_score` re-ran the full section+snippet embedding pass a second time even though `_faithfulness_rewrite` had just computed the same thing.

**Solution:**
- `relevance`/`completeness` default to `None` ("not computed"); `overall` averages only populated dimensions
- Faithfulness rewrite now returns its score alongside the report so the writer reuses it instead of re-embedding; rewrite pass targets only the specific under-grounded sections instead of regenerating the whole report blind

**Impact:** No fabricated scores in the UI; one fewer full embedding pass per report.

---

### 8. PubMed Search + Domain Routing (`tools/pubmed_tool.py`, `agents/workers.py`)

**Problem:** Only `arxiv_search` was available for literature search — arXiv barely indexes biomedical/clinical content.

**Solution:** Added `pubmed_search` alongside `arxiv_search`, with explicit routing guidance so a worker checks a result is actually about the sub-question's subject regardless of which tool it came from.

**Impact:** Medical/clinical/biomedical sub-questions get an actual literature source instead of arXiv's sparse coverage.

---

## Test Suite Changes (348 Tests)

| File | Changes |
|------|---------|
| `test_document_worker.py` | New — extraction, sentence-boundary splitting, sub-question snapping |
| `test_budget.py` | New — dual-pool independence, `_last_value` reducer behavior |
| `test_utils.py` | New — bad-escape JSON repair, LaTeX-command preservation, schema-echo recovery |
| `test_faithfulness.py` | New — per-section scoring, targeted rewrite |
| `test_llm_judge.py` | New — judge scoring, fallback on failure |
| `test_runs.py` | New — API run orchestration |
| `test_agents.py` | Updated — synthesis comparability check, source-ranking batching, supervisor exact-count generation |
| `test_graph.py` | Updated — document-pass routing, round-0 skip logic, rework-count accounting, collect-node evidence persistence |
| `test_rag.py` | Updated — dedup-aware evidence ingestion |
| `test_schemas.py` | Updated — `ReportQualityScore` None-defaults |

---

## Tech Stack

### Core
- **LangGraph 1.2+** — multi-agent state graph
- **LangChain 0.3+** — LLM abstraction
- **LlamaIndex 0.11+** — RAG indexing, query engines, and sentence-boundary splitting
- **Streamlit 1.37+** — web UI
- **FastAPI** — REST API layer (routes, run orchestration) alongside the Streamlit app

### LLM Providers
- **Ollama** (`gemma4:31b-cloud`) — default
- **Anthropic Claude**, **OpenAI GPT** — alternatives

### Data & Search
- **Tavily** — web search
- **arXiv API** — CS/physics/math preprints
- **PubMed** *(new)* — biomedical/clinical literature
- **BAAI/bge-reranker-base** — cross-encoder reranking
- **ChromaDB** — vector store for cross-round evidence persistence (not uploaded documents)
- **HuggingFace bge-small-en-v1.5** — embeddings

### Persistence & Testing
- **SQLite + aiosqlite** — checkpoints
- **pytest + pytest-asyncio** — 348 offline tests

---

## Key Design Patterns

### 1. Two-Source Evidence Model
Live web/arXiv/PubMed search feeds the sub-question dispatch loop; ingested documents feed a separate one-time extraction pass. Both converge on the same `Finding` schema and downstream critic/fact-checker/writer pipeline — no special-casing further down the graph.

### 2. Dual Budget Pools
Research (can run away across rounds) and review (a few batched calls) are counted independently, so exhausting one can't silently zero out the other's output.

### 3. Merge-by-ID Findings, Deterministic IDs Where Order Isn't Guaranteed
Fact-checker and re-research reuse an existing finding's ID to update in place. Document-worker parts, dispatched concurrently with no prior-round state to look up, instead derive a deterministic ID from `(document_url, sub_question)` so duplicates collapse the same way.

### 4. Best-Effort Recovery Over Silent Placeholders
Both `recover_from_parse_failure`'s two independent recovery paths (schema-echo, bad-escapes) and the budget-pool split follow the same philosophy: when something fails, recover or degrade gracefully rather than silently discarding real work.

### 5. Exact Counts Over Unenforced Ceilings
Where a model was given a soft ceiling ("at most N") and no pressure to use it, it under-delivered in a way that silently dropped scope. Replaced with an exact requirement plus explicit coverage instructions.

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
poetry run pytest          # all 348 tests
poetry run pytest tests/unit/test_graph.py -v  # one file
```

---

## Resume Points for Future Work

### Known Limitations
1. **Reranker latency:** `bge-reranker-base` is ~6x slower on CPU than the ms-marco model it replaced; worth revisiting if reranking latency becomes a bottleneck in a live run
2. **JSON-repair scope:** the bad-escapes recovery path relies on `OutputParserException.llm_output`, which is specific to LangChain's JSON-mode parser (Ollama) — providers whose `with_structured_output` uses tool-calling (Anthropic/OpenAI) raise a different exception shape and aren't covered
3. **Document cross-part synthesis:** a claim spanning content split across two parts of one oversized document can't be connected by either part's worker — accepted tradeoff for guaranteeing full fidelity within each slice

### Enhancement Opportunities
1. **Fast-tier model:** currently pointed at the same model as the thorough tier (a deliberate, explicit tradeoff after the original fast-tier model stopped resolving locally) — worth re-pointing at a genuinely cheap model once one is confirmed available
2. **Tool-calling JSON repair:** extend `recover_from_parse_failure` to cover Anthropic/OpenAI's tool-calling parse-failure shape
3. **Unused settings cleanup:** `settings.llm_judge_pass_threshold` is defined but never read (the judge module uses its own hardcoded constant)
4. **API/Streamlit parity:** the document-pass pipeline's backend is complete, but only the Streamlit `app.py` upload flow currently populates `ingested_documents` — the FastAPI `documents` route should be checked for the same wiring

### Testing
- All 348 tests pass, fully offline (all LLM calls mocked)
- Manual end-to-end verification via `run_research.py` recommended before relying on a new model/config combination — several of the fixes above were only caught by an actual live run, not the mocked test suite

---

## Session Metadata

**Commits** (`c77cc40..8235881`, 12 commits, July 10 – August 6):
- `c77cc40` / `983cb8c` — FastAPI app scaffold + REST routes
- `d6be076` — LLM-judge review pass
- `44fc4ad` / `faf1433` — FastAPI dependencies + reranker swap
- `9d7852a` — Per-session credential/endpoint resolution
- `8d65662` — mxbai-rerank-xsmall-v1 comparison
- `799f2e0` — Document-extraction pipeline + evidence persistence
- `d18ae98` — Comparability/citation prompt hardening
- `81816e4` — Config settings + research-intensity tuning
- `63a22b2` — Batched review pipeline, dual budget pools, PubMed search, correctness fixes
- `8235881` — Test coverage catch-up

**Total changes:** 55 files changed, 6,404 insertions / 1,070 deletions

**Test status:** 348/348 passing ✓

---

## Quick Reference: Critical Fixes

| Issue | Root Cause | Fix | Commit |
|-------|-----------|-----|--------|
| Empty report despite good findings | Single shared LLM-call budget exhausted by worker-loop overrun before critic/fact-checker/writer ran | Split into research/review pools; writer never budget-gated | `63a22b2` |
| Graph crash: `InvalidUpdateError` | `next_agent` had no reducer; ≥2 parallel branches writing it in one step is rejected even with identical values | `Annotated[AgentName \| None, _last_value]` reducer | `63a22b2` |
| Supervisor drops half a comparison topic | "AT MOST N sub-questions" gave no pressure to use the budget | "EXACTLY N" + explicit both-sides coverage instruction | `81816e4` |
| Synthesis discards real content on LaTeX | Unescaped backslashes in claims break `json.loads` | Repair-and-reparse recovery path in `recover_from_parse_failure` | `63a22b2` |
| Duplicate findings from one document | Split document parts each generate a fresh random ID | Deterministic `uuid5(document_url + sub_question)` | `63a22b2` |
| Misleading 0% quality badges | `relevance`/`completeness` defaulted to `0.0`, indistinguishable from "computed" | Default to `None`; `overall` averages only populated fields | `63a22b2` |
| Retrieval risk for uploaded documents | Chunk+embed+retrieve for a small, fully-known-upfront corpus | Single-shot full-document extraction, no vector search | `799f2e0` |

---

**For detailed architecture, see `CLAUDE.md` and `README.md` in the repo.**
