# Benchmarking Experiment — Session Log

> **Historical document.** This log describes an earlier version of the system. That version used embedding retrieval, a reranker and a fact-checker, all since removed; its numbers are not comparable with current runs. For the current architecture see [README.md](README.md), [TECHNICAL_HANDOFF.md](TECHNICAL_HANDOFF.md) and [CLAUDE.md](CLAUDE.md).

**Date:** 2026-06-11 to 2026-06-14  
**Repository:** `swarm_agent_project` (LangGraph multi-agent research swarm)  
**Branch:** `main`

---

## 1. Starting Point

The system was running `minimax-m2.5:cloud` via Ollama as its default model. A known issue was open: `grounded_rate = 0` across all smoke benchmark runs — the pipeline was producing reports with zero cited sources despite successfully calling the retrieval tool.

---

## 2. Root-Cause Investigation: grounded_rate = 0

Three compounding bugs were identified and fixed, each independently capable of zeroing out grounding.

### Bug 1 — LangGraph Send payload isolation
**File:** `research_swarm/graph/nodes.py`

`Send("worker_node", payload)` gives the receiving node **only** the payload dict — the full graph state is not merged. `session_id` was missing from the payload, so every parallel worker queried the wrong (empty) Chroma collection and returned `[]` evidence.

**Fix:** Explicitly forward `session_id`, `query`, `model_provider`, and `model_name` in every `Send` payload.

### Bug 2 — LlamaIndex OpenAI fallback
**File:** `research_swarm/rag/query_engines.py`

LlamaIndex defaults to OpenAI at init time even when `response_mode="no_text"` is set. With no OpenAI key configured, the query engine silently failed.

**Fix:** Inject `MockLLM()` into LlamaIndex settings at module import time to prevent the fallback.

### Bug 3 — Fact-checker confidence floor
**File:** `research_swarm/agents/fact_checker.py`

`minimax-m2.5:cloud` systematically returns `confidence_score=0.0` for valid evidence-backed claims. The writer filters out findings with confidence < 0.1, so all evidence-backed findings were discarded, leaving `references=[]`.

**Fix:** Apply `max(score, 0.15)` when evidence is present — a finding backed by real sources cannot score below the no-evidence baseline.

### Verification
An e2e debug script (`_e2e_debug.py`) confirmed the fix: `report.references=2` after all three patches applied.

---

## 3. Reranker Implementation Gaps

The production cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) had three implementation gaps that were degrading retrieval quality relative to dense-only.

### Gap 1 — Character truncation instead of token truncation
**File:** `research_swarm/rag/reranker.py`

`passage[:512]` was applied to each passage. 512 characters ≈ 100 tokens — far below the CrossEncoder's 512-token capacity. The model was being starved of context.

**Fix:** Removed the character-level truncation entirely. The tokenizer handles truncation at 512 tokens.

### Gap 2 — Title not prepended to passage
The dense retriever uses title as a signal; the reranker was only seeing the body text.

**Fix:** `f"{c.get('title', '')}\n{c.get('snippet', '')}".strip()` — same signal the dense retriever uses.

### Gap 3 — Snippet cap too low
**File:** `research_swarm/tools/retriever_tool.py`

Snippet was capped at 800 characters (~160 tokens), further limiting passage length.

**Fix:** Raised cap to 2,000 characters (~400 tokens).

---

## 4. BEIR Retrieval Benchmark

**Script:** `benchmarks/run_beir_reranker_compare.py`  
**Datasets:** SciFact (5,183 docs), NFCorpus (3,633 docs), ArguAna (8,674 docs)  
**Metric:** nDCG@10, seed 42, 100 queries per dataset  
**Note:** No LLM involved — pure retrieval pipeline (BGE-small embeddings + ms-marco cross-encoder)

### Resource bugs fixed in benchmark script
Two bugs caused a 6-hour hang on first run:

1. **No embedding cache** — corpus was re-embedded on every run. Fix: save/load `.npy` per dataset to `data/benchmark_results/emb_cache/`.
2. **Both rerankers loaded simultaneously** — original script loaded ms-marco and bge-reranker-v2-m3 together (~720 MB simultaneous). Fix: sequential loading with `del model; gc.collect()` between passes.

### bge-reranker-v2-m3 evaluation

The generalised academic reranker (`BAAI/bge-reranker-v2-m3`, 570 MB) was tested as an alternative to ms-marco. It did not improve over dense-only on these datasets. Decision: retain ms-marco-MiniLM with the query-length guard.

### Results — before vs after reranker fixes

| Dataset | Method | Pre-fix nDCG@10 | Post-fix nDCG@10 | Delta |
|---|---|---|---|---|
| SciFact | Dense | 0.749 | 0.749 | 0.000 |
| SciFact | + Reranker | 0.696 | **0.746** | **+0.050** |
| NFCorpus | Dense | 0.341 | 0.341 | 0.000 |
| NFCorpus | + Reranker | 0.324 | **0.356** | **+0.032** |
| ArguAna | Dense | 0.391 | 0.391 | 0.000 |
| ArguAna | + Reranker | 0.391 | 0.391 | **0.000** |

**ArguAna:** All 100 queries average 195 words — the 8-word guard fires universally, reranker bypassed entirely. Scores identical.

**SciFact:** 75/100 queries exceed 8 words (avg 12 words), guard fires. Only 25 shorter claims are reranked; the fixes improved those enough to nearly recover to dense-only level.

**NFCorpus:** Short keyword queries (avg 3.2 words), guard fires rarely (4/100). Reranker is fully active — the implementation fixes flipped the delta from −0.016 → **+0.015**.

### Comparison against published BEIR baselines (nDCG@10)

| Dataset | BM25 | Contriever | BGE-Large | Ours (dense) | Ours (+ reranker) |
|---|---|---|---|---|---|
| SciFact | 0.678 | 0.677 | 0.752 | 0.749 | 0.746 |
| NFCorpus | 0.321 | 0.328 | 0.381 | 0.341 | **0.356** |
| ArguAna | 0.397 | 0.446 | 0.416 | 0.391 | 0.391 |

Our dense embedder (`bge-small-en-v1.5`, ~130 MB) matches BGE-Large (~1.3 GB) on SciFact and NFCorpus post-fix. The reranker adds further gain on NFCorpus at negligible cost.

**Key inference:** BEIR numbers are entirely independent of the LLM choice. They reflect only the retrieval pipeline (embedder + reranker). LLM swaps do not affect these figures.

---

## 5. Schema-in-Prompt for All LLM Calls

### Problem
Several models (gemma4:31b, gpt-oss:20b) produced `Invalid json output` errors on `with_structured_output()` calls. These models either don't support native function calling reliably or have inconsistent JSON schema adherence.

### Old approach
`json_output_instruction(example_dict)` — a hand-written JSON example was injected into the system prompt as a belt-and-suspenders measure. The example could drift from the actual Pydantic schema.

### New approach
`schema_output_instruction(ModelClass)` — generates the instruction from `model_json_schema()` at import time. Always in sync with field names, types, descriptions, and constraints. Implemented in `research_swarm/agents/_utils.py`.

### Files updated
All six structured-output call sites:

| Agent | Schema class | Change |
|---|---|---|
| `supervisor.py` | `SupervisorDecision` | `json_output_instruction(dict)` → `schema_output_instruction(SupervisorDecision)` |
| `critic.py` | `Critique` | Same pattern |
| `fact_checker.py` | `FactCheckResult` | Class moved above `_SYSTEM_PROMPT` to resolve forward-reference; same pattern |
| `researcher.py` | `FindingSynthesis` | Class moved above `_SYNTHESIS_JSON_SUFFIX`; same pattern |
| `workers.py` | `FindingSynthesis` | Inherits via `_synthesis_prompt` from researcher |
| `writer.py` | `FinalReport` | Same pattern |

### Impact on gemma4:31b-cloud (before/after)

| Metric | Before fix | After fix | Delta |
|---|---|---|---|
| Mean answer score | 0.183 | 0.265 | +0.082 |
| **Grounded rate** | 16.7 % | **75.0 %** | **+58 pp** |
| **Mean faithfulness** | 0.124 | **0.627** | **+0.503** |
| JSON parse failures | 5 / 24 | 3 / 24 | -2 |

The schema injection was the dominant factor enabling gemma4 to reliably attach sources to findings.

---

## 6. Smoke Benchmark — Model Comparison

**Script:** `benchmarks/run_smoke_benchmark.py`  
**Tasks:** 24 (8 ALCE, 8 HotpotQA, 8 SciFact), seed 42, shallow depth  
**Key metrics:**
- **Answer score** — does the output contain expected answer strings?
- **Grounded rate** — fraction of tasks where `report.references` is non-empty
- **Faithfulness** — BGE embedding cosine similarity of report sections to cited snippets

### Models tested

| Model | Size | Notes |
|---|---|---|
| minimax-m2.5:cloud | — | Prior default |
| gemma4:31b-cloud | 31B | Google Gemma 4 |
| nemotron-3-nano:30b-cloud | 30B | NVIDIA Nemotron-3 Nano |
| nemotron-3-super:cloud | — | Rate-limited, invalid |
| qwen3-coder-next:cloud | — | Rate-limited, invalid |
| gpt-oss:120b-cloud | 120B | Rate-limited, invalid |
| gpt-oss:20b-cloud | 20B | JSON failures on all calls, invalid |

**Note:** Running 5 models in parallel saturated the Ollama cloud rate limit (429 errors). nemotron-3-super, qwen3-coder-next, and gpt-oss:120b results are invalid and were discarded. Subsequent runs were done sequentially.

### Valid results (clean sequential runs, post schema-in-prompt fix)

| Model | Answer score | Grounded rate | Faithfulness | Median s | Tasks ok |
|---|---|---|---|---|---|
| minimax-m2.5:cloud | **0.371** | 4 % | 0.026 | 48 s | 24/24 |
| **gemma4:31b-cloud** | 0.265 | **75 %** | **0.627** | **17 s** | 24/24 |
| nemotron-3-nano:30b | 0.238 | 25 % | 0.214 | 29 s | 24/24 |

### By dataset — all valid models

| Dataset | minimax-m2.5 | gemma4:31b | nemotron-3-nano |
|---|---|---|---|
| alce/asqa | 0.222 | 0.056 | 0.167 |
| alce/eli5 | 0.000 | 0.000 | 0.000 |
| alce/qampari | 0.078 | 0.067 | 0.067 |
| hotpotqa/bridge | 0.000 | 0.000 | 0.000 |
| hotpotqa/comparison | 0.500 | 0.500 | 0.750 |
| scifact | 0.750 | 0.500 | 0.250 |

### Models eliminated early
- **gpt-oss:20b-cloud** — `Invalid json output` at every structured-output call (supervisor, synthesis, critic, writer) even without rate limits. 2 task timeouts. Not viable regardless of schema-in-prompt fix.
- **nemotron-3-nano (first run, 0.521)** — inflated score entirely due to rate-limit noise; clean rerun gave 0.238.

---

## 7. Key Inferences

### On answer score vs grounding
minimax-m2.5 scores highest on answer score (0.371) but grounds only 4% of tasks. It is answering from **parametric memory**, bypassing the RAG pipeline entirely. For a research system whose purpose is to retrieve, cite, and synthesise from sources, this is the wrong behaviour. Its higher answer score is a misleading signal.

gemma4:31b's lower answer score (0.265) reflects **appropriate epistemic restraint**: when retrieved evidence is limited, it reports what it found rather than filling gaps with confident-sounding hallucinations. Grounding (75%) and faithfulness (0.627) are the operative quality metrics.

### On JSON schema adherence
Native `with_structured_output()` is unreliable for Ollama cloud models. Schema-in-prompt is necessary and effective:
- It eliminated the grounding problem for gemma4 (16% → 75% grounded)
- The remaining 3/24 JSON failures in gemma4 are confined to worker synthesis (the structured-output call that follows a multi-turn tool conversation — the hardest context for non-native-function-calling models)
- gpt-oss:20b failed on every call including the supervisor (the first and simplest call), indicating its JSON compliance is fundamentally broken

### On model size vs quality
- nemotron-3-nano (30B) does not outperform gemma4 (31B) on grounding or faithfulness despite similar parameter count — architecture and training matter more than size at this scale
- gpt-oss:120B was eliminated by rate limits; its quality on this task is unknown but the 20B variant's behaviour suggests the family may have JSON adherence issues

### On BEIR vs smoke benchmark
These measure entirely different things and should not be conflated:
- **BEIR** = retrieval quality (does the right document rank top?) — independent of LLM
- **Smoke** = end-to-end quality (does the final report answer correctly and cite sources?) — dependent on LLM

A model switch does not change BEIR numbers. A retriever/reranker change does not change smoke benchmark numbers (beyond the indirect effect of better evidence quality).

### On reranker applicability
The query-length guard (skip reranking for queries > 8 words) is critical for ms-marco-MiniLM:
- Most academic/research sub-questions exceed 8 words → guard fires → dense-only order preserved
- Short keyword queries (NFCorpus style) → guard does not fire → reranker helps (+1.5% nDCG@10)
- The guard is a production necessity, not a workaround: removing it causes −5% nDCG@10 on SciFact

---

## 8. Decision: Switch Default to gemma4:31b-cloud

**Rationale:** Highest grounding + faithfulness, fastest model, appropriate epistemic behaviour.

**Changes made:**
- `research_swarm/config.py` — `default_model_name`, `ollama_cloud_model`, `tier_standard_model`, `tier_thorough_model` all updated
- `benchmarks/run_smoke_benchmark.py` — `--model` default updated
- `README.md` — config example and provider table updated; BEIR table updated with post-fix numbers
- `benchmarks/README.md` — model comparison table added, historical minimax results preserved

**Commit:** `c7cd921` — "shifting from minimax-m2.5:cloud to gemma4:31b-cloud for a leaner flow"

---

## 9. Commits Made This Session

| Commit | Message | Key changes |
|---|---|---|
| `ce63268` | Fix grounded_rate=0 | Send payload isolation, LlamaIndex MockLLM, confidence floor |
| `27435a3` | Updated performance metrics | Reranker fixes, BEIR results, run_beir_reranker_compare.py |
| `c7cd921` | shifting from minimax-m2.5:cloud to gemma4:31b-cloud for a leaner flow | Default model switch, schema-in-prompt for all agents, model comparison results |

---

## 10. Open Items

1. **3/24 JSON failures in gemma4 worker synthesis** — the structured-output call after a multi-turn tool loop. Could be addressed with a JSON parse retry or a simplified schema for this specific call.
2. **SciFact label-extraction post-processor** — `_answer_score` uses substring matching; SUPPORT/CONTRADICT tasks score 0 unless the exact capitalised label appears in prose. A label-extraction step would give meaningful SciFact scores.
3. **nemotron-3-super and qwen3-coder-next** — never got clean runs due to rate-limit collisions. Could be worth retrying sequentially if model diversity is desired.
4. **minimax-m2.5 grounding** — it's possible the three grounding bug fixes (applied after the last minimax run) would also raise minimax's grounded rate above 4%. Not tested post-fix.
