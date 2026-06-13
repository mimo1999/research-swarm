# Benchmark datasets

Download the benchmark inputs with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/download_datasets.ps1
```

The script stores data under `data/benchmarks/`, which is intentionally ignored
by Git. Existing downloads and extracted directories are reused.

Included releases:

- **ALCE**: ASQA, QAMPARI, and ELI5 plus the authors' retrieved passages.
- **HotpotQA**: full distractor-setting train and validation Parquet splits.
- **SciFact**: official claim-verification release.
- **BEIR**: SciFact, NFCorpus, and ArguAna retrieval subsets.

`data/benchmarks/SHA256SUMS.txt` records hashes for the downloaded archives and
files so a benchmark run can report the exact local inputs it used.

Run the two-hour smoke benchmark with:

```powershell
poetry run python benchmarks/run_smoke_benchmark.py
```

The fixed seed-42 sample contains 8 ALCE, 8 HotpotQA, and 8 SciFact tasks.
Results, the exact task manifest, and a summary are written to
`data/benchmark_results/`.

Run the BEIR retrieval + reranking benchmark with:

```powershell
poetry run python benchmarks/run_beir_reranker_compare.py
# or a subset:
poetry run python benchmarks/run_beir_reranker_compare.py --datasets scifact nfcorpus --queries 50
# skip the embedding cache:
poetry run python benchmarks/run_beir_reranker_compare.py --no-cache
```

Corpus embeddings are cached to `data/benchmark_results/emb_cache/` on first run
(~20 min total); subsequent runs load from disk and complete in a few minutes.

---

## Results — model comparison (2026-06-14, seed 42, 24 tasks)

All runs: shallow depth, concurrency 2, sequential (no rate-limit interference).
Fixes applied: grounding bug (Send payload), schema-in-prompt for all structured LLM calls.

| Model | Answer score | Grounded rate | Faithfulness | Median s | Notes |
|---|---|---|---|---|---|
| **gemma4:31b-cloud** *(default)* | 0.265 | **75 %** | **0.627** | **17 s** | Best grounding + faithfulness; fastest |
| minimax-m2.5:cloud | **0.371** | 4 % | 0.026 | 48 s | Answers from memory; bypasses RAG |
| nemotron-3-nano:30b-cloud | 0.238 | 25 % | 0.214 | 29 s | Moderate grounding |

`gemma4:31b-cloud` is the default. Its lower answer score vs minimax reflects appropriate epistemic restraint — it reports what the retrieved evidence supports rather than filling gaps from parametric memory. Grounding (75%) and faithfulness (0.627) are the operative quality metrics for a retrieval-based research system.

### By dataset — gemma4:31b-cloud (default)

| Dataset | Tasks | Answer score | Mean s |
|---|---|---|---|
| alce/asqa | 3 | 0.056 | 71 s |
| alce/eli5 | 2 | 0.000 | 33 s |
| alce/qampari | 3 | 0.067 | 48 s |
| hotpotqa/bridge | 4 | 0.000 | 23 s |
| hotpotqa/comparison | 4 | 0.500 | 17 s |
| scifact | 8 | 0.500 | 20 s |

### Historical: minimax-m2.5:cloud (2026-06-11, pre-fix baseline)

**Model:** `minimax-m2.5:cloud` · **Fix applied:** `tool_choice="required"` in shallow mode

| Metric | Run 1 (before fix) | Run 2 (after fix) | Delta |
|---|---|---|---|
| Tasks | 24 / 24 ok | 24 / 24 ok | — |
| Elapsed | 806 s | 1244 s | +438 s |
| Median task time | 51.6 s | 81.2 s | +30 s |
| **Mean answer score** | **0.197** | **0.225** | **+0.028** |
| Grounded rate | — | 0 % (bug) | — |

### Root-cause analysis

**Remaining gaps (after tool_choice fix):**

1. **Grounded rate = 0 %** — three compounding bugs, all fixed (commit `ce63268`):

   - *JSON truncation* (commit `69f9d36`): ToolMessage content was cut mid-array;
      JSON parse failed silently; all source metadata was lost.
      Fix: truncate per-item snippets before encoding, never the array boundary.

   - *Wrong session_id* (commit `01bab66`): the `retrieve_from_rag` tool required the
     LLM to supply the session_id (a long UUID) in its tool call arguments.  The
     model hallucinated or ignored it, so every retrieval query hit an empty Chroma
     collection and returned `[]`.  **Fix: `session_id` is now pre-baked at tool
     construction time** (`build_retriever_tool(session_id=session_id)` in
     `_get_researcher_tools`); the LLM only needs to provide the `query`.

   - *LangGraph Send payload isolation* (commit `ce63268`): `Send("worker_node", payload)`
     gives the receiving node ONLY the payload dict — the full graph state is NOT merged.
     `session_id` was missing from the payload so `worker_node` queried the wrong Chroma
     collection.  Fix: explicitly forward `session_id`, `query`, `model_provider`, and
     `model_name` in every Send payload.

   - *Fact-checker confidence floor* (commit `ce63268`): `minimax-m2.5:cloud`
     systematically returns `confidence_score=0.0` for valid evidence-backed claims.
     The writer filters out findings with confidence < 0.1, so every evidence-backed
     finding was discarded, leaving `references=[]`.  Fix: `max(score, 0.15)` when
     evidence is present — a finding backed by real sources can never score below the
     no-evidence baseline (0.1).

2. **Multi-hop HotpotQA bridge questions** — shallow mode dispatches one worker
   with one tool turn.  Bridge questions require chaining two facts across
   separate documents.  A single retrieval pass cannot reliably bridge both hops.
   This is an inherent limitation of shallow depth, not a bug.  Answer score = 0.0
   for all 4 bridge tasks.

3. **SciFact label classification via substring matching** — the scoring metric
   (`answer_score`) checks whether the expected label (SUPPORT / CONTRADICT /
   NOT_ENOUGH_INFO) appears as a substring in the generated report text.  The two
   NOT_ENOUGH_INFO tasks receive an automatic 1.0 score (the label string appears
   in the prompt); the six SUPPORT/CONTRADICT tasks score 0.0 (the model generates
   prose without the exact capitalised label).  The metric needs a
   post-processing step that extracts the first capitalised label word.

---

## Results — BEIR retrieval evaluation (2026-06-13, seed 42)

**Models:** `bge-small-en-v1.5` (dense) + `ms-marco-MiniLM-L-6-v2` (reranker, query-length-guarded)
**Guard:** reranking skipped when query word count > 8

Three reranker implementation fixes applied vs the 2026-06-11 run:
- Removed `[:512]` character truncation (was ~100 tokens); tokenizer now handles truncation at 512 tokens
- Title prepended to passage (same signal used by the dense retriever)
- Snippet cap raised from 800 to 2,000 characters (~400 tokens)

### BEIR/SciFact (seed 42, 100 queries)

`mean_query_words=12.0`, `guard_fires=75/100`

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE-small) | 0.777 | 0.828 | **0.749** | — |
| + ms-marco-MiniLM (guard > 8 words) | 0.797 | 0.848 | **0.746** | **-0.002** |

75 % of queries exceed 8 words and are not reranked. The −0.002 delta on the
remaining 25 shorter claims is within noise — scientific claims are still
partially out-of-distribution for the MS MARCO encoder.

### BEIR/NFCorpus (seed 42, 100 queries)

`mean_query_words=3.2`, `guard_fires=4/100`

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE-small) | 0.130 | 0.165 | **0.341** | — |
| + ms-marco-MiniLM (guard > 8 words) | 0.139 | 0.169 | **0.356** | **+0.015** |

Short keyword queries (avg 3.2 words) sit squarely in the MS MARCO training
distribution. The implementation fixes (title prepend + full token budget)
flipped this from −0.016 (old run) to +0.015.

### BEIR/ArguAna (seed 42, 100 queries)

`mean_query_words=194.9`, `guard_fires=100/100`

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE-small) | 0.650 | 0.760 | **0.391** | — |
| + ms-marco-MiniLM (guard > 8 words) | 0.650 | 0.760 | **0.391** | **+0.000** |

All 100 queries are full argument paragraphs (avg 195 words); the guard fires
universally and the reranker is bypassed entirely.

### BEIR summary (3 datasets, seed 42, 100 queries each)

| Dataset | Corpus | Dense nDCG@10 | Reranked nDCG@10 | Δ nDCG@10 | Guard fires |
|---|---|---|---|---|---|
| SciFact | 5,183 docs | 0.749 | 0.746 | -0.002 | 75 % |
| NFCorpus | 3,633 docs | 0.341 | 0.356 | **+0.015** | 4 % |
| ArguAna | 8,674 docs | 0.391 | 0.391 | +0.000 | 100 % |

---

## Next steps

- [x] Re-run 24-task smoke benchmark after model rate limit resets — **done**
      (Run 2: mean_answer_score 0.197 → 0.225; ALCE/ASQA +167%).
- [x] Extend BEIR run with `nfcorpus` and `arguana` — **done** (2026-06-13).
- [x] Fix ToolMessage JSON truncation that dropped all source metadata — **done** (commit `69f9d36`).
- [x] Re-run smoke benchmark with all fixes applied — **done** (2026-06-14). grounded_rate 0 → 75 % with gemma4.
- [x] Pre-bake session_id into retriever tool so LLM cannot hallucinate it — **done**.
- [x] Schema-in-prompt for all structured LLM calls — **done** (2026-06-14). `schema_output_instruction(ModelClass)` injected alongside `with_structured_output` in every agent; grounded_rate on gemma4 jumped from 17 % → 75 %.
- [x] Switch default model to `gemma4:31b-cloud` — **done** (2026-06-14). Best grounding (75 %) and faithfulness (0.627) across tested models.
- [ ] Add SciFact label-extraction post-processor to `_answer_score` so that
      SUPPORT / CONTRADICT tasks are scored correctly.
- [ ] Investigate remaining 3/24 JSON parse failures in worker synthesis for gemma4
      (structured-output call after multi-turn tool loop).
- [ ] Consider a biomedical cross-encoder for scientific corpora once the query-length guard is validated on all three BEIR subsets.
