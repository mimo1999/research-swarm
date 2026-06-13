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

## Results — smoke benchmark (2026-06-11 run 2, seed 42)

**Model:** `minimax-m2.5:cloud` via Ollama · **Depth:** shallow · **Concurrency:** 2
**Fix applied:** `tool_choice="required"` in shallow mode — forces retrieval on every task

| Metric | Run 1 (before fix) | Run 2 (after fix) | Delta |
|---|---|---|---|
| Tasks | 24 / 24 ok | 24 / 24 ok | — |
| Elapsed | 806 s (13 m 26 s) | 1244 s (20 m 44 s) | +438 s |
| Median task time | 51.6 s | 81.2 s | +30 s |
| **Mean answer score** | **0.197** | **0.225** | **+0.028** |
| Grounded rate | — | 0 % → fix pending | — |

Note: longer elapsed time in Run 2 reflects the model actually using the retrieval tool
on every task rather than falling back immediately from memory.

### By dataset

| Dataset | Tasks | Answer score (run 1) | Answer score (run 2) | Delta |
|---|---|---|---|---|
| alce/asqa | 3 | 0.056 | 0.222 | **+0.167** |
| alce/eli5 | 2 | 0.000 | 0.000 | — |
| alce/qampari | 3 | 0.189 | 0.245 | +0.056 |
| hotpotqa/bridge | 4 | 0.000 | 0.000 | — |
| hotpotqa/comparison | 4 | 0.500 | 0.500 | — |
| scifact | 8 | 0.250 | 0.250 | — |

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
- [ ] Re-run 24-task smoke benchmark with all three fixes applied to measure grounded_rate > 0.
- [x] Pre-bake session_id into retriever tool so LLM cannot hallucinate it — **done** (this change).
- [ ] Add SciFact label-extraction post-processor to `_answer_score` so that
      SUPPORT / CONTRADICT tasks are scored correctly.
- [ ] Consider a biomedical cross-encoder (e.g. `cross-encoder/nboost/pt-biobert-base-msmarco`)
      for scientific corpora once the query-length guard is validated on all three BEIR subsets.
