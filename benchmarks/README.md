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
poetry run python benchmarks/run_beir_smoke.py
```

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

## Results — BEIR retrieval evaluation (2026-06-11, seed 42)

**Models:** `bge-small-en-v1.5` (dense) + `ms-marco-MiniLM-L-6-v2` (reranker, query-length-guarded)
**Guard:** reranking skipped when query word count > 8 (guard fires for SciFact and ArguAna)

### BEIR/SciFact (seed 42, 100 queries)

`mean_query_words=12.0`, `rerank_skipped=75/100`.

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE) | 0.777 | 0.828 | **0.749** | — |
| + Cross-encoder, no guard | 0.700 | 0.795 | 0.647 | −0.101 |
| + Cross-encoder, guard > 8 words | 0.737 | 0.798 | **0.696** | **−0.052** |

The guard skips 75 % of queries (> 8 words) and halves the regression.
25 shorter claims (≤ 8 words) are still reranked and remain somewhat
out-of-distribution for the MS MARCO cross-encoder.

### BEIR/NFCorpus (seed 42, 100 queries)

`mean_query_words=3.2`, `rerank_skipped=4/100`.

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE) | 0.130 | 0.165 | **0.341** | — |
| + Cross-encoder, guard > 8 words | 0.122 | 0.160 | **0.324** | **−0.016** |

Short keyword queries (avg 3.2 words) are mostly within the MS MARCO
cross-encoder's training distribution, so the guard fires rarely (4/100).
The remaining −0.016 delta is domain mismatch: the encoder was trained on
web QA, not biomedical literature.

### BEIR/ArguAna (seed 42, 100 queries)

`mean_query_words=194.9`, `rerank_skipped=100/100`.

| Method | Recall@5 | Recall@10 | nDCG@10 | Δ nDCG@10 |
|---|---|---|---|---|
| Dense (BGE) | 0.650 | 0.760 | **0.391** | — |
| + Cross-encoder, guard > 8 words | 0.650 | 0.760 | **0.391** | **+0.000** |

All 100 ArguAna queries exceed 8 words (avg ~195 words), so the guard fires
universally and the reranker is bypassed entirely — identical scores.

### BEIR summary (3 datasets, seed 42, 100 queries each)

| Dataset | Corpus | Dense nDCG@10 | Reranked nDCG@10 | Δ nDCG@10 | Guard fires |
|---|---|---|---|---|---|
| SciFact | 5,183 docs | 0.749 | 0.696 | −0.052 | 75 % |
| NFCorpus | 3,633 docs | 0.341 | 0.324 | −0.016 | 4 % |
| ArguAna | 8,674 docs | 0.391 | 0.391 | +0.000 | 100 % |

---

## Next steps

- [x] Re-run 24-task smoke benchmark after model rate limit resets — **done**
      (Run 2: mean_answer_score 0.197 → 0.225; ALCE/ASQA +167%).
- [x] Extend BEIR run with `nfcorpus` and `arguana` — **script done** (commit `c206bb0`);
      results pending the current background run.
- [x] Fix ToolMessage JSON truncation that dropped all source metadata — **done** (commit `69f9d36`).
- [ ] Re-run 24-task smoke benchmark with all three fixes applied to measure grounded_rate > 0.
- [x] Pre-bake session_id into retriever tool so LLM cannot hallucinate it — **done** (this change).
- [ ] Add SciFact label-extraction post-processor to `_answer_score` so that
      SUPPORT / CONTRADICT tasks are scored correctly.
- [ ] Consider a biomedical cross-encoder (e.g. `cross-encoder/nboost/pt-biobert-base-msmarco`)
      for scientific corpora once the query-length guard is validated on all three BEIR subsets.
