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

Run the smoke benchmark with:

```powershell
poetry run python benchmarks/run_smoke_benchmark.py
# another model, or a cheaper worker tier:
poetry run python benchmarks/run_smoke_benchmark.py --model nemotron-3-nano:30b-cloud
```

The fixed seed-42 sample contains 8 ALCE, 8 HotpotQA, and 8 SciFact tasks.
Results, the exact task manifest, and a summary are written to
`data/benchmark_results/`.

### How the benchmark works now (closed corpus, no retrieval index)

Each task's supplied evidence documents are handed to the graph as
`ingested_documents` -- the same path a user-uploaded PDF takes: one full-text claim
extraction call per document (`document_pass_node` / `document_worker_node`), no chunking,
no embeddings, no vector store. The web/literature scout is switched off
(`settings.enable_fetch_pass = False`), and any worker that still runs can only call a
`search_supplied_corpus` tool that searches the task's own documents. The score therefore
measures reading, reasoning and writing over a known evidence set, not search.

Every metric is computed **without an LLM**, so scoring is deterministic and free
(`compute_task_metrics` in `run_smoke_benchmark.py`, a pure function of the saved results). The
summary JSON groups them:

| Group | Metric | Meaning |
|---|---|---|
| correctness | `answer_score` | Fraction of expected answers found in the report. **SciFact** is scored on the verdict the report names first (SUPPORT / CONTRADICT / NOT_ENOUGH_INFO): 1.0 on an exact match, else 0.0. |
| | `evidence_answerable` | The same check run on the supplied corpus itself -- the ceiling for `answer_score`. Not defined for SciFact. |
| | `normalized_answer_score` | `answer_score / evidence_answerable`: separates model failures from tasks the corpus cannot answer. |
| trust | `number_grounding` | Share of the report's numbers (decimals, percentages, integers >= 11; citation markers ignored) that appear in the corpus or the question. The rest are *ungrounded*: invented **or** derived (a sum), so read it as an upper bound on fabrication. |
| | `abstained` / `unanswerable` | On tasks the corpus cannot answer (expected answer absent from it, or SciFact NOT_ENOUGH_INFO): did the report decline? Summary reports `abstention_rate_on_unanswerable` and `over_abstention_rate_on_answerable`. Abstention is a keyword heuristic on the executive summary. |
| citations | `support_doc_recall` / `precision` | Cited documents vs the gold supporting documents (HotpotQA: the 2 gold among 10; SciFact: the cited abstracts). |
| localization | `finding_recall`, `synthesis_loss` | Expected answers found in the *findings* vs in the final report. `synthesis_loss > 0` means the answer was extracted and then lost. |
| | `false_refute_rate`, `writer_drop_rate` | Of the findings that contain an expected answer: the share the verifier called weak/refuted, and the share the writer would never see (refuted, or confidence < 0.1). |
| efficiency | `seconds` (p50/p95), `stage_wall_s`, `llm_calls`, tokens, `reasoning_share_of_output_chars`, `llm_errors`, `success_rate`, `answer_score_per_100k_tokens` | From each task's trace file (`data/traces/<session>.jsonl`). |
| judge | `judge_overall` / `judge_verdict` | The independent LLM judge's review (coherence, relevance, completeness, citation quality; overall is the mean of the four 1-5 criteria). The only metric that costs an LLM call, and it is the pipeline's own model. |
| | `grounded` | Whether the report cites at least one source. |

Each result also keeps `finding_details` and `critique_details`, so localization metrics can be
recomputed or inspected without re-running. Use `--datasets hotpotqa,scifact` (dataset prefixes)
to run a subset.

With 24 tasks each metric carries roughly +/-20 points of sampling noise: compare runs on the same
tasks, repeat runs before trusting a small difference, and treat per-dataset numbers (3-8 tasks) as
anecdotes.

Unlike the runs below, the old `faithfulness` metric no longer exists (it scored report
sections against cited snippets by embedding similarity) and SciFact's scoring changed, so
new runs are **not comparable** with the 2026-06 numbers. Compare new runs only with each other.

**Provider concurrency:** on Ollama Cloud, long reasoning calls can effectively run one at a
time; `--concurrency 2` (the default) is safe, higher values risk `429: timed out waiting for a
concurrent request slot`.

### Generating the tables

Three tables, each produced by its own script and written into this README between
`<!-- TABLE-X:START -->` / `<!-- TABLE-X:END -->` markers (`--update-readme`). Run them one at a
time -- Ollama Cloud serves about one long request at a time per account, so overlapping jobs
distort each other's timings and cause 429s.

| Table | What it measures | Command |
|---|---|---|
| **B** relevance filter | How well the scout's LLM 0-10 scorer ranks relevant abstracts, vs word overlap and random (HotpotQA distractor pools, NFCorpus, SciFact) | `python benchmarks/relevance_benchmark.py --n 100 --update-readme` |
| **A** closed-corpus truth | Correctness, faithfulness, citation recall / correctness / completeness, critical errors, abstention (HotpotQA, ALCE, SciFact) | 1. generate: `python benchmarks/run_smoke_benchmark.py --n-per-dataset 100 --model nemotron-3-nano:30b-cloud`  2. judge: `python benchmarks/score_claims.py --results data/benchmark_results/smoke-<ts>-results.jsonl --update-readme` |
| **C** open-web coverage | Open-web NFCorpus queries: every cited source graded 0/1/2 by `gemma4:31b-cloud` (the qrels cover only ~2% of live PubMed results, so they are a sparse secondary column), against a single-PubMed-search baseline | `python benchmarks/run_nfcorpus_coverage.py --n 100 --update-readme` |

**Ablating a pipeline change.** Tunable settings can be overridden per run with `--set KEY=VALUE`
(for example `min_grounded_facts`, `extract_batch_chars`, `extract_max_facts_per_pair`,
`max_facts_for_writer`, `paper_topk_floor`; an unknown key is an error). The legacy modes the ablation
table below compares against (critic + fact-checker, ReAct worker, per-document extraction) were deleted after that comparison
(the free-form `run_writer` stays, as the attributed writer's fallback), so those rows are a record, not something to re-run. To
measure a change on a fixed task set, one job at a time:

```
python benchmarks/make_ablation_subset.py                  # once: 30 ok tasks per family
python benchmarks/run_smoke_benchmark.py --task-file data/benchmark_results/ablation-90-tasks.json \
    --model nemotron-3-nano:30b-cloud --set llm_judge_enabled=false --set min_grounded_facts=2
python benchmarks/score_claims.py --results data/benchmark_results/smoke-<ts>-results.jsonl
python benchmarks/compare_runs.py --baseline <previous run prefix> --candidate <this run prefix> [--only scifact]
```

`compare_runs.py` reports paired-bootstrap differences on the tasks both runs finished; compare
against a run made the same day, because run-to-run drift on the same model is large.
`relevance_benchmark.py --evaluate-only main --selection topk` re-scores the relevance filter's
selection rule from saved scores with no LLM calls.

Table A's claim-level columns use an **independent judge** (`--judge-model`, default
`gemma4:31b-cloud`; `score_claims.py` refuses the model that wrote the reports). The judge
splits each report into sentences and rules on every one against the supplied documents:
checkworthy, supported by its own citations, supported by any document, contradicted. Faithfulness
= supported checkworthy sentences / checkworthy; citation recall (ALCE) = sentences entailed by
their own citations / checkworthy; citation correctness = cited sentences whose citations support
them / cited sentences; citation completeness = document-supported sentences that carry a citation
/ all document-supported sentences. Dangling citations (pointing at no supplied document) are
counted without the judge.

**Validate the judge before trusting those columns:** `python benchmarks/judge_validation.py
export --claims <smoke-run>-claims.jsonl` writes ~40 stratified sentences to a CSV; label
`human_cited_support` (yes / partial / no) from the cited excerpts, then `judge_validation.py
score` reports agreement and Cohen's kappa, and Table A replaces its "not validated" warning with
the measured kappa.

### BEIR retrieval benchmark (removed)

`run_beir_smoke.py` / `run_beir_reranker_compare.py` benchmarked the embedding retriever and
cross-encoder reranker, which were removed together with Chroma and the RAG layer. The
scripts are in git history (before the commit that removed embeddings); the BEIR results
below are kept only as a record of that system.

---

> **Everything below is historical.** These results were measured on the previous
> retrieval-based pipeline (per-session Chroma index, `retrieve_from_rag`, embedding
> faithfulness score, cross-encoder reranker), all of which have since been removed.
> Grounded rate and faithfulness in particular no longer mean what they did, and the model
> recommendation below was made on those metrics. Re-run the benchmark before relying on it.

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

## Results — reranker model comparison (2026-08-23, seed 42) — current

Rerun of `run_beir_reranker_compare.py` against all three reranker methods in
one pass, same cached corpus embeddings as the 2026-07-21/22 run (SciFact's
5,183-doc cache is byte-identical, so its dense baseline is unchanged). No
production RAG/reranker code changed since that run — this rerun exists to
confirm the comparison still holds after this session's unrelated writer/
query-engine fixes, not because the reranker was touched.

| Dataset | Dense nDCG@10 | ms-marco Δ | bge-base Δ | mxbai-xsmall Δ | Guard fires |
|---|---|---|---|---|---|
| SciFact | 0.7485 | -0.0021 | +0.0067 | **+0.0095** | 75 % |
| NFCorpus | 0.3405 | +0.0153 | +0.0095 | **+0.0191** | 4 % |
| ArguAna | 0.3907 | +0.0000 | +0.0000 | +0.0000 | 100 % |

Reranker ranking (ms-marco < bge-base < mxbai-xsmall on Δ nDCG@10) is
unchanged from the 2026-07-21/22 run below — `mxbai-rerank-xsmall-v1` still
wins on quality everywhere it's exercised, `bge-reranker-base` still never
regresses vs dense, and ArguAna's queries still hit the length guard 100% of
the time. **Production reranker stays on `bge-reranker-base`** per the
latency analysis in the 2026-07-22 section below — nothing here changes that
call.

NFCorpus and ArguAna's dense nDCG@10 baselines moved (0.3131→0.3405,
0.4574→0.3907) versus the 2026-07-21/22 numbers despite loading the same
cached corpus embeddings — the query *sample* differs run to run for those
two datasets (their eligible-query pool sizes aren't fixed the way SciFact's
apparently is), so seed 42 draws a different 100-query subset each time. This
is a sampling-variance artifact of the benchmark script, not a retrieval
regression; SciFact's number is reproduced exactly, confirming the underlying
embeddings/ranking logic are unchanged.

Timing (500/1,920 pairs, SciFact/NFCorpus): ms-marco 34s/131s, bge-base
227s/887s, mxbai-xsmall 124s/500s. The ms-marco/bge-base ratio (~6-7x)
matches the 2026-07-21/22 run, but mxbai-xsmall was markedly *faster* than
bge-base this time (previously ~10x *slower*) — a big enough swing that it
looks like a real difference in this run's environment (e.g. a cold vs.
warm model-weights cache, CPU contention from another process) rather than
normal noise, though the cause wasn't isolated here. Quality ranking is
unaffected either way, and the latency-driven production choice stands, but
the "mxbai is always ~10x slower" latency claim from 2026-07-22 shouldn't be
treated as a fixed constant until this is re-checked on a quiet machine.

Results saved: `data/benchmark_results/beir-reranker-compare-20260823-151605.json`.

## Results — reranker model comparison (2026-07-21/22, seed 42)

**Production reranker (`research_swarm/rag/reranker.py`) switched from
`cross-encoder/ms-marco-MiniLM-L-6-v2` (22 MB) to `BAAI/bge-reranker-base`
(280 MB).** `bge-reranker-v2-m3` (the literal "v2" release, 2.2 GB) was
evaluated first but consistently failed to load on this machine's available
RAM (~1.9 GB free of 16 GB total) — the process died silently with no
Python exception. `bge-reranker-base` is the v1-generation, smaller BGE
reranker and loaded/ran without issue.

`benchmarks/run_beir_reranker_compare.py` now scores three reranker methods
(dense, ms-marco-MiniLM, bge-reranker-base) in one pass per dataset, same
100-query seed-42 sample and query-length guard (>8 words skips reranking)
as the 2026-06-13 run:

| Dataset | Dense nDCG@10 | ms-marco nDCG@10 | Δ | bge-base nDCG@10 | Δ | Guard fires |
|---|---|---|---|---|---|---|
| SciFact | 0.7485 | 0.7464 | -0.0021 | **0.7552** | **+0.0067** | 75 % |
| NFCorpus | 0.3131 | **0.3379** | **+0.0248** | 0.3206 | +0.0075 | 4 % |
| ArguAna | 0.4574 | 0.4574 | +0.0000 | 0.4574 | +0.0000 | 100 % |

Takeaways:
- `bge-reranker-base` never regresses nDCG@10 versus dense retrieval alone
  (ms-marco does, on SciFact: -0.0021). It's the more consistent choice.
- `ms-marco-MiniLM` still wins outright on NFCorpus's short keyword queries
  (its original training distribution).
- `bge-reranker-base` is markedly slower on CPU: ~6x the wall-clock of
  ms-marco-MiniLM for the same pair count (e.g. NFCorpus: 768s vs 128s for
  1,920 pairs; SciFact: 198s vs 30s for 500 pairs). Worth watching if
  reranking latency becomes a bottleneck in a live research run.

### mxbai-rerank-xsmall-v1 (2026-07-22, seed 42) — quality winner, latency disqualifies it

A fourth method, `mixedbread-ai/mxbai-rerank-xsmall-v1` (~70M params, ~140 MB
— smaller than `bge-reranker-base` and even most of the 2024-generation
rerankers), was added to `RERANKER_MODELS` in the same script to check
whether a newer, lighter model than `bge-reranker-base` could match or beat
it. Same 100-query seed-42 samples:

| Dataset | Dense | ms-marco Δ | bge-base Δ | mxbai-xsmall Δ | Guard fires |
|---|---|---|---|---|---|---|
| SciFact | 0.7485 | -0.0021 | +0.0067 | **+0.0095** | 75 % |
| NFCorpus | 0.3131 | +0.0248 | +0.0075 | **+0.0263** | 4 % |
| ArguAna | 0.4574 | +0.0000 | +0.0000 | +0.0000 | 100 % |

`mxbai-rerank-xsmall-v1` produced the best nDCG@10 on every dataset it was
exercised on — but at a wall-clock cost that rules it out for this CPU-only
pipeline:

| Pairs scored | ms-marco | bge-base | mxbai-xsmall |
|---|---|---|---|
| 500 (SciFact) | 26 s | 199 s | 1,970 s (~76x ms-marco, ~10x bge-base) |
| 1,920 (NFCorpus) | 121 s | 770 s | 7,581 s (~63x ms-marco, ~10x bge-base) |

Despite having roughly a quarter of `bge-reranker-base`'s parameter count,
`mxbai-rerank-xsmall-v1` runs ~10x slower per pair on CPU — parameter count
is not a reliable proxy for CPU inference cost here; the architecture isn't
optimized for the same cheap batched sequence-classification path the
BERT-style cross-encoders use. **Production reranker stays on
`bge-reranker-base`** — mxbai's quality edge doesn't justify a further 10x
latency hit on top of the 6x already paid moving off ms-marco-MiniLM.
- ArguAna's 195-word average queries hit the length guard 100% of the time
  for both models, so neither reranker is ever exercised there — dense
  retrieval numbers are unchanged by definition.

Results saved under `data/benchmark_results/beir-reranker-compare-*.json`.

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

> The checklist below dates from the June 2026 retrieval-based pipeline and is kept as history.
> Current work on open-web report quality (question frame, sectioned writer, strict-claim checks)
> is evaluated with live questions in `quality_review/` (q9-q11 are constraint-qualified) and the
> sample reports in `reports/`; see TECHNICAL_HANDOFF.md section 11 for the open issues.

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

<!-- TABLE-B:START -->
### Table B - relevance-filter quality (LLM 0-10 scorer on pools with known relevance)

Mean over pools, 95% bootstrap CI in brackets. Ties are averaged, not broken.

| Dataset | Pools | Ranker | nDCG@10 | AUC | Expected P@3 |
|---|---|---|---|---|---|
| hotpotqa | 100 | **LLM scorer** | 0.848 [0.820, 0.873] | 0.792 [0.761, 0.822] | 0.460 [0.429, 0.488] |
| hotpotqa | 100 | word overlap | 0.787 [0.754, 0.820] | 0.728 [0.683, 0.770] | 0.407 [0.368, 0.443] |
| hotpotqa | 100 | random | 0.582 [0.550, 0.615] | 0.525 [0.486, 0.565] | 0.203 [0.167, 0.240] |
| nfcorpus | 100 | **LLM scorer** | 0.438 [0.405, 0.473] | 0.556 [0.536, 0.578] | 0.367 [0.322, 0.413] |
| nfcorpus | 100 | word overlap | 0.245 [0.203, 0.292] | 0.424 [0.394, 0.457] | 0.206 [0.156, 0.260] |
| nfcorpus | 100 | random | 0.373 [0.337, 0.411] | 0.528 [0.502, 0.556] | 0.260 [0.213, 0.310] |
| scifact | 100 | **LLM scorer** | 0.598 [0.523, 0.668] | 0.777 [0.729, 0.823] | 0.212 [0.181, 0.243] |
| scifact | 100 | word overlap | 0.523 [0.445, 0.604] | 0.774 [0.725, 0.821] | 0.188 [0.156, 0.219] |
| scifact | 100 | random | 0.194 [0.143, 0.244] | 0.482 [0.419, 0.540] | 0.050 [0.027, 0.073] |

| Dataset | Selection | Precision | Recall | Mean kept | Empty rate | Topped-up rate |
|---|---|---|---|---|---|---|
| hotpotqa | strict >= 0.75 | 0.905 [0.853, 0.948] | 0.585 [0.530, 0.640] | 1.36 | 0.07 | - |
| hotpotqa | as shipped (+ top-up) | 0.865 [0.810, 0.918] | 0.605 [0.545, 0.660] | 1.51 | 0.06 | 0.11 |
| nfcorpus | strict >= 0.75 | 0.574 [0.449, 0.697] | 0.095 [0.066, 0.125] | 0.96 | 0.53 | - |
| nfcorpus | as shipped (+ top-up) | 0.578 [0.462, 0.694] | 0.112 [0.081, 0.144] | 1.14 | 0.49 | 0.1 |
| scifact | strict >= 0.75 | 0.634 [0.530, 0.732] | 0.497 [0.400, 0.587] | 1.01 | 0.38 | - |
| scifact | as shipped (+ top-up) | 0.616 [0.518, 0.708] | 0.527 [0.430, 0.620] | 1.15 | 0.36 | 0.12 |

Hard negatives are the non-relevant documents with the MOST word overlap with the query, so the word-overlap baseline is handicapped by construction (on NFCorpus and SciFact it can fall below random): compare the LLM scorer with the random row as the floor, and with word overlap only on HotpotQA, whose distractors were not chosen by overlap. NFCorpus relevance means 'linked from the same article', so many topically relevant abstracts are unlabelled, and its queries are video headlines, not questions.

Paired nDCG@10 difference, LLM minus word overlap - hotpotqa: 0.061 [0.020, 0.100]; nfcorpus: 0.193 [0.148, 0.238]; scifact: 0.075 [-0.008, 0.157]
<!-- TABLE-B:END -->

<!-- TABLE-A:START -->
### Table A - closed-corpus answer quality and truth (current pipeline)

Mean over tasks, 95% bootstrap CI in brackets. Claim-level columns are judged by `gemma4:31b-cloud` (a different model from the generator). **The judge is not yet validated against hand labels** (`benchmarks/judge_validation.py`); read faithfulness and citation columns accordingly.

| Dataset | Tasks | Answer | Answer / ceiling | Critical errors | Faithfulness | Cite recall | Cite correctness | Cite completeness | Dangling cites | Abstained: unanswerable | Abstained: answerable | Gold-doc recall | Time p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alce | 100 | 0.22 [0.16, 0.28] | 0.54 [0.44, 0.64] | 5.00% [1.00, 10.00] | 0.80 [0.74, 0.86] | 0.76 [0.69, 0.82] | 0.76 [0.70, 0.82] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1/37 | 1/63 | - | 14.247 |
| hotpotqa | 100 | 0.71 [0.62, 0.80] | 0.70 [0.60, 0.78] | 10.00% [5.00, 16.00] | 0.86 [0.81, 0.90] | 0.77 [0.71, 0.82] | 0.77 [0.71, 0.82] | 0.99 [0.97, 1.00] | 0.00 [0.00, 0.00] | 0/7 | 1/93 | 0.77 [0.70, 0.82] | 19.09 |
| scifact | 100 | 0.47 [0.38, 0.57] | - | 26.00% [17.00, 34.00] | 0.62 [0.53, 0.70] | 0.60 [0.52, 0.69] | 0.60 [0.52, 0.69] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1/33 | 2/67 | 0.93 [0.88, 0.98] | 22.123 |
| all | 300 | 0.47 [0.42, 0.52] | 0.63 [0.57, 0.70] | 13.67% [10.00, 17.67] | 0.77 [0.73, 0.80] | 0.71 [0.67, 0.75] | 0.72 [0.68, 0.76] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 2/77 | 4/223 | 0.85 [0.81, 0.89] | 18.234 |

Answer = fraction of expected answers in the report (SciFact: verdict match). Answer / ceiling divides by what the supplied documents can answer at all. Critical errors = reports with a SciFact SUPPORT<->CONTRADICT flip or a sentence a document contradicts. Abstained columns are counts of tasks where the report declined to answer (unanswerable: should be high; answerable: should be low).
<!-- TABLE-A:END -->

<!-- ABLATION:START -->
### Pipeline v2 ablation (fixed 90-task closed-corpus subset)

Means over the same 90 tasks (30 each of ALCE, HotpotQA, SciFact) generated by `nemotron-3-nano:30b-cloud`; claim-level columns judged by `gemma4:31b-cloud` (not yet validated against hand labels). Each row adds one change to the previous row; the last column lists paired-bootstrap differences (candidate minus previous row) whose 95% CI excludes zero. Run-to-run noise is large (a re-run of the unchanged pipeline on another day produced 4x fewer findings), so only paired, same-day comparisons are meaningful.

| Configuration (cumulative) | Answer | Finding recall | False-refute | Faithfulness | Cite recall | Cite complete. | Critical err. | Time (s) | LLM calls | Significant vs previous |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 pipeline (re-run baseline) | 0.27 | 0.26 | 0.53 | 0.64 | 0.07 | 0.25 | 0.16 | 50.6 | 10.2 | - |
| + question to writer/extractor | 0.30 | 0.31 | 0.67 | 0.65 | 0.23 | 0.38 | 0.19 | 58.4 | 10.0 | citation_recall +0.16 BETTER; citation_correctness +0.35 BETTER; seconds +7.83 WORSE |
| + grounded evidence windows | 0.35 | 0.27 | 0.63 | 0.63 | 0.13 | 0.33 | 0.20 | 58.8 | 10.1 | citation_correctness -0.28 WORSE |
| + single verifier | 0.30 | 0.28 | 0.25 | 0.59 | 0.24 | 0.50 | 0.26 | 67.6 | 9.5 | false_refute_rate -0.37 BETTER; over_abstention -0.13 BETTER; number_grounding -0.25 WORSE; citation_recall +0.13 BETTER; llm_calls -0.59 BETTER |
| + multi-fact packed extraction | 0.44 | 0.45 | 0.28 | 0.66 | 0.26 | 0.48 | 0.27 | 30.9 | 4.2 | answer_score +0.15 BETTER; finding_recall +0.17 BETTER; number_grounding +0.18 BETTER; faithfulness +0.15 BETTER; seconds -36.95 BETTER; llm_calls -5.15 BETTER |
| + attributed writer | 0.47 | 0.45 | 0.27 | 0.80 | 0.73 | 0.98 | 0.16 | 24.9 | 4.3 | number_grounding +0.11 BETTER; faithfulness +0.14 BETTER; citation_recall +0.47 BETTER; citation_correctness +0.15 BETTER; citation_completeness +0.49 BETTER; critical_error -0.12 BETTER; seconds -5.94 BETTER |
| + gap fill, 2 facts/sub-question (closed corpus) | 0.44 | 0.44 | 0.29 | 0.79 | 0.75 | 0.97 | 0.14 | 29.0 | 4.4 | seconds +4.14 WORSE |
<!-- ABLATION:END -->

<!-- TABLE-C:START -->
**Table C - open-web coverage, NFCorpus test queries (20 queries)**

| System | LLM precision | LLM strict precision | LLM gain | qrels precision | Unjudged rate | qrels gain | qrels hit | qrels recall |
|---|---|---|---|---|---|---|---|---|
| Swarm, open-web, shallow (n=20) | 0.950 [0.850, 1.000] | 0.579 [0.425, 0.729] | 1.529 [1.300, 1.721] | 1.000 [1.000, 1.000] | 0.163 [0.046, 0.292] | 0.100 [0.000, 0.300] | 0.050 [0.000, 0.150] | 0.004 [0.000, 0.011] |
| Baseline: one PubMed search, top 10 | 0.426 [0.231, 0.629] | 0.188 [0.081, 0.310] | 0.614 [0.329, 0.914] | 0.000 [0.000, 0.000] | 0.993 [0.979, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

| Stage | Recall of qrels-relevant docs |
|---|---|
| Scout candidates (before the relevance filter) | 0.004 [0.000, 0.011] |
| Kept by the relevance filter | 0.004 [0.000, 0.011] |
| Cited in the report | 0.004 [0.000, 0.011] |

Cited items matched to NFCorpus by PMID 100%, by title 0%. Source mix of citations: {"web": 43, "pubmed": 4, "europe_pmc": 3}; 2.35 unique domains and 2.5 citations per report; web-citation fraction 0.850 [0.717, 0.958]. Time p50 53.1 s, p95 88.468 s; 10913.05 tokens per query.

*Unjudged = a PubMed article outside NFCorpus's 3,633-abstract sample: unknown, not irrelevant, so precision covers judged citations only. Absolute recall is low by design (BEIR samples PubMed, ~38 relevant docs per query); compare configurations, not absolutes. Grade-1 relevance is noisy and PubMed overlaps model training data.*

*Sample: 20 queries, not the planned 100. The 100-query run was aborted by the Ollama Cloud account's monthly usage limit (every later LLM call failed; those runs are now flagged `llm_unavailable` instead of `ok`). "LLM" columns are gemma4:31b-cloud grades (0/1/2) of each cited source's relevance to the query and are not validated against human labels; the qrels columns are near zero because only ~2% of live PubMed results fall inside NFCorpus's sample. The web worker never triggered in this run (the paper pass answered every sub-question), so the ReAct and gap-fill configurations were the same code path; their difference (strict precision 0.72 vs 0.58) is the run-to-run noise at n=20.*
<!-- TABLE-C:END -->
