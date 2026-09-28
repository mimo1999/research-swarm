# Research Swarm — Technical Handoff

This is the reference for engineers picking up the codebase: what exists, where it lives, how it behaves and why. Related documents:
- [CLAUDE.md](CLAUDE.md): the design notes, kept in step with the code.
- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md): build chronology.
- [benchmarks/README.md](benchmarks/README.md): benchmark methodology and numbers.

*Current as of September 2026: evidence-first pipeline v2, plus the question frame, sectioned writer, deep read and strict-claim checks.*

---

## 1. System summary

A LangGraph **state machine** that turns a research question into a cited report. The design rule is that **an LLM call is spent only on judgments that need language understanding; everything checkable is done in code.** These are code, not LLM calls:
- routing;
- quote location;
- citation numbering;
- the verdict policy;
- the coverage gate;
- scope and strict-claim checks;
- deduplication;
- reference lists.

The LLM stages are fixed pipelines with bounded call counts, not agent loops:

| Stage | Calls per run (shallow, 6 sub-questions) | Model (default) |
|---|---|---|
| Question frame (query expansion) | 1 | large (`nemotron-3-nano:30b-cloud`) |
| Plan | 1 | large |
| Paper relevance scoring | 1 per sub-question | local fast tier (`gemma4:e2b`) |
| Document extraction | 1 per ~12k-char batch of uploads | local standard tier |
| Paper extraction | 1 per sub-question | local standard tier |
| Gap fill | 1 per under-covered sub-question | local standard tier |
| Verifier | 1 per ≤10 facts | local fast tier |
| Writer | outline + one per section + table + review (~7–9) | large |
| LLM judge (optional, off) | 1 | fast tier |

There are four entry points, all on the same compiled graph:
- `app.py`: Streamlit, HITL-capable;
- `run_research.py`: headless CLI;
- `api/`: FastAPI with SSE streaming, for the Next.js `frontend/`;
- `hf_space/app.py`: the Gradio Space.

### 1.1 Tech stack

| Concern | Library | Notes |
|---|---|---|
| Orchestration | LangGraph ≥0.2 (1.2 in use) | `StateGraph`, `Send` fan-out, `interrupt_before`, checkpointers |
| LLM abstraction | LangChain (`-core`, `-anthropic`, `-openai`, `-ollama`) | `.with_structured_output(method="json_schema")` everywhere |
| Schemas | Pydantic ≥2.7 + `pydantic-settings` | Every LLM output and config value is typed |
| Search | arXiv (`arxiv`), PubMed / Europe PMC (raw `httpx`), Tavily (optional) | `research_swarm/tools/` |
| Fetch / parse | `httpx`, BeautifulSoup4, `pypdf` | SSRF-validated fetcher; arXiv HTML for the deep read |
| Persistence | `aiosqlite` + `langgraph-checkpoint-sqlite` | `AsyncSqliteSaver` with an allowlisted serializer |
| UI | Streamlit ≥1.37; FastAPI + Next.js; Gradio | |
| Tracing | JSONL traces (`runtime/trace.py`); LangSmith (optional) | |
| Tests | pytest + pytest-asyncio, LLMs mocked with `AsyncMock` | 254 tests, offline |

**There are no embeddings, vector store or reranker.** ChromaDB, LlamaIndex, the BGE embedder and the cross-encoder were removed once the v2 pipeline passed its benchmark gates (see `benchmarks/README.md`). The removed retriever's BEIR numbers remain there as history.

Python ≥3.11, <3.15.

---

## 2. Repository layout

```
app.py                         Streamlit entry point (background-job runs, HITL panel)
run_research.py                Headless CLI
api/                           FastAPI: routes/research.py (start / SSE stream / status / resume),
                               routes/sessions.py, routes/config.py, routes/documents.py, runs.py
frontend/                      Next.js UI for the API
hf_space/                      Gradio Space (Ollama Cloud direct)
research_swarm/
  config.py                    Settings (pydantic-settings singleton, mutable at runtime)
  agents/
    base.py                    get_agent_llm (provider factory, base_url override) / get_tiered_llm
    _utils.py                  ainvoke_with_retry, schema_output_instruction, recover_from_parse_failure
    question.py                Split a topic into content vs. answer-format instruction
    expansion.py               probe(), expand_question(), normalize_frame(), scope_hit / mentions_any
    supervisor.py              Plan call + _enforce_plan (count cap, constraint re-attached)
    papers.py                  Tool routing, search, prefilter_candidates, score_pool, paper_key
    deep_read.py               Full-text passages for the top arXiv papers (no LLM)
    extractor.py               Packed-batch fact extraction (quote + relevance per fact)
    grounding.py               locate_quote / evidence_window / best_passage
    gap_fill.py                Search → fetch → one extraction call for thin sub-questions
    verifier.py                Enum verdict + relevance per fact, fixed policy table
    verdict.py                 Claim-check aggregation (SUPPORT / CONTRADICT / NEI style questions)
    writer.py                  Attributed writer entry, fact selection, prompts, free-form fallback
    writer_sections.py         Sectioned writer: outline → sections → comparison → review
    writer_render.py           Code render: citations, checks, REPORT_SECTIONS, deterministic_report
    text.py                    Tokenisation / keyword helpers
  graph/
    builder.py                 StateGraph assembly, serde allowlist, checkpointer
    nodes.py                   Every node + Send routers + coverage gate
    edges.py                   route_from_supervisor, route_from_collect
    rework.py                  request_rework (HITL "Edit & Re-research")
    stop.py                    Novelty stop signal + round cap
  schemas/                     state, query, plan, frame, worker, finding, source, critique,
                               report, judge
  tools/                       web_search, arxiv_tool, pubmed_tool, europe_pmc_tool,
                               url_fetcher, pdf_loader
  eval/                        llm_judge.py, claims.py (sentence-level judging), numbers.py
  runtime/                     budget.py, limits.py (LLM slots), session_ctx.py (per-session
                               credentials), trace.py, trace_stats.py, langsmith_trace.py,
                               migrations.py
  persistence/sessions.py      Session list / load / delete (raw SQLite reads)
  ui/                          sidebar, trace, graph_view (live topology), report_view,
                               sessions_view, style
  utils/                       security.py (SSRF, path traversal, injection), compat.py
tests/unit/, tests/integration/   254 offline tests
benchmarks/                    Closed-corpus smoke benchmark, claim scoring, relevance benchmark,
                               ablation tooling, quality_review/ (live-question harness)
reports/                       Sample reports from live runs
```

---

## 3. Core data model

### 3.1 `AgentState` (`schemas/state.py`)

A single `TypedDict` threaded through every node. Its reducers:

- `findings: Annotated[list[Finding], _merge_findings]` merges by `id`, so the verifier's updated findings overwrite rather than duplicate.
- `critiques: Annotated[list[Critique], _add_list]` is append-only; the latest verdict per finding wins (`_latest_verdicts`).
- `next_agent: Annotated[AgentName | None, _last_value]` tolerates several Send-fanned branches writing the same value in one step. Without it LangGraph raises `InvalidUpdateError`.
- `paper_corpus: Annotated[list[dict], _add_list]` holds the papers the scout kept.

The other fields:

| Group | Fields |
|---|---|
| Core | `query`, `plan` (carries `plan.frame`), `draft_report`, `final_report`, `messages` |
| HITL | `writer_instructions` (read once by the writer), `rework_instructions` (set by `request_rework`; while set, dispatch targets weak sub-questions; `collect_node` clears it), `human_feedback` (legacy) |
| Control | `research_rounds`, `pre_dispatch_finding_ids`, `iteration_count`, `session_id`, `model_provider`, `model_name`, `schema_version` |
| Send payload keys | `active_sub_question`, `active_batch`, `search_query`, `topic`, `frame`, `scope`, `scout_tasks`: `Send` delivers only its payload, so each fan-out lists what its node needs |
| Inputs / outputs | `ingested_documents` (uploads), `fact_conflicts` (finding-id pairs the verifier found contradictory) |

### 3.2 Schemas (`schemas/*.py`)

- **`ResearchQuery`**: `topic`, `depth` (`shallow|standard|deep`), `max_sources`, `audience` (`general|technical|academic|executive`).
- **`QuestionFrame`** (`frame.py`). Its fields:
  - `interpretation`;
  - `key_constraint`, the qualifier that separates the question from its subject;
  - `constraint_terms`, the literature's phrasings of that constraint;
  - `confusable_topics`;
  - `define_terms`, strict qualifiers such as "lossless";
  - `proof_criterion`, what would establish a strict claim; kept only when `define_terms` is non-empty;
  - `compare_items`, named items the question asks to distinguish (at most 6);
  - `search_queries`;
  - hidden (`SkipJsonSchema`): `probe_hits`, `topic`.

  `has_constraint` and `scope_phrases()` are helpers.
- **`ResearchPlan`**: `sub_questions`, `assignments: list[SubQuestionAssignment]` (sub-question, worker role, `search_query`, `domain`), `strategy`, `complexity_score`, and `frame` (hidden from the planner's schema). `assignment_for(sq)` looks up an assignment.
- **`Finding`**:
  - `id`, `claim`, `evidence: list[Source]`, `confidence`, `sub_question`;
  - `grounding` (`quote|passage|none|unknown`);
  - `relevance` (`direct|background|off_topic|unknown`).
- **`Source`**: `url`, `title`, `snippet` (the located evidence window), `source_type` (`web|arxiv|pubmed|pdf|…`), `credibility_score`.
- **`Critique`**: `finding_id`, `verdict` (`supported|weak|refuted`), `reasoning`.
- **`FinalReport`**: `title`, `exec_summary`, `sections: list[ReportSection]` (`heading`, `body_md`, `citations`), `references`, `methodology`, `limitations`, `quality_score`, `llm_judge`.

---

## 4. Execution graph

### 4.1 Topology (`graph/builder.py`)

```
START → supervisor
      → document_pass_node ── route_from_document_pass: Send ×
            ├─ document_worker_node (one per packed batch of uploads)
            └─ paper_scout_node     (one node, all sub-questions)
        (or a bounce to dispatch_node when there is no plan)
      → paper_worker_node           (both branches converge here)
      → dispatch_node ── route_from_dispatch: Send × worker_node (gap fill), or a bounce to collect
      → collect_node ── route_from_collect: dispatch_node (another round) | verifier
      → verifier → writer → END
```

`interrupt_before=["writer"]` is compiled in when HITL is on. After a run, a non-empty `graph.aget_state(config).next` means the run is paused.

Both Send-returning routers get an explicit `path_map`. It is not needed for routing, but without it `graph.get_graph()` (the UI's live diagram) silently drops those edges.

### 4.2 Nodes (`graph/nodes.py`)

- **`supervisor_node`**: a no-op if a plan exists (idempotent across resumes). Otherwise it runs `run_supervisor` in four steps:
  1. **Probe:** search the literal question on the routed tools, with no LLM.
  2. **`expand_question`:** one call produces the `QuestionFrame`, then `normalize_frame` cleans it:
     - dedupes and caps the lists;
     - treats "none" / "n/a" placeholders as empty;
     - drops confusable topics that match the scope itself;
     - drops a proof criterion when there are no strict terms.
  3. **Plan call.**
  4. **`_enforce_plan`:** caps the plan at `sub_questions_by_depth[depth]` and appends the constraint to any sub-question or query that lost it. The supervisor's `next_agent` is forced by code.
- **`route_from_document_pass`**: packs `ingested_documents` into `extract_batch_chars` batches (one `document_worker_node` each) and sends one `paper_scout_node` carrying all the scout tasks plus the frame.
- **`paper_scout_node`**, per sub-question:
  1. Searches its routed tools concurrently, getting `fetch_pass_results_per_tool` (12) results per tool.
  2. Adds the frame's `search_queries` results and the probe hits to every pool.
  3. Dedupes by `paper_key`, which merges arXiv / alphaxiv / emergentmind mirrors.
  4. Interleaves round-robin up to `paper_prefilter_pool` (48).
  5. Narrows the pool in code to `paper_max_candidates` (24) with `prefilter_candidates`.
  6. Scores the survivors 0–10 in one call against that sub-question, with the scope in the prompt.
  7. Keeps the top `paper_max_per_sub_question` papers scoring ≥ `paper_topk_floor`. A sub-question that keeps no non-web paper retries the tools its routing skipped.
- **`paper_worker_node`**: calls `deep_read` (§5.3), then makes one extraction call per sub-question over its kept papers, passing `scope=frame.key_constraint`.
- **`dispatch_node` / `route_from_dispatch`**: take the targets from `_research_targets` (§4.3) and send one `worker_node` (gap fill) per target, or bounce to `collect_node`.
- **`worker_node`**: runs `run_gap_fill` in five steps:
  1. search the routed tools;
  2. rank the results by term overlap;
  3. fetch the top pages (12 s timeout, falling back to the snippet);
  4. keep each page's two most relevant passages;
  5. make one extraction call.
- **`collect_node`**: increments `research_rounds`, clears `rework_instructions`, and runs `should_stop` (hard round cap `max_research_rounds_*` = 1/3/4, or novelty < 0.15).
- **`verifier_node`**: `run_verifier` (§5.5).
- **`writer_node`**: `run_attributed_writer` (§5.6), plus the optional LLM judge. The writer is never budget-gated; only the judge is.

### 4.3 The coverage gate: `_research_targets`

Round 0 selects every sub-question with fewer than `min_grounded_facts` findings that pass `_counts_as_coverage`. A finding counts only if all three hold:
- it is grounded (`grounding != "none"`);
- it is not background or off-topic;
- when the frame has a constraint, its claim or evidence mentions that constraint lexically (`scope_hit`).

The lexical check is a deliberate second opinion. A lenient small model labels everything "direct"; in one KV-cache run, 14 compression facts "covered" a cross-model question, so gap fill never ran. A false miss costs one gap-fill call. Misses are traced as `coverage.scope_miss`.

Later rounds target sub-questions with no finding at all. While `rework_instructions` is set, `_rework_targets` takes over: it picks the sub-questions without a supported, on-topic finding, or all of them if every one has such a finding, and appends the reviewer's keywords to their queries.

### 4.4 HITL (`app.py`, `graph/rework.py`, `api/routes/research.py`)

- **Approve:** `aupdate_state(config, {"writer_instructions": ...})`, then resume. The writer runs.
- **Edit & Re-research:** a plain resume from this pause can only run the writer. So `request_rework` writes `{"next_agent": "dispatch", "rework_instructions": text}` **as `collect_node`**, which reroutes the run to dispatch. The round cap then brings it back through the verifier to the same pause. The API's `edit` action uses the same function.
- **Discard:** abandons the run without writing. The checkpoint stays until it is deleted in Sessions; the API also clears the run's budget and job record.

---

## 5. Stages in detail

### 5.1 Question spec and frame (`agents/question.py`, `agents/expansion.py`)

`question.py` splits the topic into the content to research and an answer-format instruction ("Using only …, classify … as SUPPORT/CONTRADICT/NOT_ENOUGH_INFO"). Only the writer sees the format.

`expansion.py` turns the content into a `QuestionFrame` (§3.2). **Scope matching** (`scope_hit`, `mentions_any`) compares content-word stem prefixes and **ignores the question's general-subject stems**, so a phrasing like "KV cache migration" carries no scope and never matches. An empty frame leaves every stage behaving as before. It comes from expansion being turned off, a failed call, or a question with no qualifier.

### 5.2 Paper scout (`agents/papers.py`)

- **Tool routing:** `routed_tools_union` takes the domain's tools. It adds PubMed / Europe PMC when the text looks biomedical and arXiv when it looks CS or physics, and always adds the web. One wrong `domain` label therefore can't keep a health question off PubMed.
- **Pre-filter:** `prefilter_candidates` ranks candidates by word overlap and scope match, with primary sources before secondary. `is_secondary_source` recognises Medium, LinkedIn, YouTube, `/blog/`, `blog.` and `.pages.dev` URLs, among others.
- **Scoring:** scoring is **per sub-question**. Scoring several sub-questions in one call collapsed to near-zero scores.

### 5.3 Deep read (`agents/deep_read.py`)

The deep read takes the top `deep_read_papers` (2) primary arXiv papers in the corpus:
1. It fetches each paper's arXiv HTML through the SSRF-validated `url_fetcher._safe_get`.
2. It splits the text into paragraphs, using `alttext` for math.
3. It keeps the paragraphs that best match the question and frame, up to `deep_read_chars` (6000).
4. It appends them to the abstract as "Full-text excerpts:" and sets the canonical `arxiv.org/abs/<id>` URL.

It makes no LLM call. It exists because a paper's specifics otherwise reached reports only through blogs summarising it.

### 5.4 Extraction and grounding (`agents/extractor.py`, `agents/grounding.py`)

The extractor refers to sources and sub-questions **by number**; a paraphrased sub-question used to be silently dropped by exact-string matching. Each (source, sub-question) pair yields up to `extract_max_facts_per_pair` facts, each with a verbatim `quote` and a relevance label (`direct` / `background`).

`ground()` locates the quote (exact, then fuzzy at ≥ 0.85), or failing that the best passage for the claim, and stores the surrounding window as `Source.snippet`. A fact whose quote can't be located gets `grounding="none"` and is refuted without an LLM call.

### 5.5 Verifier (`agents/verifier.py`)

One call per ≤10 facts, returning an enum verdict and a relevance label per fact. The verdict is applied by a fixed policy in code:

| Verdict | Critique | Confidence |
|---|---|---|
| supported (quote located) | supported | 0.9 |
| supported (passage only) | supported | 0.75 |
| partial | weak | 0.5 |
| unsupported, quote located verbatim | weak (hedged, not hidden) | 0.3 |
| unsupported | refuted (hidden from the writer) | 0.1 |

`_checked_relevance` stops the verifier from downgrading a fact whose own claim states the scope; a 2B verifier had called cross-model facts "background". Facts beyond `max_facts_for_writer` (36) are dropped, best-grounded first. For claim-check questions, `verdict.py` labels each fact supports / contradicts / unrelated, and the verdict is aggregated in code.

### 5.6 Writer (`agents/writer.py`, `writer_sections.py`, `writer_render.py`)

**Fact selection:** refuted and off-topic facts are excluded, primary sources come first, and each fact is labelled direct / background and primary / secondary.

**Sectioned mode** (`writer_mode="sectioned"`, the default) runs in five steps:
1. **Outline:** the answer, the stance, and which facts go in which section. Code maps the outline onto `REPORT_SECTIONS` for academic and technical reports, gives each fact to **one** evidence section, places direct facts the outline dropped, and rebalances any section given more than max(12, an even share) facts.
2. **Sections, one at a time:** evidence sections first, then synthesis from what the evidence sections cite, then the overview. Each call sees every earlier section, so it builds on them instead of repeating them. Sentences are deduped and capped at 10 per section.
3. **Comparison table** (when the frame has `compare_items`): one row per item, 3–4 model-chosen columns, and cells citing fact numbers.
4. **Review:** sees every drafted sentence next to its cited facts' evidence. It fixes or deletes unsupported, overclaiming, mis-cited, contradictory, off-topic or duplicate sentences, and writes the final answer and summary. It must return a `claim_check` for every sentence with strong wording (`_STRONG_RE`: exact, lossless, guarantees, proves, only when, never, …). An unruled strong sentence, or every one if the review call fails, is kept only if its own cited facts contain the same wording.
5. The result is assembled into a `WriterDraft`.

If the outline fails, the writer falls back to the single-call draft. If the review fails, it keeps the unreviewed draft.

**Code render** (`render_report`) is the last gate for every writer mode:
- **Citations:** numbers references by first use, appends `[a, b]` markers, trims to at most 3 facts per sentence, and merges arXiv mirrors in the reference list.
- **Grounding drops:** drops a sentence whose numbers aren't in its facts' evidence. Also drops one that names the scope or a strict term its facts don't mention, unless the sentence negates it (`affirmed_terms`).
- **Clean-up drops:** removes near-duplicates (Jaccard ≥ 0.8, body sections before overview), markup and LaTeX residue, meta-language ("the provided facts"), and invented author attributions.
- **Direct answer:** capped at 2 sentences. With no direct fact, it opens "No retrieved source directly addresses …". Sentences asserting an unmet strict qualifier (`define_terms`) are removed and replaced with the answer facts' claims, prefixed "Not established: …".
- **Gaps:** sub-questions without a direct fact are listed in `limitations`.
- **Comparison table:** a cited cell may not add numbers. An uncited cell may only be a short, number-free definitional phrase; anything else becomes "not established".
- **Audience:** executive collapses to one paragraph. `_canonicalize_heading` maps near-miss headings onto the template.

If rendering is empty or parsing fails, the free-form writer runs, grounded by `ground_free_form_report`. `deterministic_report` is the no-LLM fallback.

### 5.7 Structured-output reliability (`agents/_utils.py`)

- `method="json_schema"` for every call: function-calling mode returns nothing with nemotron.
- `schema_output_instruction(Model)` puts a compact JSON schema in the prompt.
- `recover_from_parse_failure` repairs unescaped LaTeX backslashes and schema-echo replies.
- The schemas absorb common model mistakes: `SupervisorDecision` defaults `next_agent` / `reasoning` and lifts flattened plan fields, and `ReviewPass` accepts plain-string summaries.
- Thinking is off for structured stages (`no_thinking_stages`), with output capped at `no_thinking_max_tokens`.
- Every fallback is logged at ERROR and traced as `<stage>.fallback`.

---

## 6. LLM routing, concurrency and budgets

- **Provider factory** (`agents/base.py::get_agent_llm`): `ChatAnthropic` / `ChatOpenAI` / `ChatOllama` (with an optional `base_url`) / `ChatHFLocal` (provider `huggingface`: a `transformers` model in the app process, see below). Keys come from `runtime/session_ctx.py::resolve_api_key`, which prefers the session's own key over the process one (the Space and API are multi-tenant).
- **Tiers** (`get_tiered_llm`): `fast` (scorer, verifier), `standard` (extraction, gap fill; follows the sidebar's provider), `thorough` (supervisor, writer).
- **Large model** (`nodes._get_tiered_state_llm`): a stage whose agent label is in `large_model_stages` uses `large_model` on `large_model_ollama_base_url` (default Ollama Cloud directly, with `OLLAMA_API_KEY`). Moving a stage is a config change.
- **Concurrency** (`runtime/limits.py::llm_slot`): a process-wide cap on in-flight requests per provider. The `ollama_cloud` pool is separate from the local daemon's, so writer calls don't queue behind local extraction. `ainvoke_with_retry` holds a slot only while a request is in flight and retries 429 / 5xx / timeouts with jittered backoff. `document_worker_concurrency` caps a run's document fan-out.
- **In-process transformers** (`agents/hf_local.py`, used by the ZeroGPU Space):
  - `load()` puts the model on the device once per process; on ZeroGPU this happens at import.
  - `generate_batch()` runs one `generate()` over several prompts, each constrained to its JSON schema by lm-format-enforcer. The enforcer tables are built directly, because its transformers integration doesn't import on transformers 5. Arguments and results are plain dicts, because ZeroGPU pickles them.
  - A per-session micro-batcher groups a run's concurrent calls: up to `hf_max_batch` (6) within `hf_batch_window_s` (0.3 s). ZeroGPU bills each visitor by GPU time, so a batch costs about as much as its longest row. Batchers are keyed by session and Gradio event, so a GPU call is charged to the visitor who made it.
  - A quota error marks the session; its remaining local calls fail fast, and the Space tells the visitor the result is incomplete.
- **Budgets** (`runtime/budget.py`):
  - A **research** pool (`max_llm_calls`, 40) and a **review** pool (`max_review_llm_calls`, 10) of call counts.
  - A session-wide token cap (`max_tokens_per_session`).
  - Counted by a callback set on the model instance itself (`model_copy(update={"callbacks": …})`), because `.with_config` callbacks are lost through `with_structured_output`.

---

## 7. Security (`utils/security.py`)

- **SSRF:** `validate_url` resolves the host and rejects private, loopback, link-local, reserved and unspecified addresses, including the decimal, hex and IPv4-mapped IPv6 encodings. `url_fetcher._safe_get` follows redirects manually and validates every hop. DNS failure fails closed.
- **Path traversal:** `validate_file_path` limits reads to the temp dir and `data_dir`.
- **Prompt injection:** `sanitize_fetched_content` strips invisible and bidi characters and redacts lines matching known injection patterns before fetched text reaches a prompt. Deep-read passages pass through it too.
- **Credentials:** per-session keys are bound in `session_ctx` with a TTL sweep, never written to `settings`, and redacted in logs.

---

## 8. Persistence, UI and observability

- **Checkpoints:** `AsyncSqliteSaver` at `data/checkpoints/sessions.db`. The msgpack serializer allowlists every project schema, including `QuestionFrame`. `migrate_state` upgrades old checkpoints (schema v0 → v2) on load.
- **Streamlit (`app.py`):** one background event loop (`_BG_LOOP`) owns every asyncio object. **A run is a background `_RunJob`** in an `@st.cache_resource` registry, not part of a script run. Streamlit reruns the script on every widget change, which used to orphan the run and blank the trace and diagram. `_render_running` redraws the recorded updates and polls every 0.5 s. "Cancel run" cancels the job's future. Settings changed mid-run apply to the next run.
- **Live topology** (`ui/graph_view.py`): `get_graph().draw_mermaid()`, with the current node amber and visited nodes green.
- **Tracing:** every stage, LLM call (tokens, latency) and tool call goes to `data/traces/<session>.jsonl`. `trace_stats.analyze` summarises a trace, including `quality["scope"]` signals. LangSmith is attached per run when configured, and the UI shows the run's link.

---

## 9. Configuration (`config.py`)

A `pydantic-settings` singleton loaded from `.env`. The UI's `_apply_ui_settings()` overwrites the provider, model, max sources and Ollama URL before each run. The main knobs:

| Setting | Default | Effect |
|---|---|---|
| `sub_questions_by_depth` | 6 / 8 / 10 | Main compute knob (~55 s of local model time per sub-question) |
| `large_model`, `large_model_stages` | nemotron-3-nano:30b-cloud, [supervisor, writer] | `""` = all stages local |
| `writer_mode` | sectioned | or `single` |
| `query_expansion_enabled`, `probe_results` | true, 8 | Question frame on/off |
| `fetch_pass_results_per_tool`, `paper_prefilter_pool`, `paper_max_candidates` | 12, 48, 24 | Wide search, narrowed in code before the scorer |
| `paper_max_per_sub_question`, `paper_topk_floor` | 6, 0.5 | Papers kept per sub-question |
| `deep_read_papers`, `deep_read_chars` | 2, 6000 | Full-text passages |
| `max_facts_for_writer`, `min_grounded_facts` | 36, 1 | Writer input cap; coverage gate |
| `writer_reasoning_section` | false | Uncited "Analysis (reasoning, not from sources)" section |
| `max_concurrent_llm_calls_{ollama,ollama_cloud,anthropic,openai}` | 2 / 2 / 8 / 8 | LLM slots |
| `llm_judge_enabled` | false | Optional report judge |

---

## 10. Testing

254 tests, **fully offline**. `tests/conftest.py` switches off query expansion (its probe is a live search) and the large model, and runs the single-call writer with no deep read. The sectioned writer, expansion, deep read and rework each have their own mocked test files. Nodes are patchable because `builder.py` registers `_nodes.<name>` at `build_graph()` call time.

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy research_swarm/
```

---

## 11. Known limitations

- **A reference nobody names isn't resolved.** For "the paper" in "audit the claim that the paper proves …", the frame does not work out which paper is meant, and the writer can anchor on a different source. For now, attach the paper as a document or URL.
- **The review is lenient.** It tends to rule a strong claim supported when it is true of its own source, even if that source is about a different setting. The code-side scope and strict-term checks are the backstop.
- **Planner drift.** Sub-questions can drift toward definitions for audit-style questions.
- **Latency.** A shallow run takes about 4–8 minutes on one local GPU. Two concurrent local calls share it, so each call slows to about 40 s.
- **Tests don't cover live model behaviour.** The mocked suite checks wiring and the code guardrails; report quality is verified by live runs (`benchmarks/quality_review/`, `reports/`).
- **HITL pauses only before the writer.**
