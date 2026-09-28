from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers — stored as SecretStr so values are masked in logs/repr
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    # Bearer token for Ollama Cloud's direct API (https://ollama.com/api/...),
    # confirmed to mirror the local daemon's API surface: GET /api/tags is
    # public, POST /api/chat returns 401 {"error":"Unauthorized"} without a
    # valid token. Lets a deployment run entirely against Ollama Cloud with
    # no local `ollama serve` process -- see resolve_api_key() in
    # runtime/session_ctx.py, which prefers a session-supplied key first.
    ollama_api_key: SecretStr = SecretStr("")

    # Tools
    tavily_api_key: SecretStr = SecretStr("")
    # NCBI E-utilities: optional. Without a key, NCBI rate-limits to ~3
    # req/sec across ALL callers sharing the IP -- easy to hit with parallel
    # worker fan-out. With a key (free, from an NCBI account), the limit
    # rises to 10 req/sec. https://www.ncbi.nlm.nih.gov/account/settings/
    ncbi_api_key: SecretStr = SecretStr("")

    # Observability
    langsmith_api_key: SecretStr = SecretStr("")
    langchain_tracing_v2: bool = False
    langchain_project: str = "research-swarm"

    # App settings
    default_model_provider: str = "ollama"
    default_model_name: str = "gemma4:e2b"
    default_depth: str = "shallow"
    max_iterations: int = 1
    max_sources: int = 3
    # "research" pool: supervisor, document pass/workers, dispatch/worker loop --
    # the part that can genuinely run away (multiple rounds, multiple tool
    # turns per worker). Raises BudgetExceeded above this.
    max_llm_calls: int = 40
    # "review" pool: verifier, writer, LLM judge -- a few batched
    # calls, never an open-ended loop. Kept separate from max_llm_calls so a
    # research-loop overrun can't starve these out and leave an empty report
    # with good findings sitting unused. See runtime/budget.py.
    max_review_llm_calls: int = 10
    # Session-wide, spanning BOTH pools -- unlike the call-count limits above,
    # a call's token cost varies wildly with tool-loop context and reasoning
    # output, so capping calls alone doesn't bound actual spend. This is the
    # guardrail that matters for a shared/rate-limited key (e.g. Ollama
    # Cloud's account-wide allowance) -- see runtime/budget.py.
    max_tokens_per_session: int = 200_000
    # Paper scout (route_from_document_pass -> paper_scout_node): results requested from
    # each literature tool per query. Search is cheap (no LLM, ~1-3 s, concurrent), so this is
    # set wide; the code pre-filter below narrows it before the LLM scorer.
    fetch_pass_results_per_tool: int = 12
    relevance_threshold: float = 0.75
    # Cap on papers handed to the paper worker per sub-question (best-scoring first).
    paper_max_per_sub_question: int = 6
    # Candidates per sub-question entering its scoring call (the LLM cost: each one is read).
    paper_max_candidates: int = 24
    # Candidates per sub-question gathered (round-robin across tools / queries) before the
    # code pre-filter (papers.prefilter_candidates) picks the paper_max_candidates to score.
    paper_prefilter_pool: int = 48
    # Deep read (agents/deep_read.py): full text of this many top primary arXiv papers, cut to
    # the passages that best match the question, is added to their abstracts before extraction.
    # No extra LLM call; each adds up to deep_read_chars to one extraction call's input.
    deep_read_papers: int = 2
    deep_read_chars: int = 6000
    deep_read_timeout_s: float = 20.0
    paper_max_findings_per_sub_question: int = 6
    # A sub-question that keeps no non-web paper retries the search tools its routing skipped.
    # relevance_threshold / relevance_floor / paper_min_per_sub_question define the strict
    # (>= 0.75) rule and its top-up that benchmarks/relevance_benchmark.py compares top-k against.
    paper_min_per_sub_question: int = 3
    relevance_floor: float = 0.6
    # Off-switch with no code change, matching space_mode/llm_judge_enabled --
    # this pass adds latency (searches + one scoring call) before round 0 starts.
    enable_fetch_pass: bool = True
    # Max document workers (one per uploaded document / oversized-document slice) calling
    # the LLM at once within a run. The provider allows only a few concurrent requests per
    # account, and an uncapped fan-out over many documents drew 429s that silently dropped
    # their evidence. See runtime/limits.py.
    document_worker_concurrency: int = 3
    # Process-wide cap on in-flight LLM requests per provider, across every stage (see
    # runtime/limits.py::llm_slot). Ollama Cloud allows roughly one long request at a time per
    # account and queues or rejects the rest with 429s; hosted APIs tolerate far more.
    # 0 = unlimited. Separate processes (UI + API + a benchmark) still share one account.
    max_concurrent_llm_calls_ollama: int = 2
    max_concurrent_llm_calls_anthropic: int = 8
    max_concurrent_llm_calls_openai: int = 8
    # Ollama Cloud called directly (the writer's endpoint, below) -- its own pool, so the writer's
    # cloud calls don't queue behind the local daemon's.
    max_concurrent_llm_calls_ollama_cloud: int = 2
    # In-process transformers (provider "huggingface"): requests must reach the micro-batcher
    # together to be batched, so this matches hf_max_batch rather than capping at the GPU's 1.
    max_concurrent_llm_calls_huggingface: int = 8
    data_dir: Path = Path("data")

    # ── Hosted-deployment mode (e.g. Hugging Face Spaces) ───────────────────
    # Off by default so local/dev runs are unaffected. When enabled:
    #   - app.py prunes sessions older than space_retention_seconds (and any
    #     beyond space_max_sessions, oldest first) once per process start --
    #     needed because a public multi-tenant Space has no one around to
    #     click "delete session" and DATA_DIR is typically ephemeral storage
    #     anyway (e.g. /tmp), so nothing is lost by pruning proactively.
    #   - app.py caps concurrent graph runs at space_max_concurrent_runs via
    #     an in-process semaphore, so one Streamlit server process handling
    #     several simultaneous users can't be driven into memory exhaustion
    #     by each run's in-flight search results and LLM calls.
    space_mode: bool = False
    space_retention_seconds: int = 21600   # 6 hours
    space_max_sessions: int = 40
    space_max_concurrent_runs: int = 4

    # Ollama — shared for both local and cloud deployments.
    # In cloud mode the local daemon (same URL) proxies requests to Ollama's
    # cloud infrastructure using the credentials from `ollama login`.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_cloud_model: str = "gemma4:31b-cloud"
    ollama_deployment: str = "local"   # "local" | "cloud"
    # Reasoning/"thinking" models (see https://ollama.com/search?c=thinking)
    # otherwise interleave <think>...</think> tags into the main response
    # content by default, which lands inside whatever with_structured_output
    # is trying to parse as JSON and is a real contributor to the parse
    # failures recover_from_parse_failure exists for. Setting this segregates
    # reasoning into AIMessage.additional_kwargs['reasoning_content'] instead,
    # leaving `content` clean. No effect on models that don't support it.
    ollama_reasoning: bool = True
    # Stages that produce structured JSON run with thinking OFF regardless of ollama_reasoning.
    # Measured on nemotron-3-nano with the real prompts: valid JSON in every clean trial at
    # 2-3x the speed (writer ~22 s vs ~470 s and only 1/3 valid with thinking on). Thinking
    # holds the provider's single slot for tens of seconds per call, which feeds the 429s.
    # Stage = the `agent=` label given to _get_tiered_state_llm, minus any "[detail]"/"/role"
    # suffix. Unlisted stages keep ollama_reasoning.
    no_thinking_stages: list[str] = [
        "supervisor", "expansion", "writer", "judge", "verifier", "gap_fill",
        "paper_scout", "paper_worker", "document_worker",
    ]
    # Output cap for those stages, so a runaway generation can't hold the slot for minutes
    # (a writer once produced 131k tokens of empty output over ~7 minutes).
    no_thinking_max_tokens: int = 8192

    # ── Model tiers ──────────────────────────────────────────────────────────
    # Each tier maps to a (provider, model) pair.  Nodes pick the tier that
    # matches their role in the pipeline:
    #   fast      -- cheap/quick:  structured extraction (verifier, paper scout)
    #   standard  -- smallest capable: research workers -- called once per
    #                sub-question per tool turn, so call *volume* is highest
    #                here; keep this the cheapest tier that can still reliably
    #                do tool-calling + synthesis.
    #   thorough  -- large/expensive: the orchestrator (supervisor, called once
    #                per session to build the plan) and the writer (final
    #                synthesis over all findings) -- both need the strongest
    #                reasoning/context handling, and both are low call-volume
    #                so the larger model's cost doesn't compound.
    #
    # Defaults reuse the Ollama stack so no extra API key is required.
    tier_fast_provider:     str = "ollama"
    tier_fast_model:        str = "gemma4:e2b"
    tier_standard_provider: str = "ollama"
    # tier_standard_model is the generic fallback; get_tiered_llm overrides it
    # per-provider below with each provider's lowest-grade model, since the
    # worker tier's whole point is "smallest model that still works reliably".
    tier_standard_model:           str = "gemma4:e2b"
    tier_standard_model_local:     str = "gemma4:e2b"                    # ollama, local daemon
    tier_standard_model_cloud:     str = "gemma4:31b-cloud"             # ollama, cloud-hosted
    tier_standard_model_anthropic: str = "claude-haiku-4-5-20251001"
    tier_standard_model_openai:    str = "gpt-5-nano"
    tier_thorough_provider: str = "ollama"
    tier_thorough_model:    str = "gemma4:e2b"

    # ── In-process Hugging Face transformers (provider "huggingface", agents/hf_local.py) ────
    # For hosts with a GPU but no Ollama daemon (a ZeroGPU Space). Any tier whose provider is
    # "huggingface" runs this model in the app process. Concurrent calls within a session are
    # micro-batched: collected for up to hf_batch_window_s, at most hf_max_batch per generate().
    hf_model_id: str = "google/gemma-4-E2B-it"
    hf_max_batch: int = 8   # >= the deepest profile's sub_questions: one batch per stage
    hf_batch_window_s: float = 0.3
    # Ceiling on one ZeroGPU call's requested duration (spaces.GPU(duration=...)); quota is
    # charged on actual GPU time, but a shorter request gets better queue priority.
    hf_gpu_duration_max_s: int = 120

    # ── Large model for the few stages that need it (its own endpoint) ────────
    # Every other stage uses the tiers above (local gemma4). The stages in large_model_stages use
    # a larger model: by default Ollama Cloud called directly (https://ollama.com, authenticated
    # with OLLAMA_API_KEY) while the rest stays on the local daemon.
    #   writer     -- turns ~30 verified facts into a structured report; a 2B model used 5 of 23.
    #   supervisor -- the question frame (query expansion) and the research plan, one call each;
    #                 gemma4 wrote long, sentence-like search queries that retrieved mostly blogs,
    #                 and every later stage inherits those queries.
    # large_model="" = every stage uses its tier.
    large_model_provider: str = "ollama"
    large_model: str = "gemma4:31b-cloud"
    large_model_ollama_base_url: str = "https://ollama.com"   # "" = the normal OLLAMA_BASE_URL
    large_model_stages: list[str] = ["supervisor", "writer"]
    # "sectioned": outline call -> one call per section (with that section's facts, evidence
    # and sources) -> a final review call that fixes or deletes unsupported sentences and writes
    # the answer and summary (agents/writer_sections.py). "single": one call for the whole draft.
    writer_mode: str = "sectioned"

    # ── Research depth profiles ──────────────────────────────────────────────
    # Everything that scales with the depth the user picks, in one place (not exposed in the
    # UI). A key here overrides the global setting of the same name for runs at that depth; a
    # run with no depth, or a key a profile omits, uses the global setting.
    #   sub_questions             -- the main compute knob: each costs one scoring and one
    #                                extraction call (~55 s of local gemma4 time)
    #   gap_fill_workers          -- max gap-fill workers per round (least-covered first)
    #   research_rounds           -- max dispatch -> gap fill -> collect rounds
    #   paper_prefilter_pool      -- candidates gathered per sub-question (search only, free)
    #   paper_max_candidates      -- candidates the LLM scorer reads per sub-question
    #   paper_max_per_sub_question -- papers kept per sub-question for extraction
    #   deep_read_papers          -- papers whose full text is read
    #   max_facts_for_writer      -- sub_questions x 6 facts, so no depth drops paid-for evidence
    depth_profiles: dict[str, dict[str, int]] = {
        "shallow": {"sub_questions": 3, "gap_fill_workers": 2, "research_rounds": 1,
                    "paper_prefilter_pool": 32, "paper_max_candidates": 16,
                    "paper_max_per_sub_question": 4, "deep_read_papers": 1,
                    "max_facts_for_writer": 18},
        "standard": {"sub_questions": 5, "gap_fill_workers": 3, "research_rounds": 2,
                     "paper_prefilter_pool": 48, "paper_max_candidates": 24,
                     "paper_max_per_sub_question": 6, "deep_read_papers": 2,
                     "max_facts_for_writer": 30},
        "deep": {"sub_questions": 7, "gap_fill_workers": 5, "research_rounds": 3,
                 "paper_prefilter_pool": 64, "paper_max_candidates": 32,
                 "paper_max_per_sub_question": 8, "deep_read_papers": 3,
                 "max_facts_for_writer": 42},
    }
    # Fallbacks for the profile-only keys (a run with no depth).
    sub_questions: int = 5
    research_rounds: int = 2
    gap_fill_workers: int = 0   # 0 = one worker per under-covered sub-question

    # ── Stop-signal thresholds ───────────────────────────────────────────────
    # Fraction of new findings considered novel (below = stop).
    stop_novelty_threshold:    float = 0.15

    # ── LLM-as-a-judge review pipeline ──────────────────────────────────────
    # Independent LLM review pass over the writer's final report — catches
    # issues a mechanical check can't (wrong topic,
    # unaddressed sub-questions, incoherent prose, citations to nothing).
    # Runs on the cheap 'fast' tier since it's a review, not generation.
    # Off by default: the review is one more request on the provider's scarce slots, and offline
    # judging (benchmarks/score_claims.py) is the measurement path. The UI has a toggle.
    llm_judge_enabled: bool = False

    # ── Evidence extraction, verification and writing (see benchmarks/README.md) ────────────
    # Documents are packed into batches of this many characters, one extraction call each
    # (agents/extractor.py); each fact's evidence is located in its source (agents/grounding.py).
    extract_batch_chars: int = 12_000
    extract_max_facts_per_pair: int = 3
    # Findings passed to the verifier / writer, best grounded first (per depth: depth_profiles).
    max_facts_for_writer: int = 30
    # A sub-question with fewer grounded findings than this after the paper/document pass is
    # sent to gap fill (search -> fetch -> one extraction call).
    min_grounded_facts: int = 1
    # Paper selection keeps the best paper_max_per_sub_question papers scoring at least this.
    paper_topk_floor: float = 0.5

    # ── Question frame (agents/expansion.py) ────────────────────────────────────────────────
    # Before planning: search the literal question (probe, no LLM) and extract its distinguishing
    # constraint in one short call. Planning, search, scoring, coverage, verification and writing
    # then enforce it in code. False = plan exactly as before (empty frame everywhere).
    query_expansion_enabled: bool = True
    probe_results: int = 8
    # Optional "Analysis (reasoning, not from sources)" section: uncited, number-free sentences
    # for conceptual questions. Excluded from citation/faithfulness scoring.
    writer_reasoning_section: bool = False
    llm_judge_tier: str = "fast"
    llm_judge_pass_threshold: float = 3.5

    def for_depth(self, key: str, depth: object = None) -> int:
        """*key* for a run at *depth* (a depth string or ResearchDepth): the depth profile's
        value, else the global setting of that name."""
        profile = self.depth_profiles.get(str(getattr(depth, "value", depth) or ""), {})
        return int(profile[key]) if key in profile else int(getattr(self, key))

    def max_research_rounds(self, depth: object = None) -> int:
        """The research-loop cap for *depth*."""
        return self.for_depth("research_rounds", depth)


settings = Settings()
