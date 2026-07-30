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
    default_model_name: str = "gemma4:31b-cloud"
    default_depth: str = "shallow"
    max_iterations: int = 1
    max_sources: int = 3
    # "research" pool: supervisor, document pass/workers, dispatch/worker loop --
    # the part that can genuinely run away (multiple rounds, multiple tool
    # turns per worker). Raises BudgetExceeded above this.
    max_llm_calls: int = 40
    # "review" pool: critic, fact-checker, writer, LLM judge -- a few batched
    # calls, never an open-ended loop. Kept separate from max_llm_calls so a
    # research-loop overrun can't starve these out and leave an empty report
    # with good findings sitting unused. See runtime/budget.py.
    max_review_llm_calls: int = 10
    data_dir: Path = Path("data")

    # Local RAG -- embeddings (HuggingFace, runs fully on CPU)
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str = ""          # empty -> ~/.cache/huggingface
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Ollama — shared for both local and cloud deployments.
    # In cloud mode the local daemon (same URL) proxies requests to Ollama's
    # cloud infrastructure using the credentials from `ollama login`.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:31b-cloud"
    ollama_cloud_model: str = "gemma4:31b-cloud"
    ollama_timeout: float = 120.0      # seconds
    ollama_deployment: str = "cloud"   # "local" | "cloud"

    # ── Model tiers ──────────────────────────────────────────────────────────
    # Each tier maps to a (provider, model) pair.  Nodes pick the tier that
    # matches their role in the pipeline:
    #   fast      -- cheap/quick:  structured extraction (critic, fact-checker)
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
    tier_fast_model:        str = "gemma4:31b-cloud"
    tier_standard_provider: str = "ollama"
    # tier_standard_model is the generic fallback; get_tiered_llm overrides it
    # per-provider below with each provider's lowest-grade model, since the
    # worker tier's whole point is "smallest model that still works reliably".
    tier_standard_model:           str = "gpt-oss:20b-cloud"
    tier_standard_model_local:     str = "gemma4:4b"           # ollama, local daemon
    tier_standard_model_cloud:     str = "gpt-oss:20b-cloud"   # ollama, cloud-hosted
    tier_standard_model_anthropic: str = "claude-haiku-4-5-20251001"
    tier_standard_model_openai:    str = "gpt-5-nano"
    tier_thorough_provider: str = "ollama"
    tier_thorough_model:    str = "gemma4:31b-cloud"

    # ── Research-loop limits by depth ────────────────────────────────────────
    # Maximum dispatch→workers→collect cycles before forcing progression
    # to the critic regardless of the stop-signal score.
    max_research_rounds_shallow:  int = 1
    max_research_rounds_standard: int = 3
    max_research_rounds_deep:     int = 4

    # Per-finding cap on rework rounds after a weak/refuted critique verdict.
    # Independent of max_research_rounds (which bounds total rounds across all
    # findings) -- this bounds how many times any single finding gets
    # re-researched, so one chronically-bad finding can't hog rounds that
    # would otherwise go to others.
    max_rework_attempts: int = 3

    # ── Stop-signal thresholds ───────────────────────────────────────────────
    # Fraction of new findings considered novel (below = stop).
    stop_novelty_threshold:    float = 0.15
    # Mean cosine similarity of new vs existing finding claims (above = stop).
    stop_similarity_threshold: float = 0.85

    # ── Judge batching ───────────────────────────────────────────────────────
    # Critic and fact-checker review this many findings per LLM call instead of
    # one call each — cuts repeated system-prompt and shared-source token cost.
    judge_batch_size: int = 8

    # ── LLM-as-a-judge review pipeline ──────────────────────────────────────
    # Independent LLM review pass over the writer's final report — catches
    # issues the embedding-based faithfulness check can't (wrong topic,
    # unaddressed sub-questions, incoherent prose, citations to nothing).
    # Runs on the cheap 'fast' tier since it's a review, not generation.
    llm_judge_enabled: bool = True
    llm_judge_tier: str = "fast"
    llm_judge_pass_threshold: float = 3.5

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    def max_research_rounds(self, depth: str) -> int:
        """Return the research-loop cap for the given depth string."""
        return {
            "shallow":  self.max_research_rounds_shallow,
            "standard": self.max_research_rounds_standard,
            "deep":     self.max_research_rounds_deep,
        }.get(depth, self.max_research_rounds_standard)


settings = Settings()
