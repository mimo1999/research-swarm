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
    max_llm_calls: int = 25   # hard budget per session; raise BudgetExceeded above this
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
    ollama_model: str = "gemma4:e2b"
    ollama_cloud_model: str = "gemma4:31b-cloud"
    ollama_timeout: float = 120.0      # seconds
    ollama_deployment: str = "cloud"   # "local" | "cloud"

    # ── Model tiers ──────────────────────────────────────────────────────────
    # Each tier maps to a (provider, model) pair.  Nodes pick the cheapest tier
    # that matches their quality requirements:
    #   fast      -- cheap/quick: routing decisions, structured extraction
    #   standard  -- balanced:    research workers, critique, fact-checking
    #   thorough  -- expensive:   synthesis / writing
    #
    # Defaults reuse the Ollama stack so no extra API key is required.
    tier_fast_provider:     str = "ollama"
    tier_fast_model:        str = "gemma4:e2b"
    tier_standard_provider: str = "ollama"
    tier_standard_model:    str = "gemma4:31b-cloud"
    tier_thorough_provider: str = "ollama"
    tier_thorough_model:    str = "gemma4:31b-cloud"

    # ── Research-loop limits by depth ────────────────────────────────────────
    # Maximum dispatch→workers→collect cycles before forcing progression
    # to the critic regardless of the stop-signal score.
    max_research_rounds_shallow:  int = 1
    max_research_rounds_standard: int = 2
    max_research_rounds_deep:     int = 4

    # ── Stop-signal thresholds ───────────────────────────────────────────────
    # Fraction of new findings considered novel (below = stop).
    stop_novelty_threshold:    float = 0.15
    # Mean cosine similarity of new vs existing finding claims (above = stop).
    stop_similarity_threshold: float = 0.85

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
