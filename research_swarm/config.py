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
    default_model_provider: str = "anthropic"
    default_model_name: str = "claude-sonnet-4-6"
    default_depth: str = "standard"
    max_iterations: int = 10
    max_sources: int = 15
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
    ollama_timeout: float = 120.0      # seconds
    ollama_deployment: str = "local"   # "local" | "cloud"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


settings = Settings()
