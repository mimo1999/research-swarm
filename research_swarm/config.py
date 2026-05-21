from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Tools
    tavily_api_key: str = ""

    # Observability
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "research-swarm"

    # App settings
    default_model_provider: str = "anthropic"
    default_model_name: str = "claude-sonnet-4-6"
    default_depth: str = "standard"
    max_iterations: int = 10
    max_sources: int = 15
    data_dir: Path = Path("data")

    # Local RAG — embeddings (HuggingFace, runs fully on CPU)
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str = ""          # empty → ~/.cache/huggingface
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Local RAG — LLM for routing / decomposition (Ollama, optional)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_timeout: float = 120.0      # seconds

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


settings = Settings()
