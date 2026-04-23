"""
F1 Visualization System — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Always load project-root .env, even if backend starts from another cwd.
_ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ROOT_ENV, override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    labeled_dir: Path = data_dir / "labeled"
    models_dir: Path = data_dir / "models"
    fastf1_cache_dir: Path = raw_dir / "fastf1_cache"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    database_url: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_api_key_unprefixed: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    chat_model: str = "google/gemini-2.0-flash-exp:free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-3-small"
    chat_force_local_rag: bool = False
    chat_allow_llm: bool = True
    chat_llm_max_calls_per_minute: int = 10
    chat_llm_max_calls_per_session_per_minute: int = 5
    chat_answer_cache_ttl_seconds: int = 60
    chat_strategy_llm_min_interval_seconds: int = 30

    # ── Ollama (local LLM) ───────────────────────────────────────────────────
    # Set ollama_enabled=true to use local Ollama as the primary provider.
    # Falls back to Gemini → OpenRouter when Ollama is unavailable or times out.
    ollama_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    # How long to wait for Ollama before declaring it unavailable (seconds).
    # llama3.2 3B on CPU can take 15-40s for a full strategy response.
    ollama_timeout_s: float = 90.0
    # Ollama calls are local — they don't count against the API rate-limit budget.
    # This flag skips budget tracking for Ollama responses.
    ollama_skip_rate_limit_tracking: bool = True

    # ML inference
    overtake_ml_enabled: bool = True
    overtake_ml_min_probability: float = 0.0
    overtake_ml_blend_weight: float = 0.5

    # Telemetry processing
    default_frame_rate: float = 2.0
    chunk_duration_s: float = 60.0
    gzip_responses: bool = True

    # FastF1
    fastf1_rate_limit_delay: float = 0.3

    class Config:
        env_file = str(_ROOT_ENV)
        env_prefix = "F1VIZ_"


settings = Settings()

# Ensure directories exist
for d in [settings.raw_dir, settings.processed_dir, settings.labeled_dir,
          settings.models_dir, settings.fastf1_cache_dir]:
    d.mkdir(parents=True, exist_ok=True)
