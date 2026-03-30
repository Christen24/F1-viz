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
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str = ""
    gemini_api_key_unprefixed: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    gemini_model_unprefixed: str = Field(default="", validation_alias="GEMINI_MODEL")
    chat_force_local_rag: bool = False

    # Telemetry processing
    default_frame_rate: float = 2.0  # Hz — master timeline sample rate
    chunk_duration_s: float = 60.0  # seconds per telemetry chunk
    gzip_responses: bool = True

    # FastF1
    fastf1_rate_limit_delay: float = 0.3  # seconds between API calls

    class Config:
        env_file = str(_ROOT_ENV)
        env_prefix = "F1VIZ_"


settings = Settings()

# Ensure directories exist
for d in [settings.raw_dir, settings.processed_dir, settings.labeled_dir,
          settings.models_dir, settings.fastf1_cache_dir]:
    d.mkdir(parents=True, exist_ok=True)
