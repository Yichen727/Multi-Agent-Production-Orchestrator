"""Centralized application configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project root.
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings:
    """Application settings loaded from environment variables."""

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.5")
    LLM_MODEL_FALLBACK: str = os.getenv("LLM_MODEL_FALLBACK", "gpt-5.5-mini")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-5.5")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    VISION_MAX_ATTEMPTS: int = int(os.getenv("VISION_MAX_ATTEMPTS", "3"))

    # LangSmith
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "MAPO-Production-Orchestrator")

    # Paths
    RAW_FOOTAGE_DIR: Path = Path(os.getenv("RAW_FOOTAGE_DIR", "./app/data/raw_footage"))
    PROCESSED_OUTPUT_DIR: Path = Path(os.getenv("PROCESSED_OUTPUT_DIR", "./app/data/output"))
    PROXY_OUTPUT_DIR: Path = Path(os.getenv("PROXY_OUTPUT_DIR", "./app/data/proxies"))
    SUPPORTED_VIDEO_FORMATS: list[str] = os.getenv(
        "SUPPORTED_VIDEO_FORMATS", "mp4,mov,avi,mxf,r3d,braw"
    ).split(",")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///mapo_production.db")
    METADATA_DB_PATH: str = os.getenv("METADATA_DB_PATH", "./app/data/mapo_catalogue.db")

    # Debug
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_VERBOSE: bool = os.getenv("DEBUG_VERBOSE", "false").lower() == "true"


settings = Settings()