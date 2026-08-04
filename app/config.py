"""Centralized configuration — loads .env once, exposes typed settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings:
    """Application settings sourced from environment variables."""

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.4")
    LLM_MODEL_FALLBACK: str = os.getenv("LLM_MODEL_FALLBACK", "gpt-5.4-mini")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-5.4")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # How many times a vision call may be attempted before the clip/event is left
    # unclassified. A dropped clip loses its tags AND its embedding, so a retry is the
    # cheapest way to buy catalogue coverage; 1 disables retrying.
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
    # Where the catalogue lives. A file path persists ingested rows across restarts
    # (so incremental reuse works); ':memory:' is ephemeral (used by tests).
    METADATA_DB_PATH: str = os.getenv("METADATA_DB_PATH", "./app/data/mapo_catalogue.db")

    # Debug
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_VERBOSE: bool = os.getenv("DEBUG_VERBOSE", "false").lower() == "true"


settings = Settings()