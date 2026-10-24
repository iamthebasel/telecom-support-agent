"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # Retrieval
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers"))
    st_model: str = field(default_factory=lambda: _env("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ollama_embed_model: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "120")))      # words
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "30")))  # words

    # Routing: "rules", "semantic" or "llm"
    router_mode: str = field(default_factory=lambda: _env("ROUTER_MODE", "rules"))

    # External services and local storage
    churn_api_url: str = field(default_factory=lambda: _env("CHURN_API_URL", "http://localhost:5000"))
    churn_api_timeout: float = field(default_factory=lambda: float(_env("CHURN_API_TIMEOUT", "5")))
    kb_dir: Path = field(default_factory=lambda: Path(_env("KB_DIR", str(PROJECT_ROOT / "data" / "knowledge_base"))))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))))
    db_path: Path = field(default_factory=lambda: Path(_env("CUSTOMER_DB", str(PROJECT_ROOT / "data" / "customers.db"))))
    log_path: Path = field(default_factory=lambda: Path(_env("RUN_LOG", str(PROJECT_ROOT / "logs" / "runs.jsonl"))))


def get_settings() -> Settings:
    return Settings()
