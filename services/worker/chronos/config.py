"""Central configuration for the ChronosAudit worker.

Settings are resolved (in order of precedence):
  1. OS environment variables
  2. a local ``.env`` file (via python-dotenv)
  3. defaults defined below

The worker is deliberately runnable with **zero external credentials**: set
``MODEL_PROVIDER=none`` and ``EMBEDDING_PROVIDER=hashing`` and the entire
pipeline degrades to deterministic heuristics.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from this package's parent directory or CWD fallback.
_env_candidates = [
    Path(__file__).resolve().parents[1] / ".env",
    Path.cwd() / ".env",
]
for _p in _env_candidates:
    if _p.exists():
        load_dotenv(_p, override=False)
        break


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=tuple(str(p) for p in _env_candidates), extra="ignore")

    # --- Extraction SLM ----------------------------------------------------
    model_provider: Literal["openai", "ollama", "none"] = "none"
    model_name: str = "llama3.2"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_extraction_model: str = "gpt-4o-mini"
    openai_qa_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.0

    # --- Embeddings --------------------------------------------------------
    embedding_provider: Literal["openai", "ollama", "hashing"] = "hashing"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # used only for the hashing fallback

    # --- Storage -----------------------------------------------------------
    lance_db_path: Path = Path("./data/lancedb")
    graph_store_path: Path = Path("./data/graph/chronos.gpickle")
    graph_export_path: Path = Path("./data/graph/exports")
    doc_state_dir: Path = Path("./data/state")
    upload_dir: Path = Path("./data/uploads")

    # --- Queue / orchestration --------------------------------------------
    worker_mode: Literal["sync", "async"] = "sync"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # --- Bridge API --------------------------------------------------------
    worker_api_host: str = "0.0.0.0"
    worker_api_port: int = 8100
    worker_api_token: str = "change-me-dev-token"

    # --- Ingestion tuning --------------------------------------------------
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    ingest_batch_size: int = 8
    max_graph_context_edges: int = 60

    def ensure_directories(self) -> None:
        """Create every filesystem location the worker writes to."""
        for d in (
            self.lance_db_path,
            self.graph_store_path.parent,
            self.graph_export_path,
            self.doc_state_dir,
            self.upload_dir,
        ):
            Path(d).mkdir(parents=True, exist_ok=True) if isinstance(d, Path) else Path(str(d)).mkdir(
                parents=True, exist_ok=True
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    # Mirror a couple of knobs into the environment for library-level tools.
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
    return settings