"""Central configuration, loaded from environment / .env.

All tunables (models, retrieval thresholds, API location, demo clock) live here so the
rest of the codebase never reads ``os.environ`` directly.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL"
    )

    # --- Vector store ---
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_DIR")
    chroma_collection: str = Field(default="meridian_kb", alias="CHROMA_COLLECTION")

    # --- Source documents ---
    files_dir: Path = Field(default=PROJECT_ROOT / "files", alias="FILES_DIR")

    # --- Retrieval / reranking ---
    reranker: str = Field(default="cross_encoder", alias="RERANKER")  # cross_encoder|llm|none
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    # Floor on the top embedding cosine similarity; below this -> clarify/handoff. Embedding
    # similarity separates relevant (~0.38-0.69) from out-of-scope (~0.07-0.13) far more
    # reliably than the cross-encoder's absolute score.
    min_retrieval_score: float = Field(default=0.25, alias="MIN_RETRIEVAL_SCORE")

    # --- Mock Booking API ---
    booking_api_base_url: str = Field(
        default="http://localhost:8000/v1", alias="BOOKING_API_BASE_URL"
    )
    booking_api_token: str = Field(default="demo-token", alias="BOOKING_API_TOKEN")

    # --- Demo clock ---
    demo_date: str = Field(default="", alias="DEMO_DATE")

    @property
    def today(self) -> _dt.date:
        """The assistant's notion of 'today' (override via DEMO_DATE for reproducible demos)."""
        if self.demo_date.strip():
            return _dt.date.fromisoformat(self.demo_date.strip())
        return _dt.date.today()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
