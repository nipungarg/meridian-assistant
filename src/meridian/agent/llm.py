"""Single place to construct the chat model (temperature=0 for grounded determinism)."""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from meridian.config import get_settings


@lru_cache(maxsize=4)
def get_chat_llm(temperature: float = 0.0) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model, temperature=temperature,
                      api_key=s.openai_api_key, timeout=40, max_retries=2)
