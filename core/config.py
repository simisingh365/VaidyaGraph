"""
Arogya Sutra - Centralized configuration.

Loads environment variables from a local .env file (if present) and exposes
them as a typed `Settings` object. All other modules should import `settings`
from here instead of reading os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load variables from .env into the process environment (no-op in production
# deployments that set real env vars).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for Arogya Sutra."""

    # Chutes AI (OpenAI-compatible endpoint)
    chutes_api_key: str
    chutes_base_url: str
    model_name: str

    # Vector store
    chroma_persist_dir: str


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is required but was not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings(
        chutes_api_key=_require("CHUTES_API_KEY"),
        chutes_base_url=os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1"),
        model_name=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    )


# Convenience module-level handle. Import this in most call sites:
#   from core.config import settings
try:
    settings = get_settings()
except RuntimeError:
    # Allow importing the module (e.g. during tooling / tests) even if the
    # environment is not fully configured yet. Accessing `settings` will raise
    # a clearer error at call time via get_settings().
    settings = None  # type: ignore[assignment]
