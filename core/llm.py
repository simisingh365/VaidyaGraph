"""
VaidyaGraph - LLM client factory.

Chutes AI exposes an OpenAI-compatible endpoint, so we reuse `ChatOpenAI`
from `langchain_openai` and simply redirect it at `CHUTES_BASE_URL` with
the Chutes API key. Centralizing the client here means every agent
(specialists, safety officer, integrator) shares the same timeout, retry,
and connection-pool policy.

Timeout / retry notes
---------------------
Chutes runs community-hosted inference; a cold model or busy endpoint can
easily take 90-120 seconds to produce a full ~1k-token response. A 60s
timeout - the default we started with - was the source of the "agents are
timing out" symptom. We now give each LLM call 180s and retry up to 2
times on transient failures (connection reset, 5xx, rate-limit backoff),
which matches the FastAPI request budget in `api/diagnose.py`
(REQUEST_TIMEOUT_SEC = 180).
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_openai import ChatOpenAI

from core.config import get_settings

# Per-call budget for a single LLM completion. Five nodes run sequentially
# on the critical path (3 specialists in parallel -> safety officer ->
# integrator), so the total wall-clock is roughly
# max(specialists) + safety + integrator. 180s per call is comfortable
# headroom for a 32B model producing ~1.5k tokens.
LLM_TIMEOUT_SEC = 180

# Retry transient network failures (connection reset, read timeout,
# 429 rate-limit, 5xx). `ChatOpenAI` uses the underlying OpenAI SDK's
# exponential backoff, so 2 retries = up to ~3-4s of extra wait in the
# worst case, which is cheap insurance against a single flaky packet.
LLM_MAX_RETRIES = 2


@lru_cache(maxsize=4)
def get_llm(
    temperature: float = 0.3,
    max_tokens: int | None = 1024,
) -> ChatOpenAI:
    """Return a cached `ChatOpenAI` instance wired to Chutes AI.

    Parameters
    ----------
    temperature:
        Lower (0.1-0.4) for clinical reasoning so answers stay grounded.
        The safety officer uses 0.1; specialists use 0.3; the integrator
        uses 0.4.
    max_tokens:
        Per-response cap. `None` lets the provider decide.

    Returns
    -------
    A `ChatOpenAI` instance that speaks OpenAI's Chat Completions API
    against Chutes' base URL. Cached so the three specialists share a
    single underlying HTTP client (connection pooling).
    """
    settings = get_settings()

    return ChatOpenAI(
        # Model + endpoint come straight from the validated Settings
        # object - never read os.environ here, so tests and alternate
        # deployments can swap in their own Settings cleanly.
        model=settings.model_name,
        api_key=settings.chutes_api_key,
        base_url=settings.chutes_base_url,
        # Generation params
        temperature=temperature,
        max_tokens=max_tokens,
        # Reliability knobs - bumped from the original 60s/2 to fix the
        # "agents timing out" symptom on slower / busier Chutes nodes.
        timeout=LLM_TIMEOUT_SEC,
        max_retries=LLM_MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Output cleaner
# ---------------------------------------------------------------------------
# Some reasoning-oriented models (Qwen, DeepSeek-R1, etc.) emit their
# chain-of-thought inside <think>...</think> blocks before the real answer.
# That's great for debugging but ugly for a demo UI. We centralize a single
# regex here so every node can strip it the same way - and a future switch
# to a non-reasoning model is a no-op.
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)

# Models sometimes open <think> but get truncated before closing it. Catch
# that case too so we don't leak a half-open tag into the UI.
_OPEN_THINK_TO_END_RE = re.compile(
    r"<think\b[^>]*>.*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_llm_output(text: str) -> str:
    """Remove <think> blocks and collapse excess whitespace.

    Safe to call on any string - no-op if the model didn't emit reasoning
    tags. Returns the input unchanged on None/empty input for callers that
    want to pipe it unconditionally.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _OPEN_THINK_TO_END_RE.sub("", cleaned)
    # Collapse 3+ consecutive blank lines that often appear where the
    # think block used to be.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


__all__ = [
    "get_llm",
    "clean_llm_output",
    "LLM_TIMEOUT_SEC",
    "LLM_MAX_RETRIES",
]
