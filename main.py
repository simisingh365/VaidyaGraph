"""
Arogya Sutra - Application entry point.

Composition root only: build the FastAPI app, configure CORS, mount routers.
All actual request handling lives in the `api/` package; all agent logic
lives in `agents/`.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.diagnose import router as diagnose_router
from api.qa import router as qa_router
from core.config import get_settings
from core.llm import LLM_MAX_RETRIES, LLM_TIMEOUT_SEC

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Arogya Sutra",
    description=(
        "Agentic Preventive Diagnosis Platform combining Allopathy, Ayurveda, "
        "and Homeopathy perspectives via a fan-out/fan-in LangGraph workflow."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Wide-open during demo development. Before any real deployment, replace
# `allow_origins=["*"]` with an explicit list of trusted frontend origins -
# wildcard + credentials is also rejected by browsers, so tighten this up
# when you add auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(diagnose_router)
app.include_router(qa_router)


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Welcome / discovery endpoint."""
    return {
        "name": "Arogya Sutra",
        "tagline": "A panel of three doctors, one unified preventive plan.",
        "version": "0.1.0",
        "endpoints": {
            "diagnose": "POST /diagnose",
            "diagnose_stream": "POST /diagnose/stream  (text/event-stream)",
            "ask": "POST /ask  (follow-up Q&A, needs session_id)",
            "ask_reset": "POST /ask/reset  (clear chat history)",
            "docs": "/docs",
            "health": "/health",
            "meta": "/meta",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/meta")
def meta() -> dict:
    """Expose non-secret runtime config.

    The Streamlit frontend calls this to display which model is actually
    powering the current session in its run-metadata footer - useful as a
    trust signal for technical reviewers (they can see you're not hiding
    which LLM produced the results) and convenient during model swaps.

    NEVER include the API key or any other secret here.
    """
    settings = get_settings()
    return {
        "app": "VaidyaGraph",
        "version": "0.1.0",
        "model": settings.model_name,
        "provider": "Chutes AI (OpenAI-compatible)",
        "base_url": settings.chutes_base_url,
        "llm_timeout_sec": LLM_TIMEOUT_SEC,
        "llm_max_retries": LLM_MAX_RETRIES,
    }


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
