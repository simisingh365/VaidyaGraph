"""
Arogya Sutra - `/diagnose` API route.

Exposes the LangGraph workflow over HTTP. Two modes are supported:

    POST /diagnose            -> blocking JSON response with the final plan
    POST /diagnose/stream     -> text/event-stream of per-node updates
                                 (useful for a frontend that wants to show
                                  "Ayurveda thinking..." etc. live)

Response shape is intentionally grouped into three top-level objects so
clients can render them as distinct UI regions without parsing:

    {
      "analysis": { allopathy, ayurveda, homeopathy },
      "safety":   { severity, verdict, report },
      "plan":     { integrative_report, next_steps }
    }

Kept in its own module so `main.py` stays a thin composition root.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.graph import get_compiled_graph
from core.sessions import session_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["diagnose"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    """Incoming patient payload.

    `history` is optional because walk-in demo users often skip it; the
    specialist nodes already tolerate missing history via a placeholder.
    """

    symptoms: str = Field(
        ...,
        min_length=3,
        description="Free-text description of current symptoms.",
        examples=["Recurring acidity after meals, poor sleep, mild headaches."],
    )
    history: Optional[str] = Field(
        default="",
        description="Optional prior medical / lifestyle history.",
        examples=["32yo male, desk job, heavy tea and coffee intake."],
    )


# ---------------------------------------------------------------------------
# Response schemas - grouped by concern
# ---------------------------------------------------------------------------
# Severity is a closed enum so the frontend can drive colour / iconography
# without string-matching into free text. Mapping from the Safety Officer's
# verdict line:
#     "No major interactions found." -> "safe"
#     "Caution advised."             -> "caution"
#     "Do NOT combine as-is."        -> "unsafe"
#     anything else / missing        -> "unknown"
Severity = Literal["safe", "caution", "unsafe", "unknown"]


class PanelAnalysis(BaseModel):
    """Raw outputs from the three paradigm-isolated specialists.

    These are intentionally kept separate from the safety and plan blocks
    so the UI can render them as three side-by-side "doctor cards".
    """

    allopathy: str = Field(..., description="Modern-medicine analysis.")
    ayurveda: str = Field(..., description="Dosha-framework analysis.")
    homeopathy: str = Field(..., description="Totality-of-symptoms analysis.")


class SafetyCheck(BaseModel):
    """Output of the Safety Officer / interaction checker node.

    Lives on its own so a frontend can render it with a prominent
    red/amber/green badge without parsing markdown.
    """

    severity: Severity = Field(
        ...,
        description=(
            "Machine-readable severity derived from the Safety Officer's "
            "verdict line. Drives UI colour coding."
        ),
    )
    verdict: str = Field(
        ...,
        description=(
            "The Safety Officer's one-line verdict, verbatim, e.g. "
            "'Caution advised.'"
        ),
    )
    report: str = Field(
        ...,
        description=(
            "Full pharmacovigilance markdown: drug-herb interactions, "
            "contraindications, redundant therapies, recommended actions."
        ),
    )


class CarePlan(BaseModel):
    """The Senior Doctor's unified, patient-facing plan."""

    integrative_report: str = Field(
        ...,
        description="Full markdown Preventive Care Plan (6 sections).",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description=(
            "Parsed actionable items from Section 6 of the plan. Ready to "
            "render as a checklist."
        ),
    )


class DiagnoseResponse(BaseModel):
    """Grouped response so Analysis, Safety, and Plan are clearly distinct.

    `session_id` is a short opaque token the client should retain to make
    follow-up /ask calls. It references the frozen AgentState on the
    backend (see core/sessions.py).
    """

    session_id: str = Field(
        ...,
        description="Opaque session token for follow-up /ask calls.",
    )
    analysis: PanelAnalysis
    safety: SafetyCheck
    plan: CarePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_initial_state(req: DiagnoseRequest) -> dict[str, Any]:
    """Map the HTTP payload to the graph's `AgentState` shape."""
    return {
        "patient_symptoms": req.symptoms,
        "patient_history": req.history or "",
    }


# Regex grabs the first non-empty line under "### Verdict" in the Safety
# Officer's markdown output. Tolerant of stray whitespace / bolding.
_VERDICT_RE = re.compile(
    r"###\s*Verdict\s*\n+\s*(?:\*+\s*)?(?P<line>[^\n*]+?)(?:\s*\*+)?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _parse_safety(report: str) -> tuple[Severity, str]:
    """Extract (severity, verdict_line) from the Safety Officer report.

    The prompt contract guarantees a "### Verdict" section with one of
    three canonical lines. We still fall back gracefully to `unknown` so a
    model that deviates from the template never crashes the API.
    """
    if not report.strip():
        return "unknown", ""

    match = _VERDICT_RE.search(report)
    verdict_line = match.group("line").strip() if match else ""

    lowered = verdict_line.lower()
    if "do not combine" in lowered or "do not combine as-is" in lowered:
        return "unsafe", verdict_line
    if "caution" in lowered:
        return "caution", verdict_line
    if "no major interaction" in lowered or "no interactions" in lowered:
        return "safe", verdict_line
    return "unknown", verdict_line


def _build_response(final_state: dict[str, Any], session_id: str) -> DiagnoseResponse:
    """Map the final graph state to the grouped HTTP response."""
    interaction_report = final_state.get("interaction_report", "") or ""
    severity, verdict = _parse_safety(interaction_report)

    return DiagnoseResponse(
        session_id=session_id,
        analysis=PanelAnalysis(
            allopathy=final_state.get("allopathy_analysis", "") or "",
            ayurveda=final_state.get("ayurveda_analysis", "") or "",
            homeopathy=final_state.get("homeopathy_analysis", "") or "",
        ),
        safety=SafetyCheck(
            severity=severity,
            verdict=verdict,
            report=interaction_report,
        ),
        plan=CarePlan(
            integrative_report=final_state.get("integrative_report", "") or "",
            next_steps=final_state.get("next_steps", []) or [],
        ),
    )


# ---------------------------------------------------------------------------
# Blocking endpoint
# ---------------------------------------------------------------------------
@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    """Run the full panel and return the unified plan.

    We use `ainvoke` so the three specialists genuinely run concurrently on
    the event loop instead of blocking one worker thread each.
    """
    graph = get_compiled_graph()

    try:
        final_state: dict[str, Any] = await graph.ainvoke(_build_initial_state(req))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diagnose graph run failed")
        raise HTTPException(
            status_code=500, detail=f"Graph execution failed: {exc}"
        ) from exc

    # Register the frozen state so follow-up /ask calls can find it.
    session_id = session_store.create(final_state)
    return _build_response(final_state, session_id)


# ---------------------------------------------------------------------------
# Streaming endpoint (Server-Sent Events)
# ---------------------------------------------------------------------------
async def _sse_events(req: DiagnoseRequest) -> AsyncIterator[str]:
    """Yield one SSE frame per node completion, then a final frame.

    Frame format:
        event: node_update          |  event: final  |  event: error
        data:  {"node": "ayurveda", "state": {...}}

    Frontends can read these to light up each specialist card as its
    analysis lands, before the integrator finishes. The final frame carries
    the same grouped `DiagnoseResponse` shape as the blocking endpoint, so
    clients only need one rendering path for the end state.
    """
    graph = get_compiled_graph()

    def _frame(event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    final_state: dict[str, Any] = {}

    try:
        # `astream(..., stream_mode="updates")` yields a dict keyed by the
        # node that just produced output, mapped to that node's partial
        # state update. Perfect granularity for a "panel" UI.
        async for update in graph.astream(
            _build_initial_state(req), stream_mode="updates"
        ):
            for node_name, node_state in update.items():
                # Accumulate so we can emit a structured `final` frame that
                # matches the blocking endpoint's response shape.
                if isinstance(node_state, dict):
                    final_state.update(node_state)
                yield _frame(
                    "node_update",
                    {"node": node_name, "state": node_state},
                )

        # Register the frozen state so the streaming client can also use
        # /ask. The session id is embedded in the final frame.
        session_id = session_store.create(final_state)
        yield _frame(
            "final",
            json.loads(
                _build_response(final_state, session_id).model_dump_json()
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diagnose stream failed")
        yield _frame("error", {"message": str(exc)})


@router.post("/diagnose/stream")
async def diagnose_stream(req: DiagnoseRequest) -> StreamingResponse:
    """Stream per-agent progress as Server-Sent Events."""
    return StreamingResponse(
        _sse_events(req),
        media_type="text/event-stream",
        headers={
            # Disable buffering on common reverse proxies (nginx) so frames
            # reach the browser as they're produced.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
