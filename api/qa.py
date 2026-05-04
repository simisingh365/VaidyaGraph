"""
VaidyaGraph - `/ask` API route (follow-up Q&A over panel output).

Lifecycle
---------
1. Client calls POST /diagnose (or /diagnose/stream) with patient input.
   The response now includes a `session_id` (see api/diagnose.py).
2. Client stores the session_id locally.
3. For each follow-up question, client calls POST /ask with
       { "session_id": "...", "question": "..." }
   and receives
       { "answer": "...", "chat_history": [...] }
4. The session store on the backend retains the panel output plus the
   growing chat history. Session TTL is 2 hours; see core/sessions.py.

Errors
------
    * 404 - session not found / expired
    * 422 - malformed body (FastAPI auto-generated)
    * 500 - LLM / internal failure (bubbled from the agent)
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.qa import answer_question
from core.sessions import session_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=8, description="From /diagnose response.")
    question: str = Field(
        ...,
        min_length=2,
        description="Patient's follow-up question.",
        examples=["Can I take the PPI along with the Ashwagandha you suggested?"],
    )


class ChatTurnOut(BaseModel):
    role: str
    content: str


class AskResponse(BaseModel):
    answer: str
    chat_history: List[ChatTurnOut]


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a follow-up question grounded in the session's panel output."""
    state = session_store.get(req.session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found or expired. Run /diagnose again to "
                "start a fresh panel."
            ),
        )

    # Call the agent. It reads the current chat_history from `state`
    # internally, so we append the user turn AFTER invocation - otherwise
    # the agent would see its own question in its own history and that's
    # confusing. We append both turns now so the returned chat_history is
    # in lockstep with what the agent just reasoned over.
    answer = answer_question(state, req.question)

    # Persist the turn pair to the session.
    session_store.append_turn(req.session_id, {"role": "user", "content": req.question})
    session_store.append_turn(req.session_id, {"role": "assistant", "content": answer})

    # Return the full history so the frontend can trivially re-render the
    # chat without tracking state itself.
    refreshed = session_store.get(req.session_id) or {}
    history = refreshed.get("chat_history", []) or []
    return AskResponse(
        answer=answer,
        chat_history=[ChatTurnOut(role=t["role"], content=t["content"]) for t in history],
    )


@router.post("/ask/reset")
async def reset_history(session_id: str) -> dict:
    """Clear the chat history for a session (keep the panel output)."""
    ok = session_store.clear_history(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "reset", "session_id": session_id}


__all__ = ["router"]
