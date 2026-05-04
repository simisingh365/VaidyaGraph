"""
VaidyaGraph - Patient follow-up Q&A agent.

Runs OUTSIDE the main LangGraph workflow. After the panel has finished,
each `/ask` request invokes `answer_question(...)` with:
    * the frozen AgentState (what the panel produced)
    * the running chat history
    * the new question

Returns a plain string answer. The caller (api/qa.py) is responsible for
appending the new user/assistant pair to the session's chat_history.

Why not a graph node?
    The Q&A loop doesn't fan-out/fan-in; it's a single chain invocation.
    Wrapping it in a StateGraph would add ceremony with zero payoff. We
    still reuse core.llm, core.prompts, and AgentState so the two paths
    stay consistent.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from core.llm import clean_llm_output, get_llm
from core.prompts import QA_PROMPT
from core.state import AgentState, ChatTurn

logger = logging.getLogger(__name__)


def _missing(field: str) -> str:
    return f"_(the panel did not produce a '{field}' section in this run)_"


def _history_to_messages(history: List[ChatTurn]) -> List[BaseMessage]:
    """Convert our TypedDict chat turns to LangChain message objects.

    MessagesPlaceholder expects `BaseMessage` instances, not dicts.
    """
    out: List[BaseMessage] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        # Silently drop malformed turns - don't crash on a bad session.
    return out


def answer_question(state: AgentState, question: str) -> str:
    """Run the Q&A agent against the frozen panel state and chat history.

    Parameters
    ----------
    state:
        Completed `AgentState` from an earlier /diagnose run. Must include
        the panel outputs; chat_history may be empty or already populated.
    question:
        The new, unanswered patient question.

    Returns
    -------
    A short plain-text answer. Never raises - on failure, returns a
    user-friendly error string so the chat UI can render it as an
    assistant message.
    """
    if not question or not question.strip():
        return "_(please type a question first)_"

    chat_history = state.get("chat_history") or []

    inputs = {
        "patient_symptoms": state.get("patient_symptoms") or "(not provided)",
        "patient_history": state.get("patient_history") or "(not provided)",
        "allopathy_analysis": state.get("allopathy_analysis") or _missing("allopathy"),
        "ayurveda_analysis": state.get("ayurveda_analysis") or _missing("ayurveda"),
        "homeopathy_analysis": state.get("homeopathy_analysis") or _missing("homeopathy"),
        "interaction_report": state.get("interaction_report") or _missing("safety"),
        "integrative_report": state.get("integrative_report") or _missing("plan"),
        "chat_history": _history_to_messages(chat_history),
        "question": question.strip(),
    }

    # Temperature 0.3 matches the specialist agents: grounded, not creative.
    # max_tokens kept small because the prompt enforces 2-4 sentence answers.
    chain = QA_PROMPT | get_llm(temperature=0.3, max_tokens=500) | StrOutputParser()

    try:
        raw = chain.invoke(inputs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Q&A agent failed: %s", exc)
        return (
            "_I couldn't reach the model for that question - please try "
            "again in a moment. If it keeps happening, check the backend "
            "logs._"
        )

    return clean_llm_output(raw) or "_(empty answer)_"


__all__ = ["answer_question"]
