"""
Arogya Sutra - Shared LangGraph state.

`AgentState` is the single source of truth that flows between nodes in the
LangGraph workflow. Each specialist agent (Allopathy, Ayurveda, Homeopathy)
writes into its own field; the integrator agent consumes all three and
produces the final report and next steps.
"""

from __future__ import annotations

from typing import List, Literal, TypedDict


class ChatTurn(TypedDict):
    """One turn in a follow-up Q&A conversation.

    `role` matches OpenAI's chat-completions convention so the chat history
    can be fed into a ChatPromptTemplate via MessagesPlaceholder without
    further massaging.
    """

    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    """State passed between nodes in the VaidyaGraph workflow.

    Fields are marked optional (`total=False`) so that individual nodes can
    populate just the slice they are responsible for without having to
    re-supply every key on every update.
    """

    # --- Patient-provided inputs ---
    patient_symptoms: str
    patient_history: str

    # --- Per-system specialist analyses ---
    allopathy_analysis: str
    ayurveda_analysis: str
    homeopathy_analysis: str

    # --- Safety layer ---
    # Pharmacovigilance review that cross-checks the three specialist plans
    # for drug-herb interactions, contraindications, and redundant therapies.
    # Populated by `interaction_checker_node` after all specialists finish.
    interaction_report: str

    # --- Final synthesized output ---
    integrative_report: str
    next_steps: List[str]

    # --- Follow-up Q&A (Option 1: multi-turn dialogue over panel output) ---
    # The panel writes its frozen outputs once; subsequent /ask requests
    # append turns to `chat_history` and invoke the Q&A agent, which reads
    # the entire state but only WRITES back into chat_history. The panel
    # fields themselves are never mutated by Q&A - that way the plan the
    # patient saw first stays anchored.
    chat_history: List[ChatTurn]
