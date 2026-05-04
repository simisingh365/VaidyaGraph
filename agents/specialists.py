"""
Arogya Sutra - Specialist agent nodes for the LangGraph workflow.

Each node:
  * reads `patient_symptoms` and `patient_history` from `AgentState`
  * invokes the Chutes-hosted LLM with a role-enforcing prompt
  * returns a *partial* state dict containing only the field it owns

Because `AgentState` is declared `total=False`, returning a partial dict is
enough - LangGraph merges it into the running state.
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from core.llm import clean_llm_output, get_llm
from core.prompts import (
    ALLOPATHY_PROMPT,
    AYURVEDA_PROMPT,
    HOMEOPATHY_PROMPT,
    SAFETY_OFFICER_PROMPT,
)
from core.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_chain(prompt: ChatPromptTemplate) -> Runnable:
    """Compose prompt -> LLM -> string parser.

    A fresh chain is cheap to build (the LLM itself is cached in `get_llm`),
    and wrapping in `StrOutputParser` means the agent nodes can just return
    `{"field": chain.invoke(...)}` without poking at message objects.
    """
    return prompt | get_llm(temperature=0.3) | StrOutputParser()


def _inputs(state: AgentState) -> Dict[str, str]:
    """Extract the two fields every specialist prompt expects.

    Tolerates missing history (common for walk-in demos) by substituting a
    neutral placeholder so the prompt still renders cleanly.
    """
    return {
        "patient_symptoms": state.get("patient_symptoms", "").strip()
        or "(no symptoms provided)",
        "patient_history": state.get("patient_history", "").strip()
        or "(no prior history provided)",
    }


def _safe_invoke(chain: Runnable, inputs: Dict[str, str], label: str) -> str:
    """Invoke an LLM chain, returning a friendly error string on failure.

    The raw LLM output is always passed through `clean_llm_output` so any
    <think>...</think> reasoning tags emitted by reasoning-class models
    (Qwen, DeepSeek-R1, etc.) don't leak into the UI markdown. The clean
    step is a no-op on models that don't produce reasoning tags.

    We never want one specialist crashing to abort the whole graph - the
    integrator can still synthesize a report from the remaining two.
    """
    try:
        raw = chain.invoke(inputs)
    except Exception as exc:  # noqa: BLE001 - log & degrade gracefully
        logger.exception("%s agent failed: %s", label, exc)
        return f"_{label} analysis unavailable: {exc}_"
    return clean_llm_output(raw)


# ---------------------------------------------------------------------------
# Node: Allopathy
# ---------------------------------------------------------------------------
def allopathy_agent_node(state: AgentState) -> Dict[str, str]:
    """Produce a strictly modern-medicine analysis."""
    logger.info("Running allopathy_agent_node")
    chain = _build_chain(ALLOPATHY_PROMPT)
    analysis = _safe_invoke(chain, _inputs(state), "Allopathy")
    return {"allopathy_analysis": analysis}


# ---------------------------------------------------------------------------
# Node: Ayurveda
# ---------------------------------------------------------------------------
def ayurveda_agent_node(state: AgentState) -> Dict[str, str]:
    """Produce a Dosha-framework Ayurvedic analysis."""
    logger.info("Running ayurveda_agent_node")
    chain = _build_chain(AYURVEDA_PROMPT)
    analysis = _safe_invoke(chain, _inputs(state), "Ayurveda")
    return {"ayurveda_analysis": analysis}


# ---------------------------------------------------------------------------
# Node: Homeopathy
# ---------------------------------------------------------------------------
def homeopathy_agent_node(state: AgentState) -> Dict[str, str]:
    """Produce a totality-of-symptoms homeopathic analysis."""
    logger.info("Running homeopathy_agent_node")
    chain = _build_chain(HOMEOPATHY_PROMPT)
    analysis = _safe_invoke(chain, _inputs(state), "Homeopathy")
    return {"homeopathy_analysis": analysis}


# ---------------------------------------------------------------------------
# Node: Safety Officer / Interaction Checker
# ---------------------------------------------------------------------------
# This is NOT a specialist in the same sense as the three above - it runs
# AFTER them and reads their outputs instead of the raw patient inputs.
# It's placed in this module (rather than a separate file) because it shares
# the same chain/inputs plumbing and is conceptually still "panel-level"
# reasoning; the integrator that consumes its output remains the clear
# terminal node of the graph.
_MISSING_ANALYSIS = "_(analysis unavailable)_"


def interaction_checker_node(state: AgentState) -> Dict[str, str]:
    """Cross-check the three specialist plans for safety issues.

    Reads: allopathy_analysis, ayurveda_analysis, homeopathy_analysis
    Writes: interaction_report

    Behaves as a pharmacovigilance reviewer: flags drug-herb interactions,
    population-specific contraindications, and redundant therapies. If the
    plans are safe to combine, it says so explicitly.
    """
    logger.info("Running interaction_checker_node")

    inputs = {
        "allopathy_analysis": state.get("allopathy_analysis") or _MISSING_ANALYSIS,
        "ayurveda_analysis": state.get("ayurveda_analysis") or _MISSING_ANALYSIS,
        "homeopathy_analysis": state.get("homeopathy_analysis") or _MISSING_ANALYSIS,
    }

    # Temperature 0.1: we want deterministic, checklist-style output here -
    # this is the one node where creativity is actively harmful.
    chain = SAFETY_OFFICER_PROMPT | get_llm(temperature=0.1, max_tokens=1200) | StrOutputParser()

    report = _safe_invoke(chain, inputs, "Safety Officer")
    return {"interaction_report": report}


__all__ = [
    "allopathy_agent_node",
    "ayurveda_agent_node",
    "homeopathy_agent_node",
    "interaction_checker_node",
]
