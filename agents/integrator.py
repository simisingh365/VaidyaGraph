"""
Arogya Sutra - Integrator ("Senior Doctor") node.

Runs after all three specialist nodes have written their analyses into the
shared `AgentState`. Produces:

    * `integrative_report`  : full markdown Preventive Care Plan
    * `next_steps`          : parsed list of actionable items (the tail
                              "## 6. Next Steps (This Week)" section)

Parsing `next_steps` out of the markdown keeps the API response simple -
the frontend can render the full report AND show the action items as a
checklist without calling the LLM twice.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from core.llm import clean_llm_output, get_llm
from core.prompts import INTEGRATOR_PROMPT
from core.state import AgentState

logger = logging.getLogger(__name__)


_MISSING = "_(this specialist's analysis was unavailable)_"
_MISSING_SAFETY = (
    "_(Safety Officer review unavailable - treat combinations conservatively "
    "and consult a clinician before mixing systems.)_"
)


def _build_chain() -> Runnable:
    # Slightly higher temperature than the specialists: the integrator is
    # doing *synthesis*, not retrieval, and benefits from some rhetorical
    # room. Still well below "creative writing" territory.
    return INTEGRATOR_PROMPT | get_llm(temperature=0.4, max_tokens=1500) | StrOutputParser()


def _extract_next_steps(report: str) -> List[str]:
    """Pull the ordered list under the '## 6. Next Steps' heading.

    Falls back to an empty list if the model deviated from the template -
    we never want a formatting miss to crash the whole graph run.
    """
    # Grab everything from the Next Steps heading to end-of-string (or next
    # top-level heading, whichever comes first).
    match = re.search(
        r"##\s*6\.?\s*Next Steps.*?\n(?P<body>.*?)(?=\n#\s|\Z)",
        report,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    body = match.group("body")
    # Accept either "1. foo", "1) foo", or "- foo" style bullets.
    items = re.findall(
        r"^\s*(?:\d+[\.\)]|[-*])\s+(.+?)\s*$",
        body,
        flags=re.MULTILINE,
    )
    return [item.strip() for item in items if item.strip()]


def integrator_agent_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize the three specialist reports into one plan."""
    logger.info("Running integrator_agent_node")

    inputs = {
        "patient_symptoms": state.get("patient_symptoms", "").strip()
        or "(no symptoms provided)",
        "patient_history": state.get("patient_history", "").strip()
        or "(no prior history provided)",
        "allopathy_analysis": state.get("allopathy_analysis") or _MISSING,
        "ayurveda_analysis": state.get("ayurveda_analysis") or _MISSING,
        "homeopathy_analysis": state.get("homeopathy_analysis") or _MISSING,
        # Safety Officer output from the interaction_checker_node. If it
        # failed or was skipped, fall back to a conservative placeholder
        # instead of an empty string so the integrator doesn't silently
        # omit Section 1's safety line.
        "interaction_report": state.get("interaction_report") or _MISSING_SAFETY,
    }

    try:
        raw = _build_chain().invoke(inputs)
        # Strip <think> blocks from reasoning-class models before parsing
        # and returning. Cleaning here (rather than in app.py) means the
        # next_steps regex runs against the real plan, not a think block.
        report = clean_llm_output(raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Integrator agent failed: %s", exc)
        report = (
            "# Preventive Care Plan\n\n"
            f"_Integration step failed: {exc}. Specialist analyses are "
            "still available in the raw state._"
        )

    return {
        "integrative_report": report,
        "next_steps": _extract_next_steps(report),
    }


__all__ = ["integrator_agent_node"]
