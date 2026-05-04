"""
Arogya Sutra - LangGraph workflow (fan-out / safety-gate / fan-in).

Topology:

                       +----------------+
                       |      START     |
                       +--------+-------+
                                |
               +----------------+----------------+
               |                |                |
               v                v                v
        +-------------+  +-------------+  +--------------+
        |  Allopathy  |  |  Ayurveda   |  |  Homeopathy  |   <-- parallel
        +------+------+  +------+------+  +-------+------+       super-step
               |                |                 |
               +----------------+-----------------+
                                |
                                v
                      +---------------------+
                      | Interaction Checker |   <-- join #1
                      |  (Safety Officer)   |       (drug-herb, contra-
                      +----------+----------+        indications, stacking)
                                 |
                                 v
                         +---------------+
                         |  Integrator   |   <-- join #2 (terminal)
                         +-------+-------+
                                 |
                                 v
                               END

Why a dedicated safety gate?
    Putting the pharmacovigilance check on its own node (rather than folding
    it into the integrator prompt) has two concrete benefits:
      1. The integrator receives an *already-audited* safety report, so its
         output always surfaces warnings in a consistent place.
      2. The frontend can render `interaction_report` independently with its
         own red/amber/green badge without parsing the full plan.

Scheduling notes:
    The three specialists have independent incoming edges from START and
    write to disjoint keys, so LangGraph runs them concurrently.
    The interaction checker has three incoming edges (one per specialist) -
    LangGraph treats multiple incoming edges as an implicit barrier, so it
    only fires once ALL specialists have produced their analyses.
    The integrator likewise waits for the checker.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.integrator import integrator_agent_node
from agents.specialists import (
    allopathy_agent_node,
    ayurveda_agent_node,
    homeopathy_agent_node,
    interaction_checker_node,
)
from core.state import AgentState

logger = logging.getLogger(__name__)


# Node name constants - using symbols (not string literals) everywhere keeps
# refactors safe and typos loud. Also handy for the SSE streaming endpoint
# which emits events keyed by node name.
N_ALLOPATHY = "allopathy"
N_AYURVEDA = "ayurveda"
N_HOMEOPATHY = "homeopathy"
N_INTERACTION_CHECKER = "interaction_checker"
N_INTEGRATOR = "integrator"


def build_graph() -> CompiledStateGraph:
    """Construct and compile the Arogya Sutra workflow."""
    logger.info("Building Arogya Sutra LangGraph workflow (with safety gate)")

    graph = StateGraph(AgentState)

    # --- Nodes ---------------------------------------------------------
    graph.add_node(N_ALLOPATHY, allopathy_agent_node)
    graph.add_node(N_AYURVEDA, ayurveda_agent_node)
    graph.add_node(N_HOMEOPATHY, homeopathy_agent_node)
    graph.add_node(N_INTERACTION_CHECKER, interaction_checker_node)
    graph.add_node(N_INTEGRATOR, integrator_agent_node)

    # --- Fan-out from START -------------------------------------------
    # Three independent edges out of START => concurrent execution in the
    # same super-step.
    graph.add_edge(START, N_ALLOPATHY)
    graph.add_edge(START, N_AYURVEDA)
    graph.add_edge(START, N_HOMEOPATHY)

    # --- Join #1: specialists -> safety gate --------------------------
    # All three specialists must complete before the interaction checker
    # runs, because it needs every plan to cross-reference. LangGraph's
    # super-step scheduler enforces this automatically when a node has
    # multiple incoming edges.
    graph.add_edge(N_ALLOPATHY, N_INTERACTION_CHECKER)
    graph.add_edge(N_AYURVEDA, N_INTERACTION_CHECKER)
    graph.add_edge(N_HOMEOPATHY, N_INTERACTION_CHECKER)

    # --- Join #2: safety gate -> integrator ---------------------------
    # Single sequential edge. The integrator now receives the raw
    # specialist analyses AND the pharmacovigilance report, so it can fold
    # warnings into Section 1 ("Urgent Safety Check") of the final plan.
    graph.add_edge(N_INTERACTION_CHECKER, N_INTEGRATOR)

    # --- Terminate -----------------------------------------------------
    graph.add_edge(N_INTEGRATOR, END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph() -> CompiledStateGraph:
    """Return a process-wide cached compiled graph.

    Compilation is cheap but non-trivial (builds the channel/trigger table),
    and the API layer will call this on every request - so we memoize.
    """
    return build_graph()


__all__ = [
    "build_graph",
    "get_compiled_graph",
    "N_ALLOPATHY",
    "N_AYURVEDA",
    "N_HOMEOPATHY",
    "N_INTERACTION_CHECKER",
    "N_INTEGRATOR",
]
