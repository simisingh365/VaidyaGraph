# VaidyaGraph — Architecture

> Audience: technical reviewers, engineering leads, anyone evaluating this
> repo as a portfolio artefact.
> Scope: design decisions, not a line-by-line code walk. Pair this with
> `README.md` (product framing) and the source tree for implementation
> detail.

---

## 1. Overview

**VaidyaGraph** is an agentic AI platform for **integrative preventive
medicine in India**. It targets a real clinical pain point — what we call
the **Siloed Diagnosis problem**:

> Patients in India routinely combine allopathic prescriptions with
> Ayurvedic herbs and homeopathic remedies, often under three different
> practitioners who never talk to each other. No single layer in this
> chain checks whether the combined regimen is safe or coherent.

VaidyaGraph simulates that missing layer. It convenes a virtual panel of
three paradigm-isolated LLM specialists, subjects their combined output
to an adversarial pharmacovigilance review, synthesises a unified
preventive care plan, and supports grounded follow-up dialogue — all as
a single coherent service.

This document is about *how* the system is built, not *whether* it
should replace clinicians. It should not: the project is explicitly
scoped as an educational / portfolio artefact, and every prompt carries
safety rails enforcing that.

---

## 2. Core Architecture

### 2.1 The Topology: Fan-out / Safety-gate / Fan-in

```
                         ┌──────────┐
                         │  START   │
                         └────┬─────┘
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        ┌───────────┐  ┌───────────┐  ┌────────────┐
        │ Allopathy │  │ Ayurveda  │  │ Homeopathy │    parallel
        │    🩺     │  │    🌿     │  │     💊     │    super-step
        └─────┬─────┘  └─────┬─────┘  └──────┬─────┘
              └──────────────┼───────────────┘
                             ▼
               ┌─────────────────────────────┐
               │  Interaction Checker  🛡️    │    join #1
               │  (Safety Officer)           │    adversarial gate
               └──────────────┬──────────────┘
                              ▼
                    ┌───────────────────┐
                    │  Integrator  🧭   │    join #2 (terminal)
                    └─────────┬─────────┘
                              ▼
                           ┌─────┐
                           │ END │
                           └─────┘
```

Three design properties fall out of this topology:

1. **Real parallelism, not theatre.** The three specialists have
   independent incoming edges from `START` and write to disjoint keys on
   the shared state. LangGraph's pregel-style super-step scheduler
   schedules them concurrently on the event loop; they are not three
   sequential calls pretending to run in parallel.
2. **Joins are implicit barriers.** Multiple incoming edges on a node
   create a barrier — the Interaction Checker only fires after *all
   three* specialists have written, and the Integrator only fires after
   the safety gate is done. No explicit synchronisation primitive.
3. **Safety is a first-class node, not a prompt concern.** The
   pharmacovigilance check is a separate node with its own prompt and
   temperature, not an afterthought folded into the integrator.

### 2.2 LangGraph Implementation

**State.** A single `AgentState` TypedDict (`core/state.py`) declared
`total=False` so nodes can do partial updates. It carries the patient's
input, each specialist's markdown output, the safety report, the
integrator's plan, a parsed `next_steps` list, and the follow-up chat
history. Using `total=False` is the idiomatic LangGraph pattern — it
means each node returns `{"its_field": value}` and LangGraph merges it
into the running state without collision.

**Graph construction.** In `agents/graph.py`:

```python
graph = StateGraph(AgentState)

graph.add_node(N_ALLOPATHY, allopathy_agent_node)
graph.add_node(N_AYURVEDA, ayurveda_agent_node)
graph.add_node(N_HOMEOPATHY, homeopathy_agent_node)
graph.add_node(N_INTERACTION_CHECKER, interaction_checker_node)
graph.add_node(N_INTEGRATOR, integrator_agent_node)

# Fan-out
graph.add_edge(START, N_ALLOPATHY)
graph.add_edge(START, N_AYURVEDA)
graph.add_edge(START, N_HOMEOPATHY)

# Join #1 — all three specialists feed the safety gate
graph.add_edge(N_ALLOPATHY, N_INTERACTION_CHECKER)
graph.add_edge(N_AYURVEDA, N_INTERACTION_CHECKER)
graph.add_edge(N_HOMEOPATHY, N_INTERACTION_CHECKER)

# Join #2 — terminal synthesis
graph.add_edge(N_INTERACTION_CHECKER, N_INTEGRATOR)
graph.add_edge(N_INTEGRATOR, END)
```

The compiled graph is memoised behind `get_compiled_graph()` so repeated
API requests don't recompile the pregel channel table.

### 2.3 Paradigm Isolation (Constraint-Based Prompt Engineering)

The single most important prompt-engineering decision in the project is
that **each specialist is forbidden from referencing the others' frameworks**.
This is baked into the system prompt as an explicit prohibition, not an
aspiration:

- **Allopathy** — *"Do NOT reference Ayurveda, Homeopathy, Unani,
  Siddha, TCM, or any alternative system."* Output a differential,
  first-line investigations, drug classes (not brand names or dosages),
  and red flags for urgent referral.
- **Ayurveda** — *"Do NOT reference modern pharmacology or homeopathy."*
  Output Prakriti/Vikriti analysis of the three doshas, Agni/Ama notes,
  Ahara (diet) and Vihara (lifestyle) recommendations, and commonly
  available Indian herbs (Triphala, Ashwagandha, Guduchi, Tulsi).
- **Homeopathy** — *"Do NOT reference modern pharmacology or
  Ayurveda."* Output totality of symptoms (mentals, modalities,
  peculiars), miasmatic assessment (Psora / Sycosis / Syphilis), and
  keynote-indicated remedies.

The engineering insight: LLMs are more reliably constrained by
prohibitions than steered by aspirations. Without paradigm isolation,
three instances of the same model produce three paraphrases of the same
integrative mush, defeating the purpose of a panel.

Every prompt also inherits a shared `_SAFETY_PREAMBLE` that (a) declares
the system is a demo, (b) forbids scheduled-drug prescriptions and
specific dosages, and (c) hardcodes emergency escalation for red-flag
symptoms. One string, imported into every agent — safety policy edits
happen in one place.

### 2.4 The Adversarial Safety Officer

Runs as its own node (`interaction_checker_node` in
`agents/specialists.py`) at **temperature 0.1** — the one place where
creativity is actively harmful. Its prompt is deliberately adversarial:

> *"BE STRICT. Err on the side of flagging. A false alarm is cheap; a
> missed interaction is not."*

To anchor the model's knowledge, the prompt **enumerates high-yield
Indian-context interaction classes** rather than relying on vague recall:

| Allopathic drug class | Ayurvedic counterpart | Risk |
| --- | --- | --- |
| Anticoagulants / antiplatelets (warfarin, aspirin) | Garlic, Ginkgo, Ginger, Turmeric (high dose), Ashwagandha | Bleeding |
| Antidiabetics (sulfonylureas, insulin) | Gymnema, Fenugreek, Bitter gourd, Jamun | Additive hypoglycaemia |
| Antihypertensives | Arjuna, Sarpagandha | Additive hypotension |
| Immunosuppressants | Ashwagandha, Guduchi, Tulsi | Counteracted therapy |
| Sedatives / CNS depressants | Brahmi, Jatamansi, Ashwagandha | Additive sedation |
| Thyroid medication | Ashwagandha, Guggulu | Altered thyroid levels |

It also checks for **population contraindications** (pregnancy,
paediatric, renal/hepatic impairment), **redundant/additive therapies**
where two plans independently add e.g. a laxative, and explicitly
addresses homeopathy's special case — classical potencies are
pharmacologically inert, so the realistic harm is *delay* of
evidence-based treatment for a red-flag condition, which the prompt
tells the model to flag separately.

The Safety Officer's output ends with a **canonical verdict line** —
one of `"No major interactions found."`, `"Caution advised."`, or
`"Do NOT combine as-is."` — which the API layer parses into a typed enum
so the frontend can render a red/amber/green badge without
string-matching markdown.

### 2.5 The Integrator

Consumes all four upstream outputs plus the original patient input at
**temperature 0.4** (slightly higher than specialists — synthesis
benefits from rhetorical flexibility that diagnosis doesn't). Its prompt
enforces a **fixed six-section template**:

1. **Urgent Safety Check** — quote the Safety Officer's verdict; surface
   any red flags from the allopathic review.
2. **Points of Convergence (High Confidence)** — where all systems agree.
3. **Points of Divergence (Use Judgement)** — where they don't.
4. **Unified Lifestyle & Diet Plan**.
5. **Monitoring & When to See a Doctor**.
6. **Next Steps (This Week)** — an ordered, actionable list.

Fixing the template has two payoffs: the integrator's reasoning is
forced into an audit-friendly shape, and a simple regex in
`agents/integrator.py` (`_extract_next_steps`) can pull Section 6 out
of the markdown so the API returns `next_steps` as a structured list
without a second LLM call. The same pattern is used frontend-side to
extract Sections 2 and 3 for the dedicated Convergence / Divergence UI
cards.

An explicit rule in the prompt — *"Never silently drop a warning the
Safety Officer raised"* — guards against the integrator burying
interactions inside prose, which would be the worst possible failure
mode in a medical-adjacent demo.

---

## 3. Backend Design

### 3.1 Composition Root

`main.py` is intentionally thin: build the FastAPI app, configure CORS,
mount routers, expose meta endpoints. All request logic lives in
`api/diagnose.py` and `api/qa.py`; all agent logic lives in `agents/`;
all shared contracts live in `core/`. A reviewer opening `main.py`
should see the shape of the service in under 80 lines.

The `/meta` endpoint returns non-secret runtime config — the active
model name, provider, base URL, LLM timeout and retry policy. This is a
deliberate **trust signal**: technical reviewers can see exactly which
LLM powered a given result, and the frontend renders it in a
run-metadata footer. The API key is never exposed.

### 3.2 Grouped Response Contract

`POST /diagnose` returns a response grouped into three nested objects
plus a session token:

```json
{
  "session_id": "abc123...",
  "analysis": {
    "allopathy":  "## Allopathic Analysis ...",
    "ayurveda":   "## Ayurvedic Analysis ...",
    "homeopathy": "## Homeopathic Analysis ..."
  },
  "safety": {
    "severity": "caution",
    "verdict":  "Caution advised.",
    "report":   "## Safety & Interaction Review ..."
  },
  "plan": {
    "integrative_report": "# Preventive Care Plan ...",
    "next_steps": ["...", "..."]
  }
}
```

The grouping is load-bearing: it mirrors the three UI regions the
frontend renders (panel / safety / plan) and makes each group
independently consumable by future clients — a native mobile app could
ignore `plan.integrative_report` and render only `next_steps` as a
push-notification checklist.

`safety.severity` is a closed enum — `safe | caution | unsafe | unknown`
— parsed from the Safety Officer's verdict line by `_parse_safety()`.
Putting the classification rule in one place (the backend) means the UI
layer can't drift from it.

### 3.3 SSE Streaming

`POST /diagnose/stream` exposes the graph run as **Server-Sent Events**
so the frontend can light up each agent's card the moment that node
lands, instead of waiting for the whole panel. The implementation uses
LangGraph's native streaming:

```python
async for update in graph.astream(initial_state, stream_mode="updates"):
    for node_name, node_state in update.items():
        yield _frame("node_update", {"node": node_name, "state": node_state})

session_id = session_store.create(final_state)
yield _frame("final", build_final_payload(final_state, session_id))
```

`stream_mode="updates"` yields a dict keyed by the node that just
produced output, mapped to that node's partial state update — exactly
the granularity a live panel UI wants. The endpoint sets
`X-Accel-Buffering: no` so nginx-style reverse proxies don't buffer
frames.

The final SSE frame carries the same grouped response shape as the
blocking `/diagnose` endpoint, so clients have one rendering path for
the terminal state.

### 3.4 Frontend SSE Consumption

The Streamlit frontend (`app.py`) implements a minimal in-house SSE
parser over `requests.iter_lines` — ~60 lines, no third-party SSE
dependency. A dict of node statuses (`pending | running | done | error`)
drives a **status chip row** below the page title, and each
`node_update` frame (a) updates the relevant chip, (b) replaces the
corresponding placeholder card with real markdown. When the `final`
frame arrives the entire live layout is wiped and replaced by the
polished result layout (verdict strip, safety section, panel, consensus,
plan, chat, footer).

### 3.5 Session Management

The session store (`core/sessions.py`) holds the frozen `AgentState`
from each completed run so follow-up `/ask` calls can reach it. Three
design points:

- **Process-local by design.** An `OrderedDict` behind a `threading.Lock`,
  with LRU eviction (`_MAX_SESSIONS = 128`) and a soft TTL
  (`_SESSION_TTL_SEC = 7200` = 2 hours). Sessions are created by UUID
  inside `/diagnose` and consumed by `/ask`.
- **Honest about its limits.** The module docstring spells out that this
  is demo-grade — single-worker only, no persistence, no horizontal
  scale. Every comment that could mislead a reviewer ("this is
  production-ready!") is absent.
- **Documented upgrade path.** The abstraction is tight on purpose —
  `get` / `create` / `append_turn` / `clear_history` — so swapping
  implementations is a drop-in:

  | Deployment shape | Upgrade |
  | --- | --- |
  | Single box, real users | SQLite with a JSON state column |
  | Multi-worker or multi-box | Redis with TTL-bound keys |
  | Long-term semantic memory | Postgres + pgvector |

  Calling this out in code comments rather than a handwave in the README
  signals the author understands what production would require.

---

## 4. The Q&A Layer

### 4.1 The "Frozen State" Pattern

After a panel completes, its `AgentState` is *frozen* for the purpose of
Q&A. The main LangGraph workflow is **write-once**: it produces a plan
and never mutates it. The Q&A layer is **read-only** over that frozen
state — it cannot update the plan, it cannot call the specialists
again, it can only speak for what the panel already said.

This matters for two reasons:

1. **Auditability.** The patient saw plan X at 10:00. If the Q&A layer
   could silently update the plan at 10:05 based on follow-up
   information, no audit log would tell you which version the patient
   actually acted on. Freezing the state means the plan the patient saw
   is the plan the patient has, full stop.
2. **Safety discipline.** If the patient reports new symptoms in chat —
   *"I'm also getting chest pain now"* — the correct response is to
   invalidate the plan and recommend a re-run, not to incrementally
   patch it. Incremental patching would bypass the Safety Officer's
   cross-check over the new combined regimen.

Implementation: the Q&A agent (`agents/qa.py:answer_question`) is NOT a
node in the main graph. It's a standalone chain
(`QA_PROMPT | get_llm(...) | StrOutputParser()`) invoked directly by
`POST /ask`. The panel's state is passed as context via string
interpolation; chat history flows through `MessagesPlaceholder` so
multi-turn works without repeating the panel context in every turn.

### 4.2 Grounding Constraints

The `QA_SYSTEM_PROMPT` encodes six hard rules:

1. **Cite sources explicitly.** Every answer must attribute back to a
   specific panel section — *"The Ayurvedic analysis suggested…"*, *"The
   Safety Officer noted…"*, *"The unified plan recommends…"*. If no
   source covers the question, the agent must say so rather than
   fabricate.
2. **No new clinical claims.** If the patient asks for information the
   panel didn't produce (e.g. *"what should my HbA1c target be?"*), the
   agent says: *"The panel didn't address this. Please consult a
   clinician or re-run the panel with that question added to your
   history."*
3. **New symptoms = re-run.** Symptoms that weren't in the original
   input trigger an explicit instruction to re-run the panel, with
   red-flag symptoms (chest pain, stroke signs, severe bleeding,
   anaphylaxis, suicidal ideation) also escalating to emergency-care
   advice.
4. **Stay brief.** 2–4 sentence default; only expand on request. This
   is a constraint on the model's tendency to over-explain, not a UI
   decision.
5. **Stay in character.** Off-topic questions (weather, general
   chit-chat, legal/financial advice) are politely declined with a
   redirect.
6. **Safety Officer is authoritative.** On combination questions, the
   agent must consult the Safety Officer's report first and may never
   contradict it.

The net effect: the Q&A agent is *bounded by prompt*. An unbounded chat
in a medical-adjacent demo would be a liability; a narrowly bounded one
is a genuinely useful feature.

### 4.3 API Contract

```text
POST /diagnose         -> returns session_id in response body
POST /diagnose/stream  -> returns session_id inside the final SSE frame

POST /ask
  body:  { "session_id": "...", "question": "..." }
  reply: { "answer": "...", "chat_history": [ChatTurn, ...] }

POST /ask/reset?session_id=...
  reply: { "status": "reset", "session_id": "..." }
```

Chat history is authoritative on the backend — `/ask` returns the full
updated history, so a client can re-render without tracking state
itself. `404` on `/ask` means the session expired; the frontend
surfaces that with a "start a fresh panel" message rather than a
generic error.

---

## 5. Engineering Principles

Five principles govern every design decision in the codebase. Each one
is visible as a specific pattern in the source:

### 5.1 Centralised Contracts

Each shared concern has exactly one canonical location:

- **Agent state** → `core/state.py` (TypedDict)
- **LLM client** → `core/llm.py` (`get_llm()` factory with `@lru_cache`)
- **Prompts** → `core/prompts.py` (all templates; shared safety preamble)
- **Settings** → `core/config.py` (frozen dataclass, read once at startup)
- **Sessions** → `core/sessions.py` (singleton store)

Swapping the model, tightening safety policy, or moving sessions to
Redis each touches exactly one file.

### 5.2 Defensive Degradation

Every agent call goes through `_safe_invoke()` (or the equivalent in
the integrator / Q&A paths), which catches all exceptions and returns a
user-friendly placeholder string instead of raising. Consequence: if
Ayurveda times out, the Allopathy and Homeopathy analyses still reach
the Safety Officer, the Safety Officer still runs, the Integrator still
produces a plan, and the patient still gets a usable output with the
degradation explicitly noted. One failing specialist never aborts the
whole panel.

The same pattern applies frontend-side: all network errors are
normalised into `RuntimeError` with a human-readable message, so the
UI layer never branches on `requests` internals and never leaks a raw
traceback to the user.

### 5.3 Structured Outputs with UI-Ready Enums

Where the LLM produces free text the UI needs to switch on, the backend
parses it into a typed shape:

- **Safety severity** — regex over the Safety Officer's verdict line
  into `Literal["safe", "caution", "unsafe", "unknown"]`. The frontend
  drives its red/amber/green badge off this enum, not the markdown.
- **Next steps** — regex over Section 6 of the integrator's plan into a
  `list[str]`. The frontend renders a checklist without a second LLM
  call.
- **Convergence / Divergence** — parsed frontend-side from Sections 2
  and 3 of the plan into dedicated Panel Consensus cards.

Each piece of parsing lives in the layer closest to the concern, and
every parser has a graceful fallback if the model deviates from the
template.

### 5.4 Honest Observability

Nothing about the system hides behind an opaque layer:

- `/health` and `/meta` endpoints on the backend (model, provider,
  timeouts).
- The run-metadata footer on the frontend shows wall-clock time, model
  name, provider, API URL, and session ID on every run.
- The raw API response is always available in a collapsed expander at
  the bottom of the page.
- Backend logs one `INFO`-level line per node entry (`"Running
  allopathy_agent_node"` etc.), so a technical demo viewer can watch
  the terminal and see the graph fire in real time.

### 5.5 Prompt Engineering as Constraint Engineering

Every prompt in `core/prompts.py` spells out what the agent **must not**
do alongside what it should. Specialists are forbidden from
paradigm-crossing; the Safety Officer is told false alarms are cheaper
than missed interactions; the Q&A agent is told it may never contradict
the Safety Officer and must re-route new symptoms to a panel re-run.
LLMs respond more reliably to explicit prohibitions than to positive
framing; VaidyaGraph leans into that fact.

---

## 6. Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| Agent runtime | **LangGraph** | `StateGraph`, parallel super-steps, SSE-compatible streaming |
| LLM client | **LangChain** + `langchain-openai` | `ChatOpenAI` pointed at an OpenAI-compatible endpoint |
| Inference | **Chutes AI** | Community-hosted OpenAI-compatible inference; default `Qwen/Qwen3-32B-TEE` |
| API | **FastAPI** + **Pydantic v2** | Routers, grouped response schemas, SSE via `StreamingResponse` |
| Session store | In-memory `OrderedDict` (demo) | LRU + TTL; upgrade path → Redis / Postgres |
| Frontend | **Streamlit** ≥ 1.36 | Wide layout, `st.chat_message`, in-house SSE client, custom CSS |
| Vector store | **ChromaDB** | Reserved (unused) for future RAG extensions |
| Language | **Python 3.11+** | Type hints throughout, `TypedDict`-based state |

The entire system is ~2,500 lines across a dozen well-organised modules.
Nothing is pinned speculatively: ChromaDB sits in `requirements.txt`
unused because a half-implemented RAG layer would look worse than no
RAG, and the scaffolding is there so the next iteration can plug in a
curated Ayurvedic/allopathic corpus without re-plumbing.

---

## 7. Known Limitations & Upgrade Paths

Documented explicitly so reviewers see what a production version would
look like:

- **No authentication.** `/diagnose` and `/ask` are wide open; CORS
  wildcards everything. Real deployment needs per-user auth before either
  goes public.
- **Process-local sessions.** See §3.5. Redis is the drop-in upgrade.
- **No RAG grounding.** The agents reason from parametric knowledge
  only. Adding a retriever per paradigm (Ayurvedic formulary, WHO
  guidelines, homeopathic materia medica) is the single highest-impact
  v2 extension.
- **No clarification loop.** The graph acts on first-shot input. A
  pre-panel triage node that asks follow-up questions before running
  would materially improve output quality on vague inputs — but adds
  conditional routing complexity not yet needed for a demo.
- **Single-language.** Prompts and UI are English-only. Hindi /
  regional-language support is a natural extension for the stated
  Indian-market target.
- **No clinical validation.** The project has not been reviewed by any
  medical body and makes no clinical claim.

None of these are blockers for the project's stated scope (an engineering
portfolio artefact). Each is a clearly-bounded next step.

---

## 8. Related Documents

- `README.md` — product framing, setup, demo narrative.
- `docs/architecture.svg` — one-page architecture diagram (LinkedIn-ready).
- Source tree — `agents/`, `api/`, `core/`, `app.py`, `main.py`.
