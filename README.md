# VaidyaGraph: Agentic Integrative Diagnosis Platform

> A multi-agent AI system that convenes a virtual panel of Allopathic,
> Ayurvedic, and Homeopathic reasoners — then gates their combined advice
> through an adversarial pharmacovigilance check — to produce a single,
> patient-facing Preventive Care Plan for the Indian healthcare context.

VaidyaGraph is a **portfolio demonstration of production-grade agentic AI
patterns** applied to a high-signal domain: preventive healthcare in India,
where patients routinely combine modern medicine with Ayurvedic herbs and
homeopathic remedies without any layer between them cross-checking for
interactions.

The project showcases three non-trivial agentic-AI techniques:

- **Paradigm-isolated parallel agents** — three specialists reason
  independently, each constrained by a strict system prompt to stay within
  its own medical framework. No mushy "integrative" paraphrasing.
- **Adversarial safety gate** — a dedicated Pharmacovigilance agent runs
  *after* the specialists and its sole job is to find ways the combined
  plan could hurt the patient (drug-herb interactions, contraindications,
  redundant/additive therapies).
- **Grounded synthesis with explicit convergence/divergence** — a Senior
  Doctor integrator produces the final plan, forced by its prompt to
  surface where the three systems agree (high-confidence advice) and where
  they disagree (patient judgement required).

---

## Architecture

VaidyaGraph implements a **fan-out / safety-gate / fan-in** LangGraph
workflow:

```
                        ┌──────────┐
                        │   User   │
                        └────┬─────┘
                             │  symptoms + history
                             ▼
                        ┌──────────┐
                        │  START   │
                        └────┬─────┘
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌───────────┐  ┌───────────┐  ┌────────────┐
       │ Allopathy │  │ Ayurveda  │  │ Homeopathy │   parallel
       │    🩺     │  │    🌿     │  │     💊     │   super-step
       └─────┬─────┘  └─────┬─────┘  └──────┬─────┘
             │              │               │
             └──────────────┼───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Interaction Checker  🛡️    │   ← join #1
              │  (Safety Officer /          │     drug-herb,
              │   Pharmacovigilance)        │     contraindications,
              └──────────────┬──────────────┘     redundant therapies
                             ▼
                   ┌───────────────────┐
                   │  Integrator  🧭   │          ← join #2
                   │  (Senior Doctor)  │            (terminal)
                   └─────────┬─────────┘
                             ▼
                        ┌──────────┐
                        │   User   │   grouped JSON:
                        └──────────┘   { analysis, safety, plan }
```

### Why a dedicated Safety Gate?

Most "integrative medicine" LLM demos collapse the safety check into the
synthesis prompt. VaidyaGraph makes it a **distinct node** for two concrete
reasons:

1. **Single responsibility.** The Safety Officer runs at `temperature=0.1`
   with an adversarial prompt ("Be strict. A false alarm is cheap; a
   missed interaction is not."). It is not asked to be helpful or creative —
   it is asked to find failure modes. Folding this into the integrator
   dilutes both jobs.
2. **Structured output for the UI.** The safety verdict is parsed into a
   typed enum (`safe` / `caution` / `unsafe` / `unknown`) and surfaced as
   its own top-level field in the API response. The frontend renders it
   with a red/amber/green badge — safety is visible *before* the patient
   reads the plan, not buried inside it.

The checker's prompt enumerates the high-yield interaction classes in an
Indian context (anticoagulants ↔ Garlic/Ginkgo/Turmeric, antidiabetics ↔
Gymnema/Fenugreek, antihypertensives ↔ Arjuna/Sarpagandha,
immunosuppressants ↔ Ashwagandha/Guduchi, and more), so the model has
strong anchors rather than vague recall.

### Response contract

The backend returns a grouped response so clients can render each block
independently:

```jsonc
{
  "analysis": {
    "allopathy":  "## Allopathic Analysis ...",
    "ayurveda":   "## Ayurvedic Analysis ...",
    "homeopathy": "## Homeopathic Analysis ..."
  },
  "safety": {
    "severity": "caution",                 // safe | caution | unsafe | unknown
    "verdict":  "Caution advised.",
    "report":   "## Safety & Interaction Review ..."
  },
  "plan": {
    "integrative_report": "# Preventive Care Plan ...",
    "next_steps": ["...", "...", "..."]    // parsed from Section 6
  }
}
```

---

## Tech Stack

| Layer           | Technology                                               |
| --------------- | -------------------------------------------------------- |
| Agent runtime   | **LangGraph** (StateGraph, parallel super-steps)         |
| LLM client      | **LangChain** + `ChatOpenAI` pointed at **Chutes AI**    |
| Inference       | **Chutes AI** (OpenAI-compatible, e.g. Llama-3-70B)      |
| API             | **FastAPI** + Pydantic v2 + SSE streaming                |
| Frontend        | **Streamlit** (wide layout, 3-column panel, CSS polish)  |
| Vector store    | **ChromaDB** (reserved for RAG extensions)               |
| Language        | **Python 3.11+**                                         |

---

## Project Layout

```
VaidyaGraph/
├── agents/
│   ├── specialists.py      # 3 parallel nodes + Safety Officer
│   ├── integrator.py       # Senior Doctor synthesizer
│   └── graph.py            # StateGraph: fan-out / safety-gate / fan-in
├── api/
│   └── diagnose.py         # POST /diagnose  (+ /diagnose/stream SSE)
├── core/
│   ├── config.py           # Chutes settings loader
│   ├── llm.py              # Cached ChatOpenAI factory
│   ├── prompts.py          # 5 role-enforced templates
│   └── state.py            # AgentState TypedDict
├── tools/                  # (reserved for RAG retrievers)
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI entry point + CORS
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone

```bash
git clone <your-fork-url> VaidyaGraph
cd VaidyaGraph
```

### 2. Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
CHUTES_API_KEY=your_chutes_api_key_here
CHUTES_BASE_URL=https://llm.chutes.ai/v1
MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
```

`CHUTES_API_KEY` is the only required variable. The base URL and model name
have sensible defaults.

### 4. Run the backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's live:

```bash
curl http://localhost:8000/health
# -> {"status":"healthy"}
```

Interactive API docs are at <http://localhost:8000/docs>.

### 5. Run the frontend (in a second terminal)

```bash
streamlit run app.py
```

Opens on <http://localhost:8501>. The sidebar shows a live 🟢/🔴 badge
indicating whether the backend is reachable, plus a **Try an example**
dropdown with three scripted demo cases.

### Quick sanity test (backend only)

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
        "symptoms": "Recurring acidity after meals, poor sleep, mild headaches.",
        "history":  "32yo male, desk job, heavy tea and coffee intake."
      }'
```

---

## Demo Scenarios

The frontend's **Try an example** dropdown ships three curated cases that
exercise different parts of the graph:

| Case | Expected severity | What it demonstrates |
| --- | --- | --- |
| 🟢 Acidity & sleep issues | `safe` | Three systems converging on diet/stress — shows synthesis with agreement. |
| 🟡 Diabetic asking about herbs | `caution` | Safety gate catches metformin ↔ Gymnema/Fenugreek additive hypoglycaemia. |
| 🔴 Anticoagulated patient asking about Ashwagandha | `unsafe` | Safety gate blocks warfarin ↔ Ashwagandha/Turmeric bleeding-risk stack; integrator opens the plan with that warning. |

The third case is the single clearest demonstration of the platform's
value: three parallel agents, an adversarial safety layer catching a real
drug-herb interaction, and a senior-doctor integrator refusing to silently
bury the warning — all in one screen.

---

## Screenshots

> _Screenshots will be added here._

<!--
Suggested captures:
  1. Full landing view with the 🔴 warfarin example loaded.
  2. Safety section expanded showing the interaction report.
  3. Three-column specialist panel (zoomed).
  4. Unified Preventive Care Plan + Action Checklist.
-->

| | |
| --- | --- |
| _Landing / empty state_ | _Safety layer — `unsafe` verdict_ |
| _Three-column specialist panel_ | _Unified plan + action checklist_ |

---

## Disclaimer

**VaidyaGraph is an educational and demonstrative project. It is not a
medical device, not a diagnostic tool, and not a substitute for
professional medical advice, diagnosis, or treatment.**

- Every output is generated by a large language model and may contain
  errors, omissions, or outdated information.
- No prescription, dosage, or potency is ever recommended — only classes
  and categories, with explicit instructions to consult a licensed
  practitioner.
- **In a medical emergency, call your local emergency services
  immediately.** Do not rely on this system for any time-critical
  decision.
- The project has not been reviewed, validated, or certified by any
  medical body. No clinical claim is made or implied.

Use at your own risk. By running this code, you acknowledge these
limitations.

---

## License

This is a portfolio project. Add a license of your choice
(MIT recommended for portfolio visibility) before publishing.
