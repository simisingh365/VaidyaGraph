"""
VaidyaGraph - Streamlit frontend (live panel + follow-up chat).

Layout:

    ┌─────────────────────────────────────────────────────────────┐
    │  HERO (title, tagline, 3 example cards)                     │
    ├─────────────────────────────────────────────────────────────┤
    │  VERDICT STRIP (green / amber / red one-liner)              │
    ├─────────────────────────────────────────────────────────────┤
    │  🛡 SAFETY & INTERACTION REVIEW                             │
    ├──────────────────┬──────────────────┬───────────────────────┤
    │  🩺 Allopathy    │  🌿 Ayurveda     │  💊 Homeopathy        │
    ├──────────────────┴──────────────────┴───────────────────────┤
    │  🧩 PANEL CONSENSUS  (Convergence | Divergence)             │
    ├─────────────────────────────────────────────────────────────┤
    │  🧭 UNIFIED PREVENTIVE CARE PLAN + action checklist         │
    ├─────────────────────────────────────────────────────────────┤
    │  💬 ASK THE PANEL  (follow-up Q&A grounded in the plan)     │
    ├─────────────────────────────────────────────────────────────┤
    │  Run metadata footer                                        │
    └─────────────────────────────────────────────────────────────┘

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("VAIDYAGRAPH_API_URL", "http://localhost:8000")
DIAGNOSE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/diagnose"
STREAM_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/diagnose/stream"
ASK_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/ask"
ASK_RESET_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/ask/reset"
HEALTH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/health"
META_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/meta"

REQUEST_TIMEOUT_SEC = 600
HEALTHCHECK_TIMEOUT_SEC = 2
ASK_TIMEOUT_SEC = 120

EXAMPLE_CASES = {
    "🟢 Acidity & sleep issues": {
        "blurb": "Safe, convergent advice",
        "symptoms": (
            "Recurring acidity after meals, poor sleep for 3 weeks, mild "
            "morning headaches, occasional palpitations."
        ),
        "history": (
            "32yo male, desk job, heavy tea and coffee intake, no known "
            "allergies, father has hypertension."
        ),
    },
    "🟡 Diabetic asking about herbs": {
        "blurb": "Caution: additive hypoglycaemia risk",
        "symptoms": (
            "Type-2 diabetic on metformin, persistent fatigue, wants to "
            "add herbs for better sugar control."
        ),
        "history": (
            "45yo male, HbA1c 7.4, on metformin 1g BD for 2 years, "
            "vegetarian diet, sedentary."
        ),
    },
    "🔴 Anticoagulated + Ashwagandha": {
        "blurb": "Unsafe: bleeding-risk stack",
        "symptoms": (
            "On warfarin for atrial fibrillation, joint stiffness and "
            "fatigue, asking whether to start Ashwagandha and Turmeric "
            "supplements."
        ),
        "history": (
            "58yo male, INR stable at 2.3, hypertensive on amlodipine, "
            "no recent bleeding."
        ),
    },
}

NODE_META = {
    "allopathy": {"icon": "🩺", "label": "Allopathy",
                  "subtitle": "Modern evidence-based medicine",
                  "state_field": "allopathy_analysis"},
    "ayurveda": {"icon": "🌿", "label": "Ayurveda",
                 "subtitle": "Dosha · Agni · Dhatu framework",
                 "state_field": "ayurveda_analysis"},
    "homeopathy": {"icon": "💊", "label": "Homeopathy",
                   "subtitle": "Totality of symptoms · Miasms",
                   "state_field": "homeopathy_analysis"},
    "interaction_checker": {"icon": "🛡️", "label": "Safety Officer",
                            "subtitle": "Pharmacovigilance cross-check",
                            "state_field": "interaction_report"},
    "integrator": {"icon": "🧭", "label": "Senior Doctor",
                   "subtitle": "Unified synthesis",
                   "state_field": "integrative_report"},
}

SUGGESTED_QUESTIONS = [
    "Which parts of the plan do all three systems agree on?",
    "What was the single biggest safety concern?",
    "Explain the Ayurvedic reasoning in simple terms.",
    "What should I do this week in order of priority?",
]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VaidyaGraph",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS - overhauled design system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      /* Root tokens */
      :root {
        --vg-accent: #0d6efd;
        --vg-accent-soft: #eaf2ff;
        --vg-ink: #0f172a;
        --vg-muted: #64748b;
        --vg-line: #e5e7eb;
        --vg-surface: #ffffff;
        --vg-surface-soft: #f8fafc;
        --vg-safe: #10b981;
        --vg-caution: #f59e0b;
        --vg-unsafe: #ef4444;
      }

      .block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1280px; }

      /* Hero */
      .vg-hero {
          background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
          color: white;
          padding: 1.75rem 2rem;
          border-radius: 14px;
          margin-bottom: 1.25rem;
      }
      .vg-hero h1 {
          color: white !important;
          font-size: 2rem !important;
          margin: 0 0 0.25rem 0 !important;
          font-weight: 700 !important;
      }
      .vg-hero p {
          color: rgba(255,255,255,0.92) !important;
          margin: 0 !important;
          font-size: 1.05rem;
      }
      .vg-hero-tags {
          margin-top: 0.9rem;
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
      }
      .vg-hero-tag {
          background: rgba(255,255,255,0.18);
          color: white;
          padding: 0.25rem 0.7rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 500;
      }

      /* Section header */
      .vg-section-title {
          font-size: 1.35rem;
          font-weight: 700;
          color: var(--vg-ink);
          margin: 1.25rem 0 0.25rem 0;
          padding-left: 0.7rem;
          border-left: 4px solid var(--vg-accent);
      }
      .vg-section-caption {
          color: var(--vg-muted);
          font-size: 0.9rem;
          margin-bottom: 0.9rem;
      }

      /* Bordered containers used for cards */
      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 12px !important;
          border-color: var(--vg-line) !important;
      }

      /* Verdict strip */
      .vg-verdict {
          font-size: 1.1rem;
          font-weight: 600;
          padding: 1rem 1.2rem;
          border-radius: 12px;
          margin: 0.75rem 0 1rem 0;
          border: 1px solid transparent;
          box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      .vg-verdict-safe    { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
      .vg-verdict-caution { background: #fffbeb; color: #92400e; border-color: #fde68a; }
      .vg-verdict-unsafe  { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
      .vg-verdict-unknown { background: #f3f4f6; color: #374151; border-color: #e5e7eb; }
      .vg-verdict-sub {
          font-weight: 400; font-size: 0.9rem;
          margin-top: 0.35rem; opacity: 0.85;
      }

      /* Chips */
      .vg-chip {
          display: inline-block;
          padding: 0.18rem 0.65rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 600;
          margin-right: 0.4rem;
          margin-bottom: 0.3rem;
          border: 1px solid transparent;
      }
      .vg-chip-pending { background: #f3f4f6; color: #6b7280; border-color: #e5e7eb; }
      .vg-chip-running { background: #dbeafe; color: #1e40af; border-color: #bfdbfe; }
      .vg-chip-done    { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
      .vg-chip-error   { background: #fee2e2; color: #991b1b; border-color: #fecaca; }

      /* Hero example cards on empty state */
      .vg-example-card {
          background: var(--vg-surface);
          border: 1px solid var(--vg-line);
          border-radius: 12px;
          padding: 1rem 1.1rem;
          height: 100%;
          transition: all 0.15s ease;
      }
      .vg-example-card:hover {
          border-color: var(--vg-accent);
          box-shadow: 0 4px 10px rgba(13,110,253,0.08);
      }
      .vg-example-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.25rem; }
      .vg-example-blurb { color: var(--vg-muted); font-size: 0.85rem; margin-bottom: 0.5rem; }

      /* Run meta footer */
      .vg-meta {
          font-size: 0.78rem;
          color: var(--vg-muted);
          margin-top: 2rem;
          padding-top: 0.9rem;
          border-top: 1px solid var(--vg-line);
      }
      .vg-meta code {
          font-size: 0.78rem;
          background: var(--vg-surface-soft);
          padding: 1px 6px;
          border-radius: 4px;
      }

      /* Sidebar brand */
      .vg-brand {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 0.15rem;
      }

      /* Chat card */
      .vg-chat-hint {
          background: var(--vg-accent-soft);
          border: 1px solid #bfdbfe;
          border-radius: 10px;
          padding: 0.75rem 1rem;
          font-size: 0.88rem;
          color: #1e3a8a;
          margin-bottom: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="vg-section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="vg-section-caption">{caption}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("last_result", None),
    ("last_error", None),
    ("symptoms_seed", ""),
    ("history_seed", ""),
    ("last_elapsed", None),
    ("session_id", None),
    ("chat_history", []),          # list of {"role", "content"} dicts
    ("pending_question", None),    # a suggestion click sets this
    ("trigger_analyze", False),    # example-card click sets this
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def backend_is_up() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=HEALTHCHECK_TIMEOUT_SEC)
        return r.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def fetch_meta() -> Dict[str, Any]:
    try:
        r = requests.get(META_ENDPOINT, timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {}


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
def _iter_sse_events(resp: requests.Response) -> Iterator[Tuple[str, Dict[str, Any]]]:
    event = "message"
    data_lines: list[str] = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                try:
                    yield event, json.loads(payload)
                except json.JSONDecodeError:
                    pass
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


def stream_diagnose(symptoms: str, history: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    try:
        resp = requests.post(
            STREAM_ENDPOINT,
            json={"symptoms": symptoms, "history": history or ""},
            stream=True,
            timeout=REQUEST_TIMEOUT_SEC,
            headers={"Accept": "text/event-stream"},
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach the VaidyaGraph API at **{API_BASE_URL}**.\n\n"
            "Is the FastAPI server running? Start it with:\n\n"
            "```\npython main.py\n```"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error talking to the backend: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"API returned HTTP {resp.status_code}.\n\n**Details:** {detail}")
    with resp:
        yield from _iter_sse_events(resp)


def ask_panel(session_id: str, question: str) -> Dict[str, Any]:
    """Call /ask and return the parsed response. Raises RuntimeError on failure."""
    try:
        resp = requests.post(
            ASK_ENDPOINT,
            json={"session_id": session_id, "question": question},
            timeout=ASK_TIMEOUT_SEC,
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error talking to /ask: {exc}") from exc
    if resp.status_code == 404:
        raise RuntimeError(
            "This session has expired on the backend. Re-run the panel to "
            "start a fresh conversation."
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"/ask returned HTTP {resp.status_code}: {detail}")
    return resp.json()


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------
_SECTION_RE_TEMPLATE = (
    r"##\s*{index}\.?\s*Points\s+of\s+{name}.*?\n"
    r"(?P<body>.*?)(?=\n##\s|\Z)"
)


def _extract_section(plan_md: str, index: int, name: str) -> str:
    if not plan_md:
        return ""
    pattern = _SECTION_RE_TEMPLATE.format(index=index, name=name)
    m = re.search(pattern, plan_md, flags=re.IGNORECASE | re.DOTALL)
    return m.group("body").strip() if m else ""


def extract_convergence(md: str) -> str:
    return _extract_section(md, 2, "Convergence")


def extract_divergence(md: str) -> str:
    return _extract_section(md, 3, "Divergence")


def _count_bullets(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"^\s*[-*]\s+\S", text, flags=re.MULTILINE))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="vg-brand">🩺 VaidyaGraph</div>', unsafe_allow_html=True)
    st.caption("Agentic Preventive Diagnosis")

    st.divider()

    st.markdown("##### Patient Intake")

    symptoms = st.text_area(
        "Patient Symptoms",
        value=st.session_state.symptoms_seed,
        height=150,
        placeholder="Describe the current complaints in free text…",
        help="Required. Minimum ~3 characters.",
    )
    history = st.text_area(
        "Patient History",
        value=st.session_state.history_seed,
        height=120,
        placeholder="Age, occupation, meds, habits, family history…",
        help="Optional but improves the specialists' analyses.",
    )

    _sym_stripped = (symptoms or "").strip()
    analyze_clicked = st.button(
        "🔬 Run the Panel",
        type="primary",
        use_container_width=True,
        disabled=len(_sym_stripped) < 3,
    )

    st.divider()

    with st.expander("ℹ️ About VaidyaGraph", expanded=False):
        st.markdown(
            """
            A virtual panel of three paradigm-isolated AI specialists
            (Allopathy · Ayurveda · Homeopathy) plus an adversarial Safety
            Officer that cross-checks their plans for drug-herb
            interactions before a Senior Doctor synthesises a unified
            Preventive Care Plan.

            **Stack:** Python · FastAPI · LangGraph · Chutes AI · Streamlit.

            Portfolio / educational demo only. Not a medical device.
            """
        )

    if backend_is_up():
        st.success(f"🟢 Backend online · `{API_BASE_URL}`")
    else:
        st.error(f"🔴 Backend offline · `{API_BASE_URL}`")

    st.caption(
        "⚠️ Educational demo. Not a substitute for a licensed clinician. "
        "In emergencies, call local emergency services."
    )


# ---------------------------------------------------------------------------
# Header - always visible
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="vg-hero">
      <h1>VaidyaGraph</h1>
      <p>An agentic AI panel: Allopathy · Ayurveda · Homeopathy, synthesised into one unified plan.</p>
      <div class="vg-hero-tags">
        <span class="vg-hero-tag">LangGraph</span>
        <span class="vg-hero-tag">FastAPI</span>
        <span class="vg-hero-tag">Chutes AI</span>
        <span class="vg-hero-tag">Streamlit</span>
        <span class="vg-hero-tag">Fan-out · Safety Gate · Fan-in</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Decide whether to trigger analyze (sidebar button OR example card click)
# ---------------------------------------------------------------------------
analyze = bool(analyze_clicked) or bool(st.session_state.trigger_analyze)
if st.session_state.trigger_analyze:
    # Consume the one-shot flag.
    st.session_state.trigger_analyze = False
    # Pull the prefilled seed values the example-card click placed.
    symptoms = st.session_state.symptoms_seed
    history = st.session_state.history_seed


# ---------------------------------------------------------------------------
# Live panel rendering during a streaming run
# ---------------------------------------------------------------------------
def _chip(status: str, label: str) -> str:
    return f'<span class="vg-chip vg-chip-{status}">{label}</span>'


def _render_status_chips(statuses: Dict[str, str]) -> str:
    order = ["allopathy", "ayurveda", "homeopathy", "interaction_checker", "integrator"]
    parts = []
    for node in order:
        m = NODE_META[node]
        parts.append(_chip(statuses.get(node, "pending"), f"{m['icon']} {m['label']}"))
    return " ".join(parts)


if analyze:
    # Reset.
    st.session_state.last_result = None
    st.session_state.last_error = None
    st.session_state.session_id = None
    st.session_state.chat_history = []

    status_header = st.empty()
    chip_row = st.empty()
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Pre-claim the specialist cards, safety slot, plan slot.
    section_header(
        "👥 Specialist Panel",
        "Three paradigm-isolated specialists reasoning in parallel.",
    )
    specialist_cols = st.columns(3, gap="large")
    specialist_slots: Dict[str, Any] = {}
    for col, node in zip(specialist_cols, ["allopathy", "ayurveda", "homeopathy"]):
        with col:
            m = NODE_META[node]
            st.markdown(f"#### {m['icon']} {m['label']}")
            st.caption(m["subtitle"])
            specialist_slots[node] = st.empty()
            specialist_slots[node].info("⏳ Awaiting analysis…")

    section_header("🛡️ Safety & Interaction Review")
    safety_slot = st.empty()
    safety_slot.info("⏳ Safety Officer waiting for specialists…")

    section_header("🧭 Unified Preventive Care Plan")
    plan_slot = st.empty()
    plan_slot.info("⏳ Senior Doctor waiting for the safety review…")

    statuses = {k: "pending" for k in NODE_META.keys()}
    for n in ["allopathy", "ayurveda", "homeopathy"]:
        statuses[n] = "running"

    def _refresh(headline: str) -> None:
        status_header.markdown(f"**{headline}**")
        chip_row.markdown(_render_status_chips(statuses), unsafe_allow_html=True)

    _refresh("🧠 Convening the panel — three specialists running in parallel…")

    start = time.monotonic()
    accumulated: Dict[str, Any] = {}
    final_payload: Optional[Dict[str, Any]] = None

    try:
        for event, data in stream_diagnose(symptoms, history):
            if event == "node_update":
                node = data.get("node", "")
                node_state = data.get("state") or {}
                if isinstance(node_state, dict):
                    accumulated.update(node_state)

                m = NODE_META.get(node)
                if not m:
                    continue

                statuses[node] = "done"
                if node in {"allopathy", "ayurveda", "homeopathy"}:
                    text = node_state.get(m["state_field"]) or "_(no output)_"
                    with specialist_slots[node].container(border=True):
                        st.markdown(text)
                elif node == "interaction_checker":
                    statuses["integrator"] = "running"
                    safety_slot.success(
                        "✅ Safety Officer review complete — full details "
                        "appear below once the Senior Doctor finishes."
                    )

                if all(statuses[n] == "done" for n in ["allopathy", "ayurveda", "homeopathy"]):
                    if statuses["interaction_checker"] == "pending":
                        statuses["interaction_checker"] = "running"

                _refresh(f"🧠 Panel in progress · {time.monotonic() - start:0.0f}s elapsed")

            elif event == "final":
                final_payload = data
                for k in statuses:
                    statuses[k] = "done"
                st.session_state.last_elapsed = time.monotonic() - start
                _refresh(f"✅ Panel complete · {st.session_state.last_elapsed:0.0f}s total")

            elif event == "error":
                raise RuntimeError(data.get("message", "Stream error."))

    except RuntimeError as exc:
        st.session_state.last_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state.last_error = (
            f"Unexpected frontend error: `{type(exc).__name__}: {exc}`"
        )

    # Clear live placeholders - we're about to render the polished layout.
    status_header.empty()
    chip_row.empty()
    for s in specialist_slots.values():
        s.empty()
    safety_slot.empty()
    plan_slot.empty()

    if final_payload is not None:
        st.session_state.last_result = final_payload
        st.session_state.session_id = final_payload.get("session_id")


# ---------------------------------------------------------------------------
# Error rendering
# ---------------------------------------------------------------------------
if st.session_state.last_error:
    st.error("❌ Something went wrong")
    st.markdown(st.session_state.last_error)
    with st.expander("Troubleshooting checklist", expanded=False):
        st.markdown(
            f"""
            1. Backend up? `curl {HEALTH_ENDPOINT}` should return
               `{{"status":"healthy"}}`.
            2. `CHUTES_API_KEY` set in `.env`?
            3. Sidebar badge says "online"?
            4. Check backend terminal for a traceback.
            """
        )


result: Optional[Dict[str, Any]] = st.session_state.last_result


# ---------------------------------------------------------------------------
# Empty state - big example cards
# ---------------------------------------------------------------------------
if result is None and not st.session_state.last_error:
    st.markdown(
        '<div class="vg-section-title">Start with a scripted case</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="vg-section-caption">'
        "Each example is designed to exercise a different behaviour of the panel."
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(EXAMPLE_CASES), gap="medium")
    for col, (title, case) in zip(cols, EXAMPLE_CASES.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(case["blurb"])
                st.markdown(
                    f"<div style='color:var(--vg-muted);font-size:0.85rem;"
                    f"margin:0.5rem 0 0.75rem 0;'>"
                    f"{case['symptoms'][:120]}…</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Use this case →",
                    key=f"example_btn_{title}",
                    use_container_width=True,
                ):
                    st.session_state.symptoms_seed = case["symptoms"]
                    st.session_state.history_seed = case["history"]
                    st.session_state.trigger_analyze = True
                    st.rerun()

    st.markdown(
        '<div style="margin-top:1rem;color:var(--vg-muted);font-size:0.9rem;">'
        "👈 Or write your own symptoms in the sidebar and press "
        "<strong>Run the Panel</strong>."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if result is None:
    st.stop()


# ---------------------------------------------------------------------------
# Results - unpack
# ---------------------------------------------------------------------------
analysis = result.get("analysis", {}) or {}
safety = result.get("safety", {}) or {}
plan = result.get("plan", {}) or {}

severity = (safety.get("severity") or "unknown").lower()
verdict = safety.get("verdict") or ""
safety_report = safety.get("report") or ""
integrative_report = plan.get("integrative_report") or ""
next_steps = plan.get("next_steps") or []


# ---------------------------------------------------------------------------
# Verdict strip
# ---------------------------------------------------------------------------
_inter_match = re.search(
    r"###\s*Drug[- ]?Herb Interactions\s*\n(?P<body>.*?)(?=\n###|\Z)",
    safety_report,
    flags=re.IGNORECASE | re.DOTALL,
)
inter_bullets = _count_bullets(_inter_match.group("body")) if _inter_match else 0
convergence_text = extract_convergence(integrative_report)
divergence_text = extract_divergence(integrative_report)
conv_bullets = _count_bullets(convergence_text)
div_bullets = _count_bullets(divergence_text)

verdict_class = f"vg-verdict-{severity if severity in {'safe','caution','unsafe'} else 'unknown'}"
verdict_label = {
    "safe":    "🟢  Overall verdict: Safe to follow all three plans in parallel.",
    "caution": "🟡  Overall verdict: Caution — review interactions before combining.",
    "unsafe":  "🔴  Overall verdict: Do NOT combine as-is — material safety risk.",
    "unknown": "⚪  Overall verdict: Inconclusive — read the safety review below.",
}[severity if severity in {"safe", "caution", "unsafe"} else "unknown"]

sub_bits = []
if inter_bullets:
    sub_bits.append(f"{inter_bullets} interaction{'s' if inter_bullets != 1 else ''} flagged")
if conv_bullets:
    sub_bits.append(f"{conv_bullets} point{'s' if conv_bullets != 1 else ''} of convergence")
if div_bullets:
    sub_bits.append(f"{div_bullets} point{'s' if div_bullets != 1 else ''} of divergence")
sub_line = " · ".join(sub_bits) if sub_bits else "See sections below for details."

st.markdown(
    f'<div class="vg-verdict {verdict_class}">{verdict_label}'
    f'<div class="vg-verdict-sub">{sub_line}</div></div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Safety section
# ---------------------------------------------------------------------------
section_header(
    "🛡️ Safety & Interaction Review",
    "Adversarial pharmacovigilance check across all three plans.",
)

if severity == "safe":
    st.success(f"✅ **{verdict or 'No major interactions found.'}**  \nThe three plans appear safe to follow in parallel.")
elif severity == "caution":
    st.warning(f"⚠️ **{verdict or 'Caution advised.'}**  \nReview the interaction notes below before combining systems.")
elif severity == "unsafe":
    st.error(f"🚫 **{verdict or 'Do NOT combine as-is.'}**  \nThe Safety Officer flagged material risk.")
else:
    st.warning("ℹ️ Safety verdict could not be determined automatically.")

if safety_report:
    with st.expander("Full Safety Officer report", expanded=(severity in {"caution", "unsafe"})):
        st.markdown(safety_report)


# ---------------------------------------------------------------------------
# Specialist panel
# ---------------------------------------------------------------------------
section_header("👥 Specialist Panel", "Each specialist reasoned independently within their own paradigm.")

c_allo, c_ayur, c_homeo = st.columns(3, gap="large")
_PANEL = [
    (c_allo, "allopathy", analysis.get("allopathy")),
    (c_ayur, "ayurveda", analysis.get("ayurveda")),
    (c_homeo, "homeopathy", analysis.get("homeopathy")),
]
for col, node, body in _PANEL:
    m = NODE_META[node]
    with col:
        st.markdown(f"#### {m['icon']} {m['label']}")
        st.caption(m["subtitle"])
        with st.container(border=True):
            st.markdown(body or "_(analysis unavailable)_")


# ---------------------------------------------------------------------------
# Convergence / Divergence
# ---------------------------------------------------------------------------
if convergence_text or divergence_text:
    section_header("🧩 Panel Consensus", "Where the three systems agree vs. where they diverge.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### ✅ Points of Convergence")
        st.caption("All systems agree — strongest recommendations.")
        with st.container(border=True):
            st.markdown(convergence_text or "_(none identified)_")
    with c2:
        st.markdown("##### ⚖️ Points of Divergence")
        st.caption("Systems disagree — patient / clinician should weigh.")
        with st.container(border=True):
            st.markdown(divergence_text or "_(none identified)_")


# ---------------------------------------------------------------------------
# Unified plan
# ---------------------------------------------------------------------------
section_header("🧭 Unified Preventive Care Plan", "The Senior Doctor's synthesis.")

if integrative_report:
    with st.container(border=True):
        st.markdown(integrative_report)
else:
    st.info("_(The integrator did not produce a report for this run.)_")

if next_steps:
    st.markdown("##### ✅ Action Checklist (This Week)")
    for i, step in enumerate(next_steps, start=1):
        st.checkbox(step, key=f"next_step_{i}")


# ---------------------------------------------------------------------------
# Chat: follow-up Q&A
# ---------------------------------------------------------------------------
section_header("💬 Ask the Panel", "Follow-up questions grounded in the plan above.")

if not st.session_state.session_id:
    st.info(
        "No active session — follow-up chat is unavailable. "
        "Run the panel again to start a fresh session."
    )
else:
    st.markdown(
        '<div class="vg-chat-hint">'
        "💡 Ask about the plan. The panel will cite which specialist said what. "
        "For new symptoms, re-run the panel so the safety layer can re-evaluate."
        "</div>",
        unsafe_allow_html=True,
    )

    # Suggested-question chip row.
    q_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(q_cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(q, key=f"suggest_{q[:32]}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # History render.
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input.
    typed = st.chat_input("Ask a follow-up question about your plan…")
    question = typed or st.session_state.pending_question
    if st.session_state.pending_question:
        st.session_state.pending_question = None  # consume once

    if question:
        # Render the user turn immediately for responsiveness.
        with st.chat_message("user"):
            st.markdown(question)
        # Optimistically append - we replace with server-authoritative
        # history once /ask returns.
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Consulting the panel…"):
                try:
                    resp = ask_panel(st.session_state.session_id, question)
                    answer = resp.get("answer", "_(empty answer)_")
                    st.markdown(answer)
                    # Prefer server-side chat_history so eviction / reset
                    # on the backend stays authoritative.
                    returned_history = resp.get("chat_history") or []
                    if returned_history:
                        st.session_state.chat_history = [
                            {"role": t.get("role", "assistant"),
                             "content": t.get("content", "")}
                            for t in returned_history
                        ]
                    else:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
                    # Roll back the optimistic user turn.
                    if (st.session_state.chat_history
                            and st.session_state.chat_history[-1]["role"] == "user"):
                        st.session_state.chat_history.pop()

    # Clear-chat button beneath history, only after at least one turn.
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history", key="clear_chat"):
            try:
                requests.post(
                    ASK_RESET_ENDPOINT,
                    params={"session_id": st.session_state.session_id},
                    timeout=5,
                )
            except requests.RequestException:
                pass  # local clear still useful
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Raw JSON
# ---------------------------------------------------------------------------
with st.expander("🔧 Raw API response (JSON)", expanded=False):
    st.json(result)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
meta_info = fetch_meta()
model_name = meta_info.get("model") or "unknown"
provider = meta_info.get("provider") or "unknown"
elapsed = st.session_state.last_elapsed
elapsed_str = f"{elapsed:0.1f}s" if isinstance(elapsed, (int, float)) else "—"
sid = st.session_state.session_id or "—"

st.markdown(
    f'<div class="vg-meta">'
    f'⏱ Run: <code>{elapsed_str}</code> &nbsp;·&nbsp; '
    f'🧠 Model: <code>{model_name}</code> &nbsp;·&nbsp; '
    f'☁️ Provider: <code>{provider}</code> &nbsp;·&nbsp; '
    f'🔌 API: <code>{API_BASE_URL}</code> &nbsp;·&nbsp; '
    f'🆔 Session: <code>{sid}</code>'
    f'</div>',
    unsafe_allow_html=True,
)
