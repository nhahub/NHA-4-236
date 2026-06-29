"""Streamlit demo UI for the medical RAG assistant.

Chat-style interface for symptom exploration and general medical questions.

Features:
  • Multi-turn conversation — prior turns are sent as context, so follow-up
    answers ("3 days", "it's worse at night") are understood.
  • Optional patient info that tailors the differential — editable mid-chat
    (changes apply to your next message) and saveable as a reusable profile.
  • Live token streaming with a Stop button.
  • Per-answer response timer.

The FastAPI backend is the single entry point: every answer (including the live
token stream) is served over HTTP via the SSE endpoints, so the assistant logic
runs in exactly one place. Start the API first:

    uvicorn api.main:app

Run:
    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Make the project root importable for the local helper modules (patient form
# model + profile storage), regardless of where Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict  # noqa: E402 — must follow the sys.path insert

import storage  # noqa: E402
from patient import PatientInfo  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
_HISTORY_TURNS = 6  # must match assistant._HISTORY_MAX so nothing is silently dropped

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺")


# --- Small helpers --------------------------------------------------------
def _clean(value):
    """Empty strings / selectbox blanks become None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        label = c["title"] or c["source"]
        if c["url"]:
            st.markdown(f"- [{c['index']}] [{label}]({c['url']}) — {c['source']}")
        else:
            st.markdown(f"- [{c['index']}] {label} — {c['source']}")


def render_ml_predictions(preds: list[dict]) -> None:
    """Show the XGBoost pre-ranking as a compact expander below the answer."""
    if not preds:
        return
    with st.expander("🤖 ML pre-ranking (XGBoost — structured symptom features)", expanded=False):
        st.caption(
            "Ranked by a classifier trained on 1 M patient records. "
            "Use as a starting signal only — the answer above is grounded in "
            "retrieved medical literature, not just this list."
        )
        for i, p in enumerate(preds, start=1):
            prob = p["probability"]
            bar_pct = int(prob * 100)
            st.markdown(
                f"**{i}. {p['disease']}** — {prob:.0%} confidence"
            )
            st.progress(bar_pct)


def history_for_request() -> list[dict]:
    """Prior turns (role/content only) to send as conversation context."""
    msgs = st.session_state.get("messages", [])
    return [{"role": m["role"], "content": m["content"]} for m in msgs][-_HISTORY_TURNS:]


# --- Patient form + profiles ---------------------------------------------
_SELECT_OPTIONS = {
    "p_sex": ["", "male", "female", "other"],
    "p_severity": ["", "mild", "moderate", "severe"],
    "p_smoke": ["", "never", "former", "current"],
    "p_preg": ["", "no", "yes", "unsure", "n/a"],
}


def _apply_profile(profile: PatientInfo) -> None:
    """Push a loaded profile into the form's widget state (before they render)."""
    text_map = {
        "p_age": "" if profile.age is None else str(profile.age),
        "p_duration": profile.duration or "",
        "p_cond": profile.conditions or "",
        "p_meds": profile.medications or "",
        "p_allergy": profile.allergies or "",
        "p_alcohol": profile.alcohol or "",
        "p_other": profile.other or "",
    }
    for key, val in text_map.items():
        st.session_state[key] = val
    select_map = {
        "p_sex": profile.sex,
        "p_severity": profile.severity,
        "p_smoke": profile.smoking,
        "p_preg": profile.pregnancy,
    }
    for key, val in select_map.items():
        opts = _SELECT_OPTIONS[key]
        st.session_state[key] = val if val in opts else ""


def patient_form() -> PatientInfo | None:
    """Render profile controls + the optional patient-info form."""
    with st.expander("Patient info (optional — tailors the differential)", expanded=False):
        # Save / load a reusable profile (JSON file under data/users/).
        pid = st.text_input("Profile name", key="profile_id", placeholder="e.g. alex")
        c_load, c_save = st.columns(2)
        if c_load.button("Load", use_container_width=True) and _clean(pid):
            prof = storage.load_profile(pid)
            if prof is None:
                st.warning(f"No saved profile '{pid}'.")
            else:
                _apply_profile(prof)
                st.success(f"Loaded '{pid}'.")
                st.rerun()

        age_raw = st.text_input("Age", key="p_age")
        sex = st.selectbox("Sex", _SELECT_OPTIONS["p_sex"], key="p_sex")
        duration = st.text_input("Symptom duration", placeholder="e.g. 3 days", key="p_duration")
        severity = st.selectbox("Severity", _SELECT_OPTIONS["p_severity"], key="p_severity")
        conditions = st.text_input("Existing conditions", placeholder="e.g. diabetes", key="p_cond")
        medications = st.text_input("Current medications", key="p_meds")
        allergies = st.text_input("Allergies", key="p_allergy")
        smoking = st.selectbox("Smoking", _SELECT_OPTIONS["p_smoke"], key="p_smoke")
        alcohol = st.text_input("Alcohol", key="p_alcohol")
        pregnancy = st.selectbox("Pregnancy", _SELECT_OPTIONS["p_preg"], key="p_preg")
        other = st.text_area("Other notes", key="p_other")
        st.caption("ℹ️ Edits apply to your **next** message — you can change these mid-chat.")

    age = None
    age_clean = _clean(age_raw)
    if age_clean:
        if age_clean.isdigit():
            age = int(age_clean)
        else:
            st.warning("Age must be a number (e.g. 45).")

    info = PatientInfo(
        age=age, sex=_clean(sex), duration=_clean(duration), severity=_clean(severity),
        conditions=_clean(conditions), medications=_clean(medications),
        allergies=_clean(allergies), smoking=_clean(smoking), alcohol=_clean(alcohol),
        pregnancy=_clean(pregnancy), other=_clean(other),
    )
    # Offer to save once there's something to save.
    if not info.is_empty() and _clean(st.session_state.get("profile_id")):
        if st.button("💾 Save profile", use_container_width=True):
            storage.save_profile(st.session_state["profile_id"], info)
            st.success(f"Saved '{st.session_state['profile_id']}'.")
    return None if info.is_empty() else info


# --- Backends -------------------------------------------------------------
def api_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def run_via_api_streaming(query, use_triage, patient, history) -> dict:
    """Stream an answer from the FastAPI SSE endpoint, rendering tokens live.

    Tokens are appended to a placeholder as they arrive; the final SSE event
    carries the metadata (emergency flag, triage, citations, ML pre-ranking).
    Pressing Stop triggers a Streamlit rerun that interrupts this loop and the
    ``with`` block closes the HTTP connection; the partial answer is recovered
    at the top of the script. The server appends the disclaimer and emits it as
    a trailing token, so nothing extra is needed here.
    """
    endpoint = "/symptom-check/stream" if patient is not None else "/ask/stream"
    payload = {"query": query, "use_triage": use_triage, "history": history}
    if patient is not None:  # only /symptom-check accepts patient
        payload["patient"] = asdict(patient)

    ph = st.empty()
    acc = ""
    meta: dict = {}
    with requests.post(
        f"{API_URL}{endpoint}", json=payload, stream=True, timeout=(10, 600)
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if st.session_state.get("stop_streaming"):
                break
            if not raw or not raw.startswith("data:"):
                continue
            event = json.loads(raw[5:].strip())
            if event.get("done"):
                meta = event
                break
            token = event.get("token", "")
            if token:
                acc += token
                st.session_state["partial_answer"] = acc  # survive a Stop rerun
                ph.markdown(acc + " ▌")

    (ph.error if meta.get("emergency") else ph.markdown)(acc)
    return {
        "answer": acc,
        "emergency": meta.get("emergency", False),
        "triage": meta.get("triage", {}),
        "citations": meta.get("citations", []),
        "ml_predictions": meta.get("ml_predictions", []),
        "structured_differential": meta.get("structured_differential"),
    }


# --- Imaging / signal models (MRI / EEG / ECG) ---------------------------
def _post_file(endpoint: str, uploaded) -> requests.Response:
    files = {"file": (uploaded.name, uploaded.getvalue(),
                      uploaded.type or "application/octet-stream")}
    return requests.post(f"{API_URL}{endpoint}", files=files, timeout=180)


def _render_analysis_result(resp: requests.Response, kind: str) -> None:
    if resp.status_code == 503:
        st.warning(resp.json().get("detail", "Model weights not available."))
        return
    if resp.status_code != 200:
        st.error(f"Failed ({resp.status_code}): {resp.text[:200]}")
        return
    data = resp.json()
    if kind == "eeg":
        prob = data["seizure_probability"]
        st.metric("Seizure probability", f"{prob:.1%}")
        st.progress(int(prob * 100))
        st.write("⚠️ seizure-like activity" if data["seizure"] else "No seizure detected")
    else:  # classification (MRI / ECG)
        st.write(f"**{data['label']}** — {data['confidence']:.1%}")
        for cls, p in sorted(data["probabilities"].items(), key=lambda kv: -kv[1]):
            st.progress(int(p * 100), text=f"{cls} · {p:.0%}")
    if data.get("note"):
        st.caption("ℹ️ " + data["note"])
    st.caption(data.get("disclaimer", ""))


def analysis_panel() -> None:
    """Sidebar uploaders that run the deep-learning screeners via the API."""
    with st.expander("🧠 Imaging & signals (MRI / EEG / ECG)", expanded=False):
        st.caption("Upload a study to run the screening models. Decision-support only.")
        mri_file = st.file_uploader("Brain MRI (jpg/png)", type=["jpg", "jpeg", "png"], key="mri_up")
        if mri_file and st.button("Analyze MRI", key="mri_btn", use_container_width=True):
            _render_analysis_result(_post_file("/analyze/mri", mri_file), "class")

        eeg_file = st.file_uploader("EEG window — .npy (23×samples)", type=["npy"], key="eeg_up")
        if eeg_file and st.button("Analyze EEG", key="eeg_btn", use_container_width=True):
            _render_analysis_result(_post_file("/analyze/eeg", eeg_file), "eeg")

        ecg_file = st.file_uploader("ECG — .npy (12×samples)", type=["npy"], key="ecg_up")
        if ecg_file and st.button("Analyze ECG", key="ecg_btn", use_container_width=True):
            _render_analysis_result(_post_file("/analyze/ecg", ecg_file), "class")


# --- UI -------------------------------------------------------------------
st.title("🩺 Medical RAG Assistant")
st.caption(
    "LLM + RAG over medical literature. **Educational use only — not a "
    "diagnosis.** In an emergency, call your local emergency number."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Recover a partial answer if a previous run was interrupted by Stop.
_partial = st.session_state.pop("partial_answer", None)
if _partial and (
    not st.session_state.messages
    or st.session_state.messages[-1]["role"] != "assistant"
):
    st.session_state.messages.append(
        {"role": "assistant", "content": _partial + "\n\n_(stopped)_", "citations": []}
    )
st.session_state["stop_streaming"] = False

health = api_health()

with st.sidebar:
    st.header("Settings")
    use_triage = st.checkbox("Emergency triage layer", value=True)

    patient = patient_form()

    analysis_panel()

    st.divider()
    if health:
        st.success("Backend: FastAPI (server)")
        st.write("Ollama:", "✅" if health.get("ollama") else "❌")
        st.write("Index:", "✅" if health.get("index_loaded") else "❌")
    else:
        st.error(
            "Backend unreachable. Start the API, then reload:\n\n"
            "`uvicorn api.main:app`"
        )

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("_last_patient_sig", None)
        st.rerun()

    # Offer to reset when the patient profile changes significantly
    # (different person = stale conversation context).
    if patient is not None:
        new_sig = patient.signature()
        last_sig = st.session_state.get("_last_patient_sig")
        if last_sig is not None and last_sig != new_sig and st.session_state.get("messages"):
            st.warning("Patient profile changed. Start a new conversation?")
            if st.button("↺ Reset for new profile", use_container_width=True):
                st.session_state.messages = []
                st.session_state["_last_patient_sig"] = new_sig
                st.rerun()
        else:
            st.session_state["_last_patient_sig"] = new_sig
    if health:
        if st.button("🔄 Clear answer cache", use_container_width=True):
            try:
                requests.post(f"{API_URL}/admin/clear-cache", timeout=5)
                st.success("Cache cleared.")
            except requests.RequestException:
                st.warning("Could not reach backend to clear cache.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))
        render_ml_predictions(msg.get("ml_predictions", []))

if prompt := st.chat_input("Describe symptoms or ask a medical question…"):
    history = history_for_request()  # turns BEFORE this one
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not health:
            st.error(
                "Cannot answer — the FastAPI backend is not reachable. "
                "Start it with `uvicorn api.main:app` and reload."
            )
            st.stop()
        # Stop button — interrupts generation (triggers a rerun that closes the
        # streaming connection; the partial answer is recovered on reload).
        st.button(
            "⏹ Stop",
            key="stop_btn",
            on_click=lambda: st.session_state.update(stop_streaming=True),
        )
        started = time.time()
        try:
            data = run_via_api_streaming(prompt, use_triage, patient, history)
            render_citations(data.get("citations", []))
            render_ml_predictions(data.get("ml_predictions", []))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            st.error(f"Request failed: {exc}")
            st.stop()
        st.caption(f"⏱ {time.time() - started:.1f}s")

    st.session_state.pop("partial_answer", None)  # completed cleanly
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "citations": data.get("citations", []),
            "ml_predictions": data.get("ml_predictions", []),
        }
    )
