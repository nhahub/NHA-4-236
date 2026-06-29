"""End-to-end test suite for all bug fixes. Run with:
    python scripts/run_tests.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from safety.red_flag_detector import detect_red_flags
from safety.intent import classify_intent, CHITCHAT, QA, SYMPTOM
from rag.pipeline import retrieve_context
from patient import PatientInfo
from config import settings
import assistant as a

results: list[tuple[str, bool]] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    results.append((label, ok))
    print(("PASS" if ok else "FAIL") + "  " + label)
    if not ok:
        print("      expected=" + repr(expected) + "  got=" + repr(actual))


# ── FIX 1: Triage ─────────────────────────────────────────────────────────────
print("--- FIX 1: Triage (no false emergencies on mild/chronic headache) ---")
for q, exp in [
    ("consistent pain in my head and dizziness", False),
    ("i ve had a dull headache for a week", False),
    ("i feel dizzy when i stand up", False),
    ("worst headache of my life, came on suddenly", True),
    ("chest pain and cant breathe", True),
    ("i want to end my life", True),
]:
    r = detect_red_flags(q, use_llm=True)
    check(repr(q), r.emergency, exp)

# ── FIX 2: Reranker gate with patient hints ────────────────────────────────────
print()
print("--- FIX 2: Reranker gate passes despite patient hints ---")
patient_18f = PatientInfo(age=18, sex="female", duration="5 days", severity="mild")
hints = patient_18f.retrieval_hints()
for q in [
    "consistent pain in my head and dizziness",
    "i feel dizzy when i stand up",
    "i have had a dull headache for a week",
]:
    r = retrieve_context(q + " " + hints, rerank_query=q)
    passed = bool(r) and r[0].score >= settings.rerank_score_floor
    check(repr(q), passed, True)

# ── FIX 3: Intent routing ──────────────────────────────────────────────────────
print()
print("--- FIX 3: Intent routing ---")
for q, exp in [
    ("and the treatment?", QA),
    ("how do I manage this?", QA),
    ("what medication is used for bronchitis?", QA),
    ("what causes it?", QA),
    ("why does this happen?", QA),
    ("how long does it last?", QA),
    ("what is asthma?", QA),
    ("i also have a sore throat", SYMPTOM),
    ("i have chest pain and dizziness", SYMPTOM),
    ("hi", CHITCHAT),
    ("are you a doctor?", CHITCHAT),
]:
    check(repr(q), classify_intent(q, SYMPTOM), exp)

# ── FIX 4: Existing conditions filtered from passages ─────────────────────────
print()
print("--- FIX 4: Existing conditions excluded from retrieved passages ---")
patient_dm = PatientInfo(age=68, sex="male", conditions="type 2 diabetes", smoking="current")
q = "i feel very tired and thirsty all the time"
# Mirror assistant.prepare(): expand both top_k and top_n when filtering
pool = retrieve_context(
    q + " " + patient_dm.retrieval_hints(),
    rerank_query=q,
    top_k=settings.retrieval_top_k * 3,
    top_n=settings.rerank_top_n * 5,
)
# Call the actual function from assistant.py (not a local reimplementation)
filtered = a._filter_known_condition_passages(pool, patient_dm.conditions)[: settings.rerank_top_n]
print("      Pool size: " + str(len(pool)) + "  Filtered size: " + str(len(filtered)))
print("      Passages shown to LLM:")
for p in filtered:
    print("        " + repr(p.title) + "  score=" + str(round(p.score, 3)))
no_diabetes = bool(filtered) and not any("diabetes" in p.title.lower() for p in filtered)
check("diabetes NOT in filtered passages (and list non-empty)", no_diabetes, True)

# ── FIX 5: Mode removed — check sidebar constant removed ──────────────────────
print()
print("--- FIX 5: Mode toggle removed from dashboard ---")
app_src = Path("dashboard/streamlit_app.py").read_text(encoding="utf-8")
check("no Mode radio in source", 'st.radio("Mode"' in app_src, False)
check("patient_form() always called", "patient = patient_form()" in app_src, True)

# ── FIX 6: History turn alignment ─────────────────────────────────────────────
print()
print("--- FIX 6: History turn count aligned ---")
# Extract _HISTORY_TURNS from source without importing streamlit
src = Path("dashboard/streamlit_app.py").read_text(encoding="utf-8")
m = re.search(r"_HISTORY_TURNS\s*=\s*(\d+)", src)
ui_turns = int(m.group(1)) if m else -1
check("_HISTORY_TURNS == assistant._HISTORY_MAX", ui_turns, a._HISTORY_MAX)

# ── FIX 7: Clear-cache endpoint ───────────────────────────────────────────────
print()
print("--- FIX 7: Clear-cache endpoint in source ---")
api_src = Path("api/main.py").read_text(encoding="utf-8")
check("/admin/clear-cache defined", '"/admin/clear-cache"' in api_src, True)
check("_cache.clear() called", "_cache.clear()" in api_src, True)

# ── ROUND-2: Intent routing gaps ──────────────────────────────────────────────
print()
print("--- ROUND-2: Intent routing gaps ---")
from safety.intent import QA, SYMPTOM, CHITCHAT, classify_intent
for q, exp in [
    ("dosage for amoxicillin", QA),
    ("is this serious?", QA),
    ("should I be worried?", QA),
    ("is ibuprofen safe?", QA),
    ("can I take paracetamol?", QA),
    ("should I use antibiotics?", QA),
    ("what are the side effects of metformin?", QA),
    ("tell me more", QA),
    ("what about children?", QA),
    ("and if im pregnant?", QA),
    ("i have a sore throat", SYMPTOM),
    ("my knee hurts", SYMPTOM),
    ("hi", CHITCHAT),
]:
    check("intent: " + repr(q), classify_intent(q, SYMPTOM), exp)

# ── ROUND-2: Triage rule additions ────────────────────────────────────────────
print()
print("--- ROUND-2: Triage rules ---")
from safety.red_flag_detector import rule_based_check
for q, exp_hit in [
    ("i took too many pills", True),
    ("i swallowed too many tablets", True),
    ("i took an overdose", True),
]:
    r = rule_based_check(q)
    check("rule: " + repr(q), r is not None and r.emergency, exp_hit)

# ── ROUND-2: Condition filter plural/short ─────────────────────────────────────
print()
print("--- ROUND-2: Condition filter plural + short conditions ---")
from assistant import _filter_known_condition_passages

class _FP:
    def __init__(self, t):
        self.title = t
        self.score = 1.0

for cond, titles, expected_remaining in [
    ("migraines", ["Migraine", "Migraines and Headaches", "Fatigue"], {"Fatigue"}),
    ("asthma",    ["Asthma", "Asthmatic bronchitis", "Fatigue"],      {"Fatigue"}),
    ("flu",       ["Flu", "Influenza", "Fatigue"],                    {"Fatigue"}),
    ("asthma, hypertension", ["Asthma", "Hypertension", "Fatigue", "Diabetes"], {"Fatigue", "Diabetes"}),
]:
    passages = [_FP(t) for t in titles]
    filtered = _filter_known_condition_passages(passages, cond)
    remaining = {p.title for p in filtered}
    check("filter(" + repr(cond) + ")", remaining, expected_remaining)

# ── REGRESSION: Off-topic still blocked ───────────────────────────────────────
print()
print("--- REGRESSION: Off-topic queries still blocked ---")
for q in ["what is the capital of egypt", "who won the world cup"]:
    r = retrieve_context(q)
    passed = bool(r) and r[0].score >= settings.rerank_score_floor
    check(repr(q) + " blocked", passed, False)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
total = len(results)
passed_count = sum(1 for _, ok in results if ok)
print("=== " + str(passed_count) + "/" + str(total) + " passed ===")
if passed_count < total:
    print("FAILED:")
    for label, ok in results:
        if not ok:
            print("  " + label)

sys.exit(0 if passed_count == total else 1)
