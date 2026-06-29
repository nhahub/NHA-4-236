"""Tests for the patient-info feature: model, age-aware triage, wiring."""
from __future__ import annotations

import assistant
from patient import PatientInfo
from rag.retriever import RetrievedPassage
from safety.red_flag_detector import detect_red_flags


# --- PatientInfo model ---------------------------------------------------
def test_empty_patient_is_empty():
    assert PatientInfo().is_empty()
    assert PatientInfo().to_context() == ""


def test_to_context_lists_only_provided_fields():
    p = PatientInfo(age=65, sex="female", conditions="diabetes")
    ctx = p.to_context()
    assert "PATIENT CONTEXT" in ctx
    assert "Age: 65" in ctx
    assert "Sex: female" in ctx
    assert "Existing conditions: diabetes" in ctx
    assert "Medications" not in ctx  # not provided -> omitted


def test_retrieval_hints_and_signature():
    p = PatientInfo(age=65, sex="female", conditions="diabetes", pregnancy="no")
    hints = p.retrieval_hints()
    assert "65 year old" in hints and "female" in hints and "diabetes" in hints
    # Signature is hashable and changes with the data.
    assert hash(p.signature())
    assert p.signature() != PatientInfo(age=66).signature()


def test_is_pregnant_flag():
    assert PatientInfo(pregnancy="yes").is_pregnant is True
    assert PatientInfo(pregnancy="no").is_pregnant is False
    assert PatientInfo().is_pregnant is False


# --- Age / risk-aware triage --------------------------------------------
def test_infant_fever_is_flagged():
    r = detect_red_flags("my baby has a fever", use_llm=False, patient=PatientInfo(age=0))
    assert r.emergency is True
    assert r.reason == "infant fever"


def test_adult_fever_not_flagged():
    r = detect_red_flags("i have a fever", use_llm=False, patient=PatientInfo(age=30))
    assert r.emergency is False


def test_pregnancy_with_abdominal_pain_is_flagged():
    r = detect_red_flags(
        "i have bad abdominal pain", use_llm=False, patient=PatientInfo(pregnancy="yes")
    )
    assert r.emergency is True
    assert r.reason == "pregnancy complication"


def test_pregnancy_with_bleeding_is_flagged():
    r = detect_red_flags(
        "i noticed some bleeding", use_llm=False, patient=PatientInfo(pregnancy="yes")
    )
    assert r.emergency is True


def test_no_patient_means_no_age_rules():
    assert detect_red_flags("i have a fever", use_llm=False).emergency is False


# --- Wiring into prepare + cache ----------------------------------------
def _passage(score: float):
    return RetrievedPassage(
        id="p1", text="t", question="", title="Anemia", url="", source="NIH",
        qtype="info", score=score,
    )


def test_prepare_injects_patient_context(monkeypatch):
    monkeypatch.setattr(assistant.settings, "rerank_score_floor", -3.0)
    monkeypatch.setattr(assistant, "retrieve_context", lambda q, **kw: [_passage(8.0)])
    p = PatientInfo(age=40, conditions="diabetes")
    prep = assistant.prepare(
        "i have a cold and stomach pain", assistant.MODE_SYMPTOM,
        use_triage=False, patient=p,
    )
    assert prep.messages is not None
    user_msg = prep.messages[1]["content"]
    assert "PATIENT CONTEXT" in user_msg
    assert "diabetes" in user_msg


def test_cache_key_distinguishes_patients():
    k1 = assistant._cache_key("q", assistant.MODE_SYMPTOM, True, PatientInfo(age=30))
    k2 = assistant._cache_key("q", assistant.MODE_SYMPTOM, True, PatientInfo(age=70))
    k_none = assistant._cache_key("q", assistant.MODE_SYMPTOM, True, None)
    assert k1 != k2 != k_none and k1 != k_none
