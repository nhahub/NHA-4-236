"""Unit tests for ml_model.legacy.features and ml_model.legacy.predict.

These tests run without the trained model artifacts: features.py is always
testable; predict.py tests are skipped when artifacts are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_model.legacy.features import encode_patient, load_evidence_vocab, build_feature_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidences_json(tmp_path: Path) -> Path:
    data = {
        "E_1": {"question_en": "Do you have a fever?"},
        "E_2": {"question_en": "Do you have a cough?"},
        "E_3": {"question_en": "Do you have fatigue?"},
    }
    p = tmp_path / "release_evidences.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

class TestLoadEvidenceVocab:
    def test_returns_sorted_keys(self, tmp_path):
        p = _make_evidences_json(tmp_path)
        vocab = load_evidence_vocab(p)
        assert vocab == sorted(vocab)

    def test_length(self, tmp_path):
        p = _make_evidences_json(tmp_path)
        assert len(load_evidence_vocab(p)) == 3


class TestBuildFeatureNames:
    def test_length(self, tmp_path):
        p = _make_evidences_json(tmp_path)
        evidences = json.loads(p.read_text())
        all_codes = load_evidence_vocab(p)
        names = build_feature_names(all_codes, evidences)
        # 3 codes + 5 age bins + 1 sex
        assert len(names) == 3 + 5 + 1

    def test_age_bin_names(self, tmp_path):
        p = _make_evidences_json(tmp_path)
        evidences = json.loads(p.read_text())
        names = build_feature_names(load_evidence_vocab(p), evidences)
        assert "age_bin_0" in names
        assert "sex_male" in names


class TestEncodePatient:
    def _code_to_idx(self):
        return {"E_1": 0, "E_2": 1, "E_3": 2}

    def test_output_length(self):
        cti = self._code_to_idx()
        vec = encode_patient("['E_1']", 25, "M", cti)
        # 3 codes + 5 age bins + 1 sex = 9
        assert len(vec) == 9

    def test_dtype(self):
        vec = encode_patient("['E_1']", 25, "M", self._code_to_idx())
        assert vec.dtype == np.float32

    def test_symptom_encoding(self):
        cti = self._code_to_idx()
        vec = encode_patient("['E_1', 'E_3']", 30, "F", cti)
        assert vec[0] == 1.0  # E_1
        assert vec[1] == 0.0  # E_2 absent
        assert vec[2] == 1.0  # E_3

    def test_sex_encoding(self):
        cti = self._code_to_idx()
        male_vec = encode_patient("[]", 30, "M", cti)
        female_vec = encode_patient("[]", 30, "F", cti)
        assert male_vec[-1] == 1.0
        assert female_vec[-1] == 0.0

    def test_age_bin_encoding(self):
        cti = self._code_to_idx()
        # Age 10 -> bin 0 (0-17)
        vec = encode_patient("[]", 10, "M", cti)
        age_bins = vec[3:8]
        assert age_bins[0] == 1.0
        assert age_bins[1:].sum() == 0.0

    def test_value_suffix_stripped(self):
        """E_1_@_V_0 should map to E_1."""
        cti = self._code_to_idx()
        vec = encode_patient("['E_1_@_V_0']", 30, "M", cti)
        assert vec[0] == 1.0

    def test_unknown_code_ignored(self):
        cti = self._code_to_idx()
        vec = encode_patient("['E_99']", 30, "M", cti)
        assert vec[:3].sum() == 0.0

    def test_malformed_evidences_string(self):
        cti = self._code_to_idx()
        vec = encode_patient("NOT_A_LIST", 30, "M", cti)
        assert vec[:3].sum() == 0.0  # no symptoms encoded, no crash


# ---------------------------------------------------------------------------
# predict.py (skipped when artifacts absent)
# ---------------------------------------------------------------------------

def _artifacts_present() -> bool:
    from ml_model.legacy.predict import artifacts_available
    return artifacts_available()


@pytest.mark.skipif(not _artifacts_present(), reason="Model artifacts not trained yet")
class TestPredict:
    def test_returns_list(self):
        from ml_model.legacy.predict import predict, feature_columns
        codes = feature_columns()
        n_features = len(codes) + 5 + 1
        dummy = np.zeros(n_features, dtype=np.float32)
        result = predict(dummy)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_result_keys(self):
        from ml_model.legacy.predict import predict, feature_columns
        codes = feature_columns()
        dummy = np.zeros(len(codes) + 6, dtype=np.float32)
        result = predict(dummy, top_k=3)
        assert len(result) == 3
        for r in result:
            assert "disease" in r
            assert "probability" in r

    def test_probabilities_sum_to_one_approx(self):
        from ml_model.legacy.predict import predict, feature_columns
        codes = feature_columns()
        dummy = np.zeros(len(codes) + 6, dtype=np.float32)
        # top-k probs don't sum to 1, but each must be in [0,1]
        for r in predict(dummy):
            assert 0.0 <= r["probability"] <= 1.0

    def test_sorted_descending(self):
        from ml_model.legacy.predict import predict, feature_columns
        codes = feature_columns()
        dummy = np.zeros(len(codes) + 6, dtype=np.float32)
        probs = [r["probability"] for r in predict(dummy)]
        assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# symptom_classifier.py — the free-text symptom classifier (mocked; no heavy model)
# ---------------------------------------------------------------------------

def test_predict_text_ranks_and_shapes(monkeypatch):
    import ml_model.symptom_classifier as tp
    from config import settings

    class FakeClf:
        def predict_proba(self, X):
            return np.array([[0.1, 0.7, 0.2]])

    monkeypatch.setattr(
        tp, "_load",
        lambda: (FakeClf(), ["a", "b", "c"], {"embedding_model": settings.embedding_model}),
    )
    import rag.embeddings as emb
    monkeypatch.setattr(
        emb, "get_embedding_model",
        lambda: type("E", (), {"encode_one": staticmethod(
            lambda t: np.zeros(4, dtype="float32"))})(),
    )
    out = tp.predict_text("burning urination", top_k=2)
    assert [p["disease"] for p in out] == ["b", "c"]      # sorted descending
    assert out[0]["probability"] == 0.7
    assert all(set(p) == {"disease", "probability"} for p in out)


def test_predict_text_refuses_embedder_mismatch(monkeypatch):
    import ml_model.symptom_classifier as tp

    class FakeClf:
        def predict_proba(self, X):
            return np.array([[1.0]])

    # meta records a different embedder than the configured one -> refuse.
    monkeypatch.setattr(
        tp, "_load", lambda: (FakeClf(), ["a"], {"embedding_model": "some-other-model"})
    )
    with pytest.raises(RuntimeError, match="trained with embedder"):
        tp.predict_text("x")


# ---------------------------------------------------------------------------
# Fix 6: np.random.seed() must NOT be called at module level in evaluate.py
# ---------------------------------------------------------------------------

def test_evaluate_import_does_not_seed_numpy():
    """Importing ml_model.legacy.evaluate must not side-effect numpy's global RNG."""
    import sys
    import numpy as np

    # Force a fresh import by clearing the module from sys.modules.
    sys.modules.pop("ml_model.legacy.evaluate", None)
    state_before = np.random.get_state()[1][0]
    import ml_model.legacy.evaluate  # noqa: F401
    state_after = np.random.get_state()[1][0]
    assert state_before == state_after, (
        "ml_model.legacy.evaluate must not call np.random.seed() at module level"
    )


# ---------------------------------------------------------------------------
# Fix 7: HF_REPO constant is used (not a hardcoded duplicate string)
# ---------------------------------------------------------------------------

def test_evaluate_uses_hf_repo_constant():
    import ast
    import inspect
    import ml_model.legacy.evaluate as ev
    src = inspect.getsource(ev)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "load_dataset":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and "ddxplus" in str(arg.value):
                        pytest.fail(
                            "load_dataset called with hardcoded string; use HF_REPO constant"
                        )


# ---------------------------------------------------------------------------
# symptom_parser tests (Fix 1 prerequisite: parser feeds ML predictions)
# ---------------------------------------------------------------------------

def test_parse_returns_none_features_when_artifacts_missing(monkeypatch):
    from ml_model.legacy import symptom_parser as sp
    monkeypatch.setattr(sp, "artifacts_available", lambda: False)
    result = sp.parse("I have fever and cough")
    assert result["features"] is None
    assert result["matched_count"] == 0
    assert result["matched_symptoms"] == []


def test_parse_returns_none_features_below_min_match(monkeypatch):
    from ml_model.legacy import symptom_parser as sp
    monkeypatch.setattr(sp, "artifacts_available", lambda: True)
    # Only 2 entities match — below the 3-symptom threshold.
    monkeypatch.setattr(sp, "_build_lookup", lambda: {"fever": "E_1", "cough": "E_2"})
    monkeypatch.setattr(sp, "_ner_entities", lambda t: ["fever", "cough"])
    result = sp.parse("I have fever and cough")
    assert result["features"] is None
    assert result["matched_count"] == 2


def test_parse_returns_features_when_enough_matches(monkeypatch):
    from ml_model.legacy import symptom_parser as sp
    codes = ["E_10", "E_11", "E_12"]
    lookup = {"fever": "E_10", "cough": "E_11", "fatigue": "E_12"}
    monkeypatch.setattr(sp, "artifacts_available", lambda: True)
    monkeypatch.setattr(sp, "_build_lookup", lambda: lookup)
    monkeypatch.setattr(sp, "_ner_entities", lambda t: ["fever", "cough", "fatigue"])
    monkeypatch.setattr(sp, "feature_columns", lambda: codes)
    result = sp.parse("I have fever, cough, and fatigue")
    assert result["features"] is not None
    assert result["matched_count"] == 3
    assert set(result["matched_symptoms"]) == {"E_10", "E_11", "E_12"}


def test_parse_no_duplicate_codes(monkeypatch):
    """The same evidence code must appear only once even if two entities map to it."""
    from ml_model.legacy import symptom_parser as sp
    lookup = {"fever": "E_10", "high temperature": "E_10", "cough": "E_11", "fatigue": "E_12"}
    monkeypatch.setattr(sp, "artifacts_available", lambda: True)
    monkeypatch.setattr(sp, "_build_lookup", lambda: lookup)
    monkeypatch.setattr(sp, "_ner_entities", lambda t: ["fever", "high temperature", "cough", "fatigue"])
    monkeypatch.setattr(sp, "feature_columns", lambda: ["E_10", "E_11", "E_12"])
    result = sp.parse("fever high temperature cough fatigue")
    assert result["matched_symptoms"].count("E_10") == 1


# ---------------------------------------------------------------------------
# _match_entity quality (synthetic lookup mirroring real DDXPlus structure:
# a present-symptom question, a characterization question, and history/family
# questions that must NOT be matched as present symptoms).
# ---------------------------------------------------------------------------

# Mirrors the real failure modes: E_147-style treatment history, E_25-style
# family history, attribute questions, and the correct present-symptom codes.
_SYNTH_LOOKUP = {
    "do you have a cough?": "E_cough",
    "are you feeling nauseous or do you feel like vomiting?": "E_nausea",
    "have you been treated in hospital recently for nausea, agitation?": "E_iv_history",
    "have any of your family members been diagnosed with cluster headaches?": "E_fam_headache",
    "do you feel slightly dizzy or lightheaded?": "E_dizzy",
    "do you have swelling in one or more areas of your body?": "E_swelling",
    "where is the swelling located?": "E_swelling_loc",
}


class TestMatchEntityQuality:
    def test_exact_present_symptom(self):
        from ml_model.legacy.symptom_parser import _match_entity
        assert _match_entity("do you have a cough?", _SYNTH_LOOKUP) == "E_cough"

    def test_nausea_maps_to_symptom_not_treatment_history(self):
        """'nausea' must hit the symptom (via synonym), never the IV-history Q."""
        from ml_model.legacy.symptom_parser import _match_entity
        assert _match_entity("nausea", _SYNTH_LOOKUP) == "E_nausea"

    def test_headache_returns_none_not_family_history(self):
        """No present-symptom headache question exists -> None beats a wrong code."""
        from ml_model.legacy.symptom_parser import _match_entity
        assert _match_entity("headache", _SYNTH_LOOKUP) is None

    def test_dizziness_synonym_matches_dizzy(self):
        from ml_model.legacy.symptom_parser import _match_entity
        assert _match_entity("dizziness", _SYNTH_LOOKUP) == "E_dizzy"

    def test_prefers_presence_over_characterization(self):
        """'swelling' -> 'Do you have swelling...' not 'Where is the swelling...'."""
        from ml_model.legacy.symptom_parser import _match_entity
        assert _match_entity("swelling", _SYNTH_LOOKUP) == "E_swelling"

    def test_meta_question_never_matched_alone(self):
        from ml_model.legacy.symptom_parser import _is_meta
        assert _is_meta("Have any of your family members been diagnosed with X?")
        assert not _is_meta("Do you have a cough?")

    def test_attribute_detection(self):
        from ml_model.legacy.symptom_parser import _is_attribute
        assert _is_attribute("Where is the swelling located?")
        assert _is_attribute("How severe is the itching?")
        assert not _is_attribute("Do you have swelling in your body?")

    def test_fuzzy_handles_misspelling(self):
        from ml_model.legacy.symptom_parser import _match_entity
        # 'vomitting' (typo) should still reach the vomiting/nausea symptom.
        assert _match_entity("vomitting", _SYNTH_LOOKUP) == "E_nausea"


@pytest.mark.parametrize("text,expected_age,expected_sex", [
    ("I am a 35 year old male", 35, "M"),
    ("She is a 28 yo female", 28, "F"),
    ("patient aged 50 years old, he", 50, "M"),
    ("no demographics here", None, None),
])
def test_extract_demographic(text, expected_age, expected_sex):
    from ml_model.legacy.symptom_parser import _extract_demographic
    age, sex = _extract_demographic(text)
    assert age == expected_age
    assert sex == expected_sex
