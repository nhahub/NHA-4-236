"""Load saved model artifacts and return a ranked differential diagnosis list.

The public API is a single function:

    predict(feature_vector, top_k=5) -> list[dict]

Each dict has keys ``disease`` (str) and ``probability`` (float).
"""
from __future__ import annotations

import functools
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@functools.lru_cache(maxsize=1)
def _load_artifacts() -> tuple[XGBClassifier, object, list[str]]:
    """Load model, label encoder, and feature columns once; cache in-process."""
    model = XGBClassifier()
    model.load_model(str(ARTIFACTS_DIR / "xgb_model.json"))
    le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    all_codes: list[str] = joblib.load(ARTIFACTS_DIR / "feature_columns.pkl")
    return model, le, all_codes


def artifacts_available() -> bool:
    """Return True only when all three artifact files exist on disk."""
    return all(
        (ARTIFACTS_DIR / f).exists()
        for f in ("xgb_model.json", "label_encoder.pkl", "feature_columns.pkl")
    )


def feature_columns() -> list[str]:
    """Return the ordered evidence-code list that defines the feature vector."""
    _, _, all_codes = _load_artifacts()
    return all_codes


def predict(feature_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    """Return the top-k ranked diseases with their predicted probabilities.

    Args:
        feature_vector: 1-D float32 array produced by ml_model.features.encode_patient
        top_k: number of ranked predictions to return (default 5)

    Returns:
        List of dicts sorted by probability descending:
            [{"disease": "Influenza", "probability": 0.72}, ...]
    """
    model, le, _ = _load_artifacts()
    proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
    top_idx = np.argsort(proba)[-top_k:][::-1]
    return [
        {
            "disease": str(le.inverse_transform([i])[0]),
            "probability": round(float(proba[i]), 4),
        }
        for i in top_idx
    ]
