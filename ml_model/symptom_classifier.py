"""Serve the free-text symptom classifier — the live ML pillar.

Drop-in for the old XGBoost ``predict``: same output contract, a ranked list of
``{"disease": str, "probability": float}``. But it takes the raw user text (not a
fragile regex-parsed feature vector), embeds it with the RAG retriever's encoder,
and classifies — so train == serve and the model is in-distribution at serve time.

    predict_text(query, top_k=5) -> list[dict]
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CLF_PATH = ARTIFACTS_DIR / "symptom_text_clf.joblib"
LABELS_PATH = ARTIFACTS_DIR / "symptom_text_labels.json"
META_PATH = ARTIFACTS_DIR / "symptom_text_meta.json"


def text_artifacts_available() -> bool:
    return all(p.exists() for p in (CLF_PATH, LABELS_PATH, META_PATH))


@functools.lru_cache(maxsize=1)
def _load():
    import joblib

    clf = joblib.load(CLF_PATH)
    labels: list[str] = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    meta: dict = json.loads(META_PATH.read_text(encoding="utf-8"))
    return clf, labels, meta


def _check_embedder_matches(meta: dict) -> None:
    """Refuse to serve if the query embedder differs from the training embedder —
    the classifier lives in that embedding space, so a mismatch is nonsense
    (mirrors the RAG index sidecar check)."""
    from config import settings

    built_with = meta.get("embedding_model")
    if built_with and built_with != settings.embedding_model:
        raise RuntimeError(
            f"Symptom classifier was trained with embedder '{built_with}' but "
            f"EMBEDDING_MODEL is '{settings.embedding_model}'. Retrain with "
            "`python -m ml_model.symptom_classifier_train` or restore the matching embedder."
        )


def predict_text(query: str, top_k: int = 5) -> list[dict]:
    """Return the top-k diagnoses for a free-text symptom description.

    Returns ``[{"disease": str, "probability": float}, ...]`` sorted descending.
    """
    clf, labels, meta = _load()
    _check_embedder_matches(meta)

    from rag.embeddings import get_embedding_model

    vec = get_embedding_model().encode_one(query).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[-top_k:][::-1]
    return [
        {"disease": str(labels[i]), "probability": round(float(proba[i]), 4)}
        for i in top_idx
    ]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Free-text symptom classifier.")
    ap.add_argument("text", help="symptom description in plain language")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    if not text_artifacts_available():
        raise SystemExit("No classifier artifacts. Run `python -m ml_model.symptom_classifier_train`.")
    for p in predict_text(args.text, args.top_k):
        print(f"  {p['probability'] * 100:5.1f}%  {p['disease']}")
    print("\nSupplementary signal only — not a diagnosis. The grounded answer is "
          "produced from retrieved literature.")


if __name__ == "__main__":
    main()
