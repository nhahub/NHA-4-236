"""Train the free-text symptom classifier — the live ML pillar of the hybrid.

Fixes the train/serve mismatch that got the DDXPlus XGBoost model demoted: that
model was trained on structured binary evidence but *served* a handful of
regex-parsed symptoms (everything else "absent"), so it was out-of-distribution
and confidently wrong. This model trains AND serves on the same thing — natural
-language symptom descriptions — embedded with the very S-PubMedBert encoder the
RAG retriever already uses. A lightweight, calibrated logistic-regression head
maps the embedding to a diagnosis. Because train == serve, its held-out metrics
actually predict live behaviour, and it shares the retriever's embedding backbone
(a genuinely *hybrid* ML+RAG design).

Dataset: gretelai/symptom_to_diagnosis (853 train / 212 test, 22 diagnoses, free
text) — downloaded via HuggingFace `datasets`, no Kaggle auth needed.

    python -m ml_model.symptom_classifier_train

Saves to ml_model/artifacts/: symptom_text_clf.joblib, symptom_text_labels.json,
symptom_text_meta.json (records the embedder + test metrics, so serving can refuse
on an embedder mismatch, mirroring the RAG index sidecar).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CLF_PATH = ARTIFACTS_DIR / "symptom_text_clf.joblib"
LABELS_PATH = ARTIFACTS_DIR / "symptom_text_labels.json"
META_PATH = ARTIFACTS_DIR / "symptom_text_meta.json"
DATASET = "gretelai/symptom_to_diagnosis"


def _topk_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def main() -> int:
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import joblib

    from config import settings
    from rag.embeddings import get_embedding_model

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)
    train, test = ds["train"], ds["test"]
    labels = sorted(set(train["output_text"]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    embedder = get_embedding_model()
    print(f"Embedding {train.num_rows} train + {test.num_rows} test texts "
          f"with {settings.embedding_model} ...")
    X_train = embedder.encode(list(train["input_text"]), show_progress=True)
    X_test = embedder.encode(list(test["input_text"]), show_progress=True)
    y_train = np.array([label_to_idx[o] for o in train["output_text"]])
    y_test = np.array([label_to_idx[o] for o in test["output_text"]])

    # Multinomial logistic regression on the (already L2-normalized) embeddings.
    # class_weight balances the mildly uneven classes; probabilities are used for
    # the top-k differential and the serve-time abstention threshold.
    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    metrics = {
        "top1_accuracy": round(_topk_accuracy(proba, y_test, 1), 4),
        "top3_accuracy": round(_topk_accuracy(proba, y_test, 3), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "n_train": int(train.num_rows),
        "n_test": int(test.num_rows),
        "n_classes": len(labels),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, CLF_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    META_PATH.write_text(
        json.dumps({"dataset": DATASET, "embedding_model": settings.embedding_model,
                    **metrics}, indent=2),
        encoding="utf-8",
    )

    print("\nHeld-out test metrics (train == serve distribution):")
    print(f"  top-1 accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"  top-3 accuracy: {metrics['top3_accuracy']:.3f}")
    print(f"  macro F1:       {metrics['macro_f1']:.3f}")
    print(f"Saved -> {CLF_PATH.name}, {LABELS_PATH.name}, {META_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
