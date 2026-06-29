"""Evaluate the trained XGBoost model on the held-out test split.

Usage:
    python -m ml_model.evaluate

Prints:
    - Top-1 / Top-3 / Top-5 accuracy
    - Macro F1
    - Mean Brier score
    - Per-class precision/recall for the 10 most common diseases
    - Confusion matrix saved to ml_model/artifacts/confusion_matrix.png
    - SHAP top-20 feature importances saved to ml_model/artifacts/shap_importance.png
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    brier_score_loss,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from ml_model.features import encode_dataframe, load_evidence_vocab, build_feature_names

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "ddxplus"
EVIDENCES_PATH = DATA_DIR / "release_evidences.json"
HF_REPO = "aai530-group6/ddxplus"


def _ensure_vocab_files() -> None:
    import shutil
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EVIDENCES_PATH.exists():
        src = hf_hub_download(repo_id=HF_REPO, filename="release_evidences.json", repo_type="dataset")
        shutil.copy(src, EVIDENCES_PATH)

RANDOM_STATE = 42


def evaluate() -> None:
    np.random.seed(RANDOM_STATE)
    _ensure_vocab_files()
    # Load artifacts
    model = XGBClassifier()
    model.load_model(str(ARTIFACTS_DIR / "xgb_model.json"))
    le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    all_codes: list[str] = joblib.load(ARTIFACTS_DIR / "feature_columns.pkl")
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    # Load test split
    print("Loading test split …")
    ds = load_dataset(HF_REPO)
    test_df = pd.DataFrame(ds["test"])
    train_df = pd.DataFrame(ds["train"])

    print("Encoding test features …")
    X_test = encode_dataframe(test_df, code_to_idx)
    y_test = le.transform(test_df["PATHOLOGY"])
    n_classes = len(le.classes_)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Core metrics
    top1 = top_k_accuracy_score(y_test, y_pred_proba, k=1)
    top3 = top_k_accuracy_score(y_test, y_pred_proba, k=3)
    top5 = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    print(f"\nTop-1 Accuracy : {top1:.4f}")
    print(f"Top-3 Accuracy : {top3:.4f}")
    print(f"Top-5 Accuracy : {top5:.4f}")

    # Macro F1
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    print(f"Macro F1       : {report['macro avg']['f1-score']:.4f}")

    # Brier score (averaged across classes)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    brier = np.mean(
        [brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i]) for i in range(n_classes)]
    )
    print(f"Mean Brier Score: {brier:.4f}  (lower = better calibrated)")

    # Per-class report for top 10 diseases
    top10 = train_df["PATHOLOGY"].value_counts().head(10).index.tolist()
    report_df = pd.DataFrame(report).T
    print("\nPer-class metrics (top-10 most common diseases):")
    print(report_df.loc[top10, ["precision", "recall", "f1-score"]].to_string())

    # Confusion matrix (top 15 diseases)
    top15 = train_df["PATHOLOGY"].value_counts().head(15).index.tolist()
    top15_idx = [list(le.classes_).index(d) for d in top15]
    mask = np.isin(y_test, top15_idx)
    cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top15_idx)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=top15, yticklabels=top15,
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix — Top 15 Diseases (Test Set)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {cm_path}")

    # SHAP feature importances (sample for speed)
    import json as _json
    with open(EVIDENCES_PATH, encoding="utf-8") as f:
        evidences = _json.load(f)
    feature_names = build_feature_names(all_codes, evidences)

    sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]

    print("\nComputing SHAP values (sample n=500) …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # shape: (n, features, n_classes)

    mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    top20_idx = np.argsort(mean_shap)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in top20_idx], mean_shap[top20_idx])
    ax.set_title("Top 20 Most Influential Symptoms (Global SHAP)")
    ax.set_xlabel("Mean |SHAP value|")
    ax.invert_yaxis()
    plt.tight_layout()
    shap_path = ARTIFACTS_DIR / "shap_importance.png"
    plt.savefig(shap_path, dpi=150)
    plt.close()
    print(f"SHAP importance chart saved to {shap_path}")


if __name__ == "__main__":
    evaluate()
