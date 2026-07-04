"""Train an XGBoost multi-class classifier on DDXPlus.

Usage:
    python -m ml_model.train

Outputs (written to ml_model/artifacts/):
    xgb_model.json        — XGBoost model weights
    label_encoder.pkl     — sklearn LabelEncoder (int <-> disease name)
    feature_columns.pkl   — ordered list of evidence codes (must match inference)
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ml_model.legacy.features import encode_dataframe, load_evidence_vocab

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ddxplus"
EVIDENCES_PATH = DATA_DIR / "release_evidences.json"
CONDITIONS_PATH = DATA_DIR / "release_conditions.json"

HF_REPO = "aai530-group6/ddxplus"
RANDOM_STATE = 42


def _ensure_vocab_files() -> None:
    """Download the JSON vocabulary files from HuggingFace if not present locally."""
    import shutil
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, dest in [
        ("release_evidences.json", EVIDENCES_PATH),
        ("release_conditions.json", CONDITIONS_PATH),
    ]:
        if not dest.exists():
            print(f"Downloading {filename} from HuggingFace …")
            src = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy(src, dest)
            print(f"  saved to {dest}")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load DDXPlus splits from HuggingFace (cached locally after first run)."""
    print("Loading DDXPlus dataset …")
    ds = load_dataset(HF_REPO)
    train_df = pd.DataFrame(ds["train"])
    val_df = pd.DataFrame(ds["validate"])
    test_df = pd.DataFrame(ds["test"])
    print(
        f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights to handle class imbalance."""
    counts = Counter(y.tolist())
    total = len(y)
    n_classes = len(counts)
    weight_map = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    return np.array([weight_map[label] for label in y], dtype=np.float32)


def train() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_vocab_files()

    train_df, val_df, _ = _load_data()

    # Feature vocabulary
    all_codes = load_evidence_vocab(EVIDENCES_PATH)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    print(f"Evidence vocabulary: {len(all_codes)} codes")

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["PATHOLOGY"])
    y_val = le.transform(val_df["PATHOLOGY"])
    n_classes = len(le.classes_)
    print(f"Disease classes: {n_classes}")

    # Features
    print("Encoding training features …")
    t0 = time.time()
    X_train = encode_dataframe(train_df, code_to_idx)
    X_val = encode_dataframe(val_df, code_to_idx)
    print(f"  done in {time.time() - t0:.1f}s  shape={X_train.shape}")

    sample_weights = _sample_weights(y_train)

    # Train
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training XGBoost …")
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")

    # Save artifacts
    model.save_model(str(ARTIFACTS_DIR / "xgb_model.json"))
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.pkl")
    joblib.dump(all_codes, ARTIFACTS_DIR / "feature_columns.pkl")
    print(f"Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train()
