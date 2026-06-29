"""Feature engineering for the DDXPlus dataset.

Converts a patient row (EVIDENCES string, AGE int, SEX str) into a fixed-length
numeric vector:
  - Multi-hot over all evidence codes from release_evidences.json  (~223 dims)
  - One-hot age bins: 0-17, 18-35, 36-55, 56-75, 75+              (5 dims)
  - Binary sex: M=1, F=0                                           (1 dim)
  Total: ~229 features per patient.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

_AGE_BINS = [0, 18, 36, 56, 76, 120]  # right-exclusive upper bounds


def load_evidence_vocab(evidences_path: str | Path) -> list[str]:
    """Return a sorted list of all evidence codes from release_evidences.json."""
    with open(evidences_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return sorted(vocab.keys())


def build_feature_names(all_codes: list[str], evidences: dict) -> list[str]:
    """Human-readable feature names for SHAP and reporting."""
    names = [evidences.get(c, {}).get("question_en", c) for c in all_codes]
    names += [f"age_bin_{i}" for i in range(5)]
    names += ["sex_male"]
    return names


def encode_patient(
    evidences_str: str,
    age: int,
    sex: str,
    code_to_idx: dict[str, int],
) -> np.ndarray:
    """Encode one patient row into a fixed-length float32 feature vector.

    Args:
        evidences_str: raw string from the EVIDENCES column, e.g. "['E_1', 'E_14_@_V_0']"
        age: integer age
        sex: "M" or "F"
        code_to_idx: mapping from evidence code to index in the multi-hot vector

    Returns:
        1-D float32 array of length len(code_to_idx) + 5 + 1
    """
    n_codes = len(code_to_idx)
    vec = np.zeros(n_codes, dtype=np.float32)

    try:
        codes = ast.literal_eval(evidences_str)
    except (ValueError, SyntaxError):
        codes = []

    for code in codes:
        # Strip value suffix: "E_67_@_V_0" → "E_67"
        base = code.split("_@_")[0]
        idx = code_to_idx.get(base)
        if idx is not None:
            vec[idx] = 1.0

    # One-hot age bin
    age_bin = np.zeros(5, dtype=np.float32)
    for i in range(len(_AGE_BINS) - 1):
        if _AGE_BINS[i] <= age < _AGE_BINS[i + 1]:
            age_bin[i] = 1.0
            break

    sex_feat = np.array([1.0 if sex == "M" else 0.0], dtype=np.float32)
    return np.concatenate([vec, age_bin, sex_feat])


def encode_dataframe(df, code_to_idx: dict[str, int]) -> np.ndarray:
    """Vectorise an entire DataFrame; returns shape (n_patients, n_features)."""
    rows = [
        encode_patient(
            row["EVIDENCES"],
            int(row["AGE"]),
            row["SEX"],
            code_to_idx,
        )
        for _, row in df.iterrows()
    ]
    return np.vstack(rows)
