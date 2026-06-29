"""Imaging / signal inference endpoints (MRI, EEG, ECG).

File-upload endpoints that run the standalone ``models/`` networks:
  POST /analyze/mri  — multipart image (jpg/png)       -> tumour classification
  POST /analyze/eeg  — .npy/.csv (channels, samples)   -> seizure probability
  POST /analyze/ecg  — .npy/.csv (leads, samples)      -> rhythm/diagnostic class
  GET  /analyze/status — which model weights are present

All are decision-support only and load weights lazily on first use (the
``models`` predict helpers cache the loaded model per process).
"""
from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from models import ecg as _ecg
from models import eeg as _eeg
from models import mri as _mri

router = APIRouter(tags=["analysis"])

_DISCLAIMER = (
    "Decision-support only — not a diagnosis. These models are screening aids; "
    "confirm any finding with a qualified clinician."
)


def _require(checkpoint) -> None:
    if not checkpoint.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found at {checkpoint}. "
            "Place the trained .pth there (see models/README.md).",
        )


def _parse_csv(data: bytes) -> np.ndarray:
    """Parse a CSV/TXT signal into a 2-D float array, tolerating a header row
    and/or an index column (coerced to NaN and dropped)."""
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data), header=None)
    # If the first row is text, it's a header — re-read with it as the header.
    if df.iloc[0].map(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(io.BytesIO(data))
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.to_numpy(dtype="float32")


async def _read_signal(file: UploadFile) -> np.ndarray:
    """Read a (channels, samples) signal from an uploaded .npy or .csv file."""
    data = await file.read()
    name = (file.filename or "").lower()
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False) if name.endswith(".npy") else _parse_csv(data)
    except Exception:
        raise HTTPException(
            400, "Could not read signal — expected a .npy or .csv of shape (channels, samples)."
        )
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim != 2 or arr.size == 0 or 0 in arr.shape:
        raise HTTPException(
            400, f"Expected a non-empty 2-D (channels, samples) array, got shape {arr.shape}."
        )
    return arr


@router.get("/analyze/status")
async def analyze_status() -> dict:
    return {
        "mri": _mri.DEFAULT_CHECKPOINT.exists(),
        "eeg": _eeg.DEFAULT_CHECKPOINT.exists(),
        "ecg": _ecg.DEFAULT_CHECKPOINT.exists(),
    }


@router.post("/analyze/mri")
async def analyze_mri(file: UploadFile = File(...)) -> dict:
    _require(_mri.DEFAULT_CHECKPOINT)
    data = await file.read()
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(400, "Could not read image — expected a jpg/png brain MRI.")
    pred = await run_in_threadpool(_mri.predict, image)
    return {**pred.to_dict(), "disclaimer": _DISCLAIMER}


@router.post("/analyze/eeg")
async def analyze_eeg(file: UploadFile = File(...)) -> dict:
    _require(_eeg.DEFAULT_CHECKPOINT)
    arr = await _read_signal(file)
    pred = await run_in_threadpool(_eeg.predict, arr)
    return {
        **pred.to_dict(),
        "note": "Validation metrics pending re-training with a patient-level split.",
        "disclaimer": _DISCLAIMER,
    }


@router.post("/analyze/ecg")
async def analyze_ecg(file: UploadFile = File(...)) -> dict:
    _require(_ecg.DEFAULT_CHECKPOINT)
    arr = await _read_signal(file)
    pred = await run_in_threadpool(_ecg.predict, arr)
    note = None
    if pred.assumed_labels:
        note = "Class names are ASSUMED (PTB-XL superclasses); confirm the label order."
    out = {**pred.to_dict(), "disclaimer": _DISCLAIMER}
    if note:
        out["note"] = note
    return out
