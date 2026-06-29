"""Imaging / signal inference endpoints (MRI, EEG, ECG).

File-upload endpoints that run the standalone ``models/`` networks:
  POST /analyze/mri  — multipart image (jpg/png) -> tumour classification
  POST /analyze/eeg  — .npy (channels, samples)  -> seizure probability
  POST /analyze/ecg  — .npy (leads, samples)      -> rhythm/diagnostic class
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


async def _read_npy(file: UploadFile) -> np.ndarray:
    data = await file.read()
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception:
        raise HTTPException(400, "Could not read array — expected a NumPy .npy file.")
    if arr.ndim != 2:
        raise HTTPException(400, f"Expected a 2-D (channels, samples) array, got shape {arr.shape}.")
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
    arr = await _read_npy(file)
    pred = await run_in_threadpool(_eeg.predict, arr)
    return {
        **pred.to_dict(),
        "note": "Validation metrics pending re-training with a patient-level split.",
        "disclaimer": _DISCLAIMER,
    }


@router.post("/analyze/ecg")
async def analyze_ecg(file: UploadFile = File(...)) -> dict:
    _require(_ecg.DEFAULT_CHECKPOINT)
    arr = await _read_npy(file)
    pred = await run_in_threadpool(_ecg.predict, arr)
    note = None
    if pred.assumed_labels:
        note = "Class names are ASSUMED (PTB-XL superclasses); confirm the label order."
    out = {**pred.to_dict(), "disclaimer": _DISCLAIMER}
    if note:
        out["note"] = note
    return out
