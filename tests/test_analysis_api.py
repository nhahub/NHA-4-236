"""Tests for the MRI/EEG/ECG upload endpoints.

Weights live in models/checkpoints/ (gitignored). When present, the endpoints
return a prediction; when absent (e.g. CI), they return 503. Tests handle both.
"""
from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from models import ecg, eeg, mri

client = TestClient(app)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_analyze_status_lists_all_three():
    body = client.get("/analyze/status").json()
    assert set(body) == {"mri", "eeg", "ecg"}
    assert all(isinstance(v, bool) for v in body.values())


def test_ecg_endpoint():
    arr = np.random.randn(12, 1000).astype("float32")
    r = client.post("/analyze/ecg", files={"file": ("ecg.npy", _npy_bytes(arr), "application/octet-stream")})
    if ecg.DEFAULT_CHECKPOINT.exists():
        assert r.status_code == 200
        body = r.json()
        assert {"label", "confidence", "probabilities", "disclaimer"} <= set(body)
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    else:
        assert r.status_code == 503


def test_eeg_endpoint():
    arr = np.random.randn(23, 1280).astype("float32")
    r = client.post("/analyze/eeg", files={"file": ("eeg.npy", _npy_bytes(arr), "application/octet-stream")})
    if eeg.DEFAULT_CHECKPOINT.exists():
        assert r.status_code == 200
        body = r.json()
        assert {"seizure_probability", "seizure", "disclaimer"} <= set(body)
    else:
        assert r.status_code == 503


def test_mri_endpoint():
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(buf, format="PNG")
    r = client.post("/analyze/mri", files={"file": ("scan.png", buf.getvalue(), "image/png")})
    if mri.DEFAULT_CHECKPOINT.exists():
        assert r.status_code == 200
        body = r.json()
        assert {"label", "confidence", "probabilities", "disclaimer"} <= set(body)
    else:
        assert r.status_code == 503


def test_ecg_rejects_non_npy():
    r = client.post("/analyze/ecg", files={"file": ("bad.txt", b"not an array", "text/plain")})
    # 503 if weights absent (checked first), else 400 for the unreadable upload.
    assert r.status_code in (400, 503)
