"""Tests for dashboard signal-modality routing (pure, no Streamlit)."""
from __future__ import annotations

import io

import numpy as np

from dashboard.signal_routing import (
    infer_signal_modality,
    is_signal_file,
    scan_endpoint_for,
)


class _Upload:
    """Duck-typed stand-in for a Streamlit UploadedFile."""

    def __init__(self, name: str, data: bytes = b""):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# --- infer_signal_modality ----------------------------------------------
def test_ecg_channel_count_routes_to_ecg():
    assert infer_signal_modality(np.zeros((12, 5000))) == "/analyze/ecg"


def test_eeg_channel_count_routes_to_eeg():
    assert infer_signal_modality(np.zeros((23, 1280))) == "/analyze/eeg"


def test_transposed_signal_uses_shorter_axis():
    # (samples, channels) — channels is still the shorter axis.
    assert infer_signal_modality(np.zeros((5000, 12))) == "/analyze/ecg"


def test_mitbih_per_beat_matrix_is_rejected():
    """The reported bug: MIT-BIH (many_beats, ~188) must NOT route to EEG."""
    assert infer_signal_modality(np.zeros((21000, 188))) is None


def test_non_2d_or_empty_is_rejected():
    assert infer_signal_modality(np.zeros((1280,))) is None      # 1-D
    assert infer_signal_modality(None) is None


# --- scan_endpoint_for (type + shape) -----------------------------------
def test_image_routes_to_mri():
    assert scan_endpoint_for(_Upload("Te-no_4.jpg")) == "/analyze/mri"


def test_npy_ecg_routes_by_shape():
    up = _Upload("ecg.npy", _npy_bytes(np.zeros((12, 5000), dtype="float32")))
    assert scan_endpoint_for(up) == "/analyze/ecg"


def test_npy_dataset_shape_rejected():
    up = _Upload("mitbih_test.npy", _npy_bytes(np.zeros((21000, 188), dtype="float32")))
    assert scan_endpoint_for(up) is None
    assert is_signal_file(up) is True   # it IS a signal file — just the wrong shape


def test_unknown_extension_rejected():
    assert scan_endpoint_for(_Upload("notes.pdf")) is None
    assert is_signal_file(_Upload("notes.pdf")) is False
