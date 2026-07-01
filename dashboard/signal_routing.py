"""Pure signal-routing helpers for the dashboard (no Streamlit dependency).

Decides which ``/analyze`` endpoint an uploaded study belongs to, from the file
type and — for signals — the array shape. Kept side-effect-free and import-light
so it is unit-testable without a running Streamlit app.
"""
from __future__ import annotations

import io

import numpy as np

# The channel axis is the shorter one (a recording has far more time samples than
# channels). Expected counts: a 12-lead ECG, or a ~19-64 channel EEG montage.
ECG_CHANNELS = 12
EEG_CHANNELS = 23
# Above this, it isn't a single ECG/EEG recording — e.g. a MIT-BIH per-beat
# dataset is shaped (many_beats, ~188), which must NOT be routed to a modality.
MAX_SIGNAL_CHANNELS = 64

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_SIGNAL_EXTS = (".npy", ".csv", ".txt")


def load_signal_array(uploaded):
    """Load a .npy or .csv/.txt upload into a 2-D float array (or None on failure).

    ``uploaded`` is any object with ``.name`` and ``.getvalue()`` (a Streamlit
    UploadedFile, or a stub in tests)."""
    name = (getattr(uploaded, "name", "") or "").lower()
    raw = uploaded.getvalue()
    try:
        if name.endswith(".npy"):
            return np.load(io.BytesIO(raw), allow_pickle=False)
        import pandas as pd  # lazy: only needed for CSV/TXT

        df = pd.read_csv(io.BytesIO(raw), header=None)
        # A text first row means there's a header — re-read using it.
        if df.iloc[0].map(lambda v: isinstance(v, str)).any():
            df = pd.read_csv(io.BytesIO(raw))
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df.to_numpy(dtype="float32")
    except Exception:
        return None


def infer_signal_modality(arr) -> str | None:
    """Route a ``(channels, samples)`` signal to the ECG or EEG endpoint.

    Returns None when ``arr`` isn't a single 2-D recording, or its channel count
    matches neither modality (too many channels), so the caller can decline with a
    helpful message instead of silently mislabelling a dataset as a study.
    """
    if arr is None or getattr(arr, "ndim", None) != 2:
        return None
    channels = int(min(arr.shape))
    if not (1 <= channels <= MAX_SIGNAL_CHANNELS):
        return None
    nearer_ecg = abs(channels - ECG_CHANNELS) <= abs(channels - EEG_CHANNELS)
    return "/analyze/ecg" if nearer_ecg else "/analyze/eeg"


def is_signal_file(uploaded) -> bool:
    """True if the upload's extension is a signal type (.npy/.csv/.txt)."""
    return (getattr(uploaded, "name", "") or "").lower().endswith(_SIGNAL_EXTS)


def scan_endpoint_for(uploaded) -> str | None:
    """Pick the ``/analyze`` endpoint for an uploaded study, or None if it can't be
    recognized (unsupported extension, or a signal whose shape matches neither
    ECG nor EEG). Images route to MRI; signals route by channel count."""
    name = (getattr(uploaded, "name", "") or "").lower()
    if name.endswith(_IMAGE_EXTS):
        return "/analyze/mri"
    if name.endswith(_SIGNAL_EXTS):
        return infer_signal_modality(load_signal_array(uploaded))
    return None
