"""Generate tiny synthetic EEG/ECG .npy windows for testing the upload + chat fusion.

These are NOT real recordings — just plausibly-shaped signals so you can exercise
the /analyze/* endpoints and the chat attach flow without hunting for data.

Run:
    python -m scripts.make_sample_signals            # writes to ./samples
    python -m scripts.make_sample_signals out_dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    outdir = Path(args[0]) if args else Path("samples")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # EEG: 23 channels x 1280 samples (5 s @ 256 Hz) — alpha-ish rhythm + noise.
    t = np.arange(1280) / 256.0
    eeg = (
        np.sin(2 * np.pi * 10 * t)[None, :] * rng.uniform(0.5, 1.5, (23, 1))
        + rng.normal(0, 0.3, (23, 1280))
    ).astype("float32")
    np.save(outdir / "eeg_sample.npy", eeg)

    # ECG: 12 leads x 1000 samples — periodic QRS-like spikes + baseline noise.
    spikes = np.zeros(1000, dtype="float32")
    spikes[::200] = 1.0
    ecg = (
        spikes[None, :] * rng.uniform(0.8, 1.2, (12, 1)) + rng.normal(0, 0.05, (12, 1000))
    ).astype("float32")
    np.save(outdir / "ecg_sample.npy", ecg)

    print(f"wrote {outdir/'eeg_sample.npy'}  (shape {eeg.shape})")
    print(f"wrote {outdir/'ecg_sample.npy'}  (shape {ecg.shape})")
    print("Attach these in the chat, or: curl.exe -F file=@samples/ecg_sample.npy "
          "http://localhost:8000/analyze/ecg")


if __name__ == "__main__":
    main()
