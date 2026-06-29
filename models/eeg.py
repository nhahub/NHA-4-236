"""EEG seizure-detection model (CHB-MIT, 23 channels, binary). Decision-support only.

Outputs a seizure probability for a window of multi-channel EEG — a screening
signal to discuss with a clinician, never a diagnosis.

Two architectures are supported and auto-detected from the checkpoint:
  * "flatten" head — the original notebook model: fc1 = Linear(128*160, 128),
    which fixes the input to a 5 s @ 256 Hz window (1280 samples). This matches
    the existing ``best_eeg_model_focal.pth``.
  * "gap" head — the fixed/retrained model: AdaptiveAvgPool1d -> Linear(128, 128),
    far fewer params and length-flexible.

So this module loads today's weights as-is and the retrained weights later with
no code change.

CLI (expects a (channels, samples) array saved as .npy):
    python -m models.eeg window.npy --checkpoint models/checkpoints/eeg.pth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "eeg.pth"
N_CHANNELS = 23
SAMPLING_RATE = 256
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS  # 1280
_POOLED_LEN = WINDOW_SAMPLES // 8                 # 3x MaxPool1d(2) -> 160


class EEGNet(nn.Module):
    """1-D CNN matching both the original (flatten) and retrained (GAP) heads."""

    def __init__(self, n_channels: int = N_CHANNELS, head: str = "flatten",
                 flat_features: int = 128 * _POOLED_LEN):
        super().__init__()
        self.head = head
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        if head == "gap":
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 128)
        else:
            self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        if self.head == "gap":
            x = self.gap(x).squeeze(-1)
        else:
            x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


@dataclass
class EEGPrediction:
    seizure_probability: float
    seizure: bool
    threshold: float

    def to_dict(self) -> dict:
        return {
            "seizure_probability": self.seizure_probability,
            "seizure": self.seizure,
            "threshold": self.threshold,
        }


def _unwrap(ckpt) -> dict:
    """Return the tensor state_dict from a bare or wrapped checkpoint."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    return ckpt


def _infer_head(state: dict) -> tuple[str, int]:
    """Detect the head type from fc1's input width: 128 -> GAP, else flatten."""
    w = state.get("fc1.weight")
    if w is not None and w.shape[1] != 128:
        return "flatten", int(w.shape[1])
    return "gap", 128


def load_model(checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str = "cpu"):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = _unwrap(ckpt)
    head, flat = _infer_head(state)
    n_ch = int(state["conv1.weight"].shape[1]) if "conv1.weight" in state else N_CHANNELS
    model = EEGNet(n_channels=n_ch, head=head, flat_features=flat)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, head


def preprocess(window: np.ndarray, head: str = "flatten",
               n_channels: int = N_CHANNELS) -> torch.Tensor:
    """Per-channel z-normalize a (channels, samples) window -> (1, C, T) tensor.

    Channels are padded/trimmed to ``n_channels``. For the fixed-length flatten
    head, samples are padded/trimmed to one 5 s window.
    """
    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (channels, samples), got shape {x.shape}")

    # Auto-orient: EEG has far more time-samples than channels, so the longer
    # axis is time. Transpose a (samples, channels) array to (channels, samples).
    if x.shape[0] > x.shape[1]:
        x = x.T

    # Channel count
    if x.shape[0] < n_channels:
        x = np.pad(x, ((0, n_channels - x.shape[0]), (0, 0)))
    elif x.shape[0] > n_channels:
        x = x[:n_channels]

    # Length (only the flatten head requires a fixed length)
    if head == "flatten":
        if x.shape[1] < WINDOW_SAMPLES:
            x = np.pad(x, ((0, 0), (0, WINDOW_SAMPLES - x.shape[1])))
        elif x.shape[1] > WINDOW_SAMPLES:
            x = x[:, :WINDOW_SAMPLES]

    # Per-channel z-norm (matches the fixed training pipeline)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(x).unsqueeze(0)


_CACHE: dict = {}


def predict(window: np.ndarray, checkpoint: Path | str = DEFAULT_CHECKPOINT,
            device: str = "cpu", threshold: float = 0.5) -> EEGPrediction:
    """Seizure probability for one EEG window (a (channels, samples) array)."""
    key = (str(checkpoint), device)
    if key not in _CACHE:
        _CACHE[key] = load_model(checkpoint, device)
    model, head = _CACHE[key]

    x = preprocess(window, head=head, n_channels=model.conv1.in_channels).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return EEGPrediction(seizure_probability=prob, seizure=prob >= threshold,
                         threshold=threshold)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="EEG seizure screening (decision-support only).")
    ap.add_argument("window", type=Path, help="(channels, samples) array saved as .npy")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    pred = predict(np.load(args.window), args.checkpoint, args.device, args.threshold)
    print(f"Seizure probability: {pred.seizure_probability:.3f}  "
          f"-> {'SEIZURE' if pred.seizure else 'no seizure'} (thr {pred.threshold})")
    print("Decision-support only — not a diagnosis. Confirm with a neurologist.")


if __name__ == "__main__":
    main()
