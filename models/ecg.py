"""ECG classifier — 1-D ResNet-18, 12-lead, 5 classes. Decision-support only.

Architecture reconstructed from the checkpoint (no training notebook was
provided): a stem ``Conv1d(12, 32, 15)`` + BN, four residual stages of two
BasicBlocks each (channels 32 -> 64 -> 128 -> 256, kernel 7, 1x1 downsample
shortcuts), global average pool, and a ``Linear(256, 5)`` head.

CLASS NAMES ARE ASSUMED. With no label mapping supplied, the 5 outputs are
labelled with the PTB-XL diagnostic superclasses (the most likely source for a
12-lead / 5-class model). If the model was trained on a different scheme, set
``CLASS_NAMES`` (or pass ``class_names=``) — the weights load fine either way,
only the labels would be wrong.

CLI (expects a (leads, samples) array saved as .npy):
    python -m models.ecg ecg.npy --checkpoint models/checkpoints/ecg.pth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "ecg.pth"
N_LEADS = 12
# Assumed PTB-XL diagnostic superclasses — CONFIRM/REPLACE with the real order.
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
_KERNEL = 7


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        pad = _KERNEL // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, _KERNEL, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, _KERNEL, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ECGResNet1d(nn.Module):
    def __init__(self, n_leads: int = N_LEADS, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
        )
        self.layer1 = self._stage(32, 32, stride=1)
        self.layer2 = self._stage(32, 64, stride=2)
        self.layer3 = self._stage(64, 128, stride=2)
        self.layer4 = self._stage(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, num_classes)

    @staticmethod
    def _stage(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock1d(in_ch, out_ch, stride=stride),
            BasicBlock1d(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.stem(x)), kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


@dataclass
class ECGPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    assumed_labels: bool = True
    experimental: bool = True  # experimental research model — never a diagnosis

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "assumed_labels": self.assumed_labels,
            "experimental": self.experimental,
        }


def _unwrap(ckpt) -> dict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    return ckpt


def load_model(checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str = "cpu",
               class_names: list[str] | None = None):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = _unwrap(ckpt)
    n_leads = int(state["stem.0.weight"].shape[1]) if "stem.0.weight" in state else N_LEADS
    n_classes = int(state["head.weight"].shape[0]) if "head.weight" in state else len(CLASS_NAMES)
    model = ECGResNet1d(n_leads=n_leads, num_classes=n_classes)
    model.load_state_dict(state)
    model.eval().to(device)
    classes = class_names or (CLASS_NAMES if n_classes == len(CLASS_NAMES)
                              else [f"class_{i}" for i in range(n_classes)])
    return model, classes


def preprocess(signal: np.ndarray, n_leads: int = N_LEADS) -> torch.Tensor:
    """Per-lead z-normalize a (leads, samples) array -> (1, leads, samples)."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (leads, samples), got shape {x.shape}")
    # Auto-orient: a 12-lead ECG has far more samples than leads, so the longer
    # axis is time. Transpose a (samples, leads) array to (leads, samples).
    if x.shape[0] > x.shape[1]:
        x = x.T
    if x.shape[0] < n_leads:
        x = np.pad(x, ((0, n_leads - x.shape[0]), (0, 0)))
    elif x.shape[0] > n_leads:
        x = x[:n_leads]
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(x).unsqueeze(0)


_CACHE: dict = {}


def predict(signal: np.ndarray, checkpoint: Path | str = DEFAULT_CHECKPOINT,
            device: str = "cpu") -> ECGPrediction:
    """Classify one 12-lead ECG (a (leads, samples) array)."""
    key = (str(checkpoint), device)
    if key not in _CACHE:
        _CACHE[key] = load_model(checkpoint, device)
    model, classes = _CACHE[key]

    x = preprocess(signal, n_leads=model.stem[0].in_channels).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return ECGPrediction(
        label=classes[idx],
        confidence=float(probs[idx]),
        probabilities={c: float(p) for c, p in zip(classes, probs)},
        assumed_labels=classes == CLASS_NAMES,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classify a 12-lead ECG (decision-support only).")
    ap.add_argument("signal", type=Path, help="(leads, samples) array saved as .npy")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    pred = predict(np.load(args.signal), args.checkpoint, args.device)
    print(f"Prediction: {pred.label}  ({pred.confidence * 100:.1f}%)")
    for cls, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<6} {p * 100:5.1f}%")
    if pred.assumed_labels:
        print("\n[!] Class names are ASSUMED (PTB-XL superclasses). Confirm the label order.")
    print("\nEXPERIMENTAL — decision-support only, not a diagnosis. "
          "Confirm with a cardiologist.")


if __name__ == "__main__":
    main()
