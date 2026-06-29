"""Brain-tumor MRI classifier — EfficientNet-B0, 4 classes.

Decision-support only: outputs are "findings to discuss with a clinician", never
a definitive diagnosis.

Architecture and preprocessing mirror the training notebook (`mri_last.ipynb`):
EfficientNet-B0 backbone with a custom 2-layer head, 224x224 RGB input, ImageNet
normalization, and a black-border crop. Loads either a bare ``state_dict`` (the
original `tumor_efficientnet_v3.pth`) or the newer metadata dict
(`{"model_state_dict", "class_names", ...}`).

CLI:
    python -m models.mri path/to/scan.jpg
    python -m models.mri scan.jpg --checkpoint models/checkpoints/mri.pth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "mri.pth"
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]
_INPUT_SIZE = (224, 224)
# Out-of-distribution guard. The 4-class head softmaxes *any* image into one of
# its tumour classes, so a non-MRI (a cat photo, a chest X-ray) yields a
# confident, meaningless label. Two cheap signals reject those before the label
# is ever shown:
#   * grayscale check — brain MRIs are single-channel (R≈G≈B); colour photos are
#     not. This catches most "wrong image entirely" uploads.
#   * max-softmax floor — a genuinely ambiguous scan (low top probability) is
#     flagged rather than asserted.
_OOD_CONFIDENCE_FLOOR = 0.45
_OOD_LABEL = "not a recognized brain MRI"


@dataclass
class MRIPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    ood: bool = False  # input doesn't look like an in-distribution brain MRI
    experimental: bool = True  # unvalidated screening model — never a diagnosis

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "ood": self.ood,
            "experimental": self.experimental,
        }


class CropBlackEdges:
    """Crop the black border around an MRI by the bounding box of bright pixels.

    Identical to the transform used at training time — inference preprocessing
    must match or predictions degrade.
    """

    def __init__(self, threshold: int = 10, pad: int = 5):
        self.threshold = threshold
        self.pad = pad

    def __call__(self, img):
        gray = np.array(img.convert("L"))
        mask = gray > self.threshold
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return img
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h, w = gray.shape
        box = (
            max(0, cmin - self.pad), max(0, rmin - self.pad),
            min(w, cmax + self.pad), min(h, rmax + self.pad),
        )
        return img.crop(box)


def build_model(num_classes: int = 4) -> nn.Module:
    """EfficientNet-B0 with the notebook's custom classifier head."""
    from torchvision import models  # lazy: torchvision only needed for MRI

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def _extract_state_and_classes(ckpt) -> tuple[dict, list[str]]:
    """Support both the bare state_dict and the metadata-dict save formats."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("class_names", DEFAULT_CLASSES)
    return ckpt, DEFAULT_CLASSES


def load_model(checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str = "cpu"):
    """Load weights and return ``(model.eval(), class_names)``."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state, classes = _extract_state_and_classes(ckpt)
    model = build_model(len(classes))
    model.load_state_dict(state)
    model.eval().to(device)
    return model, classes


def _transform():
    from torchvision import transforms

    return transforms.Compose([
        CropBlackEdges(threshold=10, pad=5),
        transforms.Resize(_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])


# Process-wide cache so repeated calls don't reload the weights.
_CACHE: dict = {}


def predict(
    image,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> MRIPrediction:
    """Classify a brain MRI. ``image`` is a PIL image or a path to one."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    rgb = image.convert("RGB")
    looks_grayscale = _is_grayscale(rgb)

    key = (str(checkpoint), device)
    if key not in _CACHE:
        _CACHE[key] = load_model(checkpoint, device)
    model, classes = _CACHE[key]

    x = _transform()(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    confidence = float(probs[idx])

    # OOD: a colour image isn't a brain MRI; a low top probability is too
    # ambiguous to label. Either way, decline rather than assert a tumour class.
    ood = (not looks_grayscale) or confidence < _OOD_CONFIDENCE_FLOOR
    return MRIPrediction(
        label=_OOD_LABEL if ood else classes[idx],
        confidence=confidence,
        probabilities={c: float(p) for c, p in zip(classes, probs)},
        ood=ood,
    )


def _is_grayscale(image, tol: int = 14, frac: float = 0.98) -> bool:
    """True if the image is effectively single-channel (brain MRIs are).

    Compares the RGB channels per pixel: an MRI has R≈G≈B almost everywhere, a
    colour photo does not. ``frac`` of pixels must be within ``tol`` for the
    image to count as grayscale, tolerating minor JPEG colour artefacts.
    """
    arr = np.asarray(image, dtype=np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True  # already single-channel
    spread = arr[..., :3].max(axis=2) - arr[..., :3].min(axis=2)
    return float((spread <= tol).mean()) >= frac


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classify a brain MRI (decision-support only).")
    ap.add_argument("image", type=Path, help="path to an MRI image")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    pred = predict(args.image, args.checkpoint, args.device)
    if pred.ood:
        print(f"[OOD] {_OOD_LABEL} — this image is out-of-distribution; no class asserted.")
    else:
        print(f"Prediction: {pred.label}  ({pred.confidence * 100:.1f}%)")
    for cls, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<12} {p * 100:5.1f}%")
    print("\nEXPERIMENTAL — decision-support only, not a diagnosis. "
          "Confirm with a radiologist.")


if __name__ == "__main__":
    main()
