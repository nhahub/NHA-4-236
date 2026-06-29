"""Inspect a PyTorch ``.pth`` checkpoint to recover its architecture.

A ``.pth`` file usually stores only a ``state_dict`` (layer names + weight
tensors), not the model code. This tool reads those tensors and reports enough
to reconstruct the matching ``nn.Module``:

    python -m models.inspect_checkpoint models/checkpoints/mri.pth
    python -m models.inspect_checkpoint models/checkpoints/ecg.pth --full

What it tells you:
  * whether the file is a bare state_dict, a wrapped checkpoint, or a full model
  * every parameter tensor's name, shape, and dtype (with --full)
  * total parameter count
  * the convolution dimensionality (Conv1d/2d/3d) — a strong hint at the
    modality: 1-D ⇒ EEG/ECG signal, 2-D ⇒ image/MRI slice, 3-D ⇒ MRI volume
  * the first conv's in_channels (expected input channels)
  * the final linear layer's out_features (likely the number of output classes)
  * a best-effort guess at the architecture family
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


# --- loading -------------------------------------------------------------
# Common keys checkpoints use to wrap the weights.
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights", "params")


def load_raw(path: Path) -> Any:
    """Load a checkpoint, tolerating torch>=2.6's weights_only default.

    Tries the safe ``weights_only=True`` path first (no code execution). Falls
    back to ``weights_only=False`` for checkpoints that pickled custom objects —
    only do this with files you trust, since it can execute arbitrary code.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - report and retry
        print(
            f"[inspect] weights_only=True failed ({type(exc).__name__}); retrying "
            "with weights_only=False. Only do this for checkpoints you trust.",
            file=sys.stderr,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(obj: Any) -> tuple[OrderedDict, str]:
    """Return (tensor state_dict, description) from whatever ``torch.load`` gave.

    Handles: a bare state_dict, a dict wrapping one under a known key, and a
    full ``nn.Module``. Strips a leading ``module.`` (DataParallel) prefix.
    """
    note = "bare state_dict"

    if isinstance(obj, torch.nn.Module):
        return _strip_prefix(obj.state_dict()), "full nn.Module (architecture embedded)"

    if isinstance(obj, dict):
        # A dict that is itself the state_dict (values are tensors)?
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if tensor_vals and len(tensor_vals) >= max(1, len(obj) // 2):
            return _strip_prefix(obj), note
        # Otherwise look for a wrapped state_dict under a known key.
        for key in _STATE_DICT_KEYS:
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                extras = [k for k in obj if k not in _STATE_DICT_KEYS]
                note = f"wrapped under '{key}'"
                if extras:
                    note += f"; other keys: {', '.join(map(str, extras))}"
                return _strip_prefix(inner), note

    raise SystemExit(
        f"[inspect] Could not find a state_dict. Top-level type: {type(obj).__name__}. "
        "If this is a custom checkpoint, share its structure and I'll adapt the loader."
    )


def _strip_prefix(sd: dict) -> OrderedDict:
    if all(isinstance(k, str) and k.startswith("module.") for k in sd):
        return OrderedDict((k[len("module."):], v) for k, v in sd.items())
    return OrderedDict(sd)


# --- analysis ------------------------------------------------------------
def _conv_dims(state_dict: OrderedDict) -> dict[int, int]:
    """Count weight tensors by conv dimensionality (3->Conv1d, 4->Conv2d, 5->Conv3d)."""
    dims: dict[int, int] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() in (3, 4, 5) and k.endswith("weight"):
            dims[v.dim()] = dims.get(v.dim(), 0) + 1
    return dims


def _first_conv_in_channels(state_dict: OrderedDict) -> int | None:
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and v.dim() in (3, 4, 5):
            return int(v.shape[1])  # [out, in, *kernel]
    return None


def _final_linear_out(state_dict: OrderedDict) -> tuple[str, int] | None:
    last = None
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and k.endswith("weight"):
            last = (k, int(v.shape[0]))  # Linear weight is [out_features, in_features]
    return last


def _guess_family(state_dict: OrderedDict) -> str:
    keys = list(state_dict.keys())
    joined = " ".join(keys)
    if "fc.weight" in keys and any(k.startswith("layer1.") for k in keys):
        return "torchvision ResNet-style (layerN.*, fc)"
    if "classifier.weight" in joined or "classifier.1.weight" in joined:
        return "classifier-head net (EfficientNet/DenseNet/MobileNet family?)"
    if any("transformer" in k or "attn" in k or "attention" in k for k in keys):
        return "transformer / attention-based"
    if any("lstm" in k.lower() or "gru" in k.lower() or "rnn" in k.lower() for k in keys):
        return "recurrent (LSTM/GRU) — common for EEG/ECG sequences"
    return "custom / unrecognized — use the shapes below to reconstruct"


_MODALITY_HINT = {3: "1-D signal (EEG/ECG)", 4: "2-D image (MRI slice / X-ray)", 5: "3-D volume (MRI)"}


def report(path: Path, full: bool) -> None:
    print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB) ===")
    obj = load_raw(path)
    state_dict, note = extract_state_dict(obj)
    print(f"format        : {note}")

    tensors = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    total = sum(v.numel() for _, v in tensors)
    print(f"tensors       : {len(tensors)}")
    print(f"total params  : {total:,}")

    dims = _conv_dims(state_dict)
    if dims:
        parts = [f"Conv{d - 2}d x{n}" for d, n in sorted(dims.items())]
        modality = _MODALITY_HINT.get(max(dims), "?")
        print(f"conv layers   : {', '.join(parts)}  ->  modality looks like: {modality}")
    else:
        print("conv layers   : none found (MLP / RNN / transformer?)")

    in_ch = _first_conv_in_channels(state_dict)
    if in_ch is not None:
        print(f"input channels: {in_ch}  (first conv in_channels)")

    final = _final_linear_out(state_dict)
    if final:
        print(f"output size   : {final[1]}  (from '{final[0]}')  -> likely num classes")

    print(f"arch guess    : {_guess_family(state_dict)}")

    if full:
        print("\n-- all parameter tensors --")
        for k, v in tensors:
            print(f"  {k:<55} {tuple(v.shape)}  {str(v.dtype).replace('torch.', '')}")
    else:
        print("\n(run with --full to list every layer's name and shape)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Inspect a PyTorch .pth checkpoint.")
    ap.add_argument("paths", nargs="+", type=Path, help="checkpoint file(s) to inspect")
    ap.add_argument("--full", action="store_true", help="list every parameter tensor")
    args = ap.parse_args(argv)

    for p in args.paths:
        if not p.exists():
            print(f"[inspect] not found: {p}", file=sys.stderr)
            continue
        report(p, args.full)


if __name__ == "__main__":
    main()
