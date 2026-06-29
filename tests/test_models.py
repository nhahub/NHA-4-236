"""Tests for the models/ package (MRI classifier + checkpoint inspector).

These don't need the trained .pth weights — they validate the architecture
builder, the save-format handling, and the inspector's state_dict extraction.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from models import mri
from models.inspect_checkpoint import extract_state_dict


def test_build_model_output_shape():
    model = mri.build_model(num_classes=4).eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_extract_state_and_classes_bare_state_dict():
    sd = {"w": torch.zeros(1)}
    state, classes = mri._extract_state_and_classes(sd)
    assert state is sd
    assert classes == mri.DEFAULT_CLASSES


def test_extract_state_and_classes_metadata_dict():
    ckpt = {"model_state_dict": {"w": torch.zeros(1)}, "class_names": ["a", "b"]}
    state, classes = mri._extract_state_and_classes(ckpt)
    assert state == {"w": torch.zeros(1)}
    assert classes == ["a", "b"]


def test_crop_black_edges_trims_border():
    from PIL import Image
    import numpy as np

    arr = np.zeros((100, 100, 3), dtype="uint8")
    arr[30:70, 40:60] = 200  # bright patch in a black frame
    cropped = mri.CropBlackEdges(threshold=10, pad=0)(Image.fromarray(arr))
    # The crop uses inclusive max indices with an exclusive PIL right/lower edge
    # (a faithful port of the training notebook), so it lands 1px short — fine
    # after the downstream Resize(224).
    assert cropped.size == (19, 39)  # (width, height) of the bright box


def test_inspector_extracts_and_strips_module_prefix():
    inner = OrderedDict({"module.conv.weight": torch.zeros(4, 3, 3, 3)})
    state, note = extract_state_dict({"state_dict": inner, "epoch": 1})
    assert "conv.weight" in state and "module.conv.weight" not in state
    assert "wrapped under 'state_dict'" in note


# --- EEG model -----------------------------------------------------------
from models import eeg  # noqa: E402


def test_eeg_flatten_head_fixed_window():
    model = eeg.EEGNet(head="flatten").eval()
    out = model(torch.randn(2, 23, eeg.WINDOW_SAMPLES))
    assert out.shape == (2, 1)


def test_eeg_gap_head_is_length_flexible():
    model = eeg.EEGNet(head="gap").eval()
    assert model(torch.randn(1, 23, 1280)).shape == (1, 1)
    assert model(torch.randn(1, 23, 2048)).shape == (1, 1)  # different length OK


def test_eeg_infer_head_from_fc1_width():
    assert eeg._infer_head({"fc1.weight": torch.zeros(128, 128 * 160)}) == ("flatten", 128 * 160)
    assert eeg._infer_head({"fc1.weight": torch.zeros(128, 128)}) == ("gap", 128)


def test_eeg_preprocess_pads_channels_and_normalizes():
    import numpy as np

    win = np.random.randn(20, 1280).astype("float32") * 7 + 3  # 20 ch, off-scale
    x = eeg.preprocess(win, head="flatten")
    assert x.shape == (1, 23, eeg.WINDOW_SAMPLES)  # padded 20 -> 23 channels
    real = x[0, :20]  # padded channels are constant (std 0) -> skip them
    assert torch.allclose(real.mean(dim=1), torch.zeros(20), atol=1e-4)
