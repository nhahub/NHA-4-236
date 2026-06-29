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
