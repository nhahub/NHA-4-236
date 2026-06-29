"""Deep-learning diagnostic models (MRI / EEG / ECG).

This package is separate from ``ml_model`` (the tabular XGBoost symptom
classifier). Each modality gets its own module exposing a small, uniform
interface once its architecture is known:

    load_<modality>(checkpoint_path) -> torch.nn.Module   # weights loaded, eval()
    predict_<modality>(model, x)     -> Prediction         # preprocessed input

Drop the trained ``.pth`` files into ``models/checkpoints/`` (gitignored — they
are large binaries). Use ``python -m models.inspect_checkpoint <file>`` to read
a checkpoint's layer shapes and infer its architecture before wiring it up.
"""
from __future__ import annotations
