# Deep-learning diagnostic models (MRI / EEG / ECG)

Standalone PyTorch inference for imaging/signal models. Separate from
[`ml_model/`](../ml_model) (the tabular XGBoost symptom classifier).

> ⚠️ Like the rest of this project, these are **decision-support, not
> diagnostic**. Outputs should be framed as "findings to discuss with a
> clinician," never as a definitive diagnosis.

## Layout

```
models/
  inspect_checkpoint.py   # read a .pth's shapes + infer its architecture
  checkpoints/            # drop trained .pth files here (gitignored)
  mri.py / eeg.py / ecg.py  # arch + preprocess + predict   (added per model)
```

## Adding a model (the path with only a `.pth`)

A `.pth` is usually just a `state_dict` — weights, not code. To run it we must
rebuild the matching `nn.Module`. Workflow:

1. Drop the file in `models/checkpoints/` (e.g. `mri.pth`).
2. Inspect it:
   ```bash
   python -m models.inspect_checkpoint models/checkpoints/mri.pth --full
   ```
   This reports conv dimensionality (1-D ⇒ EEG/ECG, 2-D ⇒ MRI slice, 3-D ⇒ MRI
   volume), input channels, output classes, and a guess at the architecture.
3. Reconstruct the architecture in `models/<modality>.py`, load the weights,
   and add `preprocess()` + `predict()`.

### What makes step 3 reliable

Shapes alone pin down common backbones (ResNet, EfficientNet, standard 1-D
CNNs). For a **custom** network they only narrow it down. Anything you can share
helps — ideally:

| Need | Why |
|------|-----|
| training script / `model.py` (or repo / paper) | exact architecture |
| expected input shape | e.g. MRI `(1, 224, 224)` or volume `(1, D, H, W)`; ECG `(leads, samples)` |
| sampling rate (EEG/ECG) | windowing + resampling |
| normalization used in training | inputs must match or predictions are garbage |
| class label list | map output indices → human-readable findings |
| input file format | NIfTI/DICOM/PNG for MRI; EDF/CSV/`.npy` for EEG/ECG |
