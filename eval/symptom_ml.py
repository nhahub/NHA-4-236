"""Report the free-text symptom classifier's held-out metrics.

The metrics are computed on the dataset's test split at train time (train ==
serve distribution) and stored in the artifact sidecar, so this just surfaces
them in the eval harness — no re-embedding needed. Retrain with
`python -m ml_model.symptom_classifier_train` to refresh them.

Run:
    python -m eval.symptom_ml
"""
from __future__ import annotations

import json


def load_metrics() -> dict | None:
    from ml_model.symptom_classifier import META_PATH, text_artifacts_available

    if not text_artifacts_available():
        return None
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    m = load_metrics()
    if m is None:
        print("Symptom classifier not trained. Run `python -m ml_model.symptom_classifier_train`.")
        return 0
    print(f"Free-text symptom classifier ({m.get('dataset', '?')}, "
          f"{m.get('n_classes', '?')} classes, {m.get('n_test', '?')}-row held-out test):")
    print(f"  top-1 accuracy: {m.get('top1_accuracy', float('nan')):.3f}")
    print(f"  top-3 accuracy: {m.get('top3_accuracy', float('nan')):.3f}")
    print(f"  macro F1:       {m.get('macro_f1', float('nan')):.3f}")
    print(f"  embedder:       {m.get('embedding_model', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
