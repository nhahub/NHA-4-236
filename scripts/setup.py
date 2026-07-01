"""One-command setup: make a fresh clone runnable.

Idempotent — each step is skipped if its output already exists, so it's safe to
re-run. Steps:
  1. Download the MedQuAD knowledge base (if missing).
  2. Build the FAISS + BM25 retrieval index (if missing).
  3. (optional, --with-ml) Train the XGBoost symptom classifier (if missing).
  4. Report whether Ollama and the configured model are reachable.

Run:
    python -m scripts.setup            # data + index
    python -m scripts.setup --with-ml  # also train the symptom classifier
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "vector_store" / "index.faiss"
MEDQUAD = ROOT / "data" / "raw" / "MedQuAD"
XGB = ROOT / "ml_model" / "artifacts" / "xgb_model.json"
TEXT_CLF = ROOT / "ml_model" / "artifacts" / "symptom_text_clf.joblib"


def _run(label: str, module: str) -> bool:
    print(f"\n=== {label} ===")
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    if result.returncode != 0:
        print(f"!! {label} failed (exit {result.returncode}).")
        return False
    return True


def _check_ollama() -> None:
    print("\n=== Checking Ollama ===")
    try:
        import httpx

        from config import settings

        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        ok = any(settings.ollama_model in n for n in names)
        print(f"Ollama reachable. Models: {names or '(none)'}")
        if not ok:
            print(f"!! Configured model '{settings.ollama_model}' not pulled. "
                  f"Run: ollama pull {settings.ollama_model}")
    except Exception as exc:  # noqa: BLE001
        print(f"!! Ollama not reachable at the configured URL ({exc}). "
              "Start Ollama (and `ollama pull <model>`) before running the app.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap the project (idempotent).")
    ap.add_argument("--with-ml", action="store_true",
                    help="also train the free-text symptom classifier (small, fast)")
    ap.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    args = ap.parse_args(argv)

    ok = True
    if args.force or not MEDQUAD.exists():
        ok &= _run("1/3 Download MedQuAD", "scripts.download_medquad")
    else:
        print("=== 1/3 MedQuAD already present - skipping ===")

    if args.force or not INDEX.exists():
        ok &= _run("2/3 Build retrieval index", "rag.ingest")
    else:
        print("=== 2/3 Retrieval index already present - skipping ===")

    if args.with_ml:
        if args.force or not TEXT_CLF.exists():
            # The live pillar: free-text classifier (tiny HF dataset, ~1-2 min).
            # The DDXPlus XGBoost is a separate standalone artifact — train it via
            # `python -m ml_model.train` if you want the portfolio notebook model.
            ok &= _run("3/3 Train free-text symptom classifier", "ml_model.text_train")
        else:
            print("=== 3/3 Symptom classifier already trained - skipping ===")
    else:
        print("=== 3/3 ML training skipped (pass --with-ml to enable) ===")

    _check_ollama()

    print("\n" + ("[OK] Setup complete." if ok else "[WARN] Setup finished with errors (see above)."))
    print("Next: uvicorn api.main:app   and   streamlit run dashboard/streamlit_app.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
