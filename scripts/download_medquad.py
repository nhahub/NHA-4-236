"""Fetch the MedQuAD corpus into data/raw.

MedQuAD (https://github.com/abachaa/MedQuAD) is distributed as a GitHub repo of
XML files. The simplest, dependency-free way to get it is a shallow git clone.
This helper clones it into ``data/raw/MedQuAD`` if git is available; otherwise
it prints manual instructions.

Usage:
    python scripts/download_medquad.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/abachaa/MedQuAD.git"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "MedQuAD"


def main() -> int:
    if TARGET.exists() and any(TARGET.rglob("*.xml")):
        print(f"MedQuAD already present at {TARGET}")
        return 0

    if shutil.which("git") is None:
        print(
            "git not found. Manually download MedQuAD and unzip its XML files into:\n"
            f"  {TARGET}\n"
            f"Source: {REPO_URL}"
        )
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning MedQuAD into {TARGET} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(TARGET)],
        check=False,
    )
    if result.returncode != 0:
        print("Clone failed. Download manually from:", REPO_URL)
        return result.returncode

    n = sum(1 for _ in TARGET.rglob("*.xml"))
    print(f"Done. {n} XML files. Now run:  python -m rag.ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
