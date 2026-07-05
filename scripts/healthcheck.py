"""Environment health check — fail loud on a wrong or incomplete environment.

This project was bitten once by a *copied* virtualenv whose ``python.exe``
resolved against a different interpreter's site-packages than the one it was
launched with, so imports silently picked up the wrong (or no) packages and the
app failed deep inside a request instead of at startup.

This module verifies that the runtime-critical dependencies are importable from
the *current* interpreter and reports which interpreter that is, so a
misconfigured environment is caught at launch.

Run it directly (exits non-zero if a required dep is missing)::

    python -m scripts.healthcheck

It is also invoked, warn-only, at API startup. Always launch the app through the
active interpreter with ``python -m`` (e.g. ``python -m uvicorn api.main:app``)
rather than a bare ``uvicorn``/``streamlit`` console script, whose shebang may
point at a stale interpreter.
"""
from __future__ import annotations

import importlib
import sys

# (import name, pip/distribution hint) for each runtime-critical dependency.
# Import name != distribution name for several of these (faiss-cpu -> faiss).
_REQUIRED: list[tuple[str, str]] = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn[standard]"),
    ("pydantic_settings", "pydantic-settings"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("rank_bm25", "rank-bm25"),
    ("httpx", "httpx"),
    ("numpy", "numpy"),
]
# Optional: only needed for the imaging/signal models or the dashboard. Missing
# -> warn, not fatal, so the core RAG+LLM path still runs.
_OPTIONAL: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("PIL", "pillow"),
    ("pandas", "pandas"),
    ("streamlit", "streamlit"),
]


def _missing(mods: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for mod, hint in mods:
        try:
            importlib.import_module(mod)
        except Exception:
            out.append(hint)
    return out


def check_dependencies(
    required: list[tuple[str, str]] = _REQUIRED,
    optional: list[tuple[str, str]] = _OPTIONAL,
) -> tuple[list[str], list[str]]:
    """Return ``(missing_required, missing_optional)`` as pip-install hints."""
    return _missing(required), _missing(optional)


def format_report(missing_req: list[str], missing_opt: list[str]) -> str:
    lines = [f"Interpreter: {sys.executable}"]
    if missing_req:
        lines.append("[FAIL] MISSING required deps: " + ", ".join(missing_req))
        lines.append(
            "  -> wrong or incomplete environment. Activate the project venv and "
            "launch via `python -m` (e.g. `python -m uvicorn api.main:app`). "
            "Install with `pip install -r requirements/base.txt`."
        )
    else:
        lines.append("[OK] all required dependencies importable")
    if missing_opt:
        lines.append(
            "[note] optional deps missing (imaging/signal models, dashboard): "
            + ", ".join(missing_opt)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    missing_req, missing_opt = check_dependencies()
    print(format_report(missing_req, missing_opt))
    return 1 if missing_req else 0


if __name__ == "__main__":
    raise SystemExit(main())
