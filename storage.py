"""Tiny JSON store for user (patient) profiles.

Each profile is one file at ``data/users/<id>.json`` holding the patient fields
plus a timestamp, so a user can save their details once and reload them in a
later session instead of re-typing the form. This is a lightweight local store
— no database server — which fits the project's fully-local, free stack.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, fields
from pathlib import Path

from config import USERS_DIR
from patient import PatientInfo

_SAFE = re.compile(r"[^a-z0-9_-]+")


def _safe_id(user_id: str) -> str:
    """Filename-safe slug for a user id (lowercase, alnum/_/- only)."""
    slug = _SAFE.sub("-", user_id.strip().lower()).strip("-")
    return slug or "anonymous"


def _path(user_id: str) -> Path:
    return USERS_DIR / f"{_safe_id(user_id)}.json"


def save_profile(user_id: str, patient: PatientInfo) -> Path:
    """Persist a patient profile; returns the file path written."""
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"patient": asdict(patient), "updated_at": time.time()}
    path = _path(user_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_profile(user_id: str) -> PatientInfo | None:
    """Load a saved patient profile, or None if there is none."""
    path = _path(user_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8")).get("patient", {})
    # Keep only known fields, so an older/newer file can't break construction.
    valid = {f.name for f in fields(PatientInfo)}
    return PatientInfo(**{k: v for k, v in data.items() if k in valid})


def list_profiles() -> list[str]:
    """Return the ids of all saved profiles (sorted)."""
    if not USERS_DIR.exists():
        return []
    return sorted(p.stem for p in USERS_DIR.glob("*.json") if not p.stem.endswith("_history"))


# ---------------------------------------------------------------------------
# Session memory: persist the last conversation's differential so a returning
# user can pick up where they left off without re-describing symptoms.
# ---------------------------------------------------------------------------

def _history_path(user_id: str) -> Path:
    return USERS_DIR / f"{_safe_id(user_id)}_history.json"


def save_conversation(
    user_id: str,
    differential_summary: str,
    turns: list[dict] | None = None,
) -> Path:
    """Persist the last session's differential and turns for cross-session context.

    ``differential_summary`` should be the assistant's ranked differential text
    from the final turn.  ``turns`` is the full conversation (optional; stored
    for debugging / replay but not injected into prompts by default).
    """
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "differential_summary": differential_summary,
        "turns": turns or [],
        "saved_at": time.time(),
    }
    path = _history_path(user_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_last_session(user_id: str) -> dict | None:
    """Return the last saved session dict, or None if no history exists.

    Keys: ``differential_summary`` (str), ``turns`` (list), ``saved_at`` (float).
    """
    path = _history_path(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if "differential_summary" in data else None
