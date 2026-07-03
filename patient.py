"""Structured patient information for tailoring the symptom differential.

This is an optional, durable profile a user can supply (age, sex, chronic
conditions, medications, allergies, pregnancy) so the ranked "possible conditions
to discuss with your doctor" are more relevant. Per-visit specifics (duration,
severity, triggers) are asked conversationally by the assistant instead. It is
NOT used to produce a definitive diagnosis — the assistant stays non-diagnostic
regardless of how much detail is provided.

All fields are optional; an empty ``PatientInfo`` behaves as if no info was
given. ``to_context()`` renders a compact block for the LLM prompt;
``retrieval_hints()`` produces a short string to enrich the retrieval query;
``signature()`` makes the info hashable for the response cache.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class PatientInfo:
    """A *durable* patient profile — the health-record facts that persist across
    visits and drive the triage + differential logic. Per-visit specifics
    (duration, severity, triggers, associated symptoms) are gathered
    conversationally by the assistant's follow-up questions, not stored here."""

    # Demographics
    age: int | None = None
    sex: str | None = None          # "male" / "female" / "other"
    # History (durable, medically load-bearing)
    conditions: str | None = None   # existing conditions, e.g. "diabetes, asthma"
    medications: str | None = None
    allergies: str | None = None
    pregnancy: str | None = None    # "yes" / "no" / "unsure" / "n/a"

    # Human-readable labels for the prompt context block.
    _LABELS = {
        "age": "Age",
        "sex": "Sex",
        "conditions": "Existing conditions",
        "medications": "Current medications",
        "allergies": "Allergies",
        "pregnancy": "Pregnancy",
    }

    def _items(self) -> list[tuple[str, str]]:
        """Non-empty (label, value) pairs in declared field order."""
        out: list[tuple[str, str]] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            out.append((self._LABELS[f.name], text))
        return out

    def is_empty(self) -> bool:
        return not self._items()

    @property
    def is_pregnant(self) -> bool:
        return (self.pregnancy or "").strip().lower() in {"yes", "pregnant", "true"}

    def to_context(self) -> str:
        """Render a 'PATIENT CONTEXT' block for the prompt, or '' if empty."""
        items = self._items()
        if not items:
            return ""
        lines = "\n".join(f"- {label}: {value}" for label, value in items)
        return f"PATIENT CONTEXT (supplied by the user):\n{lines}"

    def retrieval_hints(self) -> str:
        """A short phrase appended to the retrieval query to bias recall."""
        bits: list[str] = []
        if self.age is not None:
            bits.append(f"{self.age} year old")
        if self.sex:
            bits.append(str(self.sex))
        if self.conditions:
            bits.append(str(self.conditions))
        if self.is_pregnant:
            bits.append("pregnant")
        return " ".join(bits).strip()

    def signature(self) -> tuple:
        """A hashable snapshot for cache keying."""
        return tuple(getattr(self, f.name) for f in fields(self))
