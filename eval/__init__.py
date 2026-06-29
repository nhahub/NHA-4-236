"""Evaluation harness for the hybrid medical assistant.

The keystone that turns "eyeballed demo" into "measured": a single command,

    python -m eval

prints a report across the pillars — retrieval (recall@k / MRR), citation
validity, and (stubbed, pending) groundedness, triage sensitivity/specificity,
and latency.

This is a *skeleton*: the retrieval and citation-validity checks are wired up
and runnable today; the remaining sections print a "pending" line with the
metric they will report, so the harness can grow without changing its shape.
Curated case sets live in ``eval/cases/`` (small seed sets, to expand to 30-50).
"""
from __future__ import annotations

from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"
