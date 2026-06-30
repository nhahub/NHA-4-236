"""End-to-end latency eval: p50 / p95 per flow.

Times the real pipeline so the latency picture is a tracked number, not a guess.
For each query it measures:
  * retrieval stage  — retrieve_context (embed + FAISS + BM25 + rerank), offline
  * end-to-end       — the full answer flow (prepare + grounded generation)

and reports mean / p50 / p95 over the cases. Generation dominates on CPU; the
split makes that explicit.

A warmup call (model + index load) runs first and is excluded from the stats, and
the response cache is cleared before each timed call so we measure cold work, not
a cache hit. Needs a running Ollama and is slow on CPU.

Run:
    python -m eval.latency [--flow qa|symptom] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "groundedness.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def _percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    s = sorted(values)
    idx = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[idx]


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def evaluate(cases: list[dict], flow: str = "qa", limit: int | None = None) -> dict:
    import assistant
    from rag.pipeline import retrieve_context

    answer_fn = assistant.answer_question if flow == "qa" else assistant.explore_symptoms
    if limit:
        cases = cases[:limit]

    # Warmup (model + index load) — excluded from the measured stats.
    if cases:
        retrieve_context(cases[0]["query"])
        answer_fn(cases[0]["query"], use_triage=False)

    retrieval_lat: list[float] = []
    total_lat: list[float] = []
    for case in cases:
        query = case["query"]
        assistant._cache.clear()  # measure cold work, not a cache hit

        t0 = time.perf_counter()
        retrieve_context(query)
        retrieval_lat.append(time.perf_counter() - t0)

        assistant._cache.clear()
        t0 = time.perf_counter()
        answer_fn(query, use_triage=False)
        total_lat.append(time.perf_counter() - t0)

    return {
        "flow": flow,
        "n_cases": len(cases),
        "retrieval_s": _summary(retrieval_lat),
        "end_to_end_s": _summary(total_lat),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end latency (p50/p95) per flow.")
    ap.add_argument("--flow", choices=["qa", "symptom"], default="qa")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from llm.client import get_llm

    if not get_llm().health():
        print("Latency eval skipped: Ollama server not reachable.")
        return 0
    if not args.cases.exists():
        print(f"No cases at {args.cases}")
        return 0

    r = evaluate(load_cases(args.cases), flow=args.flow, limit=args.limit)

    def _fmt(s: dict) -> str:
        return f"mean={s['mean']:.1f}s  p50={s['p50']:.1f}s  p95={s['p95']:.1f}s"

    print(f"Latency over {r['n_cases']} cases ({r['flow']} flow):")
    print(f"  retrieval : {_fmt(r['retrieval_s'])}")
    print(f"  end-to-end: {_fmt(r['end_to_end_s'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
