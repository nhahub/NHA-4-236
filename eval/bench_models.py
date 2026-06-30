"""Benchmark generator models on quality (faithfulness) vs latency.

Picks the live generation model by data instead of vibes. For each candidate
model, runs the groundedness eval (faithfulness via the same stronger judge) and
records generation latency, then prints a comparison table so the quality/latency
trade-off is explicit.

Needs a running Ollama and is **slow on CPU** (a generation + an 8B judge call per
case, per model). Defaults to the two locally pulled models.

Run:
    python -m eval.bench_models [--models qwen3:1.7b llama3.1:8b] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "groundedness.jsonl"
DEFAULT_MODELS = ["qwen3:1.7b", "llama3.1:8b"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark generator models (quality vs latency).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args(argv)

    from eval.groundedness import evaluate, load_cases
    from llm.client import get_llm

    if not get_llm().health():
        print("Bench skipped: Ollama server not reachable.")
        return 0
    if not args.cases.exists():
        print(f"No cases at {args.cases}")
        return 0

    cases = load_cases(args.cases)
    rows = []
    for model in args.models:
        r = evaluate(cases, generator_model=model, judge_model=args.judge, limit=args.limit)
        rows.append({
            "model": model,
            "faithfulness": r["mean_faithfulness"],
            "mean_latency_s": r["mean_gen_latency_s"],
            "p95_latency_s": r["p95_gen_latency_s"],
            "scored": f"{r['n_scored']}/{r['n_cases']}",
        })

    judge = rows and (args.judge or "OLLAMA_JUDGE_MODEL")
    print(f"\nModel benchmark over {cases and len(cases)} cases "
          f"(judge: {args.judge or 'configured'}):")
    print(f"  {'model':>16}  {'faithfulness':>13}  {'mean_lat_s':>11}  {'p95_lat_s':>10}  {'scored':>7}")
    for r in rows:
        f = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "n/a"
        ml = f"{r['mean_latency_s']:.1f}" if r["mean_latency_s"] is not None else "n/a"
        p95 = f"{r['p95_latency_s']:.1f}" if r["p95_latency_s"] is not None else "n/a"
        print(f"  {r['model']:>16}  {f:>13}  {ml:>11}  {p95:>10}  {r['scored']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
