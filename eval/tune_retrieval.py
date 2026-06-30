"""Tune retrieval fusion weights and top_k against the labelled eval set.

The dense/BM25 fusion split (0.5/0.5) and `retrieval_top_k` were arbitrary
guesses. This sweeps them over `eval/cases/retrieval.jsonl` and reports the
recall@k / MRR for each setting, so the values are chosen from data.

Only the *ratio* of dense_weight:bm25_weight matters to RRF, so we sweep a single
"dense fraction" f (dense_weight=f, bm25_weight=1-f). Metrics are measured on the
raw fused retriever output (pre-rerank), which is exactly what these knobs
control. The `rerank_score_floor` gate is a separate concern — calibrate it with
`python -m scripts.calibrate_gate`.

Settings are mutated in memory during the sweep and restored afterwards; nothing
is written to disk. Run (slow-ish: re-embeds each query per config):

    python -m eval.tune_retrieval [--cases PATH] [--top-k 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.retrieval import DEFAULT_CASES, _is_hit, load_cases

# Dense fraction grid (dense_weight = f, bm25_weight = 1 - f).
_DENSE_FRACTIONS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
_TOP_K_GRID = [10, 20, 30]
_RECALL_CUTOFFS = (5, 10)


def _first_hit_rank(passages, relevant_titles) -> int | None:
    return next(
        (i for i, p in enumerate(passages, start=1) if _is_hit(p.title, relevant_titles)),
        None,
    )


def _metrics(retriever, cases, top_k: int) -> dict:
    recalls = {k: 0 for k in _RECALL_CUTOFFS}
    rr_sum = 0.0
    for case in cases:
        passages = retriever.retrieve(case["query"], top_k=top_k)
        rank = _first_hit_rank(passages, case.get("relevant_titles", []))
        if rank is not None:
            rr_sum += 1.0 / rank
            for k in _RECALL_CUTOFFS:
                if rank <= k:
                    recalls[k] += 1
    n = len(cases) or 1
    return {
        "mrr": rr_sum / n,
        **{f"recall@{k}": recalls[k] / n for k in _RECALL_CUTOFFS},
    }


def sweep(cases: list[dict], top_k: int) -> dict:
    """Sweep dense fraction (fixed top_k), then top_k at the best fraction."""
    from config import settings
    from rag.retriever import get_retriever

    retriever = get_retriever()
    saved = (settings.dense_weight, settings.bm25_weight)
    try:
        by_fraction = []
        for f in _DENSE_FRACTIONS:
            settings.dense_weight, settings.bm25_weight = f, 1.0 - f
            m = _metrics(retriever, cases, top_k)
            by_fraction.append({"dense_fraction": f, **m})

        best = max(by_fraction, key=lambda r: (r["mrr"], r[f"recall@{_RECALL_CUTOFFS[-1]}"]))
        settings.dense_weight, settings.bm25_weight = (
            best["dense_fraction"], 1.0 - best["dense_fraction"],
        )
        by_topk = []
        for tk in _TOP_K_GRID:
            m = _metrics(retriever, cases, tk)
            by_topk.append({"top_k": tk, **m})
    finally:
        settings.dense_weight, settings.bm25_weight = saved

    return {"by_fraction": by_fraction, "best_fraction": best, "by_topk": by_topk}


def _print_table(rows: list[dict], key: str) -> None:
    cols = [key] + [c for c in rows[0] if c != key]
    print("  " + "  ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print("  " + "  ".join(
            f"{r[c]:>14.3f}" if isinstance(r[c], float) else f"{r[c]:>14}" for c in cols
        ))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tune fusion weights / top_k.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--top-k", type=int, default=20, help="top_k for the fraction sweep")
    args = ap.parse_args(argv)

    from config import FAISS_INDEX_PATH

    if not FAISS_INDEX_PATH.exists():
        print("Tuning skipped: FAISS index not built. Run `python -m rag.ingest`.")
        return 0
    if not args.cases.exists():
        print(f"No retrieval cases at {args.cases}")
        return 0

    cases = load_cases(args.cases)
    result = sweep(cases, args.top_k)

    print(f"Fusion sweep over {len(cases)} cases (top_k={args.top_k}); "
          "dense_fraction = dense_weight, bm25_weight = 1 - it:")
    _print_table(result["by_fraction"], "dense_fraction")
    b = result["best_fraction"]
    print(f"\nBest dense_fraction = {b['dense_fraction']} "
          f"(MRR={b['mrr']:.3f}, recall@{_RECALL_CUTOFFS[-1]}={b[f'recall@{_RECALL_CUTOFFS[-1]}']:.3f})")
    print(f"  -> set DENSE_WEIGHT={b['dense_fraction']}, BM25_WEIGHT={round(1 - b['dense_fraction'], 3)}")

    print(f"\ntop_k sweep at dense_fraction={b['dense_fraction']}:")
    _print_table(result["by_topk"], "top_k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
