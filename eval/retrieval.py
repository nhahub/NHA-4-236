"""Retrieval evaluation: recall@k and MRR over a labelled query set.

Each case in ``eval/cases/retrieval.jsonl`` pairs a query with the title(s) of
passages known to be relevant. A retrieved passage counts as a hit when its
title matches a relevant title (case-insensitive substring, either direction, to
tolerate chunk-title variance like "Urinary tract infection (adults)"). We score
against the *raw* hybrid retrieval list (pre-rerank) so recall reflects what the
retriever surfaces, independent of the reranker's top-N cut.

Run:
    python -m eval.retrieval [--k 5] [--cases PATH]

The seed case set is small and meant to be expanded to 30-50 curated queries.
If the FAISS index hasn't been built yet, this prints a clear notice and exits 0
(nothing to measure rather than a crash).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "retrieval.jsonl"
DEFAULT_K = 5


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _is_hit(passage_title: str, relevant_titles: list[str]) -> bool:
    pt = _norm(passage_title)
    return any(_norm(rt) in pt or pt in _norm(rt) for rt in relevant_titles if rt)


def evaluate(cases: list[dict], k: int = DEFAULT_K) -> dict:
    """Return aggregate ``recall@k`` and ``MRR`` plus per-case detail."""
    from rag.retriever import get_retriever

    retriever = get_retriever()
    hits = 0
    rr_sum = 0.0
    details = []
    for case in cases:
        query = case["query"]
        relevant = case.get("relevant_titles", [])
        passages = retriever.retrieve(query, top_k=k)
        rank = next(
            (i for i, p in enumerate(passages, start=1) if _is_hit(p.title, relevant)),
            None,
        )
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
        details.append({"query": query, "rank": rank})
    n = len(cases) or 1
    return {
        "k": k,
        "n_cases": len(cases),
        "recall_at_k": hits / n,
        "mrr": rr_sum / n,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval recall@k / MRR.")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = ap.parse_args(argv)

    from config import FAISS_INDEX_PATH

    if not FAISS_INDEX_PATH.exists():
        print("Retrieval eval skipped: FAISS index not built. "
              "Run `python -m rag.ingest` first.")
        return 0
    if not args.cases.exists():
        print(f"No retrieval cases at {args.cases}")
        return 0

    cases = load_cases(args.cases)
    result = evaluate(cases, k=args.k)
    print(f"Retrieval over {result['n_cases']} cases  "
          f"recall@{result['k']}={result['recall_at_k']:.2f}  "
          f"MRR={result['mrr']:.2f}")
    for d in result["details"]:
        mark = f"rank {d['rank']}" if d["rank"] else "MISS"
        print(f"  [{mark:>7}] {d['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
