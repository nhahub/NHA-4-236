"""``python -m eval`` — one-command report across the assistant's pillars.

Skeleton: the retrieval and citation-validity sections run today; groundedness,
triage sensitivity/specificity, and latency are stubbed with the metric each
will report (P1 work). The goal is a stable shape that prints something honest
now and fills in without restructuring.

    python -m eval            # full report
    python -m eval report     # same
"""
from __future__ import annotations

import sys

from eval import CASES_DIR


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _run_retrieval() -> None:
    _section("Retrieval (recall@k / MRR)")
    from eval import retrieval

    retrieval.main([])


def _run_citations() -> None:
    _section("Citation validity")
    from eval import citations

    cases = CASES_DIR / "citations.jsonl"
    if not cases.exists():
        print("pending — no eval/cases/citations.jsonl")
        return
    citations.main([])


def _run_pending() -> None:
    _section("Pending (P1)")
    print("groundedness/faithfulness — hallucination rate (LLM-as-judge / ragas)")
    print("triage sensitivity / specificity — labelled emergency vs non-emergency")
    print("latency — p50/p95 end-to-end per flow")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in {"report"}:
        print(f"unknown command {argv[0]!r}; usage: python -m eval [report]")
        return 2
    print("Hybrid Medical Assistant — eval report")
    _run_retrieval()
    _run_citations()
    _run_pending()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
