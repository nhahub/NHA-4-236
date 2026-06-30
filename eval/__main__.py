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


def _run_triage() -> None:
    _section("Triage (rules sensitivity / specificity)")
    from eval import triage

    cases = CASES_DIR / "triage.jsonl"
    if not cases.exists():
        print("pending — no eval/cases/triage.jsonl")
        return
    triage.main([])


def _run_groundedness() -> None:
    _section("Groundedness (faithfulness, LLM-as-judge)")
    from eval import groundedness

    cases = CASES_DIR / "groundedness.jsonl"
    if not cases.exists():
        print("pending — no eval/cases/groundedness.jsonl")
        return
    groundedness.main([])


def _run_pending(with_llm: bool) -> None:
    _section("Pending (P1)")
    if not with_llm:
        print("groundedness/faithfulness — run `python -m eval --with-llm` "
              "(slow, needs Ollama) or `python -m eval.groundedness`")
    print("latency — p50/p95 end-to-end per flow")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_llm = "--with-llm" in argv
    positional = [a for a in argv if a != "--with-llm"]
    if positional and positional[0] not in {"report"}:
        print(f"unknown command {positional[0]!r}; usage: python -m eval [report] [--with-llm]")
        return 2
    print("Hybrid Medical Assistant — eval report")
    _run_retrieval()
    _run_citations()
    _run_triage()
    if with_llm:  # slow LLM-judge pass, opt-in
        _run_groundedness()
    _run_pending(with_llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
