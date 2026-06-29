"""Citation-validity check: every ``[n]`` in an answer must map to a real source.

Reuses the production ``enforce_citation_integrity`` pass so the eval measures
exactly what ships. A case is *valid* when the answer cites no out-of-range
marker (i.e. the integrity pass strips nothing). The aggregate metric is the
fraction of cases with no invented citations.

Run:
    python -m eval.citations [--cases PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from assistant import _CITATION_MARKER_RE, enforce_citation_integrity
from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "citations.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def check_case(answer: str, n_sources: int) -> dict:
    """One case: how many cited markers, how many were invalid (invented)."""
    citations = [{"index": i} for i in range(1, n_sources + 1)]
    valid = {c["index"] for c in citations}
    cited = {int(n) for n in _CITATION_MARKER_RE.findall(answer)}
    invalid = sorted(cited - valid)
    clean, _ = enforce_citation_integrity(answer, citations)
    return {"cited": sorted(cited), "invalid": invalid, "valid": not invalid,
            "stripped": clean != answer}


def evaluate(cases: list[dict]) -> dict:
    results = [check_case(c["answer"], c["n_sources"]) for c in cases]
    n = len(results) or 1
    valid = sum(r["valid"] for r in results)
    return {
        "n_cases": len(results),
        "validity_rate": valid / n,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Citation-validity rate.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = ap.parse_args(argv)

    if not args.cases.exists():
        print(f"No citation cases at {args.cases}")
        return 0
    cases = load_cases(args.cases)
    result = evaluate(cases)
    print(f"Citation validity over {result['n_cases']} cases: "
          f"{result['validity_rate']:.0%} clean")
    for case, r in zip(cases, result["results"]):
        if not r["valid"]:
            print(f"  INVALID {r['invalid']} in: {case['answer'][:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
