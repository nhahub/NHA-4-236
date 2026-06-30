"""Triage evaluation: sensitivity / specificity of the rule-based red-flag pass.

Measures the **offline rules layer** (`detect_red_flags(text, use_llm=False)`),
which is the deterministic, reproducible safety net — the LLM second pass needs a
running Ollama and is layered on top, not measured here.

Cases (`eval/cases/triage.jsonl`) are labelled emergency / self_harm /
non_emergency. The "should flag" positives are emergency ∪ self_harm.

Metrics:
  * sensitivity = TP / (TP + FN) over positives — MUST be ~1.0 (a missed
    emergency is the dangerous failure).
  * specificity = TN / (TN + FP) over non_emergency.
  * self-harm routing = fraction of self_harm cases that route to the crisis
    message (reason contains self-harm/suicid), not the generic ER message.

False negatives (missed emergencies) are printed explicitly.

Run:
    python -m eval.triage [--cases PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "triage.jsonl"
_POSITIVE = {"emergency", "self_harm"}


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def evaluate(cases: list[dict]) -> dict:
    from safety.red_flag_detector import SELF_HARM_MESSAGE, detect_red_flags, emergency_message

    tp = fn = tn = fp = 0
    self_harm_total = self_harm_routed = 0
    false_negatives: list[str] = []
    false_positives: list[str] = []

    for case in cases:
        text, label = case["text"], case["label"]
        result = detect_red_flags(text, use_llm=False)
        flagged = result.emergency
        should_flag = label in _POSITIVE

        if should_flag and flagged:
            tp += 1
        elif should_flag and not flagged:
            fn += 1
            false_negatives.append(text)
        elif not should_flag and not flagged:
            tn += 1
        else:
            fp += 1
            false_positives.append(text)

        if label == "self_harm":
            self_harm_total += 1
            if flagged and emergency_message(result) == SELF_HARM_MESSAGE:
                self_harm_routed += 1

    sensitivity = tp / (tp + fn) if (tp + fn) else 1.0
    specificity = tn / (tn + fp) if (tn + fp) else 1.0
    return {
        "n_cases": len(cases),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "self_harm_routing": (self_harm_routed / self_harm_total) if self_harm_total else 1.0,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Triage rules sensitivity/specificity.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = ap.parse_args(argv)

    if not args.cases.exists():
        print(f"No triage cases at {args.cases}")
        return 0

    r = evaluate(load_cases(args.cases))
    print(f"Triage rules over {r['n_cases']} cases  "
          f"sensitivity={r['sensitivity']:.2f}  specificity={r['specificity']:.2f}  "
          f"self-harm routing={r['self_harm_routing']:.2f}")
    print(f"  TP={r['tp']} FN={r['fn']} TN={r['tn']} FP={r['fp']}")
    for t in r["false_negatives"]:
        print(f"  MISSED (false negative): {t}")
    for t in r["false_positives"]:
        print(f"  over-flagged (false positive): {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
