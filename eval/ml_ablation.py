"""ML pre-ranking ablation — does the free-text classifier actually help?

Answers the honest question: is the ML "pillar" earning its keep, or is it
decoration? For each first-person symptom query we generate the answer twice —
**with** ML pre-ranking on the live path and **without** it — and judge the
faithfulness of both (same LLM-as-judge as `eval.groundedness`). We also report
how many answers changed at all.

Reading the result:
  * If faithfulness barely moves AND few answers change, ML pre-ranking is
    decoration — keep the classifier as a standalone showcase, not a live pillar.
  * If it moves the number (up = helps, down = hurts), you have a real data point.

Caveat: ML also biases which passages are retrieved, so this isolates the
*answer*-level effect against a fixed reference context (the raw-query retrieval),
not the full retrieval-side effect. Directional, not a precise causal estimate.

Needs a running Ollama and is slow (4 model calls per case). Run:

    python -m eval.ml_ablation [--limit N] [--judge MODEL]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "ml_ablation.jsonl"


def load_cases(path: Path) -> list[dict]:
    import json

    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def _faithfulness(context: str, answer: str, judge_model: str) -> float | None:
    from eval.groundedness import judge_answer

    verdicts = judge_answer(context, answer, judge_model)
    if not verdicts:
        return None
    total = len(verdicts)
    return sum(1 for v in verdicts if v.get("supported")) / total if total else 1.0


def evaluate(cases: list[dict], judge_model: str | None = None, limit: int | None = None) -> dict:
    import assistant
    from config import settings
    from llm.client import get_llm
    from rag.pipeline import format_context, retrieve_context

    judge_model = judge_model or settings.ollama_judge_model
    if limit:
        cases = cases[:limit]

    saved_flag = settings.ml_in_live_path
    with_scores, without_scores = [], []
    n_ml_fired = n_changed = 0
    details = []
    try:
        for case in cases:
            q = case["query"]
            # Fixed reference context (raw query) both answers are judged against.
            context = format_context(retrieve_context(q))

            answers = {}
            for ml_on in (True, False):
                settings.ml_in_live_path = ml_on
                prep = assistant.prepare(q, assistant.MODE_SYMPTOM, use_triage=False)
                if prep.messages is None:
                    answers[ml_on] = None
                    continue
                if ml_on and prep.ml_predictions:
                    n_ml_fired += 1
                answers[ml_on] = get_llm().chat(prep.messages)

            a_on, a_off = answers.get(True), answers.get(False)
            f_on = _faithfulness(context, a_on, judge_model) if a_on else None
            f_off = _faithfulness(context, a_off, judge_model) if a_off else None
            if f_on is not None:
                with_scores.append(f_on)
            if f_off is not None:
                without_scores.append(f_off)
            changed = bool(a_on and a_off and a_on.strip() != a_off.strip())
            n_changed += changed
            details.append({"query": q, "faith_with": f_on, "faith_without": f_off,
                            "changed": changed})
    finally:
        settings.ml_in_live_path = saved_flag

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_cases": len(cases),
        "n_ml_fired": n_ml_fired,
        "n_answers_changed": n_changed,
        "mean_faithfulness_with_ml": _mean(with_scores),
        "mean_faithfulness_without_ml": _mean(without_scores),
        "judge_model": judge_model,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ablate ML pre-ranking (faithfulness with vs without).")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args(argv)

    from llm.client import get_llm

    if not get_llm().health():
        print("Ablation skipped: Ollama server not reachable.")
        return 0
    if not args.cases.exists():
        print(f"No cases at {args.cases}")
        return 0

    r = evaluate(load_cases(args.cases), args.judge, args.limit)
    w, wo = r["mean_faithfulness_with_ml"], r["mean_faithfulness_without_ml"]
    delta = (w - wo) if (w is not None and wo is not None) else None

    print(f"ML pre-ranking ablation over {r['n_cases']} symptom cases "
          f"(judge: {r['judge_model']}):")
    print(f"  ML fired on:              {r['n_ml_fired']}/{r['n_cases']} cases")
    print(f"  answers changed with ML:  {r['n_answers_changed']}/{r['n_cases']}")
    print(f"  faithfulness WITH ML:     {w:.3f}" if w is not None else "  faithfulness WITH ML: n/a")
    print(f"  faithfulness WITHOUT ML:  {wo:.3f}" if wo is not None else "  faithfulness WITHOUT ML: n/a")
    if delta is not None:
        verdict = ("helps" if delta > 0.02 else "hurts" if delta < -0.02 else "no meaningful change")
        print(f"  delta (with - without):   {delta:+.3f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
