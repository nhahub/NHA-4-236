"""Groundedness / faithfulness eval (LLM-as-judge).

Measures how well generated answers stick to the retrieved context — i.e. the
hallucination rate, as a tracked number. This is a direct implementation of the
ragas-style *faithfulness* metric (no heavy ragas dependency):

  1. Run the real pipeline for each query (retrieve -> grounded generation).
  2. A **stronger** judge model (``OLLAMA_JUDGE_MODEL``, default llama3.1:8b)
     extracts the distinct factual claims the answer makes and labels each as
     supported / not-supported *by the retrieved context only* (general world
     knowledge doesn't count, and the boilerplate safety disclaimer is ignored).
  3. faithfulness = supported_claims / total_claims, averaged over cases.

Needs a running Ollama and is **slow on CPU** (a generation + an 8B judge call
per case), so it is opt-in: not part of the fast ``python -m eval`` report unless
``--with-llm`` is passed. Cases are queries the corpus covers
(``eval/cases/groundedness.jsonl``).

Run:
    python -m eval.groundedness [--limit N] [--generator MODEL] [--judge MODEL]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from eval import CASES_DIR

DEFAULT_CASES = CASES_DIR / "groundedness.jsonl"

_JUDGE_PROMPT = """You are a strict fact-checker grading whether an ANSWER is \
grounded in the provided CONTEXT.

Extract the distinct factual claims the ANSWER makes about the medical topic. \
For each claim, decide if it is supported by the CONTEXT alone. Rules:
- Use ONLY the CONTEXT. Your own world knowledge does NOT count as support.
- Ignore generic safety boilerplate (e.g. "this is not a medical diagnosis, \
consult a professional") — it is not a factual claim to grade.
- A claim is supported only if the CONTEXT states or directly implies it.

Return STRICT JSON, no prose, in exactly this shape:
{{"claims": [{{"claim": "<short paraphrase>", "supported": true|false}}]}}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def _parse_judge_json(raw: str) -> list[dict] | None:
    # Strip ```json ... ``` fences some models wrap JSON in, then take the
    # outermost {...} object and parse it.
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    claims = data.get("claims")
    if not isinstance(claims, list):
        return None
    # Keep only well-formed claim entries (dict with a boolean-ish "supported").
    return [c for c in claims if isinstance(c, dict) and "supported" in c]


def judge_answer(context: str, answer: str, judge_model: str) -> list[dict] | None:
    """Return the judge's per-claim verdicts, or None if it couldn't be parsed.

    One repair retry: LLM-as-judge JSON is flaky, so on a parse miss we re-ask
    once with a stricter "JSON only" nudge before giving up.
    """
    from llm.client import get_llm

    prompt = _JUDGE_PROMPT.format(context=context, answer=answer)
    raw = get_llm().generate(prompt, model=judge_model, temperature=0.0)
    verdicts = _parse_judge_json(raw)
    if verdicts is not None:
        return verdicts
    raw = get_llm().generate(
        prompt + "\n\nReturn ONLY the JSON object, no prose, no code fences.",
        model=judge_model,
        temperature=0.0,
    )
    return _parse_judge_json(raw)


def evaluate(
    cases: list[dict],
    generator_model: str | None = None,
    judge_model: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run generation + judging over the cases and aggregate faithfulness."""
    import assistant
    from config import settings
    from llm.client import get_llm
    from rag.pipeline import format_context, retrieve_context

    judge_model = judge_model or settings.ollama_judge_model
    if limit:
        cases = cases[:limit]

    details = []
    faithfulness_sum = 0.0
    scored = 0
    latencies: list[float] = []
    for case in cases:
        query = case["query"]
        prep = assistant.prepare(query, assistant.MODE_QA, use_triage=False)
        if prep.messages is None:  # refused / no grounding — nothing to judge
            details.append({"query": query, "status": "no_grounding"})
            continue
        t0 = time.perf_counter()
        answer = get_llm().chat(prep.messages, model=generator_model)
        gen_latency = time.perf_counter() - t0
        latencies.append(gen_latency)
        context = format_context(retrieve_context(query))
        verdicts = judge_answer(context, answer, judge_model)
        if not verdicts:
            details.append({"query": query, "status": "judge_unparsed",
                            "gen_latency_s": gen_latency})
            continue
        supported = sum(1 for v in verdicts if v.get("supported"))
        total = len(verdicts)
        faith = supported / total if total else 1.0
        faithfulness_sum += faith
        scored += 1
        details.append({
            "query": query, "status": "ok", "faithfulness": faith,
            "supported": supported, "total": total, "gen_latency_s": gen_latency,
            "unsupported_claims": [v.get("claim") for v in verdicts if not v.get("supported")],
        })

    return {
        "n_cases": len(cases),
        "n_scored": scored,
        "mean_faithfulness": (faithfulness_sum / scored) if scored else None,
        "generator_model": generator_model or settings.ollama_model,
        "judge_model": judge_model,
        "mean_gen_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "p95_gen_latency_s": (sorted(latencies)[int(0.95 * (len(latencies) - 1))]
                              if latencies else None),
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Groundedness / faithfulness (LLM-as-judge).")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--generator", default=None, help="generator model (default: configured)")
    ap.add_argument("--judge", default=None, help="judge model (default: OLLAMA_JUDGE_MODEL)")
    args = ap.parse_args(argv)

    from llm.client import get_llm

    if not get_llm().health():
        print("Groundedness eval skipped: Ollama server not reachable.")
        return 0
    if not args.cases.exists():
        print(f"No groundedness cases at {args.cases}")
        return 0

    cases = load_cases(args.cases)
    r = evaluate(cases, args.generator, args.judge, args.limit)
    mf = r["mean_faithfulness"]
    mf_str = f"{mf:.2f}" if mf is not None else "n/a"
    print(f"Groundedness over {r['n_scored']}/{r['n_cases']} cases  "
          f"mean_faithfulness={mf_str}  (judge: {r['judge_model']})")
    for d in r["details"]:
        if d["status"] != "ok":
            print(f"  [{d['status']:>14}] {d['query']}")
            continue
        print(f"  [{d['faithfulness']:.2f} {d['supported']}/{d['total']}] {d['query']}")
        for c in d["unsupported_claims"]:
            print(f"        unsupported: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
