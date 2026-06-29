"""Ragas-based faithfulness / answer-relevance / context-precision eval.

This is an *integration* eval, not a unit test: it needs Ragas installed and a
live LLM + embeddings. It is skipped automatically when those aren't available
(e.g. on a minimal CI runner) so the suite still passes. Run it locally after
building the index and starting Ollama:

    RUN_RAGAS=1 pytest tests/test_faithfulness.py
"""
from __future__ import annotations

import os

import pytest

ragas = pytest.importorskip("ragas", reason="ragas not installed")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAGAS") != "1",
    reason="Set RUN_RAGAS=1 to run the (slow, model-dependent) faithfulness eval.",
)

# A tiny gold set; expand with real MedQuAD QA pairs for a meaningful score.
EVAL_QUESTIONS = [
    "What is influenza?",
    "What are common symptoms of type 2 diabetes?",
]

# Minimum acceptable scores — tune to your stack. CI can gate PRs on these.
THRESHOLDS = {
    "faithfulness": 0.6,
    "answer_relevancy": 0.6,
    "context_precision": 0.5,
}


def test_faithfulness_above_threshold():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    from assistant import answer_question
    from rag.pipeline import retrieve_context

    rows = []
    for q in EVAL_QUESTIONS:
        passages = retrieve_context(q)
        resp = answer_question(q, use_triage=False)
        rows.append(
            {
                "question": q,
                "answer": resp.answer,
                "contexts": [p.text for p in passages],
                "ground_truth": "",
            }
        )

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    for metric, minimum in THRESHOLDS.items():
        score = result[metric]
        assert score >= minimum, f"{metric}={score:.3f} below threshold {minimum}"
