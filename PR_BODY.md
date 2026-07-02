# P0–P3 credibility pass — from "impressive demo" to honest & measurable

Turns the RAG+LLM pillar from eyeballed to **measured**, fixes the broken ML
pillar (train/serve mismatch), wires the imaging models into the hybrid, hardens
the safety net, and documents everything for reproduction and interview defence.

**28 commits. Suite: 241 passed, 1 skipped.** Educational use only — not a medical device.

## Results at a glance
Every number is produced by the eval harness (`python -m eval`), not eyeballed.

| What | Metric |
|------|--------|
| Retrieval | recall@5 **0.97**, MRR **0.90** (30 cases) |
| Symptom ML (free-text, train==serve) | top-1 **0.915**, top-3 **1.00** |
| Grounding / hallucination | faithfulness **0.85** (LLM-as-judge) |
| Emergency triage | sensitivity **1.00**, specificity **1.00** (38 cases) |
| Prompt-injection resistance | **8/8 (100%)** |
| Latency (QA, CPU) | end-to-end p50 **39s** / p95 60s |

## The headline: the ML pillar is genuinely fixed (train == serve)
The DDXPlus **XGBoost** was demoted for a train/serve mismatch — trained on
complete structured evidence vectors but served ~3 regex-parsed symptoms
(everything else "absent") → out-of-distribution → confidently wrong (flu → "TB
91%"). It's replaced on the live path by a **free-text classifier**: it embeds the
raw symptom text with the retriever's **own S-PubMedBert encoder** → logistic-
regression head → 22 conditions (`gretelai/symptom_to_diagnosis`). Because train ==
serve, its held-out 0.915 top-1 reflects live behaviour. It biases the RAG recall
query, is cross-checked against retrieved passages, and **abstains** when
unconfident. XGBoost is kept as a standalone portfolio artifact (notebook + CLI),
fully off the live path. The 3 scan models (MRI/EEG/ECG) feed the same hybrid loop
via `scan_findings`.

## P0 — honesty & safety
- **Citation-integrity pass** — strips invented `[n]` markers and prunes citations
  to those actually referenced (blocking + streaming paths).
- **XGBoost off the live path**; **MRI out-of-distribution guard** (non-MRI →
  "not a recognized brain MRI"); **`experimental` label** on every scan output;
  env/launch **healthcheck** + `python -m` launch docs.

## P1 — measured (one command: `python -m eval`)
- **Retrieval quality**: diversified reranked passages (no more "UTI×3"), a
  **two-stage grounding gate** (off-topic vs medical-but-uncovered via a lexical
  `looks_medical` signal), a context-token budget, and fusion weights **chosen by a
  sweep** (dense/BM25 0.4/0.6, not guessed). `num_predict` 512→768.
- **Groundedness** faithfulness 0.85 (ragas-style LLM-judge); **triage** sens/spec
  1.00 (the eval **caught a missed stroke**, fixed + regression-guarded); **model
  choice by data** (qwen3:1.7b 0.75@53s vs llama3.1:8b 0.86@94s); **latency** p50 39s.

## P2 / P3
- **Prompt-injection / identity suite** — identity attacks caught deterministically;
  disclaimer code-enforced; behavioural eval **8/8** (`eval.injection`).
- **`prepare()` decomposed** into individually-tested helpers (pure, no behaviour
  change). **Free-text symptom model** (the ML fix above) — the P3 "real fix".
- Hosting-only P2 (Redis/SQLite/auth/observability) intentionally skipped — this is
  a local, educational deliverable.

## Fixes surfaced by manual testing
- **Attached scans no longer refused** — a scan-only upload was declined by the
  grounding gate (empty text → weak rerank); now reranks against the finding and
  never gate-declines when a study is attached. MRI → grounded answer + citations.
- **ECG/EEG mis-routing fixed** — a MIT-BIH-style dataset (~21000×188) was routed to
  the EEG model; now rejected with a clear message (pure, unit-tested
  `dashboard/signal_routing.py`).
- **Cardiac triage gap closed** — "heart ache / chest ache / my chest hurts" now
  trigger the emergency stop (was: only "chest pain"). One-word "heartache" and
  "chest tightness" correctly excluded.
- **Client-side crisis fallback** — if the API is unreachable, a self-harm/emergency
  message still surfaces crisis / urgent-care guidance (offline regex rules) instead
  of a bare "backend unreachable" error.
- `/health` reports the live classifier; imaging labels reworded to plain, non-emoji
  disclaimers.

## Documentation
- `docs/PROJECT_GUIDE.md` — mentor-grade walkthrough (architecture, every module,
  the request lifecycle, ML/RAG/LLM/safety deep-dives, interview Q&A, knowledge-gap
  report, learning roadmap, 2/5/10-min presentation scripts).
- `docs/MANUAL_TESTING.md` — type-this/expect-this verification checklist.
- `notebooks/text_classifier_analysis.ipynb` — the live model's data-science
  writeup (EDA, pipeline, per-class metrics, confusion matrix), companion to the
  existing XGBoost notebook.

## Notes
- Run tests/evals with `.venv/Scripts/python.exe`; groundedness/latency/injection/
  bench need a running Ollama (`qwen3:1.7b` + `llama3.1:8b`). Restart the app after
  pulling — code is loaded at process start and the response cache is in-memory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
