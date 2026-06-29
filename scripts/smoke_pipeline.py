"""End-to-end pipeline smoke test — run with: python scripts/smoke_pipeline.py

Named `smoke_pipeline` (not `test_`) on purpose: it is a manual script that runs
the full pipeline (and calls Ollama) at import, so it must never be collected as
a pytest test.
"""
# This script imports modules step-by-step between printed section headers (and
# after configuring stdout below), so module-level imports are intentionally not
# all at the top.
# ruff: noqa: E402
import io
import sys

# Force UTF-8 output so emoji in prompt templates don't crash on Windows cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

QUERY = "I have a sore throat, fever, and swollen glands for 3 days"

# ── 1. Retrieval ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Retrieval")
print("=" * 60)
from rag.pipeline import retrieve_context, format_context

passages = retrieve_context(QUERY)
print(f"Retrieved {len(passages)} passage(s)")
for p in passages:
    print(f"  score={p.score:.3f}  title={p.title[:65]}")

# ── 2. Context formatting ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2 — Formatted context (first 800 chars)")
print("=" * 60)
ctx = format_context(passages)
print(ctx[:800])

# ── 3. Prompt assembly ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 — Prompt assembly")
print("=" * 60)
from llm.client import load_prompt
template = load_prompt("symptom_prompt")
user_prompt = template.format(context=ctx, query=QUERY)
print(f"Prompt length: {len(user_prompt)} chars / ~{len(user_prompt)//4} tokens (estimate)")
print("--- Prompt tail (last 300 chars) ---")
print(user_prompt[-300:])

# ── 4. LLM call ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4 — LLM generation (this may take a minute on CPU)")
print("=" * 60)
from llm.client import get_llm, load_prompt as lp
system = lp("system_prompt")
messages = [
    {"role": "system", "content": system},
    {"role": "user",   "content": user_prompt},
]
llm = get_llm()
answer = llm.chat(messages)
print(answer)

# ── 5. Full assistant flow ────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5 — Full assistant.explore_symptoms()")
print("=" * 60)
from assistant import explore_symptoms
resp = explore_symptoms(QUERY)
print(f"emergency={resp.emergency}")
print(f"citations={resp.citations}")
print(f"triage={resp.triage}")
print()
print("--- Answer ---")
print(resp.answer)
