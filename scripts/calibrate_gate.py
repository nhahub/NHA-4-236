"""Calibrate the rerank confidence gate (``settings.rerank_score_floor``).

Runs the real retrieve + rerank pipeline over a handful of clearly on-topic
medical queries and clearly off-topic ones, and prints the top reranked score
for each. A good floor sits in the gap between the two clusters: above the
off-topic scores, below the on-topic ones.

Run (from the project root, with models cached):
    python -m scripts.calibrate_gate
"""
from __future__ import annotations

from rag.pipeline import retrieve_context
from safety.intent import CHITCHAT, QA, classify_intent

ON_TOPIC = [
    "what is anemia",
    "what causes high blood pressure",
    "symptoms of type 2 diabetes",
    "how is asthma treated",
    "what is indigestion",
    "i have a fever and a cough",
]

OFF_TOPIC = [
    "hi",
    "how do i make pasta",
    "what is the capital of egypt",
    "who won the world cup",
    "can i ask something else",
    "asdfghjkl",
]


def _top_score(query: str) -> float:
    passages = retrieve_context(query)
    return passages[0].score if passages else float("-inf")


def _reaches_gate(query: str) -> bool:
    """The gate only sees queries the intent router did NOT divert to chit-chat."""
    return classify_intent(query, QA) != CHITCHAT


def main() -> None:
    print("Loading models + index (first call is slow)...\n")
    print("(queries marked [router] are caught before the gate — excluded)\n")

    print("ON-TOPIC (want scores ABOVE the floor):")
    on = []
    for q in ON_TOPIC:
        s = _top_score(q)
        on.append(s)
        print(f"  {s:+8.3f}  {q}")

    print("\nOFF-TOPIC (want scores BELOW the floor):")
    off = []
    for q in OFF_TOPIC:
        s = _top_score(q)
        if not _reaches_gate(q):
            print(f"  {s:+8.3f}  {q}   [router: chit-chat — not gated]")
            continue
        off.append(s)
        print(f"  {s:+8.3f}  {q}")

    lo_on = min(on)
    hi_off = max(off)
    print("\n--- summary ---")
    print(f"  lowest on-topic : {lo_on:+.3f}")
    print(f"  highest off-topic: {hi_off:+.3f}")
    if lo_on > hi_off:
        suggested = round((lo_on + hi_off) / 2, 2)
        print(f"  clean gap -> suggested rerank_score_floor = {suggested}")
    else:
        print("  clusters OVERLAP — no perfect cutoff; pick to favour recall or"
              " precision and review the borderline queries above.")


if __name__ == "__main__":
    main()
