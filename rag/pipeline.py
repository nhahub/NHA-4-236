"""Retrieval pipeline: query -> hybrid retrieve -> rerank -> formatted context.

Kept separate from the LLM and API layers so it can be unit-tested in isolation
and reused by both the FastAPI routes and the Streamlit demo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from config import settings
from rag.reranker import maybe_rerank
from rag.retriever import RetrievedPassage, get_retriever

logger = logging.getLogger("rag.pipeline")

# Rough chars-per-token for English text; good enough for a budget guard without
# pulling in the model's real tokenizer (which we don't have for Ollama models).
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def apply_context_budget(
    passages: list[RetrievedPassage], max_tokens: int
) -> tuple[list[RetrievedPassage], bool]:
    """Trim passages so their combined text fits ``max_tokens`` (estimated).

    Passages are kept in rank order; once the running budget is exhausted the
    next passage is truncated to whatever fits (on a word boundary) and the rest
    are dropped. Returns ``(kept, trimmed)`` where ``trimmed`` flags that any
    text was cut, so the caller can log it. ``max_tokens <= 0`` disables the cap.
    Truncation copies the passage (the shared retriever objects aren't mutated).
    """
    if max_tokens <= 0 or not passages:
        return passages, False

    kept: list[RetrievedPassage] = []
    used = 0
    trimmed = False
    for p in passages:
        cost = estimate_tokens(p.text)
        if used + cost <= max_tokens:
            kept.append(p)
            used += cost
            continue
        remaining = max_tokens - used
        if remaining > 0:  # partially fit this passage, truncated to a word edge
            cut_chars = remaining * _CHARS_PER_TOKEN
            text = p.text[:cut_chars].rsplit(" ", 1)[0].rstrip()
            if text:
                kept.append(replace(p, text=text + " …"))
        trimmed = True
        break  # budget exhausted; drop any remaining passages

    return kept, trimmed


@dataclass
class Citation:
    """A source reference surfaced alongside an answer."""

    index: int
    title: str
    url: str
    source: str


def retrieve_context(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
    rerank_query: str | None = None,
) -> list[RetrievedPassage]:
    """Run hybrid retrieval followed by optional cross-encoder reranking.

    ``query`` is used for retrieval (may include patient hints / history context
    to widen recall). ``rerank_query`` is used for reranking and should be the
    clean symptom-only user message — demographic phrases confuse the
    cross-encoder and tank scores for legitimate queries.
    """
    retriever = get_retriever()
    candidates = retriever.retrieve(query, top_k=top_k or settings.retrieval_top_k)
    return maybe_rerank(
        rerank_query or query, candidates, top_n=top_n or settings.rerank_top_n
    )


def format_context(passages: list[RetrievedPassage]) -> str:
    """Render passages as a numbered, citable context block for the prompt."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] {p.title}".strip()
        if p.url:
            header += f" ({p.url})"
        blocks.append(f"{header}\n{p.text}")
    return "\n\n".join(blocks)


def build_citations(passages: list[RetrievedPassage]) -> list[Citation]:
    return [
        Citation(index=i, title=p.title, url=p.url, source=p.source)
        for i, p in enumerate(passages, start=1)
    ]
