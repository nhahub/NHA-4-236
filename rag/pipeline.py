"""Retrieval pipeline: query -> hybrid retrieve -> rerank -> formatted context.

Kept separate from the LLM and API layers so it can be unit-tested in isolation
and reused by both the FastAPI routes and the Streamlit demo.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from rag.reranker import maybe_rerank
from rag.retriever import RetrievedPassage, get_retriever


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
