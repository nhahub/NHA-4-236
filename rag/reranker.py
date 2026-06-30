"""Cross-encoder re-ranking of candidate passages.

A bi-encoder retriever is fast but approximate. A cross-encoder scores each
(query, passage) pair jointly and is much more precise, so we use it to reorder
the retriever's top-k down to the final top-n that the LLM actually sees.

The reranker is optional (``USE_RERANKER``); when disabled, the retriever's
fusion order is passed through unchanged.
"""
from __future__ import annotations

from functools import lru_cache

from config import settings
from rag.retriever import RetrievedPassage


def _norm_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def diversify(
    passages: list[RetrievedPassage],
    top_n: int,
    jaccard_threshold: float,
) -> list[RetrievedPassage]:
    """Pick ``top_n`` passages preferring distinct sources over near-duplicates.

    ``passages`` must be pre-sorted by relevance (best first). A candidate is a
    duplicate of an already-selected passage when it shares the (non-empty) title
    or its text token-set overlaps above ``jaccard_threshold``. Duplicates are
    held back and only used to backfill if there aren't enough distinct sources
    to fill ``top_n`` — so the slot count is preserved, but distinct sources win.
    """
    selected: list[RetrievedPassage] = []
    seen_titles: set[str] = set()
    seen_tokens: list[set[str]] = []
    deferred: list[RetrievedPassage] = []

    for p in passages:
        if len(selected) >= top_n:
            break
        title = _norm_title(p.title)
        tokens = set(p.text.lower().split())
        title_dup = bool(title) and title in seen_titles
        text_dup = any(_jaccard(tokens, st) >= jaccard_threshold for st in seen_tokens)
        if title_dup or text_dup:
            deferred.append(p)
            continue
        selected.append(p)
        if title:
            seen_titles.add(title)
        seen_tokens.append(tokens)

    for p in deferred:  # backfill only if distinct sources were too few
        if len(selected) >= top_n:
            break
        selected.append(p)
    return selected[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        top_n: int | None = None,
    ) -> list[RetrievedPassage]:
        top_n = top_n or settings.rerank_top_n
        if not passages:
            return []

        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(passages, scores), key=lambda ps: ps[1], reverse=True
        )
        # Write the cross-encoder score back onto every candidate first, so the
        # gate (which reads passages[0].score) and diversification both see the
        # reranked score rather than the stale fusion score.
        scored: list[RetrievedPassage] = []
        for passage, score in reranked:
            passage.score = float(score)
            scored.append(passage)
        if settings.rerank_dedup:
            return diversify(scored, top_n, settings.dedup_jaccard_threshold)
        return scored[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def maybe_rerank(
    query: str,
    passages: list[RetrievedPassage],
    top_n: int | None = None,
) -> list[RetrievedPassage]:
    """Rerank when enabled, else truncate the fusion order to top_n."""
    top_n = top_n or settings.rerank_top_n
    if settings.use_reranker:
        return get_reranker().rerank(query, passages, top_n)
    return passages[:top_n]
