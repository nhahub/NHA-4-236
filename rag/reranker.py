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
        out: list[RetrievedPassage] = []
        for passage, score in reranked[:top_n]:
            passage.score = float(score)
            out.append(passage)
        return out


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
