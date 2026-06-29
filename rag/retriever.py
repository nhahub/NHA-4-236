"""Hybrid dense + BM25 retrieval over the MedQuAD knowledge base.

Dense ranking comes from a FAISS inner-product index of normalized embeddings;
sparse ranking from a BM25 corpus. The two rankings are combined with
Reciprocal Rank Fusion (RRF), which is robust to the different score scales of
the two retrievers and needs no score normalization.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from config import (
    BM25_CORPUS_PATH,
    FAISS_INDEX_PATH,
    INDEX_META_PATH,
    PASSAGES_PATH,
    settings,
)
from rag.embeddings import get_embedding_model


@dataclass
class RetrievedPassage:
    """A passage returned from retrieval, with its fusion score."""

    id: str
    text: str
    question: str
    title: str
    url: str
    source: str
    qtype: str
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    """Loads the persisted index/corpus and serves fused top-k results."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self._load()

    def _load(self) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        for path in (PASSAGES_PATH, FAISS_INDEX_PATH, BM25_CORPUS_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing index artifact: {path}. Run `python -m rag.ingest` first."
                )

        self._check_embedder_matches_index()

        with PASSAGES_PATH.open(encoding="utf-8") as fh:
            self.passages = [json.loads(line) for line in fh]

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with BM25_CORPUS_PATH.open(encoding="utf-8") as fh:
            corpus = json.load(fh)
        self.bm25 = BM25Okapi([_tokenize(t) for t in corpus])

        self.embedder = get_embedding_model()

    def _check_embedder_matches_index(self) -> None:
        """Refuse to serve if the query embedder differs from the index builder.

        Two different models can share an embedding dimension, so the FAISS dim
        check alone won't catch a swapped ``EMBEDDING_MODEL`` — the vectors would
        live in an incompatible space and retrieval would silently return
        nonsense. The ingest step records the builder model in a sidecar; if it
        disagrees with the configured embedder, fail loudly with a fix. Older
        indexes built before the sidecar existed are allowed through unchecked.
        """
        if not INDEX_META_PATH.exists():
            return
        try:
            meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        built_with = meta.get("embedding_model")
        configured = settings.embedding_model
        if built_with and built_with != configured:
            raise RuntimeError(
                "Embedding model mismatch: the index was built with "
                f"'{built_with}' but EMBEDDING_MODEL is '{configured}'. "
                "Retrieval over a mismatched embedder returns garbage. Either "
                f"set EMBEDDING_MODEL='{built_with}' or rebuild the index with "
                "`python -m rag.ingest`."
            )

    def _dense_ranking(self, query: str, top_k: int) -> list[int]:
        vector = self.embedder.encode_one(query).reshape(1, -1)
        _scores, ids = self.index.search(vector, top_k)
        return [int(i) for i in ids[0] if i != -1]

    def _sparse_ranking(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedPassage]:
        """Return the top-k passages by reciprocal-rank fusion of both rankers."""
        top_k = top_k or settings.retrieval_top_k

        dense = self._dense_ranking(query, top_k)
        sparse = self._sparse_ranking(query, top_k)

        # Weighted Reciprocal Rank Fusion.
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + settings.dense_weight / (
                self.rrf_k + rank + 1
            )
        for rank, idx in enumerate(sparse):
            fused[idx] = fused.get(idx, 0.0) + settings.bm25_weight / (
                self.rrf_k + rank + 1
            )

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[RetrievedPassage] = []
        for idx, score in ordered:
            p = self.passages[idx]
            results.append(
                RetrievedPassage(
                    id=p["id"],
                    text=p["text"],
                    question=p["question"],
                    title=p["title"],
                    url=p["url"],
                    source=p["source"],
                    qtype=p["qtype"],
                    score=float(score),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Process-wide singleton retriever (index loaded once)."""
    return HybridRetriever()
