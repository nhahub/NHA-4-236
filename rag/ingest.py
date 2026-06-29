"""Build the knowledge base from one or more medical datasets.

Pipeline:
  1. Load the selected sources (MedQuAD by default; optionally PubMedQA,
     MedlinePlus, Symptom2Disease) and map each record onto the shared
     ``Passage`` schema, chunked into ~300-500 token passages with source
     metadata (title, url, source, qtype).
  2. Embed all passages and write a FAISS inner-product index plus a BM25
     corpus for hybrid retrieval.

MedQuAD alone is enough for a working system. The other sources are optional
and only loaded when their data is present in ``data/raw`` (and, for PubMedQA,
when ``datasets`` is installed) — see ``rag/sources.py``.

Run:
    python -m rag.ingest                              # MedQuAD only (default)
    python -m rag.ingest --sources medquad pubmedqa   # add PubMedQA
    python -m rag.ingest --sources all                # everything available
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from config import (
    BM25_CORPUS_PATH,
    FAISS_INDEX_PATH,
    INDEX_META_PATH,
    PASSAGES_PATH,
    RAW_DIR,
)


def _write_index_meta(embedder, n_passages: int) -> None:
    """Record the embedding model + dim that built the index, for load-time checks."""
    INDEX_META_PATH.write_text(
        json.dumps(
            {
                "embedding_model": embedder.model_name,
                "dim": embedder.dim,
                "n_passages": n_passages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

# Re-exported for backward compatibility (tests and external callers import
# these from rag.ingest).
from rag.chunking import Passage, chunk_text  # noqa: F401
from rag.sources import LOADERS, load_medquad as parse_medquad, load_sources  # noqa: F401


def build_index(passages: list[Passage]) -> None:
    """Embed passages and persist the FAISS index + BM25 corpus + passages."""
    import faiss  # imported here so parsing-only runs don't need faiss

    from rag.embeddings import get_embedding_model

    if not passages:
        raise SystemExit(
            "No passages loaded. Check that at least one source's data is in "
            "data/raw (e.g. run scripts/download_medquad.py)."
        )

    PASSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Persist passages (the source of truth for retrieval results).
    with PASSAGES_PATH.open("w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    # 2. Dense index (cosine via normalized vectors + inner product).
    embedder = get_embedding_model()
    vectors = embedder.encode([p.text for p in passages], show_progress=True)
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # 3. BM25 corpus (tokenized lazily at load time by the retriever).
    with BM25_CORPUS_PATH.open("w", encoding="utf-8") as fh:
        json.dump([p.text for p in passages], fh)

    # 4. Sidecar so retrieval can detect a query/index embedder mismatch.
    _write_index_meta(embedder, len(passages))

    print(
        f"Indexed {len(passages)} passages\n"
        f"  passages : {PASSAGES_PATH}\n"
        f"  faiss    : {FAISS_INDEX_PATH}\n"
        f"  bm25     : {BM25_CORPUS_PATH}"
    )


def append_index(new_passages: list[Passage]) -> None:
    """Embed only ``new_passages`` and append them to the existing index.

    Avoids re-embedding the whole corpus when adding a source. Requires an
    existing index built with the *same* embedding model (the dimension is
    checked; a mismatch aborts so you rebuild fully instead of corrupting the
    vector space). Passages whose id is already indexed are skipped.
    """
    import faiss

    from rag.embeddings import get_embedding_model

    for path in (PASSAGES_PATH, FAISS_INDEX_PATH, BM25_CORPUS_PATH):
        if not path.exists():
            raise SystemExit(
                f"No existing index ({path} missing). Run a full build first."
            )

    existing_ids = set()
    with PASSAGES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            existing_ids.add(json.loads(line)["id"])

    fresh = [p for p in new_passages if p.id not in existing_ids]
    if not fresh:
        print("Nothing to append — all passages already indexed.")
        return

    embedder = get_embedding_model()
    vectors = embedder.encode([p.text for p in fresh], show_progress=True)

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    if index.d != vectors.shape[1]:
        raise SystemExit(
            f"Embedding dim mismatch: index={index.d}, embedder={vectors.shape[1]}. "
            "The existing index was built with a different model — rebuild fully "
            "(`python -m rag.ingest --sources ...`) instead of appending."
        )
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with PASSAGES_PATH.open("a", encoding="utf-8") as fh:
        for p in fresh:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    with BM25_CORPUS_PATH.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    corpus.extend(p.text for p in fresh)
    with BM25_CORPUS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh)

    _write_index_meta(embedder, index.ntotal)

    print(f"Appended {len(fresh)} passages. Index now holds {index.ntotal}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the medical knowledge base.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=RAW_DIR,
        help="Directory containing the dataset dumps.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["medquad"],
        help=f"Sources to ingest. Options: {', '.join(LOADERS)}, or 'all'.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Embed only the new sources and append to the existing index "
        "(instead of rebuilding the whole index). Requires the same embedding "
        "model as the existing index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of passages (for quick smoke tests).",
    )
    args = parser.parse_args()

    sources = list(LOADERS) if args.sources == ["all"] else args.sources
    print(f"Loading sources: {', '.join(sources)}")
    passages = load_sources(sources, args.raw)
    if args.limit:
        passages = passages[: args.limit]

    if args.append:
        append_index(passages)
    else:
        build_index(passages)


if __name__ == "__main__":
    main()
