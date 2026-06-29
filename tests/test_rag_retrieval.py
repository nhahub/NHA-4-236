"""Unit tests for ingestion/chunking and the retrieval fusion logic.

These run without the heavy embedding/LLM models or a built index — they cover
the deterministic pieces (chunking, MedQuAD parsing, RRF fusion).
"""
from __future__ import annotations

import textwrap

from rag.ingest import chunk_text, parse_medquad


def test_chunk_text_respects_target_size():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    # No chunk should be wildly larger than target (sentence packing + overlap).
    assert all(len(c.split()) <= 160 for c in chunks)


def test_chunk_text_short_input_single_chunk():
    assert chunk_text("A short sentence.", target_tokens=400) == ["A short sentence."]


def test_parse_medquad_skips_empty_answers(tmp_path):
    xml = textwrap.dedent(
        """\
        <Document id="1" source="NIH" url="https://example.org/flu">
          <Focus>Influenza</Focus>
          <QAPairs>
            <QAPair>
              <Question qtype="information">What is influenza?</Question>
              <Answer>Influenza is a contagious respiratory illness caused by viruses.</Answer>
            </QAPair>
            <QAPair>
              <Question qtype="treatment">How is it treated?</Question>
              <Answer></Answer>
            </QAPair>
          </QAPairs>
        </Document>
        """
    )
    (tmp_path / "flu.xml").write_text(xml, encoding="utf-8")

    passages = parse_medquad(tmp_path)

    assert len(passages) == 1  # the empty-answer pair is dropped
    p = passages[0]
    assert p.title == "Influenza"
    assert p.url == "https://example.org/flu"
    assert p.source == "NIH"
    assert "respiratory illness" in p.text


def test_rrf_fusion_combines_rankings(monkeypatch):
    """The fusion math should reward passages ranked highly by both retrievers."""
    from rag import retriever as retriever_module

    # Build a retriever without touching disk/models.
    r = retriever_module.HybridRetriever.__new__(retriever_module.HybridRetriever)
    r.rrf_k = 60
    r.passages = [
        {
            "id": str(i),
            "text": f"passage {i}",
            "question": "",
            "title": f"T{i}",
            "url": "",
            "source": "test",
            "qtype": "general",
        }
        for i in range(5)
    ]
    monkeypatch.setattr(r, "_dense_ranking", lambda q, k: [0, 1, 2])
    monkeypatch.setattr(r, "_sparse_ranking", lambda q, k: [2, 0, 4])

    from config import settings

    settings.dense_weight = 0.5
    settings.bm25_weight = 0.5

    results = r.retrieve("anything", top_k=5)
    ids = [x.id for x in results]

    # Passage 0 (rank 1 dense, rank 2 sparse) and 2 (rank 3 dense, rank 1 sparse)
    # should outrank passages found by only one retriever.
    assert ids[0] in {"0", "2"}
    assert set(ids[:3]) == {"0", "1", "2"}


# --- Embedding-model mismatch guard --------------------------------------
def test_embedder_mismatch_raises(tmp_path, monkeypatch):
    """A query embedder different from the index builder must fail loudly."""
    import json
    import config
    from rag.retriever import HybridRetriever

    meta = tmp_path / "index_meta.json"
    meta.write_text(json.dumps({"embedding_model": "model-A", "dim": 768}))
    monkeypatch.setattr("rag.retriever.INDEX_META_PATH", meta)
    monkeypatch.setattr(config.settings, "embedding_model", "model-B")

    r = HybridRetriever.__new__(HybridRetriever)
    try:
        r._check_embedder_matches_index()
        raise AssertionError("expected RuntimeError on embedder mismatch")
    except RuntimeError as e:
        assert "model-A" in str(e) and "model-B" in str(e)


def test_embedder_match_passes(tmp_path, monkeypatch):
    import json
    import config
    from rag.retriever import HybridRetriever

    meta = tmp_path / "index_meta.json"
    meta.write_text(json.dumps({"embedding_model": "same-model", "dim": 768}))
    monkeypatch.setattr("rag.retriever.INDEX_META_PATH", meta)
    monkeypatch.setattr(config.settings, "embedding_model", "same-model")

    r = HybridRetriever.__new__(HybridRetriever)
    r._check_embedder_matches_index()  # must not raise


def test_missing_meta_is_allowed(tmp_path, monkeypatch):
    """Indexes built before the sidecar existed load without the check."""
    from rag.retriever import HybridRetriever

    monkeypatch.setattr("rag.retriever.INDEX_META_PATH", tmp_path / "absent.json")
    r = HybridRetriever.__new__(HybridRetriever)
    r._check_embedder_matches_index()  # must not raise
