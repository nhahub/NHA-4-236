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


# --- reranker diversification -------------------------------------------
def _passage(title: str, text: str, score: float):
    from rag.retriever import RetrievedPassage

    return RetrievedPassage(
        id=title + text[:5], text=text, question="", title=title, url="",
        source="medquad", qtype="info", score=score,
    )


def test_diversify_prefers_distinct_titles():
    """The 'UTI x3' case: distinct sources should win the limited slots."""
    from rag.reranker import diversify

    passages = [
        _passage("UTI", "urinary tract infection symptoms include burning", 9.0),
        _passage("UTI", "urinary tract infection is treated with antibiotics", 8.5),
        _passage("UTI", "urinary tract infection prevention tips", 8.0),
        _passage("Anemia", "anemia is a low red blood cell count", 7.5),
        _passage("Stroke", "stroke is a medical emergency with FAST signs", 7.0),
    ]
    out = diversify(passages, top_n=3, jaccard_threshold=0.85)
    assert [p.title for p in out] == ["UTI", "Anemia", "Stroke"]


def test_diversify_backfills_when_not_enough_distinct():
    """If distinct sources can't fill top_n, dupes backfill (slot count kept)."""
    from rag.reranker import diversify

    passages = [
        _passage("UTI", "urinary tract infection symptoms one", 9.0),
        _passage("UTI", "urinary tract infection symptoms two", 8.0),
    ]
    out = diversify(passages, top_n=3, jaccard_threshold=0.85)
    assert len(out) == 2  # only two existed; no padding invented
    assert out[0].score == 9.0  # best-scored kept first


def test_diversify_dedups_near_identical_text_across_titles():
    """Same text under different titles is still a near-duplicate."""
    from rag.reranker import diversify

    shared = "the patient presents with fever cough fatigue and body aches today"
    passages = [
        _passage("Flu", shared, 9.0),
        _passage("Influenza", shared, 8.0),  # different title, identical text
        _passage("Cold", "a common cold causes a runny nose and sneezing", 7.0),
    ]
    out = diversify(passages, top_n=2, jaccard_threshold=0.85)
    assert [p.title for p in out] == ["Flu", "Cold"]


# --- medical-topicality (two-stage gate) --------------------------------
def test_looks_medical_true_for_health_queries():
    from rag.topicality import looks_medical

    assert looks_medical("what are IVF success rates by age")
    assert looks_medical("how is type 2 diabetes treated")
    assert looks_medical("persistent cough and fever for five days")
    assert looks_medical("symptoms of appendicitis")  # suffix -itis
    assert looks_medical("treatment for leukemia")     # suffix -emia


def test_looks_medical_false_for_off_topic_queries():
    from rag.topicality import looks_medical

    assert not looks_medical("what is the capital of Egypt")
    assert not looks_medical("who won the world cup in 2022")
    assert not looks_medical("write me a poem about the sea")
    assert not looks_medical("qwerty asdf zxcv")


# --- context-token budget ------------------------------------------------
def test_context_budget_keeps_all_when_under_limit():
    from rag.pipeline import apply_context_budget

    passages = [_passage("A", "short text", 1.0), _passage("B", "more text", 0.9)]
    kept, trimmed = apply_context_budget(passages, max_tokens=1000)
    assert kept == passages
    assert trimmed is False


def test_context_budget_drops_overflow_passages_and_flags_trim():
    from rag.pipeline import apply_context_budget

    # ~10 tokens each (40 chars / 4). Budget of 12 tokens fits one whole + a
    # truncated slice of the next.
    big = "word " * 12  # 60 chars -> ~15 tokens
    passages = [_passage("A", big, 1.0), _passage("B", big, 0.9), _passage("C", big, 0.8)]
    kept, trimmed = apply_context_budget(passages, max_tokens=18)
    assert trimmed is True
    assert len(kept) < len(passages)  # at least one dropped
    assert kept[0].title == "A"       # rank order preserved


def test_context_budget_disabled_with_nonpositive_limit():
    from rag.pipeline import apply_context_budget

    passages = [_passage("A", "x " * 500, 1.0)]
    kept, trimmed = apply_context_budget(passages, max_tokens=0)
    assert kept == passages and trimmed is False


# --- groundedness judge parsing (offline) -------------------------------
def test_judge_json_parses_plain_and_fenced():
    from eval.groundedness import _parse_judge_json

    plain = '{"claims": [{"claim": "x", "supported": true}, {"claim": "y", "supported": false}]}'
    assert _parse_judge_json(plain) == [
        {"claim": "x", "supported": True}, {"claim": "y", "supported": False}
    ]
    fenced = "```json\n" + plain + "\n```"
    assert len(_parse_judge_json(fenced)) == 2
    prose = "Here is my assessment:\n" + plain + "\nDone."
    assert len(_parse_judge_json(prose)) == 2


def test_judge_json_rejects_junk_and_malformed_entries():
    from eval.groundedness import _parse_judge_json

    assert _parse_judge_json("no json here") is None
    assert _parse_judge_json('{"claims": "not a list"}') is None
    # Entries missing "supported" are dropped, not crashed on.
    assert _parse_judge_json('{"claims": [{"claim": "x"}, {"claim": "y", "supported": true}]}') == [
        {"claim": "y", "supported": True}
    ]
