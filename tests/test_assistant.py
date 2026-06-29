"""Offline tests for streaming think-stripping and the response cache."""
from __future__ import annotations

import json

import pytest

import assistant
from llm.client import OllamaClient, _consume_think


# --- <think> stripping in the stream ------------------------------------
def test_consume_think_passes_plain_text():
    emit, pending, in_think = _consume_think("hello world", False)
    assert emit == "hello world"
    assert pending == ""
    assert in_think is False


def test_consume_think_strips_inline_block():
    emit, pending, in_think = _consume_think("<think>reason</think>answer", False)
    assert emit == "answer"
    assert in_think is False


def test_consume_think_handles_tag_split_across_chunks():
    # "<thi" arrives first and must be held back, not emitted as text.
    emit1, pending1, in_think1 = _consume_think("ok<thi", False)
    assert emit1 == "ok"
    assert pending1 == "<thi"
    assert in_think1 is False
    # Completing the open tag flips us into the think block.
    emit2, pending2, in_think2 = _consume_think(pending1 + "nk>secret", in_think1)
    assert emit2 == ""
    assert in_think2 is True


def _fake_stream(monkeypatch, deltas):
    """Patch httpx.stream so chat_stream reads our canned NDJSON lines."""
    lines = [json.dumps({"message": {"content": d}}) for d in deltas]

    class _Resp:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield from lines

    class _Ctx:
        def __enter__(self):
            return _Resp()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "llm.client.httpx.stream", lambda *a, **k: _Ctx()
    )


def test_chat_stream_concatenates_and_strips_thinking(monkeypatch):
    _fake_stream(
        monkeypatch,
        ["<think>", "step ", "one</think>", "The ", "answer", " is 42."],
    )
    out = "".join(OllamaClient().chat_stream([{"role": "user", "content": "x"}]))
    assert out == "The answer is 42."


# --- Response cache ------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_cache():
    assistant._cache.clear()
    yield
    assistant._cache.clear()


def test_repeat_query_is_served_from_cache(monkeypatch):
    calls = {"prepare": 0, "chat": 0}

    def fake_prepare(query, mode_hint, use_triage, patient=None, history=None, structured=False):
        calls["prepare"] += 1
        return assistant.Prepared(
            emergency=False,
            triage={"emergency": False, "source": "rules"},
            citations=[],
            messages=[{"role": "user", "content": query}],
        )

    class _LLM:
        def chat(self, messages, **kw):
            calls["chat"] += 1
            return "cached answer"

    monkeypatch.setattr(assistant, "prepare", fake_prepare)
    monkeypatch.setattr(assistant, "get_llm", lambda: _LLM())

    first = assistant.answer_question("What is anemia?")
    second = assistant.answer_question("What is anemia?")

    assert first.answer == second.answer
    assert "cached answer" in first.answer
    assert calls["prepare"] == 1  # second call skipped preparation
    assert calls["chat"] == 1  # and skipped the LLM


# --- Disclaimer guarantee -----------------------------------------------
def test_ensure_disclaimer_appends_when_missing():
    out = assistant.ensure_disclaimer("Anemia is low red blood cells [1].")
    assert out.endswith(assistant.DISCLAIMER)
    assert "not a medical diagnosis" in out.lower()


def test_ensure_disclaimer_is_idempotent():
    once = assistant.ensure_disclaimer("Some answer.")
    twice = assistant.ensure_disclaimer(once)
    assert once == twice
    assert twice.lower().count("not a medical diagnosis") == 1


def test_generated_answer_gets_disclaimer(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "prepare",
        lambda q, m, t, p=None, h=None, structured=False: assistant.Prepared(
            emergency=False,
            triage={"source": "rules"},
            citations=[],
            messages=[{"role": "user", "content": q}],
        ),
    )
    monkeypatch.setattr(
        assistant,
        "get_llm",
        lambda: type("X", (), {"chat": staticmethod(lambda m, **k: "Truncated answer")})(),
    )
    resp = assistant.answer_question("what is anemia")
    assert resp.answer.endswith(assistant.DISCLAIMER)


# --- Confidence gate ----------------------------------------------------
def _passage(score: float):
    from rag.retriever import RetrievedPassage

    return RetrievedPassage(
        id="p1",
        text="some grounding text",
        question="",
        title="Anemia",
        url="https://x.org",
        source="NIH",
        qtype="info",
        score=score,
    )


def test_gate_declines_when_top_score_below_floor(monkeypatch):
    monkeypatch.setattr(assistant.settings, "rerank_score_floor", -3.0)
    monkeypatch.setattr(assistant, "retrieve_context", lambda q, **kw: [_passage(-5.0)])
    prep = assistant.prepare("what is qwerty", assistant.MODE_QA, use_triage=False)
    assert prep.messages is None
    assert prep.citations == []
    assert prep.static_answer == assistant.NO_GROUNDING_MESSAGE


def test_gate_allows_when_top_score_above_floor(monkeypatch):
    monkeypatch.setattr(assistant.settings, "rerank_score_floor", -3.0)
    monkeypatch.setattr(assistant, "retrieve_context", lambda q, **kw: [_passage(8.0)])
    prep = assistant.prepare("what is anemia", assistant.MODE_QA, use_triage=False)
    assert prep.messages is not None
    assert len(prep.citations) == 1


# --- Multi-turn history --------------------------------------------------
def test_history_folds_into_retrieval_and_messages(monkeypatch):
    monkeypatch.setattr(assistant.settings, "rerank_score_floor", -100.0)
    seen = {}

    def fake_retrieve(q, **kw):
        seen["q"] = q
        return [_passage(8.0)]

    monkeypatch.setattr(assistant, "retrieve_context", fake_retrieve)
    history = [
        {"role": "user", "content": "i have stomach pain"},
        {"role": "assistant", "content": "Could you tell me the duration?"},
    ]
    prep = assistant.prepare(
        "3 days", assistant.MODE_SYMPTOM, use_triage=False, history=history
    )
    # Retrieval query folds in the prior user turn so the follow-up retrieves.
    assert "stomach pain" in seen["q"] and "3 days" in seen["q"]
    # Prior turns are replayed to the LLM before the final templated message.
    assert [m["role"] for m in prep.messages] == ["system", "user", "assistant", "user"]


def test_cache_key_distinguishes_history():
    k1 = assistant._cache_key("3 days", assistant.MODE_SYMPTOM, True, None, None)
    k2 = assistant._cache_key(
        "3 days", assistant.MODE_SYMPTOM, True, None,
        [{"role": "user", "content": "stomach pain"}],
    )
    assert k1 != k2


def test_self_harm_routes_to_crisis_message():
    from safety.red_flag_detector import SELF_HARM_MESSAGE

    # Rule-based triage fires (no LLM / no retrieval), so this is deterministic.
    resp = assistant.answer_question("I am thinking about suicide")
    assert resp.emergency is True
    assert resp.answer == SELF_HARM_MESSAGE
    assert resp.citations == []


def test_emergency_is_not_cached(monkeypatch):
    def fake_prepare(query, mode_hint, use_triage, patient=None, history=None, structured=False):
        return assistant.Prepared(
            emergency=True,
            triage={"emergency": True, "source": "rules"},
            citations=[],
            messages=None,
            static_answer="call 911",
        )

    monkeypatch.setattr(assistant, "prepare", fake_prepare)
    resp = assistant.answer_question("I have crushing chest pain")
    assert resp.emergency is True
    assert (
        assistant.cached_response("I have crushing chest pain", assistant.MODE_QA, True)
        is None
    )


# ---------------------------------------------------------------------------
# Fix 1-3: ml_predictions wiring through AssistantResponse
# ---------------------------------------------------------------------------

def _make_static_prep(ml_preds=None):
    return assistant.Prepared(
        emergency=False,
        triage={"emergency": False, "reason": "none", "confidence": 0.0, "source": "none"},
        citations=[],
        messages=None,
        static_answer="Static test answer.",
        ml_predictions=ml_preds,
    )


def test_assistant_response_has_ml_predictions_field():
    from dataclasses import fields
    field_names = {f.name for f in fields(assistant.AssistantResponse)}
    assert "ml_predictions" in field_names


def test_record_stream_propagates_ml_predictions():
    preds = [{"disease": "Influenza", "probability": 0.72}]
    prep = _make_static_prep(ml_preds=preds)
    resp = assistant.record_stream(
        query="flu test", mode_hint=assistant.MODE_SYMPTOM, use_triage=False,
        prep=prep, answer="Static test answer.",
    )
    assert resp.ml_predictions == preds


def test_record_stream_ml_predictions_empty_when_none():
    prep = _make_static_prep(ml_preds=None)
    resp = assistant.record_stream(
        query="flu test2", mode_hint=assistant.MODE_SYMPTOM, use_triage=False,
        prep=prep, answer="Static test answer.",
    )
    assert resp.ml_predictions == []


def test_to_dict_includes_ml_predictions():
    preds = [{"disease": "COVID-19", "probability": 0.55}]
    prep = _make_static_prep(ml_preds=preds)
    resp = assistant.record_stream(
        query="flu test3", mode_hint=assistant.MODE_SYMPTOM, use_triage=False,
        prep=prep, answer="Static test answer.",
    )
    d = resp.to_dict()
    assert "ml_predictions" in d
    assert d["ml_predictions"] == preds


def test_answer_propagates_ml_predictions(monkeypatch):
    """_answer() must copy ml_predictions from prep into AssistantResponse."""
    preds = [{"disease": "Bronchitis", "probability": 0.60}]
    prep = _make_static_prep(ml_preds=preds)
    monkeypatch.setattr(assistant, "prepare", lambda *a, **kw: prep)
    resp = assistant.answer_question("I have a cough")
    assert resp.ml_predictions == preds


# ---------------------------------------------------------------------------
# Fix 4-5: Streaming routes cache via record_stream and include ml_predictions
# ---------------------------------------------------------------------------

def _run_sse(coro):
    """Collect all parsed SSE data events from an async streaming route."""
    import asyncio

    async def _collect():
        response = await coro
        events = []
        async for chunk in response.body_iterator:
            text = chunk if isinstance(chunk, str) else chunk.decode()
            for line in text.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    # asyncio.run() is self-contained (creates and closes its own loop), so this
    # is immune to other tests that close the default loop via asyncio.run().
    return asyncio.run(_collect())


class _FakeRequest:
    """Stands in for the Starlette Request the stream routes take; the client
    never disconnects in these tests."""

    async def is_disconnected(self):
        return False


def test_ask_stream_meta_includes_ml_predictions(monkeypatch):
    from api.routes.medical_qa import ask_stream
    from api.schemas import QueryRequest

    preds = [{"disease": "Influenza", "probability": 0.8}]
    prep = _make_static_prep(ml_preds=preds)
    monkeypatch.setattr(assistant, "prepare", lambda *a, **kw: prep)

    request = QueryRequest(query="what is flu", use_triage=False)
    events = _run_sse(ask_stream(request, _FakeRequest()))

    meta = next(e for e in events if e.get("done"))
    assert "ml_predictions" in meta
    assert meta["ml_predictions"] == preds


def test_symptom_check_stream_meta_includes_ml_predictions(monkeypatch):
    from api.routes.symptom_check import symptom_check_stream
    from api.schemas import SymptomCheckRequest

    preds = [{"disease": "Bronchitis", "probability": 0.6}]
    prep = _make_static_prep(ml_preds=preds)
    monkeypatch.setattr(assistant, "prepare", lambda *a, **kw: prep)

    request = SymptomCheckRequest(query="I have cough and fever", use_triage=False)
    events = _run_sse(symptom_check_stream(request, _FakeRequest()))

    meta = next(e for e in events if e.get("done"))
    assert "ml_predictions" in meta
    assert meta["ml_predictions"] == preds


def test_ask_stream_response_is_cached(monkeypatch):
    """After streaming, the same query should be served from cache."""
    prep = _make_static_prep(ml_preds=None)
    monkeypatch.setattr(assistant, "prepare", lambda *a, **kw: prep)

    from api.routes.medical_qa import ask_stream
    from api.schemas import QueryRequest

    request = QueryRequest(query="unique cache test query xyz", use_triage=False)
    _run_sse(ask_stream(request, _FakeRequest()))

    cached = assistant.cached_response(
        request.query, assistant.MODE_QA, False, None, None
    )
    assert cached is not None


# --- Phase 2: ML <-> RAG feedback loop -----------------------------------
class _FakePassage:
    """Duck-typed RetrievedPassage for offline assistant tests."""

    def __init__(self, title, text, score=5.0):
        self.title = title
        self.text = text
        self.score = score
        self.url = ""
        self.source = "test"
        self.id = "x"
        self.question = ""
        self.qtype = ""


def test_annotate_ml_support_flags_unsupported():
    from assistant import _annotate_ml_support

    passages = [_FakePassage("Influenza", "Influenza is a viral infection causing fever.")]
    preds = [
        {"disease": "Influenza", "probability": 0.7},
        {"disease": "Tuberculosis", "probability": 0.2},  # absent from passages
    ]
    _annotate_ml_support(preds, passages)
    assert preds[0]["supported"] is True
    assert preds[1]["supported"] is False


def test_annotate_ml_support_unverifiable_label_is_supported():
    """A label with no significant word tokens can't be checked -> not flagged."""
    from assistant import _annotate_ml_support

    passages = [_FakePassage("Something", "unrelated text")]
    preds = [{"disease": "URTI", "probability": 0.5}]  # all-caps abbreviation < 4 chars
    _annotate_ml_support(preds, passages)
    assert preds[0]["supported"] is True


def _stub_symptom_path(monkeypatch, preds):
    """Force the SYMPTOM intent + ML predictions through prepare() offline."""
    import assistant as a
    from safety import intent as intent_router

    monkeypatch.setattr(intent_router, "classify_intent", lambda q, m: intent_router.SYMPTOM)
    monkeypatch.setattr(a, "_ML_AVAILABLE", True)
    monkeypatch.setattr(
        a, "_symptom_parser",
        type("P", (), {"parse": staticmethod(lambda q: {"features": [1, 2, 3]})}),
    )
    monkeypatch.setattr(
        a, "_ml_predict",
        type("M", (), {"predict": staticmethod(lambda f: [dict(p) for p in preds])}),
    )
    monkeypatch.setattr(a, "load_prompt", lambda name: "{context}\n\nQ:{query}")


def test_ml_disease_terms_appended_to_retrieval_query(monkeypatch):
    """ML -> RAG: top disease names widen the recall query, NOT the rerank query."""
    import assistant as a

    captured = {}

    def fake_retrieve(query, rerank_query=None, top_k=None, top_n=None):
        captured["query"] = query
        captured["rerank_query"] = rerank_query
        return [_FakePassage("Croup", "Croup causes a barking cough.")]

    _stub_symptom_path(monkeypatch, [{"disease": "Croup", "probability": 0.6}])
    monkeypatch.setattr(a, "retrieve_context", fake_retrieve)

    a.prepare("my child has a barking cough", a.MODE_SYMPTOM, use_triage=False)

    assert "Croup" in captured["query"]            # disease folded into recall
    assert captured["rerank_query"] == "my child has a barking cough"  # raw query preserved


def test_unsupported_high_conf_prediction_flagged_in_prompt(monkeypatch):
    """Retrieval -> ML: a confident prediction absent from context is flagged."""
    import assistant as a

    def fake_retrieve(query, rerank_query=None, top_k=None, top_n=None):
        return [_FakePassage("Common cold", "The common cold is a mild viral illness.")]

    _stub_symptom_path(monkeypatch, [{"disease": "Tuberculosis", "probability": 0.6}])
    monkeypatch.setattr(a, "retrieve_context", fake_retrieve)

    prep = a.prepare("persistent cough and night sweats", a.MODE_SYMPTOM, use_triage=False)
    prompt = prep.messages[-1]["content"]
    assert "Tuberculosis" in prompt
    assert "no supporting passage" in prompt


# --- Structured request logging ------------------------------------------
def test_prepare_logs_structured_line_for_chitchat(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="assistant"):
        assistant.prepare("hello there", assistant.MODE_QA, use_triage=False)
    msgs = [r.getMessage() for r in caplog.records if r.name == "assistant"]
    assert any("outcome=chitchat" in m and "latency_ms=" in m for m in msgs)


def test_prepare_logs_answered_with_ml_and_citations(monkeypatch, caplog):
    import logging
    import assistant as a

    def fake_retrieve(query, rerank_query=None, top_k=None, top_n=None):
        return [_FakePassage("Croup", "Croup causes a barking cough.")]

    _stub_symptom_path(monkeypatch, [{"disease": "Croup", "probability": 0.6}])
    monkeypatch.setattr(a, "retrieve_context", fake_retrieve)

    with caplog.at_level(logging.INFO, logger="assistant"):
        a.prepare("barking cough", a.MODE_SYMPTOM, use_triage=False)
    line = next(r.getMessage() for r in caplog.records if "outcome=answered" in r.getMessage())
    assert "ml_matched=1" in line
    assert "citations=1" in line


def test_symptom_stream_parses_structured_differential(monkeypatch):
    """structured=True over the stream path must parse the JSON into meta
    (previously always None because record_stream doesn't parse it)."""
    from api.routes.symptom_check import symptom_check_stream
    from api.schemas import SymptomCheckRequest

    monkeypatch.setattr(assistant, "cached_response", lambda *a, **k: None)
    prep = assistant.Prepared(
        emergency=False,
        triage={"emergency": False, "reason": "none", "confidence": 0.0, "source": "none"},
        citations=[],
        messages=[{"role": "user", "content": "x"}],
        ml_predictions=[],
    )
    monkeypatch.setattr(assistant, "prepare", lambda *a, **kw: prep)
    monkeypatch.setattr(
        assistant, "stream_tokens", lambda p: iter(['{"conditions": ["Influenza"]}'])
    )

    request = SymptomCheckRequest(query="cough and fever", use_triage=False, structured=True)
    events = _run_sse(symptom_check_stream(request, _FakeRequest()))
    meta = next(e for e in events if e.get("done"))
    assert meta["structured_differential"] == {"conditions": ["Influenza"]}
