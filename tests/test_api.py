"""API tests using FastAPI's TestClient with the assistant layer mocked.

The routes are exercised end-to-end (request validation, threadpool offload,
response shaping) while the retrieval + Ollama calls are stubbed, so these run
fast and offline in CI.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from assistant import AssistantResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Isolate each test from the shared in-memory rate limiter — otherwise the
    cumulative throttled calls across this file (now including the /stream
    endpoints) could trip the limit and cause spurious 429s."""
    import api.main as m

    m._ip_timestamps.clear()
    m._last_sweep = 0.0
    yield


@pytest.fixture
def stub_assistant(monkeypatch):
    def fake(*args, **kwargs) -> AssistantResponse:
        return AssistantResponse(
            answer="Grounded answer [1].\nThis is general information...",
            emergency=False,
            triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
            citations=[
                {"index": 1, "title": "Influenza", "url": "https://x.org", "source": "NIH"}
            ],
        )

    # The routes call these via `import assistant as _a` (e.g. _a.answer_question),
    # so patch the functions on the assistant module itself, not on the route
    # module. `fake` takes *args/**kwargs to tolerate the full call signatures
    # (including `structured` for explore_symptoms).
    monkeypatch.setattr("assistant.answer_question", fake)
    monkeypatch.setattr("assistant.explore_symptoms", fake)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "ollama" in body and "index_loaded" in body
    assert "ml_model_loaded" in body  # README advertises ML status here


def test_ask_returns_grounded_answer(stub_assistant):
    resp = client.post("/ask", json={"query": "What is the flu?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["emergency"] is False
    assert body["citations"][0]["title"] == "Influenza"
    assert "[1]" in body["answer"]


def test_symptom_check_returns_response(stub_assistant):
    resp = client.post("/symptom-check", json={"query": "fever and cough"})
    assert resp.status_code == 200
    assert resp.json()["triage"]["source"] == "rules"


def test_symptom_check_accepts_patient_info(stub_assistant):
    resp = client.post(
        "/symptom-check",
        json={
            "query": "stomach pain",
            "patient": {"age": 65, "sex": "female", "conditions": "diabetes"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["emergency"] is False


def test_symptom_check_rejects_bad_age(stub_assistant):
    resp = client.post(
        "/symptom-check",
        json={"query": "stomach pain", "patient": {"age": 999}},
    )
    assert resp.status_code == 422


def test_ask_accepts_history(stub_assistant):
    resp = client.post(
        "/ask",
        json={
            "query": "what about treatment",
            "history": [
                {"role": "user", "content": "what is anemia"},
                {"role": "assistant", "content": "Anemia is..."},
            ],
        },
    )
    assert resp.status_code == 200


def test_empty_query_rejected():
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


# --- Rate limiter: no unbounded growth -----------------------------------
def test_rate_limiter_evicts_stale_ips(monkeypatch):
    """IPs that stop sending requests must not accumulate forever."""
    import api.main as m

    m._ip_timestamps.clear()
    m._last_sweep = 0.0

    t = {"now": 1000.0}
    monkeypatch.setattr(m.time, "monotonic", lambda: t["now"])

    # 50 one-off IPs make a single request each.
    for i in range(50):
        assert m._check_rate_limit(f"10.0.0.{i}") is True
    assert len(m._ip_timestamps) == 50

    # Advance well past the window; a fresh request triggers the sweep.
    t["now"] += m._RATE_LIMIT_WINDOW + 5
    assert m._check_rate_limit("10.0.0.999") is True
    # All 50 departed IPs are gone; only the active one remains.
    assert len(m._ip_timestamps) == 1


def test_rate_limiter_still_throttles(monkeypatch):
    import api.main as m

    m._ip_timestamps.clear()
    m._last_sweep = 0.0
    monkeypatch.setattr(m.time, "monotonic", lambda: 5000.0)

    ip = "203.0.113.7"
    allowed = [m._check_rate_limit(ip) for _ in range(m._RATE_LIMIT_REQUESTS + 3)]
    assert allowed[: m._RATE_LIMIT_REQUESTS] == [True] * m._RATE_LIMIT_REQUESTS
    assert allowed[m._RATE_LIMIT_REQUESTS:] == [False, False, False]


# --- SSE streaming endpoints + cache pre-check ---------------------------
def test_ask_stream_serves_cache_without_generating(monkeypatch):
    """A cache hit streams the stored answer in one chunk and never prepares."""
    cached = AssistantResponse(
        answer="Cached flu answer.",
        emergency=False,
        triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
        citations=[],
        ml_predictions=[],
    )
    monkeypatch.setattr("assistant.cached_response", lambda *a, **k: cached)

    def _no_prepare(*a, **k):
        raise AssertionError("prepare must not run on a cache hit")

    monkeypatch.setattr("assistant.prepare", _no_prepare)

    resp = client.post("/ask/stream", json={"query": "what is the flu"})
    assert resp.status_code == 200
    assert "Cached flu answer." in resp.text
    assert '"done": true' in resp.text


def test_ask_stream_streams_tokens_then_meta(monkeypatch):
    from assistant import Prepared

    monkeypatch.setattr("assistant.cached_response", lambda *a, **k: None)
    prep = Prepared(
        emergency=False,
        triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
        citations=[{"index": 1, "title": "Influenza", "url": "https://x", "source": "NIH"}],
        messages=[{"role": "user", "content": "x"}],
        ml_predictions=[],
    )
    monkeypatch.setattr("assistant.prepare", lambda *a, **k: prep)
    monkeypatch.setattr("assistant.stream_tokens", lambda p: iter(["Influenza ", "is viral."]))

    resp = client.post("/ask/stream", json={"query": "what is the flu"})
    assert resp.status_code == 200
    body = resp.text
    assert "Influenza " in body and "is viral." in body  # streamed tokens
    assert '"done": true' in body                          # final meta event
    assert "Influenza" in body                             # citation in meta


def test_symptom_stream_cache_includes_structured_differential(monkeypatch):
    cached = AssistantResponse(
        answer="Cached differential.",
        emergency=False,
        triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
        citations=[],
        ml_predictions=[{"disease": "Croup", "probability": 0.6}],
        structured_differential={"conditions": ["Croup"]},
    )
    monkeypatch.setattr("assistant.cached_response", lambda *a, **k: cached)
    monkeypatch.setattr("assistant.prepare", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("prepare must not run on a cache hit")))

    resp = client.post("/symptom-check/stream", json={"query": "barking cough"})
    assert resp.status_code == 200
    assert "Cached differential." in resp.text
    assert "structured_differential" in resp.text
    assert "Croup" in resp.text


# --- Stop -> server-side cancellation (tokens_until_disconnect) -----------
def test_tokens_until_disconnect_stops_and_closes_upstream():
    """On client disconnect the loop stops early and closes the LLM generator."""
    import asyncio
    from api.sse import tokens_until_disconnect

    closed = {"v": False}

    def gen():
        try:
            for i in range(100):
                yield f"t{i}"
        finally:
            closed["v"] = True  # GeneratorExit when .close() is called

    # Disconnected on the 3rd check, so t0 and t1 are emitted, then we stop.
    checks = iter([False, False, True, True, True])

    class _Req:
        async def is_disconnected(self):
            return next(checks)

    async def _run():
        return [tok async for tok in tokens_until_disconnect(gen(), _Req())]

    out = asyncio.run(_run())
    assert out == ["t0", "t1"]
    assert closed["v"] is True


def test_tokens_until_disconnect_passes_all_when_connected():
    import asyncio
    from api.sse import tokens_until_disconnect

    closed = {"v": False}

    def gen():
        try:
            yield "a"
            yield "b"
            yield "c"
        finally:
            closed["v"] = True

    class _Req:
        async def is_disconnected(self):
            return False

    async def _run():
        return [tok async for tok in tokens_until_disconnect(gen(), _Req())]

    out = asyncio.run(_run())
    assert out == ["a", "b", "c"]
    assert closed["v"] is True  # closed on normal exhaustion too — no leak


def test_stream_endpoint_is_rate_limited(monkeypatch):
    """The /stream endpoints (the UI's only traffic path) must be throttled."""
    from api.main import _RATE_LIMIT_REQUESTS

    cached = AssistantResponse(
        answer="x",
        emergency=False,
        triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
        citations=[],
        ml_predictions=[],
    )
    monkeypatch.setattr("assistant.cached_response", lambda *a, **k: cached)

    codes = [
        client.post("/ask/stream", json={"query": "hi"}).status_code
        for _ in range(_RATE_LIMIT_REQUESTS + 1)
    ]
    assert codes.count(200) == _RATE_LIMIT_REQUESTS
    assert codes[-1] == 429


def test_ask_ignores_unknown_scan_findings_field(monkeypatch):
    """scan_findings is no longer part of the API — an extra field is ignored
    (pydantic drops unknowns) and the request still succeeds."""
    def fake(query, use_triage=True, history=None):
        return AssistantResponse(
            answer="ok", emergency=False,
            triage={"emergency": False, "reason": "ok", "confidence": 0.5, "source": "rules"},
            citations=[],
        )

    monkeypatch.setattr("assistant.answer_question", fake)
    r = client.post("/ask", json={
        "query": "what does my scan mean",
        "scan_findings": "Brain MRI analysis: meningioma (87% confidence).",  # ignored
    })
    assert r.status_code == 200
