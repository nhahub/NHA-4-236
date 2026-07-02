"""FastAPI application entrypoint.

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET  /health         — liveness + dependency status
    POST /ask            — general medical Q&A
    POST /symptom-check  — symptom exploration / differential
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import assistant as _assistant
import ml_model.symptom_classifier as _text_predict

from api.routes import analysis, medical_qa, symptom_check
from api.schemas import HealthResponse
from config import FAISS_INDEX_PATH
from llm.client import get_llm

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter: max N requests per IP per window (seconds).
# Protects the local Ollama process from being overwhelmed if the API is ever
# exposed beyond localhost.
# ---------------------------------------------------------------------------
_RATE_LIMIT_REQUESTS = 10   # max requests
_RATE_LIMIT_WINDOW   = 60   # per this many seconds
_ip_timestamps: dict[str, deque] = defaultdict(deque)
_last_sweep = 0.0  # monotonic time of the last stale-entry sweep

# Endpoints that trigger an Ollama generation and must be throttled. The /stream
# variants matter most: the Streamlit UI now sends every request there, so
# omitting them would leave the primary traffic path unprotected.
_THROTTLED_PATHS = frozenset(
    {"/ask", "/symptom-check", "/ask/stream", "/symptom-check/stream"}
)


def _sweep_stale(now: float) -> None:
    """Drop IPs whose timestamps have all expired so the dict can't grow without
    bound. An IP that stops sending requests would otherwise keep its (stale)
    entry forever, since pruning only happens when that same IP returns."""
    stale = [
        ip for ip, dq in _ip_timestamps.items()
        if not dq or now - dq[-1] > _RATE_LIMIT_WINDOW
    ]
    for ip in stale:
        del _ip_timestamps[ip]


def _check_rate_limit(ip: str) -> bool:
    """Return True if the IP is within the allowed rate, False if throttled."""
    global _last_sweep
    now = time.monotonic()
    # Periodic global cleanup (at most once per window) to evict departed IPs.
    if now - _last_sweep > _RATE_LIMIT_WINDOW:
        _sweep_stale(now)
        _last_sweep = now
    dq = _ip_timestamps[ip]
    while dq and now - dq[0] > _RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT_REQUESTS:
        return False
    dq.append(now)
    return True

app = FastAPI(
    title="Medical RAG Assistant",
    description=(
        "LLM + RAG medical Q&A and symptom-exploration assistant, grounded in "
        "retrieved medical literature. Educational use only — not a diagnosis."
    ),
    version="0.1.0",
)

# Permissive CORS so the Streamlit demo (or any local UI) can call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Throttle generation endpoints (blocking + streaming) to protect Ollama."""
    if request.url.path in _THROTTLED_PATHS:
        ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a moment before trying again."},
            )
    return await call_next(request)


app.include_router(medical_qa.router, tags=["qa"])
app.include_router(symptom_check.router, tags=["symptom"])
app.include_router(analysis.router)


@app.on_event("startup")
def _env_healthcheck() -> None:
    """Warn loudly at boot if a runtime-critical dependency can't be imported
    from the current interpreter — catches a wrong/incomplete (e.g. copied)
    virtualenv before it fails deep inside a request. Warn-only: never blocks
    startup, and skipped under pytest."""
    import sys

    if "pytest" in sys.modules:
        return
    try:
        from scripts.healthcheck import check_dependencies, format_report

        missing_req, missing_opt = check_dependencies()
        if missing_req or missing_opt:
            print("[healthcheck]\n" + format_report(missing_req, missing_opt),
                  file=sys.stderr)
    except Exception:
        pass


@app.on_event("startup")
def _warmup_models() -> None:
    """Pre-load the embedder, reranker and LLM at boot so the first real request
    is fast — a cold start otherwise takes minutes on CPU. Runs in a daemon
    thread (the server is ready immediately) and is skipped under pytest so it
    never triggers heavy model loads during tests."""
    import sys
    import threading

    if "pytest" in sys.modules:
        return

    def _run() -> None:
        try:
            from rag.pipeline import retrieve_context

            retrieve_context("warmup")  # loads embedder + reranker + FAISS index
        except Exception:
            pass
        try:
            get_llm().generate("ok", temperature=0.0)  # loads the Ollama model
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="warmup").start()


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama=get_llm().health(),
        index_loaded=FAISS_INDEX_PATH.exists(),
        # Report the LIVE ML pillar (the free-text classifier), not the demoted
        # DDXPlus XGBoost — that's what setup --with-ml trains and what answers use.
        ml_model_loaded=_text_predict.text_artifacts_available(),
    )


@app.post("/admin/clear-cache", tags=["meta"])
async def clear_cache() -> dict:
    """Clear the in-memory response cache (useful after code/prompt changes)."""
    _assistant._cache.clear()
    return {"cleared": True}


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": "Medical RAG Assistant",
        "docs": "/docs",
        "endpoints": ["/ask", "/symptom-check", "/health"],
        "disclaimer": "Educational use only. Not a medical diagnosis.",
    }
