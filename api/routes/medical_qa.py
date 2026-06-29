"""POST /ask — retrieve grounding passages and answer a medical question."""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

import assistant as _a
from api.schemas import AssistantResponseModel, QueryRequest
from api.sse import tokens_until_disconnect

router = APIRouter()


@router.post("/ask", response_model=AssistantResponseModel)
async def ask(request: QueryRequest) -> AssistantResponseModel:
    # Retrieval, reranking and the Ollama call are blocking/CPU-bound, so run
    # them in a worker thread to keep the event loop responsive.
    history = [m.model_dump() for m in request.history] if request.history else None
    result = await run_in_threadpool(
        _a.answer_question, request.query, request.use_triage, history,
        request.scan_findings,
    )
    return AssistantResponseModel(**result.to_dict())


@router.post("/ask/stream")
async def ask_stream(request: QueryRequest, raw_request: Request) -> StreamingResponse:
    """SSE stream of answer tokens, followed by a final metadata event."""
    history = [m.model_dump() for m in request.history] if request.history else None

    async def _generate():
        # Serve a previously computed answer in one chunk so repeated queries
        # stay instant (the blocking /ask does this via the cache; the stream
        # path must too now that the UI streams everything).
        cached = await run_in_threadpool(
            _a.cached_response, request.query, _a.MODE_QA, request.use_triage, None,
            history, request.scan_findings,
        )
        if cached is not None:
            yield f"data: {json.dumps({'token': cached.answer})}\n\n"
            yield "data: " + json.dumps({
                "done": True,
                "emergency": cached.emergency,
                "triage": cached.triage,
                "citations": cached.citations,
                "ml_predictions": cached.ml_predictions,
            }) + "\n\n"
            return

        prep = await run_in_threadpool(
            _a.prepare, request.query, _a.MODE_QA, request.use_triage, None, history,
            False, request.scan_findings,
        )
        acc = ""
        async for chunk in tokens_until_disconnect(_a.stream_tokens(prep), raw_request):
            acc += chunk
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        # Client pressed Stop / went away mid-stream: the upstream Ollama stream
        # is already closed; don't finalize, cache, or send meta for a partial.
        if await raw_request.is_disconnected():
            return
        # Only generated answers get the disclaimer (matching the blocking path).
        # Static replies — chit-chat, emergency, no-grounding — keep their own
        # wording: a greeting shouldn't carry a medical disclaimer, and an
        # emergency message shouldn't have its urgency diluted.
        answer = acc
        if prep.messages is not None:
            answer = _a.ensure_disclaimer(acc)
            if answer != acc:
                tail = answer[len(acc):]
                yield f"data: {json.dumps({'token': tail})}\n\n"
        resp = await run_in_threadpool(
            _a.record_stream,
            request.query, _a.MODE_QA, request.use_triage, prep, answer, None, history,
            request.scan_findings,
        )
        meta = {
            "done": True,
            "emergency": resp.emergency,
            "triage": resp.triage,
            "citations": resp.citations,
            "ml_predictions": resp.ml_predictions,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
