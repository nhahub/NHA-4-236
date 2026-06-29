"""POST /symptom-check — retrieve passages and produce a ranked differential."""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

import assistant as _a
import storage as _storage
from api.schemas import AssistantResponseModel, SymptomCheckRequest
from patient import PatientInfo

router = APIRouter()


def _build_patient(request: SymptomCheckRequest) -> PatientInfo | None:
    return PatientInfo(**request.patient.model_dump()) if request.patient is not None else None


def _inject_last_session(history: list[dict] | None, session_id: str | None) -> list[dict] | None:
    """Prepend the last session's differential as a synthetic assistant turn so
    the follow-up prompt can reference it, even in a brand-new browser session."""
    if not session_id:
        return history
    last = _storage.load_last_session(session_id)
    if not last or not last.get("differential_summary"):
        return history
    synthetic = {"role": "assistant", "content": last["differential_summary"]}
    return [synthetic, *(history or [])]


@router.post("/symptom-check", response_model=AssistantResponseModel)
async def symptom_check(request: SymptomCheckRequest) -> AssistantResponseModel:
    patient = _build_patient(request)
    history = [m.model_dump() for m in request.history] if request.history else None
    history = _inject_last_session(history, request.session_id)
    result = await run_in_threadpool(
        _a.explore_symptoms,
        request.query,
        request.use_triage,
        patient,
        history,
        request.structured,
    )
    # Persist the differential for cross-session memory.
    if request.session_id and result.answer and not result.emergency:
        await run_in_threadpool(
            _storage.save_conversation,
            request.session_id,
            result.answer,
            (history or []) + [{"role": "assistant", "content": result.answer}],
        )
    return AssistantResponseModel(**result.to_dict())


@router.post("/symptom-check/stream")
async def symptom_check_stream(request: SymptomCheckRequest) -> StreamingResponse:
    """SSE stream of differential tokens, followed by a final metadata event."""
    patient = _build_patient(request)
    history = [m.model_dump() for m in request.history] if request.history else None
    history = _inject_last_session(history, request.session_id)

    async def _generate():
        # Serve a cached differential in one chunk so repeats stay instant,
        # mirroring the blocking /symptom-check. Uses the post-injection history
        # so the key matches what record_stream caches under.
        cached = await run_in_threadpool(
            _a.cached_response,
            request.query, _a.MODE_SYMPTOM, request.use_triage, patient, history,
        )
        if cached is not None:
            yield f"data: {json.dumps({'token': cached.answer})}\n\n"
            yield "data: " + json.dumps({
                "done": True,
                "emergency": cached.emergency,
                "triage": cached.triage,
                "citations": cached.citations,
                "ml_predictions": cached.ml_predictions,
                "structured_differential": cached.structured_differential,
            }) + "\n\n"
            return

        prep = await run_in_threadpool(
            _a.prepare,
            request.query,
            _a.MODE_SYMPTOM,
            request.use_triage,
            patient,
            history,
            request.structured,
        )
        acc = ""
        for chunk in _a.stream_tokens(prep):
            acc += chunk
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        # Only generated answers get the disclaimer (matching the blocking path);
        # static replies (chit-chat / emergency / no-grounding) keep their own
        # wording.
        answer = acc
        if prep.messages is not None:
            answer = _a.ensure_disclaimer(acc)
            if answer != acc:
                tail = answer[len(acc):]
                yield f"data: {json.dumps({'token': tail})}\n\n"
        resp = await run_in_threadpool(
            _a.record_stream,
            request.query, _a.MODE_SYMPTOM, request.use_triage, prep, answer, patient, history,
        )
        # Persist differential for cross-session memory.
        if request.session_id and answer and not resp.emergency:
            await run_in_threadpool(
                _storage.save_conversation,
                request.session_id,
                answer,
                (history or []) + [{"role": "assistant", "content": answer}],
            )
        meta = {
            "done": True,
            "emergency": resp.emergency,
            "triage": resp.triage,
            "citations": resp.citations,
            "ml_predictions": resp.ml_predictions,
            "structured_differential": resp.structured_differential,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
