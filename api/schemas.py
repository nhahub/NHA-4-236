"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MessageModel(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'.")
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="User text.")
    use_triage: bool = Field(
        True, description="Run the emergency red-flag triage layer first."
    )
    history: list[MessageModel] | None = Field(
        None, description="Prior conversation turns, for follow-up context."
    )
    scan_findings: str | None = Field(
        None,
        max_length=2000,
        description=(
            "Summary of an attached imaging/signal study (from /analyze/*), fused "
            "into the grounded answer and used to bias retrieval. Decision-support."
        ),
    )


class PatientInfoModel(BaseModel):
    """Optional structured patient context to tailor the differential.

    Used only to make the ranked possibilities more relevant and to feed the
    age-aware triage — never to produce a definitive diagnosis.
    """

    age: int | None = Field(None, ge=0, le=120)
    sex: str | None = None
    duration: str | None = None
    severity: str | None = None
    conditions: str | None = None
    medications: str | None = None
    allergies: str | None = None
    smoking: str | None = None
    alcohol: str | None = None
    pregnancy: str | None = None
    other: str | None = Field(None, max_length=2000)


class SymptomCheckRequest(QueryRequest):
    patient: PatientInfoModel | None = Field(
        None, description="Optional structured patient information."
    )
    structured: bool = Field(
        False,
        description=(
            "When true, ask the LLM for machine-readable JSON output. "
            "The parsed differential is returned in `structured_differential`."
        ),
    )
    session_id: str | None = Field(
        None,
        description="User/session id for cross-session memory (save/load last differential).",
    )


class CitationModel(BaseModel):
    index: int
    title: str
    url: str
    source: str


class TriageModel(BaseModel):
    emergency: bool
    reason: str
    confidence: float
    source: str


class AssistantResponseModel(BaseModel):
    answer: str
    emergency: bool
    triage: TriageModel
    citations: list[CitationModel] = []
    ml_predictions: list[dict] = []
    structured_differential: dict | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    index_loaded: bool
    ml_model_loaded: bool
