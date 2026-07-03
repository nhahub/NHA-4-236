"""High-level assistant orchestration.

Ties together intent routing, safety triage, the retrieval pipeline, and the
LLM into the two user-facing flows (general Q&A and symptom exploration).
Shared by the FastAPI routes and the Streamlit demo so the logic lives in
exactly one place.

Request handling, in order:
    1. Intent routing  — chit-chat/greetings get a template reply, no retrieval.
                          Definition questions are answered plainly even in
                          Symptom mode; first-person symptom reports get the
                          differential even in General mode (the UI toggle is
                          only a hint).
    2. Safety triage   — emergencies short-circuit to an urgent-care message.
    3. Retrieve + gate — if the best reranked passage is too weak, decline
                          instead of forcing a citation over irrelevant text.
    4. Generate        — grounded answer, blocking (`chat`) or streaming.

Generation is split so the same triage + retrieval work backs both the blocking
API path and the token-streaming Streamlit path:

    prepare()       -> route + triage + retrieve + build messages (no main LLM)
    stream_tokens() -> run the LLM (streaming)
    _answer()       -> run the LLM (blocking) + cache
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Iterator

from config import settings
from llm.client import get_llm, load_prompt
from patient import PatientInfo
from rag.topicality import looks_medical
from rag.pipeline import (
    Citation,
    apply_context_budget,
    build_citations,
    format_context,
    retrieve_context,
)
import json as _json
import re

from safety import intent as intent_router
from safety.red_flag_detector import (
    TriageResult,
    detect_red_flags,
    emergency_message,
)

# Symptom ML — the free-text classifier (train==serve on symptom text). Optional:
# if the artifacts aren't trained yet, ML is skipped and the answer is pure
# LLM+RAG. The legacy DDXPlus XGBoost (ml_model.legacy) is kept as a standalone
# portfolio artifact and is NOT on the live path.
try:
    import ml_model.symptom_classifier as _text_predict
    _TEXT_ML_AVAILABLE = _text_predict.text_artifacts_available()
except Exception:  # pragma: no cover
    _text_predict = None  # type: ignore[assignment]
    _TEXT_ML_AVAILABLE = False

logger = logging.getLogger("assistant")


def _log_request(
    *,
    start: float,
    intent: str,
    outcome: str,
    ml_matched: int = 0,
    citations: int = 0,
) -> None:
    """Emit one structured line per prepared request for observability.

    Captures the decisions that determine answer quality and cost — routed
    intent, whether ML pre-ranking fired, how the retrieval gate resolved, and
    end-to-end prepare latency — as key=value pairs that are trivial to grep or
    ship to a log aggregator. ``outcome`` is one of: chitchat, emergency,
    no_grounding, answered.
    """
    logger.info(
        "request intent=%s outcome=%s ml_matched=%d citations=%d latency_ms=%d",
        intent, outcome, ml_matched, citations, int((time.perf_counter() - start) * 1000),
    )


# Prompt files for each routed intent.
QA_PROMPT = "qa_prompt"
SYMPTOM_PROMPT = "symptom_prompt"
SYMPTOM_FOLLOWUP_PROMPT = "symptom_followup_prompt"

# Marker text that indicates a prior assistant turn contained a differential.
_DIFFERENTIAL_MARKER = "RANKED DIFFERENTIAL"

# Off-topic: the query doesn't look medical at all (e.g. "capital of Egypt").
NO_GROUNDING_MESSAGE = (
    "I couldn't find relevant medical information for that in my sources, so I "
    "can't give a grounded answer. I can help with questions about symptoms, "
    "conditions, medications, and other health topics — try rephrasing, or ask "
    "about a specific medical topic.\n\n"
    "This is general information, not a medical diagnosis. Consult a qualified "
    "healthcare professional for any health concern."
)

# Medical-but-uncovered: a genuine health question the corpus doesn't cover
# (e.g. IVF success rates). Be honest that it's a gap in *my sources*, not a
# refusal to engage with the topic.
MEDICAL_UNCOVERED_MESSAGE = (
    "That looks like a health question, but I don't have sources covering it in "
    "my knowledge base, so I can't give a grounded, cited answer without risking "
    "making something up. A clinician or an up-to-date medical resource would be "
    "a better source here. If you have a related question about a common "
    "condition, symptom, or medication, I may have grounded information on that."
    "\n\n"
    "This is general information, not a medical diagnosis. Consult a qualified "
    "healthcare professional for any health concern."
)

_NEUTRAL_TRIAGE = TriageResult(False, "not triaged", 0.0, "none")

# "No grounding" refusals we deliberately do NOT cache: they're borderline (a
# query that scores just below the floor may become answerable after an index or
# config change), and they're cheap to recompute since they short-circuit before
# the LLM. Caching them would freeze a refusal in place until the cache is cleared.
_NO_GROUNDING_ANSWERS = frozenset({NO_GROUNDING_MESSAGE, MEDICAL_UNCOVERED_MESSAGE})


def _is_no_grounding_refusal(prep: "Prepared") -> bool:
    return prep.messages is None and prep.static_answer in _NO_GROUNDING_ANSWERS


_STOP_WORDS = {"type", "and", "or", "the", "of", "in", "with", "for", "to",
               "not", "are", "was", "has", "had", "its", "can", "may", "due"}


def _filter_known_condition_passages(passages, conditions: str | None):
    """Drop passages whose title is primarily about an existing condition.

    When every retrieved passage is about a condition the patient already has,
    the model will list that condition as a differential no matter what the
    prompt says. Filtering forces the model to reason from other material.
    Falls back to the original list if filtering would leave nothing.
    """
    if not conditions or not passages:
        return passages
    # Extract significant keywords from conditions: drop numbers and stop words.
    # "type 2 diabetes, hypertension" → {"diabetes", "hypertension"}
    raw_words = re.findall(r"[a-zA-Z]{3,}", conditions.lower())
    keywords = {w for w in raw_words if w not in _STOP_WORDS}
    if not keywords:
        return passages

    def _title_is_about_known(title: str) -> bool:
        t = title.lower()
        # Match singular and plural: "migraines" matches "Migraine" and vice versa.
        return any(
            kw in t or (kw.endswith("s") and kw[:-1] in t) or (not kw.endswith("s") and kw + "s" in t)
            for kw in keywords
        )

    filtered = [p for p in passages if not _title_is_about_known(p.title)]
    return filtered if filtered else passages


# Generic words in DDXPlus disease labels that carry no disease-specific signal;
# excluded from the support check so e.g. "Acute laryngitis" is verified on
# "laryngitis", not the ubiquitous "acute".
_ML_LABEL_FILLER = frozenset({
    "acute", "chronic", "possible", "stable", "unstable", "initial", "viral",
    "allergic", "localized", "spontaneous", "syndrome", "disease", "infection",
    "exacerbation", "reaction", "reactions", "attack", "food", "poisoning",
})


def _annotate_ml_support(ml_preds: list[dict], passages) -> None:
    """Mark each ML prediction with whether the retrieved passages mention it.

    Closes the hybrid loop in the other direction: a high-confidence prediction
    with zero supporting passages is a signal the LLM (and the caller) should
    treat with extra scrutiny. Verification uses only genuine disease words
    (>= 5 chars, excluding generic filler), so abbreviation-only labels (URTI,
    GERD, SLE) are treated as unverifiable -> supported, avoiding false caution
    flags. Mutates ``ml_preds`` in place, adding a ``supported`` key.
    """
    blob = " ".join(f"{p.title} {p.text}" for p in passages).lower()
    for pred in ml_preds:
        tokens = [
            w for w in re.findall(r"[a-z]{5,}", pred["disease"].lower())
            if w not in _ML_LABEL_FILLER and w not in _STOP_WORDS
        ]
        # No checkable token (abbreviation/filler-only label) -> can't verify.
        pred["supported"] = (not tokens) or any(tok in blob for tok in tokens)

# The system prompt instructs the model to end every answer with this exact
# disclaimer, but an 8B model (or the num_predict cap on a long answer) can
# drop it. We guarantee it in code so a generated answer never ships without it.
DISCLAIMER = (
    "This is general information, not a medical diagnosis. Consult a qualified "
    "healthcare professional for any health concern."
)


def ensure_disclaimer(text: str) -> str:
    """Append the safety disclaimer unless the answer already contains it."""
    if "not a medical diagnosis" in text.lower():
        return text
    sep = "" if text.endswith("\n") else "\n\n"
    return f"{text.rstrip()}{sep}{DISCLAIMER}"


# Matches a single inline citation marker like ``[3]`` (not a JSON list ``[1, 3]``).
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def enforce_citation_integrity(
    answer: str, citations: list[dict]
) -> tuple[str, list[dict]]:
    """Drop ``[n]`` markers that don't map to a retrieved source, and prune the
    citation list to those the answer actually references.

    The model is told to cite passages by their context number ``[1]..[N]``, but
    a weak model can invent a ``[7]`` when only 3 passages were retrieved. This
    pass parses every marker, removes any whose number is out of range (so no
    answer cites a source that doesn't exist), and returns only the citations the
    answer genuinely used.

    Indices are deliberately **not** renumbered: on the streaming path the user
    has already seen ``[3]`` meaning "source 3", so the surviving citation list
    must keep the original numbers to stay consistent with the displayed text.
    If the answer used no valid markers at all, the full citation list is kept
    (the sources still grounded the answer even if the model omitted markers).

    Returns ``(clean_answer, kept_citations)``.
    """
    valid = {c["index"] for c in citations}
    cited = {int(n) for n in _CITATION_MARKER_RE.findall(answer)}
    invalid = cited - valid

    clean = answer
    if invalid:
        clean = _CITATION_MARKER_RE.sub(
            lambda m: "" if int(m.group(1)) in invalid else m.group(0), clean
        )
        # Tidy whitespace/punctuation left dangling where a marker was removed.
        clean = re.sub(r"[ \t]{2,}", " ", clean)
        clean = re.sub(r"[ \t]+([.,;:)])", r"\1", clean)

    referenced = cited & valid
    kept = [c for c in citations if c["index"] in referenced] if referenced else citations
    return clean, kept


@dataclass
class AssistantResponse:
    answer: str
    emergency: bool
    triage: dict
    citations: list[dict] = field(default_factory=list)
    ml_predictions: list[dict] = field(default_factory=list)
    structured_differential: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prepared:
    """Everything needed to produce an answer, minus the main LLM call.

    When ``messages`` is ``None`` the answer is already decided (chit-chat,
    emergency, or no-grounding) and lives in ``static_answer`` — no generation
    needed.
    """

    emergency: bool
    triage: dict
    citations: list[dict]
    messages: list[dict[str, str]] | None
    static_answer: str | None = None
    ml_predictions: list[dict] | None = None  # Phase 2: ranked ML differential


# --- Response cache -------------------------------------------------------
# A tiny LRU keyed on the request. Repeated demo queries (slow on CPU) then
# return instantly. Emergencies are never cached — triage must always re-run.
_CACHE_MAX = 50
_cache: "OrderedDict[tuple, AssistantResponse]" = OrderedDict()


def _cache_key(
    query: str,
    mode_hint: str,
    use_triage: bool,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
) -> tuple:
    sig = patient.signature() if patient is not None else None
    hist = (
        tuple((m.get("role"), m.get("content")) for m in history) if history else None
    )
    return (query.strip().lower(), mode_hint, use_triage, sig, hist)


def _cache_get(key: tuple) -> AssistantResponse | None:
    resp = _cache.get(key)
    if resp is not None:
        _cache.move_to_end(key)
    return resp


def _cache_put(key: tuple, resp: AssistantResponse) -> None:
    _cache[key] = resp
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


# --- Preparation + generation --------------------------------------------
# How many prior messages to replay to the LLM, and how many prior *user* turns
# to fold into the retrieval query so short follow-ups ("3 days") still retrieve.
_HISTORY_MAX = 6
_RETRIEVAL_CONTEXT_TURNS = 2

# How many predictions the classifier returns, how many top diseases to fold
# into the retrieval recall query, and the probability above which an ML
# prediction with no supporting passage is flagged to the LLM for extra scrutiny.
_ML_TOP_K = 5
_ML_RETRIEVAL_TERMS = 3
_ML_SUPPORT_FLAG_FLOOR = 0.15
# Only surface ML predictions the retrieved literature supports. Even the
# free-text classifier is limited to its 22 training conditions, so on an
# uncovered presentation its top guess can be wrong. When the grounded passages
# don't back a prediction, hide it from the user-facing differential rather than
# show a misleading number. The full list (with caution flags) is still given to
# the LLM, which reasons over the discrepancy. Set False to surface every one.
_ML_SHOW_ONLY_SUPPORTED = True


SYMPTOM_JSON_PROMPT = "symptom_json_prompt"


def _parse_structured_differential(text: str) -> dict | None:
    """Try to extract a JSON object from the LLM response.

    The model is instructed to return pure JSON but may wrap it in prose or
    markdown fences. We strip those and attempt a parse; on failure return None
    so the caller can surface the raw text instead of crashing.
    """
    # Strip optional ```json … ``` fences.
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the first { … } block.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return _json.loads(stripped[start : end + 1])
    except _json.JSONDecodeError:
        return None


def _has_prior_differential(history: list[dict] | None) -> bool:
    """True when any prior assistant turn contains a ranked differential."""
    if not history:
        return False
    return any(
        _DIFFERENTIAL_MARKER in m.get("content", "")
        for m in history
        if m.get("role") == "assistant"
    )


def _extract_prior_differential(history: list[dict]) -> str:
    """Return the differential section from the most recent assistant message that has one."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        idx = content.find(_DIFFERENTIAL_MARKER)
        if idx != -1:
            return content[idx:]
    return ""


def _retrieval_query(query: str, history: list[dict] | None) -> str:
    """Fold recent user turns into the query so follow-ups retrieve in context."""
    if not history:
        return query
    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    recent = user_turns[-_RETRIEVAL_CONTEXT_TURNS:]
    return " ".join([*recent, query]).strip()


def _build_retrieval_query(
    query: str,
    history: list[dict] | None,
    patient: PatientInfo | None,
    ml_preds: list[dict] | None,
) -> str:
    """Assemble the *recall* query: base query + history + patient/ML biases.

    Only the recall query is widened; the caller keeps the raw user message as the
    rerank query so cross-encoder scoring (and the confidence gate) are unaffected.
    """
    rq = _retrieval_query(query, history)
    if patient is not None and patient.retrieval_hints():
        rq = f"{rq} {patient.retrieval_hints()}"
    # ML -> RAG feedback: fold the top predicted diseases into recall so the LLM
    # sees passages about them and can confirm or contradict the ML signal.
    if ml_preds:
        ml_terms = " ".join(p["disease"] for p in ml_preds[:_ML_RETRIEVAL_TERMS])
        rq = f"{rq} {ml_terms}"
    return rq


def _ml_prompt_block(ml_preds: list[dict] | None) -> str:
    """The ML PRE-RANKING block prepended to the prompt, or "" when ML is off."""
    if not ml_preds:
        return ""
    lines = [
        "\nML PRE-RANKING (free-text symptom classifier, supplementary signal):",
        "Use as a starting signal only — reason from the retrieved context.",
        "If context contradicts a high-ranked ML prediction, note the discrepancy"
        " explicitly rather than ignoring it.",
    ]
    for pred in ml_preds:
        line = f"  - {pred['disease']}: {pred['probability']:.2f} confidence"
        # Flag confident predictions the retrieved context does NOT support, so
        # the model weighs the discrepancy instead of echoing the number.
        if (
            pred.get("supported") is False
            and pred["probability"] >= _ML_SUPPORT_FLAG_FLOOR
        ):
            line += " (no supporting passage retrieved — treat with caution)"
        lines.append(line)
    return "\n".join(lines)


def _patient_prompt_block(patient: PatientInfo | None, intent: str) -> str:
    """The labelled patient-context block, or "" when there's no patient info.

    Known conditions and current medications are code-enforced here (not just in
    the prompt text) because small models ignore the prompt rule alone.
    """
    if patient is None or patient.is_empty():
        return ""
    block = patient.to_context()
    if patient.conditions and intent == intent_router.SYMPTOM:
        block += (
            f"\n\nCRITICAL — DO NOT list any of the following as differential "
            f"candidates (patient already has them): {patient.conditions}"
        )
    if patient.medications and intent == intent_router.SYMPTOM:
        block += (
            f"\n\nMEDICATION ALERT — Patient is currently taking: "
            f"{patient.medications}. "
            f"In SECTION 5 (next steps), flag any suggested treatments or "
            f"common medications (e.g. NSAIDs, anticoagulants, steroids) that "
            f"may interact with or be contraindicated alongside these medications."
        )
    return block


def prepare(
    query: str,
    mode_hint: str,
    use_triage: bool,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
    structured: bool = False,
) -> Prepared:
    """Route + triage + retrieve and build chat messages (no main LLM call)."""
    start = time.perf_counter()
    intent = intent_router.classify_intent(query, mode_hint)

    # 1. Chit-chat / greetings: redirect with a template, no retrieval, no LLM.
    #    (These patterns can never be an emergency, so triage is skipped.)
    if intent == intent_router.CHITCHAT:
        _log_request(start=start, intent=intent, outcome="chitchat")
        return Prepared(
            emergency=False,
            triage=_NEUTRAL_TRIAGE.to_dict(),
            citations=[],
            messages=None,
            static_answer=intent_router.chitchat_reply(query),
        )

    # 2. Safety first for anything we will actually answer. Patient context
    #    (age / pregnancy) feeds the age-aware red-flag checks.
    triage = (
        detect_red_flags(query, patient=patient)
        if use_triage
        else TriageResult(False, "triage disabled", 0.0, "none")
    )
    if triage.emergency:
        _log_request(start=start, intent=intent, outcome="emergency")
        return Prepared(
            emergency=True,
            triage=triage.to_dict(),
            citations=[],
            messages=None,
            static_answer=emergency_message(triage),
        )

    # 2b. ML pre-ranking: the free-text symptom classifier (embeds the raw text
    #     with the RAG encoder, so train == serve). Supplementary signal only —
    #     the grounded LLM answer stays authoritative. Abstains below a confidence
    #     floor so vague / out-of-scope text (the model covers 22 conditions)
    #     doesn't yield a confident guess. Skipped when artifacts aren't trained.
    ml_preds: list[dict] | None = None
    if settings.ml_in_live_path and _TEXT_ML_AVAILABLE and intent == intent_router.SYMPTOM:
        preds = _text_predict.predict_text(query, top_k=_ML_TOP_K)
        if preds and preds[0]["probability"] >= settings.ml_min_confidence:
            ml_preds = preds

    # 3. Retrieve + rerank, then gate on the best score: weak grounding means
    #    the query is off-topic / out-of-corpus — decline instead of fabricating.
    #    Patient hints + recent turns bias recall without changing the display.
    retrieval_query = _build_retrieval_query(query, history, patient, ml_preds)
    # Pass the raw user query as rerank_query: patient hints / history context
    # widen retrieval recall but confuse the cross-encoder scoring.
    # When filtering existing conditions from the differential, fetch a larger
    # reranked pool first so non-condition passages survive after the filter.
    needs_filter = (
        intent == intent_router.SYMPTOM
        and patient is not None
        and bool(patient.conditions)
    )
    # When filtering existing conditions, expand BOTH the initial retrieval pool
    # (top_k) and the reranked pool (top_n) so non-condition passages survive.
    # With a default top_k=20 and a query biased toward the existing condition,
    # all 20 candidates can be about that condition, leaving nothing after filter.
    rerank_top_n = settings.rerank_top_n * 5 if needs_filter else None
    top_k = settings.retrieval_top_k * 3 if needs_filter else None
    passages = retrieve_context(
        retrieval_query, rerank_query=query, top_k=top_k, top_n=rerank_top_n
    )
    if needs_filter:
        passages = _filter_known_condition_passages(passages, patient.conditions)
        passages = passages[: settings.rerank_top_n]
    if (
        settings.use_reranker
        and passages
        and passages[0].score < settings.rerank_score_floor
    ):
        # Two-stage gate: weak grounding alone can't tell an off-topic query
        # ("capital of Egypt") from a genuine health question the corpus doesn't
        # cover ("IVF success rates"). A lexical medical-topic check supplies the
        # orthogonal signal, so the uncovered case gets an honest "not in my
        # sources" instead of the same flat off-topic refusal.
        uncovered = looks_medical(query)
        _log_request(
            start=start,
            intent=intent,
            outcome="uncovered_medical" if uncovered else "off_topic",
            ml_matched=len(ml_preds or []),
        )
        return Prepared(
            emergency=False,
            triage=triage.to_dict(),
            citations=[],
            messages=None,
            static_answer=MEDICAL_UNCOVERED_MESSAGE if uncovered else NO_GROUNDING_MESSAGE,
        )

    # Context-token budget: cap injected context so the stacked prompt can't
    # silently overflow the model window. Applied before format/citations so both
    # see the same (possibly trimmed) passage set and stay numbered consistently.
    n_before = len(passages)
    passages, trimmed = apply_context_budget(passages, settings.max_context_tokens)
    if trimmed:
        dropped = n_before - len(passages)
        logger.warning(
            "context budget ~%d tokens hit: dropped %d passage(s)%s",
            settings.max_context_tokens, dropped,
            ", truncated the last to fit" if dropped == 0 else "",
        )

    context = format_context(passages)
    citations: list[Citation] = build_citations(passages)

    # Retrieval -> ML cross-check: tag predictions the retrieved context supports.
    if ml_preds:
        _annotate_ml_support(ml_preds, passages)

    # 4. Build the messages the LLM will answer from. Supplied patient details
    #    are prepended as a labelled context block (used to tailor the
    #    differential, never to assert a definitive diagnosis).
    is_followup = (
        intent == intent_router.SYMPTOM and _has_prior_differential(history)
    )
    if intent == intent_router.QA:
        prompt_name = QA_PROMPT
    elif structured and intent == intent_router.SYMPTOM:
        prompt_name = SYMPTOM_JSON_PROMPT
    elif is_followup:
        prompt_name = SYMPTOM_FOLLOWUP_PROMPT
    else:
        prompt_name = SYMPTOM_PROMPT

    system = load_prompt("system_prompt")
    if is_followup:
        prior_diff = _extract_prior_differential(history)
        user_prompt = load_prompt(prompt_name).format(
            context=context, query=query, prior_differential=prior_diff
        )
    else:
        user_prompt = load_prompt(prompt_name).format(context=context, query=query)

    # Prepend the optional context blocks, innermost first so the final order is
    # patient → ML → base prompt (each block sits above the one before it).
    if ml_block := _ml_prompt_block(ml_preds):
        user_prompt = f"{ml_block}\n\n{user_prompt}"
    if patient_block := _patient_prompt_block(patient, intent):
        user_prompt = f"{patient_block}\n\n{user_prompt}"
    messages = [{"role": "system", "content": system}]
    if history:  # replay recent turns so follow-ups are answered in context
        messages.extend(history[-_HISTORY_MAX:])
    messages.append({"role": "user", "content": user_prompt})
    _log_request(
        start=start, intent=intent, outcome="answered",
        ml_matched=len(ml_preds or []), citations=len(citations),
    )
    # User-facing differential: drop predictions the literature doesn't support
    # so a confidently-wrong classifier guess isn't shown as a ranked disease.
    # (The LLM already received the full list with caution flags above.)
    display_preds = ml_preds
    if ml_preds and _ML_SHOW_ONLY_SUPPORTED:
        display_preds = [p for p in ml_preds if p.get("supported")] or None

    return Prepared(
        emergency=False,
        triage=triage.to_dict(),
        citations=[asdict(c) for c in citations],
        messages=messages,
        ml_predictions=display_preds,
    )


def stream_tokens(prep: Prepared) -> Iterator[str]:
    """Yield the answer text for a prepared request as it is generated.

    For short-circuited requests (chit-chat / emergency / no-grounding) this
    yields the static answer in one piece; otherwise it streams LLM tokens.
    """
    if prep.messages is None:
        yield prep.static_answer or ""
        return
    yield from get_llm().chat_stream(prep.messages)


def _answer(
    query: str,
    mode_hint: str,
    use_triage: bool,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
    structured: bool = False,
) -> AssistantResponse:
    key = _cache_key(query, mode_hint, use_triage, patient, history)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    prep = prepare(
        query, mode_hint, use_triage, patient, history, structured=structured,
    )
    if prep.messages is None:
        answer = prep.static_answer or ""
        structured_diff = None
    else:
        raw = get_llm().chat(prep.messages)
        if structured:
            structured_diff = _parse_structured_differential(raw)
            # Surface a human-readable fallback alongside the JSON attempt.
            answer = raw if structured_diff is None else ensure_disclaimer(raw)
        else:
            answer = ensure_disclaimer(raw)
            structured_diff = None

    # Citation-integrity pass: strip invented [n] markers and prune the source
    # list to those actually cited. Skipped for structured JSON, whose citation
    # arrays use a different mechanism (validated at parse time).
    citations = prep.citations
    if structured_diff is None and prep.messages is not None:
        answer, citations = enforce_citation_integrity(answer, prep.citations)

    resp = AssistantResponse(
        answer=answer,
        emergency=prep.emergency,
        triage=prep.triage,
        citations=citations,
        ml_predictions=prep.ml_predictions or [],
        structured_differential=structured_diff,
    )
    if not prep.emergency and not _is_no_grounding_refusal(prep):
        _cache_put(key, resp)
    return resp


def cached_response(
    query: str,
    mode_hint: str,
    use_triage: bool,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
) -> AssistantResponse | None:
    """Return a previously computed response for this request, if cached."""
    return _cache_get(
        _cache_key(query, mode_hint, use_triage, patient, history)
    )


def record_stream(
    query: str,
    mode_hint: str,
    use_triage: bool,
    prep: Prepared,
    answer: str,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
    structured: bool = False,
) -> AssistantResponse:
    """Build the response for a streamed answer and cache it for next time.

    Runs the same citation-integrity pass as the blocking path so the cached /
    re-served answer and its source list are consistent (invented [n] markers
    dropped, citations pruned to those referenced). Skipped for static replies
    (chit-chat / emergency / no-grounding) and for ``structured`` JSON answers,
    whose citation arrays use a different mechanism (validated at parse time).
    """
    citations = prep.citations
    if prep.messages is not None and not structured:
        answer, citations = enforce_citation_integrity(answer, prep.citations)
    resp = AssistantResponse(
        answer=answer,
        emergency=prep.emergency,
        triage=prep.triage,
        citations=citations,
        ml_predictions=prep.ml_predictions or [],
    )
    if not prep.emergency and not _is_no_grounding_refusal(prep):
        _cache_put(
            _cache_key(query, mode_hint, use_triage, patient, history),
            resp,
        )
    return resp


# --- Public flows ---------------------------------------------------------
# Mode hints passed from the UI / API; intent routing may still override them.
MODE_QA = intent_router.QA
MODE_SYMPTOM = intent_router.SYMPTOM


def answer_question(
    query: str,
    use_triage: bool = True,
    history: list[dict] | None = None,
) -> AssistantResponse:
    """General medical question -> grounded answer with citations."""
    return _answer(
        query, mode_hint=MODE_QA, use_triage=use_triage, history=history,
    )


def explore_symptoms(
    query: str,
    use_triage: bool = True,
    patient: PatientInfo | None = None,
    history: list[dict] | None = None,
    structured: bool = False,
) -> AssistantResponse:
    """Symptom description -> grounded, ranked condition exploration.

    ``patient`` is optional structured context (age, history, etc.) used to
    tailor the differential and the age-aware triage. ``history`` is the prior
    conversation turns, so follow-ups are answered in context. Never diagnostic.
    When ``structured=True``, the LLM is asked for machine-readable JSON and the
    parsed result is returned in ``structured_differential``.
    """
    return _answer(
        query,
        mode_hint=MODE_SYMPTOM,
        use_triage=use_triage,
        patient=patient,
        history=history,
        structured=structured,
    )
