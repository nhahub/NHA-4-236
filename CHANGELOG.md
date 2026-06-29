# Changelog

All notable pipeline improvements are documented here. Dates are ISO-8601.

## [Unreleased] — 2026-06-29

Hybrid (ML + RAG + LLM) pipeline hardening: the ML path now actually fires in
practice, the two retrieval systems inform each other, and the index/limits/logs
are production-safer. Offline test suite: **145 passing, 0 failing** (was ~138
with ~12 broken).

### Added
- **Single entry point: Streamlit now talks to FastAPI for everything.** The
  dashboard consumes the `/ask/stream` and `/symptom-check/stream` SSE endpoints
  for live token streaming and no longer imports the `assistant` module
  in-process. The assistant logic runs in exactly one place (the API); the UI is
  a pure HTTP client. The Stop button now works over HTTP, and the UI requires
  the backend to be running. (`dashboard/streamlit_app.py`)
- **Cache pre-check on the SSE stream routes.** `/ask/stream` and
  `/symptom-check/stream` serve a previously computed answer in one chunk on a
  cache hit, so repeated queries stay instant — parity with the blocking
  endpoints, which the all-streaming UI would otherwise have lost.
  (`api/routes/medical_qa.py`, `api/routes/symptom_check.py`)
- **ML → RAG feedback loop.** The top-3 XGBoost disease predictions are folded
  into the retrieval *recall* query so RAG pulls passages about the predicted
  conditions. The cross-encoder `rerank_query` stays the raw user message, so
  scoring and the confidence gate are unaffected. (`assistant.py`)
- **RAG → ML cross-check.** Each ML prediction is tagged `supported` based on
  whether the retrieved passages mention it; confident-but-unsupported
  predictions get a `(no supporting passage retrieved — treat with caution)`
  flag injected into the prompt. Conservative on abbreviation/filler labels
  (URTI, GERD, "Acute …") to avoid false flags. (`assistant.py`)
- **Embedding-model mismatch guard.** Ingest writes `index_meta.json` (embedder
  name + dim); the retriever refuses to serve if `EMBEDDING_MODEL` disagrees
  with the model that built the index. Catches the silent same-dimension model
  swap that the existing FAISS dim-check misses. (`rag/ingest.py`,
  `rag/retriever.py`, `config.py`)
- **Structured per-request logging.** One greppable line per request —
  `intent=… outcome=… ml_matched=… citations=… latency_ms=…` — at every
  `prepare()` exit (chitchat / emergency / no_grounding / answered).
  (`assistant.py`)

### Changed
- **Symptom→evidence-code matcher rewritten.** Replaced "first substring wins"
  with: lay→clinical synonym expansion, history/family/treatment-question
  demotion, presence-over-characterization tiering, whole-word matching, and a
  best-scored fuzzy fallback. (`symptom_parser.py`)
- **Robust NER extraction.** Symptom extraction now uses noun chunks + head
  nouns (with a pronoun/time-word ignore list), so symptoms are still captured
  when only the general `en_core_web_sm` spaCy model is installed instead of the
  medical `en_core_sci_md`. Previously the ML path silently no-opped in that
  setup. (`symptom_parser.py`)
- **Triage ordering.** Age/pregnancy checks now run before text rules, so an
  infant (age < 1) with fever yields the precise `"infant fever"` reason rather
  than the conversational `"young child fever"` rule. Both still short-circuit
  to an emergency. (`safety/red_flag_detector.py`)

### Fixed
- **Symptom-mapping correctness bugs** (fed wrong evidence to the classifier):
  - `nausea` mapped to a hospital-IV-treatment-history question (E_147) instead
    of the symptom (E_148).
  - `headache` mapped to a family-history question (E_25); DDXPlus has no
    headache symptom, so it now correctly resolves to *no match*.
  - `dizziness` matched nothing; now resolves via synonym.
- **Rate-limiter memory leak.** `_ip_timestamps` never evicted entries for IPs
  that stopped sending requests; added a once-per-window sweep. (`api/main.py`)
- **Broken test stubs** out of sync with production signatures: `prepare`'s
  `structured` kwarg, `retrieve_context`'s `rerank_query` kwarg, the API routes'
  `import assistant as _a` indirection, and two assertions referencing renamed
  prompt sections / re-routed intents.
- **Disclaimer no longer stapled onto static replies in streaming.** The SSE
  routes appended the medical disclaimer to *every* answer, so once the UI
  streamed everything, greetings carried a "not a medical diagnosis" line and
  emergency messages had their urgency diluted. The disclaimer is now added only
  to generated answers, matching the blocking path. (`api/routes/medical_qa.py`,
  `api/routes/symptom_check.py`)
- Removed a dead `numpy` import. (`symptom_parser.py`)

### Tests
- New coverage: matcher quality (8), ML↔RAG feedback + cross-check (12),
  embedding-model guard (3), rate-limiter eviction/throttle (2), structured
  logging (2).
