# Changelog

All notable pipeline improvements are documented here. Dates are ISO-8601.

## [Unreleased] — 2026-06-30

### P1 retrieval quality (2026-06-30)
- **Diversified reranked passages** (`rag/reranker.py` `diversify`). MedQuAD has
  many Q&A pairs per disease, so the top-N was often "UTI, UTI, UTI". After
  cross-encoder scoring, greedily select passages preferring distinct titles and
  non-duplicate text (Jaccard over tokens); backfill with dupes only if too few
  distinct sources exist, so the slot count is preserved. `RERANK_DEDUP` (default
  on), `DEDUP_JACCARD_THRESHOLD`.
- **Two-stage grounding gate** (`rag/topicality.py` + `assistant.prepare`). A weak
  grounding score alone can't tell an off-topic query ("capital of Egypt") from a
  genuine health question the corpus doesn't cover ("IVF success rates") — both
  score low. A lexical medical-topic check (curated terms + medical morphology
  like *-itis*/*-emia*) now distinguishes them: the uncovered case gets an honest
  "that's a health question, but it's not in my sources" instead of the same flat
  off-topic refusal. Log outcome split into `uncovered_medical` vs `off_topic`.
- **Context-token budget** (`rag/pipeline.py` `apply_context_budget`). Caps the
  injected retrieval context (~`MAX_CONTEXT_TOKENS`, est. 4 chars/token) so the
  stacked prompt (system + ML/patient/scan blocks + context + query + answer
  room) can't silently overflow the model window. Passages are kept in rank order;
  the overflow one is truncated on a word boundary and the rest dropped, applied
  before formatting/citations so both stay consistent. Trims are logged, never
  silent.

### P0 credibility pass — honest & measurable (2026-06-30)
The jump from "impressive-looking but unproven" to "honest and measurable".

- **Citation-integrity pass** (`assistant.enforce_citation_integrity`). Every
  `[n]` marker in a generated answer is validated against the retrieved passages;
  an invented marker (e.g. `[7]` when 3 passages were retrieved) is stripped from
  the text, and the source list is pruned to the citations actually referenced.
  Applied on the blocking path and the streaming path (`record_stream`); the SSE
  routes now emit the cleaned canonical `answer` in the final event and the
  Streamlit UI repaints with it, so displayed text and citations stay aligned.
  Skipped for structured-JSON answers (validated at parse time).
- **XGBoost demoted off the live path** (`ML_IN_LIVE_PATH`, default `false`). The
  DDXPlus classifier is fed only the few symptoms parsed from free text
  (everything else marked *absent*) — a train/serve mismatch that makes it
  confidently wrong (flu → "TB 91%"). The LLM already builds the differential from
  grounded context, so the classifier is kept as a standalone artifact (CLI +
  notebook). Re-enable with the env flag.
- **Imaging/signal honesty.**
  - **MRI out-of-distribution guard**: a non-grayscale image (photo) or a
    low-confidence scan returns *"not a recognized brain MRI"* (`"ood": true`)
    instead of asserting a tumour class (`models/mri.py`).
  - **`experimental` flag on every MRI/EEG/ECG output** (API JSON, CLI footer,
    dashboard caption) — visible uncertainty everywhere. ECG keeps its
    `assumed_labels` flag for the unverified PTB-XL class names.
- **Eval harness skeleton** (`python -m eval`). One command reports retrieval
  `recall@k`/`MRR` (over `eval/cases/retrieval.jsonl`) and citation-validity rate
  (reusing the shipped integrity pass); groundedness, triage sens/spec, and
  latency are stubbed for P1. Sub-commands: `python -m eval.retrieval`,
  `python -m eval.citations`.
- **Env/launch healthcheck** (`python -m scripts.healthcheck`): reports the active
  interpreter and any missing runtime deps, catching a wrong/copied virtualenv
  before it fails deep in a request. Also runs warn-only at API startup. README
  now documents launching via `python -m`.

### CSV signal inputs + louder scan failures (2026-06-29)
- **EEG/ECG endpoints accept `.csv`** (and `.txt`) as well as `.npy`, tolerating a
  header row / index column (`api/routes/analysis.py`); empty/garbage uploads are
  rejected with 400. UI uploaders + chat attach accept `.csv` too.
- **Attached-study failures are now surfaced loudly** in the chat (a 500/400 from a
  model shows an explicit error instead of silently falling back to a text-only
  answer — which previously looked like the model was ignored). Root cause of that
  confusion: the MRI endpoint 500s if the server was started before `torchvision`
  was installed → **restart the API** to fix.

### MRI/EEG/ECG fused into the chat pipeline (earlier same day)
- **Attached studies now flow into the grounded answer.** A new `scan_findings`
  field (threaded through `prepare` / `_answer` / `cached_response` /
  `record_stream` / `answer_question` / `explore_symptoms` and both routes) injects
  the model's finding into the LLM prompt and biases retrieval toward it — so the
  answer discusses the MRI/EEG/ECG result alongside the literature.
- **Streamlit chat accepts file attachments** (`st.chat_input(accept_file=…)`):
  an attached image → MRI, a `.npy` → EEG/ECG (inferred from channel count); the
  raw model result is shown and its finding fused into the streamed answer.
- A scan-bearing message routes through `/symptom-check` (triage + differential).
  `scan_findings` is part of the cache key.
- **Robust signal orientation + sample data.** EEG/ECG preprocessing now
  auto-orients transposed arrays (longer axis = time), and the UI infers modality
  from the *shorter* axis (nearest of 12-lead ECG / 23-channel EEG).
  `scripts/make_sample_signals.py` writes tiny synthetic `eeg_sample.npy` /
  `ecg_sample.npy` for testing the attach flow. Offline suite: **178 passing**.

### MRI/EEG/ECG integrated into the app (earlier same day)
- **Three upload endpoints** (`api/routes/analysis.py`): `POST /analyze/mri`
  (image), `POST /analyze/eeg` (.npy), `POST /analyze/ecg` (.npy), plus
  `GET /analyze/status`. Lazy-load the weights, run in a threadpool, return
  probabilities + a decision-support disclaimer; 503 if weights are absent.
- **`models/ecg.py`**: 1-D ResNet-18 reconstructed from the checkpoint (12-lead,
  5-class). Loads the supplied weights with zero missing/unexpected keys. Class
  names default to the **assumed** PTB-XL superclasses (flagged; configurable).
- **Streamlit**: an "Imaging & signals" sidebar panel uploads a study to the
  endpoints and renders the result (`dashboard/streamlit_app.py`).
- Weights staged in `models/checkpoints/{mri,eeg,ecg}.pth` (gitignored).
  `python-multipart` pinned for uploads. Offline suite: **174 passing**.

### Imaging/signal models + triage fix (earlier same day)
- **Triage no longer over-flags asthma as a cardiac emergency.** "wheezing,
  shortness of breath and chest tightness" was hard-firing a 911 emergency
  labelled "possible cardiac event". Bare `chest tightness` and bare `shortness
  of breath` (the ambiguous descriptors — classic asthma) now flow to normal
  symptom analysis; acute phrasings (`chest pain`/`pressure`, `can't`/`difficulty
  breathing`, etc.) still hard-block. (`safety/red_flag_detector.py`, +8 tests)
- **`models/` package for deep-learning diagnostics** (separate from `ml_model/`):
  checkpoint inspector + **`models/mri.py`** (EfficientNet-B0 brain-tumor
  classifier) + **`models/eeg.py`** (CHB-MIT seizure screener that auto-detects
  the flatten vs GAP head, so it loads today's weights and the retrained ones
  unchanged). Each has load/preprocess/predict + CLI and handles bare-state_dict
  and wrapped/metadata saves. Added `torchvision` (CPU) to requirements + Dockerfile.
- Reviewed the MRI/EEG/ECG notebooks and produced fixed copies (`*_fixed.ipynb`):
  MRI (drop vertical flip, safe loader, seeding, metadata save); EEG (patient-level
  split to remove leakage, per-channel norm, GAP head 57k vs 2.66M params,
  FocalLoss, AUROC/AUPRC/sensitivity metrics — **requires retraining**).
- Tooling notes: set `OLLAMA_NUM_GPU=0` + `OLLAMA_TIMEOUT=300` in the env when the
  GPU CUDA build is broken (Ollama runs on CPU). Offline suite: **165 passing**.

### Hybrid pipeline hardening (earlier)

Hybrid (ML + RAG + LLM) pipeline hardening: the ML path now actually fires in
practice, the two retrieval systems inform each other, and the index/limits/logs
are production-safer. Offline test suite: **145 passing, 0 failing** (was ~138
with ~12 broken).

### Added
- **`/health` now reports ML model status** (`ml_model_loaded`), which the
  README already advertised but the response was missing. (`api/main.py`,
  `api/schemas.py`)
- **Stop now cancels generation server-side.** The SSE stream routes detect a
  client disconnect (Stop pressed → connection closed) between tokens and close
  the upstream Ollama stream, so the model stops generating immediately instead
  of finishing a discarded answer. A partial answer is never cached or
  finalized. (`api/sse.py`, `api/routes/medical_qa.py`,
  `api/routes/symptom_check.py`)
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
- **Rate limiter now covers the streaming endpoints.** The throttle middleware
  only checked `/ask` and `/symptom-check`; after the UI moved to SSE, all real
  traffic went to `/ask/stream` and `/symptom-check/stream`, leaving the primary
  path unprotected. Added both to the throttled set. (`api/main.py`)
- **Docker: dashboard volume + image bloat.** The dashboard mounted the now-
  unused `ml_model/artifacts` (it no longer loads the ML model) and never mounted
  `./data`, so saved patient profiles were ephemeral — swapped the mount to
  `./data`. Added `ml_model/artifacts/` to `.dockerignore` so the ~28 MB model
  isn't baked into the image (it's volume-mounted at runtime).
  (`docker-compose.yml`, `.dockerignore`)
- **`huggingface-hub` pinned explicitly** in requirements — imported directly by
  `ml_model.train`/`evaluate` and `symptom_parser` but previously only present
  transitively. (`requirements.txt`)
- **Docker ML path now functional.** The image shipped no spaCy model, so the
  symptom parser fell back to a weak regex tokenizer and the classifier rarely
  fired in-container. The Dockerfile now installs `spacy` + `en_core_web_sm`
  (biomedical `en_core_sci_md` remains an optional upgrade). (`Dockerfile`)
- **`structured_differential` now populated on the streaming symptom path.** It
  was always `None` over SSE because `record_stream` doesn't parse JSON; the
  route now parses the streamed output when `structured=True`.
  (`api/routes/symptom_check.py`)
- Removed the unused `pytest-asyncio` dependency (no async-marker tests exist).
  (`requirements.txt`)
- Renamed `scripts/test_pipeline.py` → `scripts/smoke_pipeline.py` so pytest
  can't accidentally collect a manual script that calls Ollama at import; dropped
  a dead `build_citations` import there too.
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
  logging (2), SSE streaming + cache pre-check (3), Stop/cancellation helper (2),
  streaming rate-limit + `/health` ML field.
- Added an autouse fixture resetting the in-memory rate limiter between API
  tests so the now-throttled `/stream` calls can't cause cross-test 429s.
