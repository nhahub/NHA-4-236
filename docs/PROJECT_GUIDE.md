# Hybrid Medical Assistant — The Complete Mentor Guide

> A deep, teach-me-everything walkthrough of the project so you can explain it in
> interviews, defend every design choice, modify any component, and continue
> without AI help. Read top-to-bottom once; use it as a reference after.

**What the project is, in one sentence:** a locally-runnable medical Q&A and
symptom-exploration assistant built on a *hybrid* of three model types — a
free-text **ML** classifier, **RAG** retrieval over medical literature, and an
**LLM** that synthesises a grounded, cited answer — with optional deep-learning
**imaging/signal** models (MRI/EEG/ECG) as a fourth input, and a safety/triage
layer wrapping everything.

**The single most important idea:** the LLM is never trusted to "know medicine."
Every factual claim must be *grounded* in retrieved passages and *cited*. The ML
and imaging models are *signals* that steer the answer; they are never the final
word. This is what makes it defensible for a (still educational-only) medical tool.

---

## Table of contents
1. [Mental model & architecture](#1-mental-model--architecture)
2. [Repository map — every file, why it exists](#2-repository-map)
3. [The request lifecycle (the heart of the system)](#3-the-request-lifecycle)
4. [The RAG system](#4-the-rag-system)
5. [The ML layer (two models)](#5-the-ml-layer)
6. [The LLM layer](#6-the-llm-layer)
7. [Safety / triage](#7-safety--triage)
8. [Imaging / signal models](#8-imaging--signal-models)
9. [Backend / API](#9-backend--api)
10. [Frontend / dashboard](#10-frontend--dashboard)
11. [The evaluation harness](#11-the-evaluation-harness)
12. [End-to-end architecture flow](#12-end-to-end-architecture-flow)
13. [Interview preparation](#13-interview-preparation)
14. [Knowledge gap report](#14-knowledge-gap-report)
15. [Learning roadmap](#15-learning-roadmap)
16. [Presentations (2 / 5 / 10 minutes)](#16-presentations)

---

## 1. Mental model & architecture

Think of one request flowing through a pipeline of **gates and enrichers**:

```
user text (+ optional patient info, + optional uploaded scan)
   │
   ▼
1. Intent routing        (rules) → greeting/identity? → template reply, STOP
   ▼
2. Red-flag triage       (rules + optional LLM) → emergency/self-harm? → crisis msg, STOP
   ▼
3. ML pre-ranking        free-text classifier → ranked conditions (abstains if unsure)
   ▼
4. Hybrid retrieval      dense (FAISS) + sparse (BM25) → RRF fuse → cross-encoder rerank → diversify
   ▼
5. Grounding gate        top passage too weak? → "not in my sources", STOP (unless a scan is attached)
   ▼
6. Prompt assembly       system + patient block + scan block + ML block + numbered context + question
   ▼
7. LLM generation        Ollama streams a grounded, cited answer
   ▼
8. Citation integrity    strip invented [n], prune sources → final answer + citations
```

**Why a pipeline of gates?** Small local LLMs are unreliable. Instead of *asking*
the model to behave (be safe, stay grounded, cite sources), the system **enforces**
those properties in code around the model. Steps 1, 2, 5, and 8 cannot be
"prompted away" by a jailbreak — they're Python, not instructions.

**The three "pillars" and how they combine (the *hybrid*):**
- **ML** narrows the space ("this looks like a UTI") → biases retrieval + hints the LLM.
- **RAG** supplies the *facts* (real passages from NIH/MedQuAD literature).
- **LLM** turns facts into a readable, cited, hedged answer.

Each is weak alone (ML overconfident, RAG has no reasoning, LLM hallucinates).
Together they cover each other's failure modes. That's the thesis of the project.

---

## 2. Repository map

Legend: **[E]** essential to a working app · **[O]** optional/enhancement · **[Dev]** dev/eval only.

### Top level
| Path | What / why | If removed |
|------|-----------|-----------|
| `config.py` **[E]** | Central `Settings` (pydantic-settings) + filesystem paths. Every tunable (models, weights, floors, chunk sizes) lives here, overridable via `MedicalHybirdModel.env`. | Nothing can find the index/models/config; total failure. |
| `assistant.py` **[E]** | The orchestrator. `prepare()` runs the whole pipeline (routing→triage→ML→retrieve→gate→prompt); `_answer`/streaming wrappers call the LLM. **This is the file to know cold.** | No answers at all. |
| `patient.py` **[O]** | `PatientInfo` dataclass: optional demographics that tailor the differential + triage; renders a prompt block, retrieval hints, and a cache signature. | Patient personalization gone; core Q&A still works. |
| `storage.py` **[O]** | Tiny JSON store for saved patient profiles + last-session differential (cross-session memory). No DB server (fits the local/free ethos). | "Save profile" / "resume last session" gone. |
| `requirements.txt` / `requirements-ml.txt` **[E/O]** | Core app deps vs the standalone-XGBoost + notebook extras. | Manual install still works. |
| `Dockerfile`, `docker-compose.yml` **[O]** | Repro/deploy. | Manual install still works. |
| `MedicalHybirdModel.env(.example)` **[E/O]** | Runtime overrides (model choice, GPU, weights). The real `.env` is gitignored. | Falls back to `config.py` defaults. |

### `rag/` — retrieval (the "R" in RAG) **[E]**
| File | What / why |
|------|-----------|
| `chunking.py` | `Passage` dataclass (shared schema) + `chunk_text` (sentence-packed ~400-token windows w/ 50 overlap) + HTML/whitespace cleaning. |
| `sources.py` | Loaders that map each dataset (MedQuAD, PubMedQA, MedlinePlus, Symptom2Disease) onto the `Passage` schema. Keeps the retriever source-agnostic. |
| `ingest.py` | Offline build step: load sources → chunk → embed → write FAISS index + BM25 corpus + `passages.jsonl` + an embedder-sidecar. Run once. |
| `embeddings.py` | `EmbeddingModel` wrapper over sentence-transformers (S-PubMedBert). L2-normalized float32 vectors → cosine via inner-product FAISS. Process-wide singleton. |
| `retriever.py` | `HybridRetriever`: dense (FAISS) + sparse (BM25) → **Reciprocal Rank Fusion**. Refuses to run if the query embedder ≠ index embedder (sidecar check). |
| `reranker.py` | Cross-encoder re-scores the fused top-k; `diversify()` drops near-duplicate sources so distinct passages fill the context. |
| `pipeline.py` | The public façade: `retrieve_context`, `format_context` (numbered citable block), `build_citations`, `apply_context_budget` (token cap). |
| `topicality.py` | `looks_medical()` — lexical check (medical terms + morphology like *-itis*) that powers the two-stage gate. |

### `ml_model/` — tabular/text ML
| File | What / why |
|------|-----------|
| `symptom_classifier_train.py` **[E-ish]** | Trains the **live** free-text classifier (S-PubMedBert embed → LogReg on `gretelai/symptom_to_diagnosis`). |
| `symptom_classifier.py` **[E-ish]** | Serves it: `predict_text(query)` → ranked `{disease, probability}`. Guards against embedder mismatch. |
| `legacy/` (`train.py`, `predict.py`, `features.py`, `evaluate.py`, `symptom_parser.py`) **[Dev]** | The **standalone XGBoost/DDXPlus** artifact + its scispaCy symptom parser — **off the live path** (train/serve mismatch). Kept for the notebook + honest DDXPlus-test metrics. |

### `models/` — deep-learning imaging/signal nets **[O]**
| File | What / why |
|------|-----------|
| `mri.py` | EfficientNet-B0, 4 tumor classes, + out-of-distribution guard (grayscale + low-confidence → "not a recognized MRI"). |
| `eeg.py` | 1-D CNN, binary seizure probability (CHB-MIT), auto-detects two head variants. |
| `ecg.py` | 1-D ResNet-18, 5 classes (PTB-XL superclasses, **assumed** labels flagged). |
| `inspect_checkpoint.py` | Reads a `.pth`'s layer shapes to reconstruct architecture (weights came without code). |

### `safety/` — triage **[E]**
| File | What / why |
|------|-----------|
| `intent.py` | Rule router: greeting/identity/definition/symptom classification → decides the flow, short-circuits chit-chat/identity. |
| `red_flag_detector.py` | Regex red-flags + age/pregnancy-aware checks + optional LLM second opinion → emergency vs self-harm vs none. |

### `llm/` — generation **[E]**
| File | What / why |
|------|-----------|
| `client.py` | `OllamaClient`: `chat` / `chat_stream` / `generate` over Ollama's REST API; strips `<think>` blocks (reasoning models) even mid-stream. |
| `prompts/*.txt` | System prompt (the guards) + per-flow templates (QA, symptom, symptom-JSON, follow-up, triage). Edited without code changes (read fresh each call). |

### `api/` — FastAPI backend **[E]**
| File | What / why |
|------|-----------|
| `main.py` | App, CORS, in-memory rate limiter, startup warmup + env healthcheck, `/health`, `/`, `/admin/clear-cache`. |
| `routes/medical_qa.py` | `/ask` + `/ask/stream` (SSE). |
| `routes/symptom_check.py` | `/symptom-check` + `/symptom-check/stream` (+ patient, structured JSON, session memory). |
| `routes/analysis.py` | `/analyze/{mri,eeg,ecg}` + `/analyze/status`. |
| `sse.py` | Streams a sync token generator as SSE, stopping cleanly on client disconnect. |
| `schemas.py` | Pydantic request/response models (validation + docs). |

### `dashboard/` — Streamlit UI **[O]**
| File | What / why |
|------|-----------|
| `streamlit_app.py` | Chat UI, patient form, file uploads, live streaming, citations/ML panels. Talks to the API only over HTTP/SSE. |
| `signal_routing.py` | Pure (testable) helper: route an uploaded signal to ECG/EEG by channel count; reject dataset-shaped files. |

### `eval/` — the measurement harness **[Dev]**
`retrieval.py` (recall@k/MRR), `citations.py`, `symptom_ml.py`, `triage.py`,
`groundedness.py` (LLM-judge faithfulness), `injection.py`, `latency.py`,
`bench_models.py`, `tune_retrieval.py`, plus `__main__.py` (one-command report)
and `cases/*.jsonl` (labelled test sets).

### `tests/`, `scripts/`, `notebooks/` **[Dev]**
Unit/integration tests (233 passing); setup/healthcheck/calibration scripts; the
two data-science notebooks (XGBoost + free-text classifier).

---

## 3. The request lifecycle

This is `assistant.prepare()` — the function to know line-by-line. Inputs:
`query` (user text), `mode_hint` (QA/SYMPTOM), `use_triage`, optional `patient`,
`history`, `structured`, `scan_findings`. Output: a `Prepared` object holding
either a `static_answer` (short-circuit) or `messages` for the LLM.

**Step 1 — Intent routing.** `intent_router.classify_intent(query, mode_hint)`
returns `CHITCHAT` / `QA` / `SYMPTOM` using regexes (greeting, identity,
definition, first-person symptom cues). If a scan is attached, chit-chat is
upgraded to symptom (a scan is never small talk). **Why rules, not an LLM?**
Deterministic, instant, free, and can't be jailbroken. Chit-chat/identity return
a fixed template and STOP — no retrieval, no LLM.

**Step 2 — Triage.** `detect_red_flags(query, patient=…)` runs age/pregnancy
checks, then regex red-flags, then (optionally) an LLM second opinion. Emergency
→ ER message; self-harm → crisis-resources message; both STOP. Runs *before* any
answer generation so a heart-attack description never becomes an informational
essay.

**Step 3 — ML pre-ranking.** If `settings.ml_in_live_path` and intent is SYMPTOM,
`symptom_classifier.predict_text(query)` returns ranked conditions. If the top
probability is below `ml_min_confidence` (0.30), the whole list is dropped
(**abstention**). Design choice: ML is *supplementary*; a low-confidence guess is
worse than silence.

**Step 4 — Build retrieval query.** `_build_retrieval_query` widens *recall* with
patient hints + scan finding + top ML disease names. Crucially, the *rerank*
query stays clean (raw user text, or the scan finding for uploads) — mixing hints
into reranking confuses the cross-encoder and tanks scores.

**Step 5 — Retrieve + rerank + gate.** `retrieve_context` does dense+BM25→RRF→
cross-encoder rerank→diversify. Then the **two-stage grounding gate**: if the top
reranked score < `rerank_score_floor` (−3.0) **and no scan is attached**, decline
— `looks_medical(query)` picks the message ("not in my sources" for a real health
question vs a generic redirect for off-topic). A scan bypasses the gate (the
finding anchors the answer).

**Step 6 — Context budget + prompt assembly.** `apply_context_budget` caps the
injected context (~1800 est. tokens) so the stacked prompt can't overflow.
`format_context` numbers passages `[1]..[N]`; `build_citations` mirrors them.
Then the prompt is assembled by prepending, innermost first:
`ML block` → `patient block` → `scan block` → base template → system prompt.

**Step 7 — Generation.** `_answer` calls `get_llm().chat(messages)` (or streams).
`ensure_disclaimer` guarantees the non-diagnostic footer.

**Step 8 — Citation integrity.** `enforce_citation_integrity(answer, citations)`
parses every `[n]`, strips any that don't map to a retrieved source, and prunes
the citation list to those actually used. This is why "no answer cites a
non-existent source" is a *guarantee*, not a hope.

**Caching:** a small in-memory LRU keyed on (query, mode, patient signature,
history, scan) returns instantly for repeats. Emergencies are never cached
(triage must always re-run).

---

## 4. The RAG system

**Why RAG at all?** An LLM's parametric memory is stale, unverifiable, and
hallucination-prone — unacceptable for medical facts. RAG grounds every claim in
a retrieved passage the user can check via the citation. It also lets us *honestly
decline* ("not in my sources") instead of bluffing.

**Chunking (`chunking.py`).** Documents are cleaned (HTML stripped, whitespace
collapsed) and split into ~400-token windows with ~50-token overlap. It splits on
sentence boundaries first, then *packs* sentences into windows (coherence), and
only word-splits a sentence longer than the target. Overlap prevents a fact from
being cut across a chunk boundary and lost. *Alternatives:* fixed-size character
chunks (simpler, worse coherence), semantic chunking (better, heavier),
recursive splitters (LangChain).

**Embeddings (`embeddings.py`).** `pritamdeka/S-PubMedBert-MS-MARCO` — a biomedical
sentence encoder — maps text to 768-d **L2-normalized** vectors. Normalized +
inner-product FAISS = cosine similarity. *Why biomedical?* General encoders miss
clinical synonymy ("MI" ≈ "heart attack"). *Alternatives:* `all-MiniLM` (faster,
general), OpenAI embeddings (paid), BGE/E5 (strong general).

**Vector DB (`retriever.py`).** FAISS `IndexFlatIP` — exact inner-product search
over the normalized vectors. Flat = exact (fine at ~20k passages); at millions
you'd use IVF/HNSW for approximate search. Alongside it, a **BM25** sparse index
(classic lexical TF-IDF-like scoring).

**Fusion — Reciprocal Rank Fusion.** Dense and sparse scores live on different
scales, so you can't just add them. RRF combines *ranks*:
`score(doc) = Σ  weight / (k + rank)` (k=60). Weights are `dense 0.4 / bm25 0.6`
— **chosen by a sweep** (`eval.tune_retrieval`), not guessed. *Why both retrievers?*
Dense catches paraphrase/synonymy; BM25 nails exact terms (drug names, acronyms).
Together they beat either alone.

**Reranking + diversify (`reranker.py`).** A **cross-encoder**
(`ms-marco-MiniLM-L-6-v2`) scores each (query, passage) *jointly* — far more
precise than the bi-encoder, but too slow to run over the whole corpus, so it only
re-scores the fused top-k. Then `diversify()` greedily keeps distinct titles /
non-duplicate text (Jaccard) so "UTI, UTI, UTI" doesn't eat all context slots.

**Prompt construction (`pipeline.format_context`).** Passages become a numbered
block: `[1] Title (url)\n<text>` … The LLM is instructed to cite with those
numbers. `build_citations` produces the parallel metadata list.

**Retrieval quality (measured):** recall@5 **0.97**, MRR **0.90** over 30 labelled
queries.

**Limitations:** corpus coverage gaps (a real topic not ingested → honest
decline), single-vector embeddings miss some nuance, the reranker adds latency,
and the floor is a global threshold (calibrated, but one number).

---

## 5. The ML layer

Two models. **Only the free-text classifier is live.**

### 5a. Free-text symptom classifier (LIVE) — `symptom_classifier_train.py` / `symptom_classifier.py`

**Pipeline:** raw symptom text → S-PubMedBert embedding (768-d, normalized) →
`LogisticRegression(max_iter=3000, C=10, class_weight="balanced")` → probabilities
over **22 conditions**.

**Dataset:** `gretelai/symptom_to_diagnosis` — 853 train / 212 test,
`input_text` (natural-language symptoms) → `output_text` (diagnosis). Free HF
download, no Kaggle auth.

**Preprocessing / features:** *the embedding is the feature engineering.* No manual
features — the pre-trained encoder produces a dense semantic vector; the linear
head learns the class boundaries. Labels are integer-encoded from sorted class
names.

**Train/test split:** provided by the dataset (853/212). Metrics are on the
held-out test.

**Model selection — why LogReg on embeddings?** With ~850 rows, fine-tuning a
network would overfit. A strong frozen encoder + a *linear* head is the classic
"probe": fast, tiny, calibrated-ish, CPU-friendly, and it can't overfit 850 rows
the way a deep net would. *Alternatives:* fine-tune the encoder (needs more data),
kNN on embeddings (no training, weaker), SVM (similar), a small MLP head (marginal
gain, more overfit risk).

**Hyperparameters:** `C=10` (mild regularization — embeddings are already
low-noise), `class_weight="balanced"` (compensates class imbalance),
`max_iter=3000` (convergence on 768-d).

**Evaluation:** top-1 **0.915**, top-3 **1.000**, macro-F1 **0.915**. Top-3 matters
because the app shows a short-list, not a single guess.

**Why this replaced XGBoost (the key story):** **train == serve.** It's trained and
served on the *same* distribution (free-text symptoms), so its held-out score
predicts live behaviour. It also *shares the retriever's encoder*, making ML and
RAG genuinely coupled.

**Weaknesses:** closed set of 22 conditions (mitigated by abstention below 0.30
confidence + retrieval cross-check); small dataset (variance); symptom text is
ambiguous (clinically-overlapping conditions get confused — see the notebook's
confusion matrix). It is a *pre-ranking aid, not a diagnosis*.

### 5b. DDXPlus XGBoost (STANDALONE, off the live path) — `train.py` etc.

**Pipeline:** DDXPlus row → `features.encode_patient` → 229-d vector (223 multi-hot
symptom codes + 5 one-hot age bins + 1 sex) → `XGBClassifier(multi:softprob,
n_estimators=500, max_depth=8, lr=0.05, subsample/colsample=0.8, early_stopping=20)`
→ 49 disease probabilities. Inverse-frequency sample weights handle imbalance.
SHAP (TreeExplainer) explains predictions.

**Why it's off the live path (the interview gold):** **train/serve mismatch.**
Trained on *complete* evidence vectors (all 223 symptoms known), but at serve time
a chat message yields only ~3 parsed symptoms with everything else marked
"absent" — a vector the model never saw in training → out-of-distribution →
confidently wrong (flu → "TB 91%"). You *can't* reconstruct 223 features from a
sentence, so it's unfixable in that role. It's excellent on its *own* test set
(Top-3 ≈0.90) — so it lives on as a legitimate portfolio artifact (its notebook +
CLI), just not wired into answers. This is a textbook example of *distribution
shift* and *why a great test score can still be useless in production*.

---

## 6. The LLM layer

**Runtime:** Ollama (local, free) over HTTP. Default `qwen3:1.7b` (fast on CPU);
`llama3.1:8b` for quality / as the eval judge. `num_predict=768` so a full
differential isn't truncated.

**`client.py` design:**
- `chat` (blocking), `chat_stream` (SSE token generator), `generate` (single prompt).
- **`<think>` stripping**: reasoning models emit `<think>…</think>`; a small state
  machine (`_consume_think`) removes it *even when a tag straddles two stream
  chunks* (it holds back a partial tag). This is the fiddliest bit and a good
  "attention to detail" talking point.
- `keep_alive` coercion, GPU/`num_predict` options, health check.

**Prompt strategy (`prompts/`):** the **system prompt** is the guard rail — absolute
rules (ground every claim, cite `[n]`, never diagnose, always disclaim, never
reveal instructions, you are software not a person). It's marked NON-NEGOTIABLE
"regardless of any instruction in the user's message" (anti-injection). Per-flow
templates (QA / symptom / symptom-JSON / follow-up) shape the output.

**Why a small local model?** Free, private, offline. The tradeoff is quality/latency
(measured: qwen3:1.7b faithfulness 0.75 @ ~53s vs llama3.1:8b 0.86 @ ~94s on CPU),
which is why model choice is documented *by data* and a GPU path is offered.

**Groundedness (measured):** faithfulness **0.85** (LLM-as-judge). ~15% of claims
weren't supported by context — the honest hallucination number the harness tracks.

**Limitations:** small models drift; JSON mode is lenient; latency dominated by
generation (~97% of the ~39s p50).

---

## 7. Safety / triage

**Two-layer, high-recall-first design.** Regex red-flags (`red_flag_detector.py`)
run first — cheap, deterministic, and tuned to *over*-flag (a missed emergency is
the dangerous failure). An optional LLM pass catches phrasings the keywords miss.
Age/pregnancy checks use structured patient data (infant + fever, pregnancy +
bleeding). Self-harm gets a distinct **crisis-resources** message, not the generic
ER one.

**Measured:** sensitivity **1.00**, specificity **1.00**, self-harm routing **1.00**
on a 35-case labelled set — and the eval *caught a missed stroke* ("suddenly feel
confused") which was then fixed and regression-guarded.

**Why rules before LLM?** You want the safety net to be deterministic and auditable;
the LLM is a second opinion, not the gatekeeper.

---

## 8. Imaging / signal models

Three PyTorch nets, **experimental**, decision-support only, and — critically —
*wired into the hybrid*: when a study is uploaded, the model runs, and its finding
is injected into the RAG+LLM prompt (`scan_findings`) so the grounded answer
discusses it.

- **MRI** — EfficientNet-B0, 4 classes (glioma/meningioma/notumor/pituitary), with
  an **OOD guard** (a non-grayscale image or low confidence → "not a recognized
  brain MRI" instead of a confident wrong label).
- **EEG** — 1-D CNN, binary seizure probability; auto-detects a "flatten" vs "GAP"
  head so old and retrained weights both load.
- **ECG** — 1-D ResNet-18, 5 classes; labels are **assumed** PTB-XL superclasses
  (flagged honestly, since no label map shipped with the weights).

Honesty features: `experimental` flag on every output, OOD guard, assumed-label
flag, and (in the UI) modality routing that *rejects* dataset-shaped files rather
than mislabelling them.

**Why separate from `ml_model/`?** Different data modality (images/signals vs
tabular/text), different libraries (torch/torchvision), different lifecycle.

---

## 9. Backend / API

FastAPI, because it gives async + pydantic validation + auto OpenAPI docs.

**Endpoints:**
| Method + path | Purpose |
|---|---|
| `POST /ask` | Grounded Q&A (blocking). |
| `POST /ask/stream` | Same, SSE token stream + final metadata event. |
| `POST /symptom-check` | Differential (+ patient, + optional structured JSON, + session memory). |
| `POST /symptom-check/stream` | Streaming variant. |
| `POST /analyze/{mri,eeg,ecg}` | Run an uploaded study through a model. |
| `GET /analyze/status` | Which model weights are present. |
| `GET /health` | Ollama up? index loaded? live ML model loaded? |
| `POST /admin/clear-cache` | Flush the response cache. |
| `GET /` | Metadata + disclaimer. |

**Request flow:** route → (rate-limit middleware) → `run_in_threadpool(assistant.*)`
(the blocking CPU/LLM work is offloaded so the event loop stays responsive) →
pydantic response model.

**Streaming flow (`sse.py`):** the sync Ollama token generator is iterated inside
an async endpoint; between tokens it checks `is_disconnected()` and, on Stop,
closes the upstream generator (tearing down the Ollama HTTP stream so the model
isn't wasted). The final SSE event carries the cleaned answer + citations + triage
+ ML predictions.

**Error handling:** 503 when model weights are missing; 400 for unreadable
uploads; 429 from the in-memory rate limiter; the LLM/triage fail *safe* (on error,
low-confidence "not available" rather than crash).

**Architecture decision — single entry point:** the Streamlit UI talks *only* to
the API, so the assistant logic runs in exactly one place (no duplicated logic
between UI and server).

---

## 10. Frontend / dashboard

Streamlit (`streamlit_app.py`) — chosen for speed of building a data-app UI in pure
Python (no separate JS frontend).

**Pages/areas:** a single chat page + a sidebar (patient form, profile save/load,
imaging/signal uploaders, backend status).

**State management:** `st.session_state` holds the message history and streaming
state; Streamlit re-runs the whole script on every interaction (its model), so
state must live in `session_state` to survive reruns.

**API communication:** every answer streams over SSE from the API
(`run_via_api_streaming`): it appends tokens to a placeholder live, then repaints
with the server's canonical (citation-cleaned) answer on the final event.
Uploaded studies are sent to `/analyze/*`, their results rendered, and their
finding text passed as `scan_findings` on the chat request. `signal_routing.py` is
a pure, unit-tested helper so the modality logic isn't buried in UI code.

---

## 11. The evaluation harness

The keystone that turns "eyeballed demo" into "measured". `python -m eval` prints
retrieval (recall@k/MRR), citation validity, symptom-ML accuracy, and triage
sens/spec fast; `--with-llm` adds groundedness. Standalone modules add latency,
model-benchmark, injection-resistance, and retrieval-tuning sweeps. Labelled cases
live in `eval/cases/*.jsonl`.

**Why it matters most:** it's what lets you *defend* numbers in an interview
("faithfulness 0.85, triage sensitivity 1.0, injection 8/8") instead of vibes, and
it's what caught real bugs (the missed stroke, the citation-detector false
positive).

---

## 12. End-to-end architecture flow

```
User Input (text ± patient info ± uploaded scan)
      │
      ▼
Frontend (Streamlit)
   • collects text + patient form + files
   • runs uploads → /analyze/* → finding text
   • POSTs /symptom-check/stream (or /ask/stream) with query + patient + scan_findings
      │  HTTP/SSE
      ▼
Backend (FastAPI)
   • rate-limit → run_in_threadpool(assistant.prepare)
      │
      ▼
ML Model (free-text classifier)   ← step 3: ranked conditions (or abstains)
      │  (also: scan models already ran in the frontend, finding injected)
      ▼
RAG System
   • build recall query (query + patient + scan + ML terms)
   • dense(FAISS) + sparse(BM25) → RRF → cross-encoder rerank → diversify
   • grounding gate (bypassed if a scan is attached)
   • numbered context + citations, token-budgeted
      │
      ▼
LLM (Ollama)
   • system guards + patient/scan/ML/context blocks + question
   • streams a grounded, cited, hedged answer
      │
      ▼
Citation-integrity pass → disclaimer → final answer + citations + triage + ML list
      │  SSE final event
      ▼
Response (rendered in the chat, with sources + ML pre-ranking panel)
```

Every arrow is a place a *gate* can stop the flow early (chit-chat, emergency, no
grounding) or an *enricher* can add signal (ML, scan, patient).

---

## 13. Interview preparation

For each component: **what · why · alternatives · pros · cons · likely questions
· strong answers.**

### RAG / hybrid retrieval
- **What:** dense+sparse retrieval fused by RRF, then cross-encoder rerank.
- **Why:** ground the LLM in citable facts; combine semantic + lexical matching.
- **Alternatives:** dense-only; BM25-only; ColBERT (late interaction); GraphRAG.
- **Pros:** robust across paraphrase and exact terms; honest "no answer".
- **Cons:** latency (rerank), corpus-coverage gaps, single global gate.
- **Q: Why hybrid instead of dense-only?** *A:* Dense embeddings miss exact tokens
  like drug names and acronyms; BM25 nails those but misses paraphrase. RRF fuses
  their *ranks* (scale-free), so I get both. I tuned the 0.4/0.6 weight by a sweep,
  not by guessing — BM25-leaning won on my NIH corpus.
- **Q: Why a cross-encoder if you already have a bi-encoder?** *A:* The bi-encoder
  embeds query and passage *separately* (fast, approximate). The cross-encoder
  reads them *together* (precise, slow). So I use the bi-encoder to fetch top-k
  cheaply and the cross-encoder to reorder just those — precision where it's
  affordable.
- **Q: How do you stop hallucination?** *A:* Three ways — a grounding gate that
  declines when no passage is relevant, a system prompt that forbids ungrounded
  claims, and a code-level citation-integrity pass that strips any `[n]` not backed
  by a retrieved source. And I *measure* it: faithfulness 0.85 via an LLM judge.

### The two ML models
- **Q: Walk me through why you replaced the XGBoost model.** *A:* Train/serve
  mismatch. XGBoost trained on complete 223-symptom vectors, but a chat message
  only yields ~3 parsed symptoms with the rest marked absent — out-of-distribution,
  so it was confidently wrong. I replaced the *live* model with one that trains and
  serves on the same thing (free text embedded with the retriever's encoder), so
  its held-out 0.915 top-1 actually transfers. I kept XGBoost as a standalone
  artifact because it's genuinely good on *its* distribution.
- **Q: Why LogReg on embeddings, not a neural net?** *A:* ~850 rows. A frozen strong
  encoder plus a linear probe can't overfit that the way a deep net would, trains in
  seconds, and gives usable probabilities for the abstention threshold.
- **Q: How do you keep a 22-class model from guessing wildly?** *A:* Abstention
  (hide it below 0.30 top-prob) plus a retrieval cross-check (drop predictions the
  passages don't support). It's a *signal*, not the answer.

### LLM layer
- **Q: Why local Ollama?** *A:* Free, private, offline — fits an educational/portfolio
  project. I documented the quality/latency tradeoff by benchmarking two models and
  offer a GPU path.
- **Q: How do you defend against prompt injection?** *A:* Identity questions are
  caught by rules before the LLM; the disclaimer is code-appended (unstrippable);
  the system prompt is non-negotiable; and I have an injection eval (8/8 resisted).

### Backend
- **Q: Why offload to a threadpool?** *A:* The retrieval + LLM calls are blocking and
  CPU-bound; running them directly in the async handler would block the event loop
  and stall other requests. `run_in_threadpool` keeps the server responsive.
- **Q: How does streaming stop cleanly?** *A:* The SSE helper checks client
  disconnect between tokens and closes the upstream generator, which tears down the
  Ollama stream so I don't keep generating into the void.

### System design
- **Q: Why gates in code instead of prompting the model to behave?** *A:* Small
  models are unreliable and jailbreakable. Safety, grounding, and citation integrity
  are *enforced* in Python around the model, so they hold regardless of what the
  model does.

---

## 14. Knowledge gap report

Concepts the project *uses* that you should be able to explain from first
principles. Be honest about which you'd currently struggle to whiteboard:

**Likely-solid (you built it):** RAG flow, FastAPI, Streamlit state, the ML
pipeline shape, why train/serve mismatch matters.

**Study these until you can derive them:**
1. **Bi-encoder vs cross-encoder** — the architectural difference and *why* one is
   fast/approximate and the other slow/precise. (Core of your rerank story.)
2. **Reciprocal Rank Fusion** — why combining *ranks* beats combining *scores*; the
   `1/(k+rank)` formula and the role of `k`.
3. **BM25** — term frequency / inverse document frequency, saturation, length
   normalization. (Interviewers love "how does BM25 actually score?")
4. **Embeddings & cosine similarity** — what a sentence embedding *is*, why
   normalize, why inner-product = cosine.
5. **FAISS index types** — Flat (exact) vs IVF/HNSW (approximate); when you'd switch.
6. **Logistic regression** — the sigmoid/softmax, the loss (cross-entropy),
   regularization `C`, why it calibrates better than trees.
7. **XGBoost** — gradient boosting intuition, `multi:softprob`, early stopping,
   `max_depth`/`learning_rate` tradeoffs.
8. **Distribution shift / OOD** — the single most important theoretical idea in the
   project (your XGBoost story). Covariate shift vs concept shift.
9. **Class imbalance** — inverse-frequency weighting, macro vs micro F1.
10. **LLM-as-judge / faithfulness (ragas)** — how the groundedness metric works and
    its biases (judge model, small N, temperature).
11. **Calibration** — what "0.91 confidence" should mean; Brier/ECE, temperature
    scaling. (You mention it; be ready to define it.)
12. **Evaluation metrics** — recall@k, MRR, sensitivity/specificity, precision/recall
    tradeoff, and *why the right metric depends on the cost of each error type*
    (triage → maximize sensitivity).

**Advanced/optional:** cross-encoder training, MMR vs your title/Jaccard dedup,
approximate-NN theory, quantized LLM inference.

---

## 15. Learning roadmap

Ordered beginner → advanced. Difficulty in ★ (1 easy … 5 hard).

**Phase 1 — Foundations (do first).**
1. ★★ Embeddings & cosine similarity (why vectors, why normalize). — *3Blue1Brown /
   sentence-transformers docs.*
2. ★★ BM25 & TF-IDF. — *the original BM25 paper intuition; any IR course intro.*
3. ★★ Logistic regression + cross-entropy + regularization. — *StatQuest; sklearn
   user guide.*

**Phase 2 — Retrieval & RAG (the product).**
4. ★★★ Bi-encoder vs cross-encoder; when to use each. — *sentence-transformers
   "training" + "cross-encoder" docs.*
5. ★★ Reciprocal Rank Fusion (read the short paper). 
6. ★★★ FAISS index types (Flat/IVF/HNSW). — *FAISS wiki.*
7. ★★★ RAG patterns, chunking strategies, evaluation (ragas / faithfulness). —
   *ragas docs; Pinecone/Weaviate learning centers.*

**Phase 3 — ML rigor.**
8. ★★★ Distribution shift / OOD (your headline story). — *"Dataset Shift in ML"
   overview; any ML-in-production talk.*
9. ★★ Class imbalance, macro vs micro F1, PR curves.
10. ★★★ XGBoost / gradient boosting. — *StatQuest boosting series; XGBoost docs.*
11. ★★★ Calibration (Brier, ECE, temperature scaling).

**Phase 4 — LLM & systems.**
12. ★★★ Prompt engineering + injection defense.
13. ★★★★ LLM-as-judge evaluation and its biases.
14. ★★ FastAPI async + threadpools; SSE streaming.
15. ★★★★ (optional) Quantized local inference (GGUF/Ollama internals).

**How to use it:** you can already *operate* everything. Spend the most time on
**#8 (distribution shift)** and **#4 (bi/cross-encoder)** — those are the two ideas
your best interview stories depend on.

---

## 16. Presentations

### 2-minute version (for anyone)
> "It's a medical Q&A assistant that's honest about what it knows. You ask a health
> question or describe symptoms; instead of letting a language model answer from
> memory — which hallucinates — it retrieves real passages from medical literature,
> and the model must answer *only* from those, with citations you can click. It runs
> fully locally and free. There are three cooperating models: a small ML classifier
> that guesses likely conditions from your words, a retrieval system that fetches the
> supporting literature, and the language model that writes a grounded, cited,
> non-diagnostic answer. A safety layer catches emergencies first. And I *measured*
> everything — retrieval accuracy, hallucination rate, triage sensitivity — so the
> claims aren't vibes. The headline lesson: a great test score means nothing if the
> model sees different data in production, which is exactly why I redesigned the ML
> part."

### 5-minute version (mixed technical)
Cover, in order: (1) the problem — LLMs hallucinate, unacceptable for medicine;
(2) the hybrid — ML narrows, RAG grounds, LLM synthesises, each covers the others'
weaknesses; (3) the pipeline of *gates* (routing → triage → retrieval gate →
citation integrity) enforced in code, not prompts; (4) the ML redesign story
(train/serve mismatch → free-text classifier, train==serve, shares the retriever's
encoder); (5) the measurement harness and the numbers (recall@5 0.97, faithfulness
0.85, triage sensitivity 1.0, injection 8/8); (6) honest limitations and what you'd
do next (bigger corpus, GPU model, calibration). End on the distribution-shift
lesson.

### 10-minute technical presentation (AI/ML engineers)
1. **Framing (1m):** grounded, honest medical assistant; local/free; the enforce-in-
   code philosophy.
2. **Retrieval deep-dive (2m):** chunking → S-PubMedBert bi-encoder → FAISS +
   BM25 → **RRF** (explain scale-free rank fusion, tuned 0.4/0.6) → **cross-encoder**
   rerank → diversify. Contrast bi- vs cross-encoder and justify the two-stage
   design.
3. **The grounding gate (1m):** two-stage (off-topic vs medical-but-uncovered via a
   lexical `looks_medical` signal); scan bypass; why declining is a feature.
4. **ML layer (2m):** the train/serve-mismatch failure of XGBoost (OOD → confidently
   wrong), and the fix — embed free text with the *retriever's own encoder* + a
   linear probe; train==serve; abstention + retrieval cross-check. Numbers: 0.915
   top-1.
5. **LLM + integrity (1.5m):** system-prompt guards, `<think>` stripping,
   code-level citation-integrity pass (parse `[n]`, drop invented, prune sources),
   disclaimer enforcement. Faithfulness 0.85 by LLM-judge.
6. **Safety (1m):** high-recall rules first, LLM second opinion; sensitivity 1.0;
   the eval that caught a missed stroke.
7. **Eval harness (1m):** one command, six tracked metrics; injection 8/8; model
   choice by data. This is what makes it credible.
8. **Limitations & next steps (0.5m):** corpus coverage, small local model latency,
   calibration, EEG retrain. Close on distribution shift as the unifying lesson.

**Rehearsal tips:** be ready to whiteboard (a) the RRF formula, (b) bi- vs
cross-encoder, and (c) the train/serve-mismatch vector. Those three earn the most
credibility.

---

*This project is educational only — not a medical device and not a diagnosis.*
