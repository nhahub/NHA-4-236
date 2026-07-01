# 🩺 Hybrid Medical Assistant — LLM + RAG + ML

A medical Q&A and symptom-exploration assistant built on a **three-layer hybrid
architecture**: a free-text ML classifier pre-ranks candidate conditions from the
user's symptom description, RAG retrieves grounded clinical passages, and an LLM
synthesises both into a cited, ranked differential exploration. The classifier
shares the retriever's S-PubMedBert encoder, so ML and RAG are genuinely coupled
— and because it trains and serves on the *same* free text, it stays in
-distribution at serve time (unlike the earlier DDXPlus model, now a standalone
artifact — see [ML model details](#ml-model-details)).

Runs **fully locally and free**: a local open model via [Ollama](https://ollama.com),
[FAISS](https://github.com/facebookresearch/faiss) for vector search,
`sentence-transformers` for embeddings, [MedQuAD](https://github.com/abachaa/MedQuAD)
+ other NIH sources as the knowledge base, and
[gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
for the live symptom classifier.

> ⚠️ **Educational use only. This is not a medical device and does not provide a
> diagnosis.** Always consult a qualified healthcare professional. In an
> emergency, call your local emergency number.

---

## Results at a glance

Every claim below is measured by the eval harness (`python -m eval` +
[the eval modules](#evaluation)) — not eyeballed. Numbers are on this machine
(CPU, `qwen3:1.7b` generator, `llama3.1:8b` judge).

| What | Metric | How |
|------|--------|-----|
| Retrieval | recall@5 **0.97**, MRR **0.90** (30 cases) | `eval.retrieval` |
| Symptom ML (free-text) | top-1 **0.915**, top-3 **1.00** (held-out) | `eval.symptom_ml` |
| Grounding / hallucination | faithfulness **0.85** | `eval.groundedness` |
| Emergency triage | sensitivity **1.00**, specificity **1.00** | `eval.triage` |
| Prompt-injection resistance | **8/8 (100%)** | `eval.injection` |
| Citation integrity | invented `[n]` markers stripped | `eval.citations` |
| Latency (QA, CPU) | retrieval ~1.2 s, end-to-end p50 **39 s** / p95 60 s | `eval.latency` |

Retrieval fusion weights and the generator model were **chosen from data**
(`eval.tune_retrieval`, `eval.bench_models`), not guessed. See
[Evaluation](#evaluation) and [Model choice](#model-choice-benchmarked).

---

## Architecture overview

```
 user query  (+ optional patient info, + mode toggle as a hint)
        │
        ▼
 1. Intent routing (rules, no LLM)
      • greeting / identity questions  → template reply (stop)
      • definition ("what is X")       → Q&A flow
      • first-person symptoms          → differential flow
        │
        ▼
 2. Red-flag triage (keyword rules + age/pregnancy-aware + light LLM)
      • emergency    → urgent-care message (stop)
      • self-harm    → crisis-resources message (stop)
        │ not an emergency
        ▼
 3. [ML] ml_model/text_predict.py: free-text symptoms → S-PubMedBert embedding
        │ → logistic-regression classifier → ranked conditions (top-5)
        │ top-1 below the abstention floor → skip ML (stay pure LLM+RAG)
        ▼
 4. [ML→RAG] top predictions bias the retrieval recall query; each prediction is
        │ cross-checked against the retrieved passages (unsupported ones hidden)
        ▼
 5. Hybrid retrieval (dense FAISS + BM25, RRF fusion)
 6. Cross-encoder rerank → top-N passages
      • confidence gate: top score below floor → "no relevant info" (stop)
        │ grounded
        ▼
 7. LLM (Ollama) STREAMS a grounded, cited answer
      • ML pre-ranking injected into prompt as a starting signal
      • LLM must reason from retrieved context; notes any contradiction with ML output
      • /ask           → direct grounded answer
      • /symptom-check → ranked differential (+ patient context, + ML hints)
      • non-diagnostic disclaimer guaranteed on every answer
```

Steps 1, 2, 6's gate, and the disclaimer are **enforced in code** — an 8B model
cannot skip them. The ML layer is **gracefully optional**: if the model has not
been trained yet, or if fewer than 3 symptoms were matched from the user's text,
the system falls back silently to pure LLM+RAG.

---

## Key behaviours

- **ML pre-ranking** — XGBoost trained on DDXPlus (49 disease classes, 223
  symptom codes) pre-ranks candidate conditions from structured symptom features.
  Predictions are injected into the LLM prompt as a signal, not a final answer.
  The LLM is explicitly instructed to flag contradictions between ML output and
  retrieved evidence.
- **Symptom parser** — converts free-text symptom descriptions to DDXPlus
  feature vectors using scispaCy NER + exact/fuzzy code matching. Extracts
  demographic hints (age, sex) via regex. Falls back to pure RAG when matched
  symptom count is below the confidence threshold (< 3).
- **Intent router** — chit-chat, greetings, identity questions, and gibberish
  get template replies (no retrieval, no hallucinated sources). Treatment /
  dosage / severity follow-ups are answered with the plain Q&A flow even when a
  patient form is filled in — they never re-run the differential.
- **Confidence gate** — off-topic / out-of-corpus queries are declined instead
  of fabricating an answer over irrelevant passages.
- **Streaming + Stop** — answers stream token-by-token with a Stop button; with
  the model kept resident (`keep_alive`) the first token lands in ~1 s after
  warm-up. The API also exposes SSE streaming endpoints for custom frontends.
- **Multi-turn** — prior turns are sent as context, so follow-ups ("3 days",
  "it's worse at night") are understood; recent turns also fold into the
  retrieval query so a bare follow-up still retrieves in context.
- **Patient info (optional)** — structured age / sex / duration / history /
  meds / lifestyle that tailors the differential, feeds age-aware triage
  (infant + fever, pregnancy + bleeding), and seeds the ML demographic features.
- **Existing-condition guard** — known conditions are excluded from retrieved
  passages and the differential via passage filtering, prompt injection, and
  reranked-pool expansion.
- **Safety** — emergencies and self-harm short-circuit before any LLM call;
  the disclaimer is appended in code if the model ever drops it.
- **Rate limiting** — the API throttles `/ask` and `/symptom-check` to
  10 requests per IP per 60 s.

---

## Project layout

```
.
├── assistant.py               # orchestration: route → triage → ML → retrieve → gate → LLM
├── symptom_parser.py          # free text → DDXPlus feature vector (scispaCy + fuzzy match)
├── patient.py                 # optional structured patient info
├── storage.py                 # JSON user-profile store
├── config.py                  # central settings (env-overridable)
│
├── ml_model/                  # XGBoost ML classifier (Phase 2)
│   ├── features.py            # multi-hot symptom encoding + age/sex features (~229 dims)
│   ├── train.py               # training script: python -m ml_model.train
│   ├── evaluate.py            # metrics: top-k accuracy, F1, Brier score, SHAP, confusion matrix
│   ├── predict.py             # inference: load artifacts, return ranked disease list
│   └── artifacts/             # xgb_model.json, label_encoder.pkl, feature_columns.pkl
│
├── rag/
│   ├── ingest.py              # parse/clean/chunk MedQuAD → FAISS + BM25
│   ├── embeddings.py          # sentence-transformers wrapper
│   ├── retriever.py           # hybrid dense + BM25 (RRF fusion)
│   ├── reranker.py            # cross-encoder re-ranking
│   └── pipeline.py            # retrieve → rerank → formatted context
│
├── llm/
│   ├── client.py              # Ollama REST client (blocking + streaming, think-strip)
│   └── prompts/               # system / qa / symptom / triage prompts
│
├── safety/
│   ├── intent.py              # pre-retrieval intent router + chit-chat templates
│   └── red_flag_detector.py   # rule + age-aware + LLM triage; self-harm template
│
├── api/                       # FastAPI app (/ask, /symptom-check, /health)
├── dashboard/streamlit_app.py # chat-style demo UI (streaming + patient form)
│
├── data/
│   └── raw/
│       ├── MedQuAD/           # RAG knowledge base (Phase 1)
│       └── ddxplus/           # ML training corpus (Phase 2, downloaded automatically)
│
├── notebooks/
│   └── ml_model_analysis.ipynb  # standalone data science portfolio notebook
│
├── tests/
│   ├── test_ml_model.py       # unit tests for features.py + predict.py
│   ├── test_rag_retrieval.py
│   ├── test_faithfulness.py   # Ragas eval
│   └── ...
│
├── scripts/
│   ├── download_medquad.py
│   ├── calibrate_gate.py
│   └── run_tests.py
│
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Quickstart (local)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~16 GB RAM recommended for an 8B model (CPU works; GPU is faster)

### 2. Install

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy MedicalHybirdModel.env.example MedicalHybirdModel.env
```

**Linux / macOS:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp MedicalHybirdModel.env.example MedicalHybirdModel.env
```

Phase 2 ML dependencies (install once, same on all platforms):

```bash
pip install xgboost shap
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

### 3. Pull a model

```bash
ollama pull llama3.1:8b
# lighter:  ollama pull mistral:7b-instruct-q4
```

### 4. Build the RAG knowledge base

**Fast path (one command, idempotent):**

```bash
python -m scripts.setup            # download MedQuAD + build the index
python -m scripts.setup --with-ml  # also train the XGBoost symptom classifier
```

It skips any step whose output already exists and reports whether Ollama and the
configured model are reachable. Or do the steps manually:

```bash
python scripts/download_medquad.py    # clones MedQuAD into data/raw/
python -m rag.ingest                  # MedQuAD only → FAISS + BM25 + passages
```

The knowledge base is multi-source and pluggable:

| Source | `--sources` name | Get the data | Extra deps |
|--------|-----------------|--------------|------------|
| **MedQuAD** (default) | `medquad` | `python scripts/download_medquad.py` | — |
| **PubMedQA** | `pubmedqa` | auto-downloads from Hugging Face | `pip install datasets` |
| **MedlinePlus** | `medlineplus` | topic XML from [medlineplus.gov/xml.html](https://medlineplus.gov/xml.html) | — |
| **Symptom2Disease** | `symptom2disease` | [Kaggle CSV](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) | — |

```bash
python -m rag.ingest --sources medquad pubmedqa   # multiple sources
python -m rag.ingest --sources all                # every source present in data/raw/
```

### 5. Train the symptom classifier (recommended, fast)

```bash
# Downloads a small HF dataset, embeds it with S-PubMedBert, trains the
# logistic-regression head, and saves the artifacts to ml_model/artifacts/.
python -m ml_model.text_train      # ~1-2 min; prints held-out metrics
```

This trains the **live** ML pillar (the free-text classifier). It's optional —
without it the assistant runs as pure LLM+RAG — but recommended, and it's on the
live path by default (`ML_IN_LIVE_PATH=true`) once trained. Held-out metrics:
top-1 **0.915**, top-3 **1.000**, macro F1 **0.915**.

<details><summary>Optional: the standalone DDXPlus XGBoost artifact</summary>

The original XGBoost model is a separate portfolio artifact, **not** on the live
path (see [ML model details](#ml-model-details) for why). To reproduce it:

```bash
python -m ml_model.train           # downloads DDXPlus (~4 GB), ~10-30 min on CPU
python -m ml_model.evaluate        # Top-3 ~0.90+, Macro F1 ~0.85+, SHAP plot
```
</details>

### 6. Run

Always launch through the **active interpreter** with `python -m` rather than a
bare `uvicorn`/`streamlit` console script — a copied or stale virtualenv can ship
an `.exe` whose shebang points at the wrong interpreter, silently importing the
wrong packages. Verify the environment first:

```bash
python -m scripts.healthcheck   # reports the interpreter + any missing deps
```

The API also runs this check (warn-only) at startup. Then:

```bash
python -m uvicorn api.main:app --reload        # API at http://localhost:8000/docs
python -m streamlit run dashboard/streamlit_app.py   # UI at http://localhost:8501
```

---

## API

| Endpoint | Method | Body | Purpose |
|----------|--------|------|---------|
| `/ask` | POST | `{"query": "...", "use_triage": true}` | Grounded answer to a question |
| `/ask/stream` | POST | same as `/ask` | SSE token stream + metadata |
| `/symptom-check` | POST | `{"query": "...", "use_triage": true, "patient": {...}}` | Ranked differential (+ optional ML pre-ranking, off by default) |
| `/symptom-check/stream` | POST | same as `/symptom-check` | SSE token stream + metadata |
| `/analyze/mri` | POST | multipart image | Brain-tumour classification (4 classes) |
| `/analyze/eeg` | POST | multipart `.npy` `(23, samples)` | Seizure-probability screening |
| `/analyze/ecg` | POST | multipart `.npy` `(12, samples)` | 12-lead rhythm/diagnostic class |
| `/analyze/status` | GET | — | Which imaging/signal model weights are present |
| `/health` | GET | — | Ollama + index + ML model status |
| `/admin/clear-cache` | POST | — | Flush in-memory response cache |

`/ask` and `/symptom-check` also accept an optional `scan_findings` string — the
summary of an attached study, which is fused into the grounded answer and used to
bias retrieval (the Streamlit chat does this automatically when you attach a file).

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is influenza and how does it spread?"}'
```

`/symptom-check` accepts an optional `patient` object (all fields optional):

```bash
curl -X POST http://localhost:8000/symptom-check \
  -H "Content-Type: application/json" \
  -d '{
        "query": "persistent cough, fatigue, and low-grade fever for 5 days",
        "patient": {"age": 68, "sex": "male", "conditions": "type 2 diabetes",
                    "smoking": "current"}
      }'
```

Patient fields: `age, sex, duration, severity, conditions, medications,
allergies, smoking, alcohol, pregnancy, other` — all optional.

Both endpoints accept an optional `history` array for multi-turn context:

```json
{ "query": "and the treatment?",
  "history": [
    {"role": "user",      "content": "what is anemia"},
    {"role": "assistant", "content": "Anemia is..."}
  ] }
```

---

## ML model details

### Live pillar — free-text symptom classifier

The model on the live path (`ml_model/text_predict.py`) embeds the user's raw
symptom text with the **same S-PubMedBert encoder the retriever uses**, then a
calibrated logistic-regression head maps that embedding to a condition. Because
it trains and serves on the same distribution (natural-language symptom
descriptions), it is in-distribution at serve time — the fix for the DDXPlus
model's train/serve mismatch.

| Property | Value |
|----------|-------|
| Training data | [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) — 853 train / 212 test, free text |
| Features | 768-d S-PubMedBert sentence embedding (shared with RAG) |
| Head | multinomial logistic regression (`class_weight="balanced"`) |
| Classes | 22 conditions |
| **Held-out top-1** | **0.915** |
| **Held-out top-3** | **1.000** |
| **Macro F1** | **0.915** |

Train it (small, ~1–2 min): `python -m ml_model.text_train`. Metrics surface in
the eval harness via `python -m eval.symptom_ml`.

**Integration (hybrid loop):** the top predictions (a) widen the RAG recall
query so the retriever fetches passages about them, and (b) are prepended to the
LLM prompt as a *starting signal* — the model is told to reason from the
retrieved context and flag any contradiction. Two guards keep it honest:
- **Abstention** — if the top class is below `ML_MIN_CONFIDENCE` (0.30), no ML is
  shown (vague / out-of-scope text gets no confident guess).
- **Retrieval cross-check** — predictions the retrieved passages don't support are
  hidden from the user-facing differential (the LLM still sees the full list with
  caution flags). Toggle via `assistant._ML_SHOW_ONLY_SUPPORTED`.

It is a *supplementary* signal: the grounded, cited LLM answer is authoritative,
and if the classifier isn't trained the answer is pure LLM+RAG.

### Standalone artifact — DDXPlus XGBoost

The original XGBoost classifier (trained on [DDXPlus](https://github.com/mila-iqia/ddxplus),
~1M synthetic cases, 49 conditions, 229 structured features) is **kept as a
standalone portfolio artifact, off the live path**. It was demoted because it was
fed only a few regex-parsed symptoms at serve time (everything else "absent") →
out-of-distribution and confidently wrong. Train it with `python -m ml_model.train`,
evaluate with `python -m ml_model.evaluate` (Top-3 ~0.90+, Macro F1 ~0.85+, SHAP
explainability → `ml_model/artifacts/shap_importance.png`). The
`notebooks/ml_model_analysis.ipynb` notebook reproduces its full pipeline (EDA,
training, evaluation, SHAP) and runs independently on Kaggle/Colab.

---

## Imaging & signal models (MRI / EEG / ECG)

Three optional deep-learning **screening** models live in the `models/` package,
separate from the tabular symptom classifier. Drop their PyTorch weights into
`models/checkpoints/` (`mri.pth`, `eeg.pth`, `ecg.pth`) and they light up
automatically.

> 🧪 **EXPERIMENTAL.** Every output from these models is labelled `experimental`
> (in the API JSON, the CLI footer, and the dashboard) and is decision-support
> only — never a diagnosis. Specific safeguards:
> - **MRI** has an out-of-distribution guard: a non-grayscale image (e.g. a
>   photo) or a low-confidence scan returns *"not a recognized brain MRI"* with
>   `"ood": true` instead of asserting a tumour class.
> - **ECG** class names are **assumed** (PTB-XL superclasses); with no label map
>   supplied, outputs carry `"assumed_labels": true`. Confirm the order before
>   trusting the class names.
> - **EEG** metrics are pending a patient-level-split retrain.

| Modality | Architecture | Input | Output |
|----------|-------------|-------|--------|
| **MRI** | EfficientNet-B0 | brain-MRI image (jpg/png) | glioma / meningioma / notumor / pituitary |
| **EEG** | 1-D CNN | `.npy` `(23, samples)` @ 256 Hz | seizure probability (binary) |
| **ECG** | 1-D ResNet-18 | `.npy` `(12, samples)` | 5-class (assumed PTB-XL superclasses) |

**Two ways to use them:**

1. **In the chat** — attach a file to a message (📎). The result is shown and its
   finding is fused into the grounded answer (the LLM discusses it against the
   retrieved literature).
2. **Direct endpoints** — `POST /analyze/{mri,eeg,ecg}` (see the API table).

```bash
# Generate tiny synthetic test signals, then try an endpoint:
python -m scripts.make_sample_signals
curl.exe -F "file=@samples/ecg_sample.npy" http://localhost:8000/analyze/ecg
```

Inputs are auto-oriented (a transposed `(samples, channels)` array is handled).
Inspect any checkpoint with `python -m models.inspect_checkpoint <file>`.

> ⚠️ **These are decision-support screeners, not diagnostic tools.** Known
> limitations shipped as-is: **ECG** class names are *assumed* (set
> `models.ecg.CLASS_NAMES` once you confirm the real order); **EEG** validation
> metrics are optimistic until retrained with a patient-level split. See
> `models/README.md`.

---

## Evaluation

One command prints a report across the assistant's pillars:

```bash
python -m eval                 # full report (fast)
python -m eval --with-llm      # also run groundedness (slow, needs Ollama)
python -m eval.retrieval       # recall@k / MRR only
python -m eval.citations       # citation-validity rate only
python -m eval.triage          # triage sensitivity / specificity only
python -m eval.symptom_ml      # free-text classifier held-out metrics
python -m eval.groundedness    # faithfulness via LLM-as-judge (slow, needs Ollama)
python -m eval.tune_retrieval  # sweep fusion weights / top_k for the best config
```

Implemented today:

- **Retrieval** — `recall@k` and `MRR` over a labelled query set
  (`eval/cases/retrieval.jsonl`; query → known-relevant passage titles).
- **Citation validity** — fraction of answers that cite only real sources, run
  through the same `enforce_citation_integrity` pass that ships (an invented
  `[7]` when 3 passages were retrieved is stripped from the answer and flagged).
- **Triage sensitivity / specificity** — the rule-based red-flag layer over a
  labelled emergency / self-harm / non-emergency set (`eval/cases/triage.jsonl`).
  Sensitivity must stay ~1.0 (a missed emergency is the dangerous failure);
  self-harm cases are also checked to route to the crisis-resources message. The
  LLM second pass needs a running Ollama and is layered on top, not measured here.

- **Groundedness / faithfulness** *(LLM-as-judge, needs Ollama)* — a stronger
  judge model (`OLLAMA_JUDGE_MODEL`, default llama3.1:8b) extracts each answer's
  factual claims and checks them against the retrieved context; faithfulness =
  supported / total. A direct ragas-style metric, no heavy ragas dependency.
- **Retrieval tuning** — `python -m eval.tune_retrieval` sweeps the dense/BM25
  fusion split and `top_k` over the eval set so they're chosen from data. (The
  defaults shipped — `DENSE_WEIGHT=0.4`, BM25-leaning — came from this sweep.)
- **Latency** *(needs Ollama)* — `python -m eval.latency` times the pipeline per
  flow (mean / p50 / p95), splitting the retrieval stage from end-to-end. Measured
  (CPU, qwen3:1.7b, QA flow): retrieval ~1.2 s, **end-to-end p50 39 s / p95 60 s** —
  generation is ~97 % of the time, so model choice and `num_predict` are the
  latency levers, not retrieval.

### Model choice (benchmarked)

`python -m eval.bench_models` runs the groundedness eval per generator model and
reports faithfulness vs latency. Measured on this machine (CPU, 5 cases, llama3.1:8b
judge):

| Generator | Faithfulness | Mean latency | p95 latency |
|-----------|-------------|--------------|-------------|
| `qwen3:1.7b`  | 0.75 | 53 s | 59 s |
| `llama3.1:8b` | **0.86** | 94 s | 103 s |

**The choice, from data:** `llama3.1:8b` is meaningfully more faithful (+0.11, ~15 %
relative fewer unsupported claims) but ~75 % slower on CPU. So:

- **CPU demo → `qwen3:1.7b`** (the shipped default): ~53 s/answer is already the
  ceiling of tolerable; the citation-integrity pass and grounding gate offset its
  lower faithfulness.
- **Quality / GPU / cloud → `llama3.1:8b`**: the faithfulness win is worth it, and
  on a GPU its latency drops to seconds, making it the clear pick. Set
  `OLLAMA_MODEL=llama3.1:8b` (and unset `OLLAMA_NUM_GPU=0`).

(Faithfulness has run-to-run variance at this small N and non-zero temperature —
treat these as directional, and re-run the bench on your hardware.)

---

## Configuration

All settings live in `config.py` and are overridable via env vars or `MedicalHybirdModel.env`:

| Setting | Env var | Default | Purpose |
|---------|---------|---------|---------|
| LLM model | `OLLAMA_MODEL` | `llama3.1:8b` | Generation model |
| Force CPU | `OLLAMA_NUM_GPU` | `0` | Set to `0` if CUDA build crashes |
| Output cap | `OLLAMA_NUM_PREDICT` | `768` | Max tokens per response; 512 truncates long differentials |
| Keep resident | `OLLAMA_KEEP_ALIVE` | `-1` | Keep model loaded between calls |
| Timeout | `OLLAMA_TIMEOUT` | `300` | Seconds; CPU generation is slow |
| Embedder | `EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Must match what the index was built with |
| Retrieval | `RETRIEVAL_TOP_K` | `20` | Candidates fused from dense + BM25 |
| Fusion split | `DENSE_WEIGHT` / `BM25_WEIGHT` | `0.4` / `0.6` | Dense:BM25 RRF weights (tuned via `eval.tune_retrieval`) |
| Rerank | `RERANK_TOP_N` | `3` | Passages sent to the LLM |
| Confidence gate | `RERANK_SCORE_FLOOR` | `-3.0` | Decline if top reranked score is below this |
| ML in live path | `ML_IN_LIVE_PATH` | `true` | Use the free-text symptom classifier as a pre-ranking signal |
| ML abstention | `ML_MIN_CONFIDENCE` | `0.30` | Hide ML unless the top class clears this probability |

For a snappier demo on CPU: `OLLAMA_MODEL=qwen3:1.7b` (its `<think>` output is
stripped automatically) and lower `OLLAMA_NUM_PREDICT`.

---

## Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin v2.20+)
- ~20 GB free disk space (Ollama model ~5 GB, MedQuAD data, Python deps)
- 8 GB RAM minimum; 16 GB recommended for the 8B model

There are two modes depending on whether Ollama is already installed on your machine.

---

### Mode A — Ollama already running locally (recommended if model is already pulled)

The `api` container points to `host.docker.internal:11434` by default, so it
reaches the Ollama process on your host without any port conflict.

**1. Create your env file (once):**

```powershell
# Windows
copy MedicalHybirdModel.env.example MedicalHybirdModel.env
```
```bash
# Linux / macOS
cp MedicalHybirdModel.env.example MedicalHybirdModel.env
```

**2. Build the RAG index (once):**

```bash
docker compose run --rm api python scripts/download_medquad.py
docker compose run --rm api python -m rag.ingest
```

**3. Train the ML classifier (optional, once):**

```bash
docker compose run --rm api python -m ml_model.train
```

**4. Start the stack:**

```bash
docker compose up api dashboard
```

---

### Mode B — Fully containerised (Ollama runs inside Docker)

Use this on a machine where Ollama is **not** installed locally.

**1. Start the containerised Ollama and wait for it to be healthy:**

```bash
docker compose --profile dockerized-ollama up -d ollama
docker compose ps          # wait until Status shows "healthy" (~20 s)
```

**2. Pull the LLM into the container:**

```bash
docker compose exec ollama ollama pull llama3.1:8b
# Lighter alternative: ollama pull mistral:7b-instruct-q4
```

**3. Create your env file, build the index, train ML (same as Mode A steps 1–3 above).**

**4. Start the full stack:**

```bash
docker compose --profile dockerized-ollama up api dashboard
```

> For Mode B, edit `MedicalHybirdModel.env` and set:
> `OLLAMA_BASE_URL=http://ollama:11434`
> so the api container uses the Ollama container instead of the host.

---

### Services

| Service | URL |
|---------|-----|
| API + interactive docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Health check | http://localhost:8000/health |

### Useful commands

```bash
# Rebuild the image after code changes:
docker compose build api

# Check logs:
docker compose logs -f api

# Stop everything:
docker compose down

# Stop and remove volumes (wipes pulled models — requires re-pull):
docker compose down -v
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `api` exits immediately on first start | Run `docker compose ps` — if `ollama` isn't `healthy` yet, wait 30 s and retry. The healthcheck retries 10 times with 10 s intervals. |
| Answers truncate mid-sentence | `OLLAMA_NUM_PREDICT` in the compose file is set to `768`; lower values (≤512) cut off long differentials. |
| `FileNotFoundError: Missing index artifact` | Step 4 (ingest) was skipped or wrote to the wrong path — re-run `docker compose run --rm api python -m rag.ingest`. |
| GPU / CUDA crash inside container | `OLLAMA_NUM_GPU: 0` is set in `docker-compose.yml` (force CPU). Remove or set to a positive value only if your Docker host has NVIDIA Container Toolkit installed. |

---

## Tests & evaluation

```bash
pytest -q                                       # all offline unit tests (includes ML tests)
python scripts/run_tests.py                     # integration checks (needs the FAISS index)
RUN_RAGAS=1 pytest tests/test_faithfulness.py  # Ragas faithfulness eval (needs Ollama)
python -m ml_model.evaluate                     # ML evaluation on test split (needs trained model)
ruff check .                                    # lint
```

The ML test suite (`tests/test_ml_model.py`) runs fully offline — no model
artifacts, no network, no Ollama needed. Tests that require trained artifacts
auto-skip until `python -m ml_model.train` has been run.

---

## Safety & limitations

- **Not a diagnosis.** All output is a non-diagnostic exploration, never a
  definitive diagnosis. The LLM is instructed never to diagnose, prescribe,
  claim to be human/a doctor, reveal its prompt, or be talked out of these rules.
- **ML is synthetic-data trained.** DDXPlus was generated from a medical
  knowledge base, not real patient records. Performance on real clinical data
  would likely be lower. ML predictions are a starting signal, not a clinical
  finding.
- **Code-enforced guardrails.** Triage, intent routing, the confidence gate, and
  the disclaimer are enforced in code — not left to model instruction.
- **Emergency triage is a safety net, not a guarantee** — it errs toward
  flagging and should never delay calling emergency services.
- **Off-topic queries are declined** (confidence gate) rather than answered over
  irrelevant passages.
- Answer quality is bounded by the indexed corpus and the local model. Use the
  Ragas faithfulness eval to track grounding quality when changing models or
  retrieval settings.

---

## Data & licensing

**MedQuAD** — created by Asma Ben Abacha and Dina Demner-Fushman. Review the
[MedQuAD repository](https://github.com/abachaa/MedQuAD) for license and usage
terms.

**DDXPlus** — Fansi Tchango, A. et al. (NeurIPS 2022). Released for **research
use only**. Do not use this system commercially or clinically. Dataset:
[HuggingFace](https://huggingface.co/datasets/aai530-group6/ddxplus) /
[Figshare](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585) /
[GitHub](https://github.com/mila-iqia/ddxplus).

If you enable the optional RAG sources, review each one's terms:
[PubMedQA](https://github.com/pubmedqa/pubmedqa) (MIT),
[MedlinePlus](https://medlineplus.gov/about/developers/webservices/) (NLM terms),
[Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)
(Kaggle dataset license). Symptom2Disease is used only as retrievable reference
text, never as classifier training data.
