"""Central configuration for the medical RAG assistant.

All settings are read from environment variables (optionally a local `.env`)
with local-friendly defaults, so the project runs out of the box without any
paid API keys.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Filesystem layout ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
USERS_DIR = DATA_DIR / "users"  # persisted patient profiles (JSON, one per user)

# Concrete artifact paths produced by `rag/ingest.py`.
PASSAGES_PATH = PROCESSED_DIR / "passages.jsonl"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
BM25_CORPUS_PATH = VECTOR_STORE_DIR / "bm25_corpus.json"
# Sidecar recording which embedding model built the index, so retrieval can
# refuse to run a query embedder that disagrees with the indexed vectors.
INDEX_META_PATH = VECTOR_STORE_DIR / "index_meta.json"

PROMPTS_DIR = ROOT_DIR / "llm" / "prompts"


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a `.env` file."""

    model_config = SettingsConfigDict(env_file="MedicalHybirdModel.env", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_triage_model: str = "llama3.1:8b"
    ollama_timeout: int = 120
    # Number of model layers to offload to GPU. None = let Ollama auto-detect.
    # Set to 0 to force CPU (use this if Ollama crashes with a CUDA error).
    ollama_num_gpu: int | None = None
    # Cap generated tokens. CPU inference cost scales with output length, so a
    # bound keeps answers (and latency) in check. 512 leaves room for a full
    # symptom differential (restatement + conditions + care guidance +
    # disclaimer) without overflowing; QA answers finish well under it.
    ollama_num_predict: int = 512
    # How long Ollama keeps the model resident after a call. "-1" = forever,
    # which avoids a multi-second reload penalty between queries.
    ollama_keep_alive: str = "-1"

    # Embeddings
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    # Confidence gate: if the top reranked passage scores below this, the query
    # has no good grounding (off-topic / out-of-corpus) and we decline instead
    # of forcing a citation. CrossEncoder logits — calibrated with
    # `python -m scripts.calibrate_gate` (on-topic >= +0.8, off-topic <= -3.9;
    # -3.0 keeps real queries while declining gibberish/off-topic). Only
    # applied when use_reranker=True.
    rerank_score_floor: float = -3.0

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_n: int = 3
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    # Diversify the reranked passages so near-duplicate sources don't eat the
    # limited context slots (MedQuAD has many Q&A pairs per disease, so an
    # un-diversified top-N was often "UTI, UTI, UTI"). Greedy selection prefers
    # distinct titles and non-duplicate text; backfills with dupes only if there
    # aren't enough distinct sources to fill top_n. Jaccard is over text tokens.
    rerank_dedup: bool = True
    dedup_jaccard_threshold: float = 0.85
    # Cap the retrieved context injected into the prompt so the stacked prompt
    # (system + ML/patient/scan blocks + context + query + room to answer) can't
    # silently overflow the model window. Budget is in *estimated* tokens
    # (~4 chars/token); passages are kept in rank order until the budget is hit,
    # then the next passage is truncated to fit. Trims are logged, never silent.
    max_context_tokens: int = 1800

    # Chunking
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # Symptom ML (XGBoost on DDXPlus). Off the live path by default: it is fed
    # only the handful of symptoms parsed from free text (everything else marked
    # absent), a train/serve mismatch that makes it confidently wrong. The LLM
    # already builds the differential from grounded context. Kept as a standalone
    # portfolio artifact (CLI + notebook). Set true to re-inject it into answers.
    ml_in_live_path: bool = False


settings = Settings()
