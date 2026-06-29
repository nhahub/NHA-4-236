# Image for the FastAPI backend (and, optionally, the Streamlit UI).
# Ollama runs as a separate service — see docker-compose.yml.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps occasionally needed by faiss/torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch BEFORE requirements.txt.
# Without this, pip resolves torch's Linux CUDA variant as a transitive dep of
# sentence-transformers and pulls ~2 GB of CUDA libraries (cudnn, cublas, nccl,
# triton, etc.) that are useless in a container with no GPU.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# spaCy + a small English model so the symptom parser's NER works in-container,
# which is what enables the ML pre-ranking path. Without a model it falls back to
# a weak regex tokenizer and the classifier rarely fires (it needs >=3 matched
# symptoms). The larger biomedical model (en_core_sci_md — see README) extracts
# symptoms more accurately but is a big download, so we ship the small model.
RUN pip install "spacy>=3.7,<4" \
 && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

# Default: serve the API. Override `command` in compose for Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
