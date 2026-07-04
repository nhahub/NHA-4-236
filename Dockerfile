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
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# NOTE: the live ML pre-ranking is the free-text symptom classifier
# (ml_model/symptom_classifier.py) — it embeds the raw text with S-PubMedBert and
# needs no spaCy/NER. The legacy DDXPlus XGBoost (ml_model/legacy/) and its spaCy
# symptom parser are off the live path, so they aren't installed here.

COPY . .

EXPOSE 8000

# Default: serve the API. Override `command` in compose for Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
