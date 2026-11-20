# Dockerfile — runs the RAG Assistant web app on Hugging Face Spaces (Docker SDK).
# HF Spaces serves whatever listens on port 7860.

FROM python:3.11-slim

# onnxruntime (used by fastembed for the embedder + reranker) needs libgomp1.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as a non-root user (uid 1000). Create it so caches are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install Python deps first so this layer caches between code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the two models at BUILD time → the first user request is fast, not a 2-min download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
 && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the app (secrets and heavy folders are excluded via .dockerignore).
COPY --chown=user . .

EXPOSE 7860

# Production server: ONE worker (so the models load once), several threads for concurrency,
# a generous timeout (model + LLM calls), bound to the port HF Spaces expects.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", \
     "--bind", "0.0.0.0:7860", "--pythonpath", "app,query", "server:app"]
