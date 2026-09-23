FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
# torch/numpy/faiss default their BLAS/OMP thread pool to the *visible*
# CPU count, which in a container often reflects the host's full core
# count, not the fraction actually allocated (Render's free tier: 0.1
# CPU). Each extra thread allocates its own working buffers — real
# memory, not just wasted CPU scheduling — and this project's traffic is
# single-request-at-a-time on a machine this small, so there's no
# parallelism benefit being given up. Found this after a real deploy:
# Render's free tier OOM-killed the app at >512MB despite this same image
# peaking at ~497MB under a local `docker run --memory=512m` test — the
# gap is exactly this kind of per-thread overhead that a synthetic local
# memory cap doesn't reproduce the same way real host CPU visibility does.

WORKDIR /app

# faiss-cpu and torch (pulled in by sentence-transformers) link against
# libgomp at runtime, which python:3.11-slim doesn't ship by default.
# tesseract-ocr is the actual OCR engine pytesseract shells out to — it's
# a system binary, not a pip package, and document_service.py's OCR
# fallback degrades gracefully (skips OCR, logs a warning) if it's ever
# missing, but it's installed here so the image supports it out of the box.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# sentence-transformers pulls in torch, and plain `pip install torch` on
# Linux defaults to the CUDA build — several GB of GPU libraries this
# CPU-only container will never use. Installing the CPU wheel first
# satisfies that dependency before requirements.txt gets a chance to
# trigger the CUDA one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image at build time. Without
# this, embedding_service.py pulls it from Hugging Face Hub on first use —
# fine for a long-lived process, but on a platform with an ephemeral
# filesystem (Render's free tier wipes local disk, and any HF cache with
# it, on every spin-down) that download would repeat on every single cold
# start. Baking it into the image turns a ~76s cold start (measured: ~58s
# download + 18s actual encoding) into just the ~18s encoding step.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
# Alembic migration files + config (used by app/core/database.py's
# init_db() on startup to keep the schema current when DATABASE_URL is set).
COPY alembic ./alembic
COPY alembic.ini .
# One small bundled PDF, auto-ingested on startup only if the vector store
# is empty (app/services/demo_seed_service.py) — exists so a deployment
# with an ephemeral filesystem (e.g. Render's free tier, which wipes local
# disk on every spin-down) always has something to query against without
# requiring a real upload first. Never touches a store that already has
# data (local dev, or any platform with real persistent storage).
COPY demo_corpus ./demo_corpus

# AWS Lambda Web Adapter — a Lambda *extension* (not a layer; layers can't
# attach to package_type="Image" functions), so this is the only way to
# add it to a container-image function. Inert outside Lambda: nothing
# invokes this binary unless Lambda's own runtime bootstrap is present, so
# docker-compose.yml/local `docker run` are unaffected. Verify the current
# release tag at https://github.com/aws/aws-lambda-web-adapter/releases
# before rebuilding against a newer one.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter
COPY entrypoint.sh ./entrypoint.sh

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/uploads /app/vector_store /app/feedback \
    && chmod +x /app/entrypoint.sh \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# entrypoint.sh handles the Lambda-specific /tmp redirection (read-only
# filesystem outside /tmp) and always execs uvicorn at the end — so this
# works unchanged for Render, Cloud Run, Lambda, and local docker-compose.
CMD ["/app/entrypoint.sh"]
