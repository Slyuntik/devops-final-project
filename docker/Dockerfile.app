FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.4.4

RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    transformers==5.5.3 \
    tokenizers==0.22.2 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    prometheus-client==0.19.0 \
    prometheus-fastapi-instrumentator==6.1.0 \
    python-dotenv==1.0.0 \
    pydantic==2.5.0 \
    sentencepiece==0.2.1 \
    safetensors==0.7.0 \
    huggingface_hub==1.10.1 \
    asyncpg==0.29.0

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY data/ ./data/

RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/data/scibert_quality_model

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
