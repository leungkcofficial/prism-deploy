FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/

# Create empty models directory (populated via volume mount at runtime)
RUN mkdir -p models

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV PRISM_MODELS_DIR=/app/models
EXPOSE 8000

# Single worker — model loading is expensive; scale via replicas instead
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
