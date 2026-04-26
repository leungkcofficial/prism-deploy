FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Trained models (baked into the image — ~750 MB excl. RSF) ─────────────────
# The RSF DR-Learner (rsf_dr/dr_learner.pkl, 3.6 GB) is the dominant artefact.
# All models are copied at build time so the container runs standalone.
COPY models/ ./models/

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV PRISM_MODELS_DIR=/app/models
EXPOSE 8000

# Single worker — all models load into one process (~2 GB RSS). Scale by running
# multiple container replicas behind a load balancer, not via --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
