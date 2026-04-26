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

# ── Download trained models from Hugging Face Hub ─────────────────────────────
# Models are hosted at https://huggingface.co/datasets/PRISM-CKD/prism-models
# This layer is cached by Docker; models are only re-downloaded when the
# HF_MODELS_REPO build arg changes or the cache is explicitly cleared.
ARG HF_MODELS_REPO=leungkc/prism-models
RUN python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("HF_MODELS_REPO") or "PRISM-CKD/prism-models"
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir="/app/models",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Models downloaded OK.")
PYEOF

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV PRISM_MODELS_DIR=/app/models
EXPOSE 8000

# Single worker — all models load into one process (~2 GB RSS). Scale by running
# multiple container replicas behind a load balancer, not via --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
