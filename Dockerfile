# ── Hugging Face Spaces – FastAPI backend ──────────────────────────────────────
# HF Spaces exposes port 7860 by default.
# The model is downloaded from HF Hub at first startup (~223 MB) and then
# cached inside the container for subsequent requests.
# ───────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# System deps: build tools needed for some torch/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/       ./app/
COPY src/       ./src/

# HF Spaces runs as a non-root user; make cache dir accessible
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Expose the HF Spaces default port
EXPOSE 7860

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
