#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Creates a micromamba env called "bucket_classify", installs vLLM + deps,
# then runs the classification script on all 8 GPUs via tensor parallelism.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="bucket_classify"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/classify_buckets.py"

# ── 1. Create env (skip if it already exists) ─────────────────────────────────
if micromamba env list | grep -q "^${ENV_NAME}"; then
    echo "[setup] Environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "[setup] Creating micromamba env '${ENV_NAME}' with Python 3.11 …"
    micromamba create -n "$ENV_NAME" python=3.11 -c conda-forge -y
fi

# ── 2. Install Python deps ────────────────────────────────────────────────────
echo "[setup] Installing dependencies …"
micromamba run -n "$ENV_NAME" pip install --upgrade pip --quiet

# vLLM ships its own optimised CUDA kernels; install the pre-built wheel.
# pinning to a known-good version; bump as needed.
micromamba run -n "$ENV_NAME" pip install \
    "vllm>=0.4.3" \
    "pandas>=2.0" \
    "tqdm>=4.66" \
    "huggingface_hub>=0.23" \
    "transformers>=4.40" \
    "accelerate>=0.29" \
    --quiet

echo "[setup] All deps installed."

# ── 3. Run classification ─────────────────────────────────────────────────────
echo ""
echo "[run] Starting classification with Mistral-7B-Instruct across 8 GPUs …"
echo "[run] Output will be written to: $SCRIPT_DIR/bucketed/"
echo ""

micromamba run -n "$ENV_NAME" python "$PYTHON_SCRIPT"
