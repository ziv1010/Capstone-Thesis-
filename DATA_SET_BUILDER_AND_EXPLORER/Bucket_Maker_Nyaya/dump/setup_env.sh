#!/usr/bin/env bash
# setup_env.sh — Create and populate the bucket_maker micromamba environment

set -e

ENV_NAME="bucket_maker"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════"
echo " Bucket Maker — Environment Setup"
echo "════════════════════════════════════════"
echo ""

# Check micromamba is available
if ! command -v micromamba &> /dev/null; then
    echo "ERROR: micromamba not found. Please install it first."
    exit 1
fi

echo "[setup] Creating micromamba env: ${ENV_NAME}"
micromamba create -n "${ENV_NAME}" \
    -f "${SCRIPT_DIR}/environment.yml" \
    --yes \
    -q

echo ""
echo "[setup] Installing pip packages inside env..."
micromamba run -n "${ENV_NAME}" pip install \
    tqdm \
    huggingface_hub>=0.23.0 \
    pandas \
    requests \
    --quiet

echo ""
echo "[setup] Done! Environment '${ENV_NAME}' is ready."
echo ""
echo "To run the full pipeline:"
echo "  bash run_pipeline.sh"
echo ""
echo "Or run individual steps:"
echo "  micromamba run -n ${ENV_NAME} python 01_csv_to_json.py"
echo "  micromamba run -n ${ENV_NAME} python 02_bucket_classifier.py"
echo "  micromamba run -n ${ENV_NAME} python 03_build_bucket_datasets.py"
