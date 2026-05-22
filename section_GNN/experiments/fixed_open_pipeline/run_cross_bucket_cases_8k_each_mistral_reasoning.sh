#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATED_GRAPH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="$SCRIPT_DIR/cross_bucket_cases_8k_each_mistral_reasoning_config.yaml"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
ENV_NAME="${CONDA_ENV:-thesis_work}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="$DEFAULT_CONFIG"
LIMIT_ARG=()

usage() {
    cat <<'EOF'
Usage:
  bash run_cross_bucket_cases_8k_each_mistral_reasoning.sh
  bash run_cross_bucket_cases_8k_each_mistral_reasoning.sh --limit 5000
  bash run_cross_bucket_cases_8k_each_mistral_reasoning.sh --env thesis_work
  bash run_cross_bucket_cases_8k_each_mistral_reasoning.sh --config /path/to/config.yaml

Pipeline:
  1. Preprocess cross-bucket FIXED_OPEN JSONs into cleaned cases
  2. Build the updated reasoning-focused graph bundle
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --limit)
            LIMIT_ARG=(--limit "$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
    echo "micromamba not found: $MICROMAMBA_BIN" >&2
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi
if ! "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Micromamba environment not found: $ENV_NAME" >&2
    exit 1
fi

echo "============================================================"
echo "  Cross-Bucket Updated Graph Pipeline"
echo "  Micromamba   : $MICROMAMBA_BIN"
echo "  Environment  : $ENV_NAME"
echo "  Config       : $CONFIG_PATH"
if [ ${#LIMIT_ARG[@]} -gt 0 ]; then
    echo "  Limit        : ${LIMIT_ARG[1]}"
fi
echo "============================================================"
echo ""

echo ">>> Step 1/2: preprocess_fixed_open.py"
"$MICROMAMBA_BIN" run -n "$ENV_NAME" "$PYTHON_BIN" \
    "$SCRIPT_DIR/preprocess_fixed_open.py" \
    --config "$CONFIG_PATH" \
    "${LIMIT_ARG[@]}"

echo ""
echo ">>> Step 2/2: build_graph.py"
"$MICROMAMBA_BIN" run -n "$ENV_NAME" "$PYTHON_BIN" \
    "$UPDATED_GRAPH_ROOT/build_graph.py" \
    --config "$CONFIG_PATH" \
    "${LIMIT_ARG[@]}"

echo ""
echo "Pipeline complete."
