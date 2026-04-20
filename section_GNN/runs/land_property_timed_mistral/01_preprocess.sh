#!/usr/bin/env bash
# Step 1: Preprocess raw JSON files into cleaned cases.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
LIMIT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARGS+=(--limit "$2"); shift 2 ;;
    --env)   MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
echo "[01_preprocess] env=$MAMBA_ENV config=$CONFIG"
micromamba run -n "$MAMBA_ENV" python \
  "$SECTION_GNN/experiments/fixed_open_pipeline/preprocess_fixed_open.py" \
  --config "$CONFIG" \
  "${LIMIT_ARGS[@]}"
echo "[01_preprocess] Done."
