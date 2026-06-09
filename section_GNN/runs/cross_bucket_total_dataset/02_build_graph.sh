#!/usr/bin/env bash
# Step 2: Build the reasoning-focused graph.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SECTION_GNN"
CONFIG="$SCRIPT_DIR/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LIMIT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARGS+=(--limit "$2"); shift 2 ;;
    --gpus)  GPUS="$2"; shift 2 ;;
    --env)   MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
echo "[02_build_graph] env=$MAMBA_ENV gpus=$GPUS"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
  micromamba run -n "$MAMBA_ENV" python \
  "$SECTION_GNN/src/scripts/build_graph.py" \
  --config "$CONFIG" \
  "${LIMIT_ARGS[@]}"
echo "[02_build_graph] Done."
