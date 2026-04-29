#!/usr/bin/env bash
# Step 03 — Build a self-contained reasoning-focused graph from the cleaned
# stage-tagged cases. Output: data/graph_cache/stage_test_graph.reasoning_focused.pt
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECTION_GNN="$(cd "$EXP_ROOT/.." && pwd)"
CONFIG="$EXP_ROOT/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="${GPUS:-0}"

LIMIT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARGS+=(--limit "$2"); shift 2 ;;
    --gpus)  GPUS="$2"; shift 2 ;;
    --env)   MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "[03_build_graph] env=$MAMBA_ENV gpus=$GPUS"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
  micromamba run -n "$MAMBA_ENV" python \
  "$SECTION_GNN/final_graph/build_graph.py" \
  --config "$CONFIG" \
  "${LIMIT_ARGS[@]}"
echo "[03_build_graph] Done."
