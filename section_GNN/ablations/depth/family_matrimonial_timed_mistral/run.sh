#!/usr/bin/env bash
# Ablation: GNN depth (num_layers 1, 2, 3). Reuses existing full graph — no rebuild.
# Usage: bash run.sh [--depths "1 2 3"] [--k K] [--env ENV]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE_CONFIG="$SECTION_GNN/runs/family_matrimonial_timed_mistral/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K=5
DEPTHS="1 2 3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --depths) DEPTHS="$2"; shift 2 ;;
    --k)      K="$2"; shift 2 ;;
    --env)    MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY="micromamba run -n $MAMBA_ENV python"
GRAPH_CACHE_DIR=$($PY -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['paths']['graph_cache_dir'])")
GRAPH_CACHE_NAME=$($PY -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['graph']['cache_name'])")
GRAPH_CACHE="$GRAPH_CACHE_DIR/$GRAPH_CACHE_NAME"
OUTPUTS_DIR=$($PY -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['paths']['outputs_dir'])")
KFOLD_SCRIPT="$SECTION_GNN/src/scripts/kfold_cv.py"

echo ">>> Depth ablation using graph: $GRAPH_CACHE"

for depth in $DEPTHS; do
  CONFIG="$SCRIPT_DIR/config_depth${depth}.yaml"
  RUN_NAME="ablation_depth${depth}_family_matrimonial_kfold"
  LOG_DIR="$OUTPUTS_DIR/logs"
  mkdir -p "$LOG_DIR"

  echo ""
  echo "=== Depth $depth: run=$RUN_NAME ==="
  pids=()
  for fold_idx in $(seq 0 $((K - 1))); do
    log_file="$LOG_DIR/${RUN_NAME}_fold_${fold_idx}.log"
    (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$fold_idx" \
      $PY "$KFOLD_SCRIPT" \
      --config "$CONFIG" --run-name "$RUN_NAME" \
      --k "$K" --fold "$fold_idx" \
      --graph-cache "$GRAPH_CACHE") 2>&1 | tee -a "$log_file" &
    pids+=("$!")
    echo "  Fold $fold_idx -> GPU $fold_idx  (pid ${pids[-1]})"
  done

  status=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "Fold $i FAILED" >&2; status=1; }
  done
  [[ "$status" -ne 0 ]] && exit "$status"

  $PY "$KFOLD_SCRIPT" --config "$CONFIG" --run-name "$RUN_NAME" --k "$K" \
    --aggregate-only --graph-cache "$GRAPH_CACHE"
  echo "Depth $depth done. Summary: $OUTPUTS_DIR/models/$RUN_NAME/kfold/kfold_summary.json"
done

echo ""
echo "All depth ablations complete."
