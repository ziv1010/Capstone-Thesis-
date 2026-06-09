#!/usr/bin/env bash
# Step 3: Stratified K-fold cross-validation. One fold per GPU, all run in parallel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SECTION_GNN"
CONFIG="$SCRIPT_DIR/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
RUN_NAME="${RUN_NAME:-ablation_hierarchical_enc_motor_accidents_kfold}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GRAPH_CACHE="${GRAPH_CACHE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)          MAMBA_ENV="$2";    shift 2 ;;
    --run-name)     RUN_NAME="$2";     shift 2 ;;
    --k)            K="$2";            shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --graph-cache)  GRAPH_CACHE="$2";  shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

KFOLD_SCRIPT="$SECTION_GNN/src/scripts/kfold_cv.py"
GRAPH_CACHE_ARGS=()
[[ -n "$GRAPH_CACHE" ]] && GRAPH_CACHE_ARGS+=(--graph-cache "$GRAPH_CACHE")

# Derive outputs_dir from config
OUTPUTS_DIR="$(grep 'outputs_dir:' "$CONFIG" | awk '{print $2}')"
LOG_DIR="$OUTPUTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[03_kfold] env=$MAMBA_ENV  run=$RUN_NAME  k=$K"
echo "[03_kfold] Launching $K folds in parallel on GPUs 0-$((K-1)) ..."

pids=()
for fold_idx in $(seq 0 $((K - 1))); do
  log_file="$LOG_DIR/${RUN_NAME}_fold_${fold_idx}.log"
  (
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$fold_idx" \
      micromamba run -n "$MAMBA_ENV" python "$KFOLD_SCRIPT" \
      --config "$CONFIG" \
      --run-name "$RUN_NAME" \
      --k "$K" \
      --fold "$fold_idx" \
      --val-fraction "$VAL_FRACTION" \
      "${GRAPH_CACHE_ARGS[@]}"
  ) 2>&1 | tee -a "$log_file" &
  pids+=("$!")
  echo "[03_kfold] Fold $fold_idx -> GPU $fold_idx  (pid ${pids[-1]})  log: $log_file"
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[03_kfold] Fold $i complete."
  else
    echo "[03_kfold] Fold $i FAILED." >&2
    status=1
  fi
done

[[ "$status" -ne 0 ]] && { echo "[03_kfold] One or more folds failed." >&2; exit "$status"; }

echo "[03_kfold] All folds done. Aggregating ..."
micromamba run -n "$MAMBA_ENV" python "$KFOLD_SCRIPT" \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --k "$K" \
  --aggregate-only \
  "${GRAPH_CACHE_ARGS[@]}"

SUMMARY="$OUTPUTS_DIR/models/$RUN_NAME/kfold/kfold_summary.json"
echo "[03_kfold] Done. Summary -> $SUMMARY"
