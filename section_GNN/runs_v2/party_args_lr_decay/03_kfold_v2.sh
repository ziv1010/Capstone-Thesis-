#!/usr/bin/env bash
# K-fold runner for party_args_lr_decay experiment.
# Uses kfold_cv_v2.py (ReduceLROnPlateau + 90 epochs).
# One fold per GPU (GPUs 0-4), then aggregates.
set -euo pipefail

V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$V2_DIR/../.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
RUN_NAME="${RUN_NAME:-kfold_run_v2}"
CONFIG="${CONFIG:-}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GRAPH_CACHE="${GRAPH_CACHE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)          MAMBA_ENV="$2";   shift 2 ;;
    --run-name)     RUN_NAME="$2";    shift 2 ;;
    --config)       CONFIG="$2";      shift 2 ;;
    --k)            K="$2";           shift 2 ;;
    --val-fraction) VAL_FRACTION="$2";shift 2 ;;
    --graph-cache)  GRAPH_CACHE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config required" >&2; exit 1; }

KFOLD_SCRIPT="$V2_DIR/scripts/kfold_cv_v2.py"
GRAPH_CACHE_ARGS=()
[[ -n "$GRAPH_CACHE" ]] && GRAPH_CACHE_ARGS+=(--graph-cache "$GRAPH_CACHE")

OUTPUTS_DIR="$(grep 'outputs_dir:' "$CONFIG" | awk '{print $2}')"
LOG_DIR="$OUTPUTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[kfold_v2] env=$MAMBA_ENV  run=$RUN_NAME  k=$K"
echo "[kfold_v2] Launching $K folds in parallel on GPUs 0-$((K-1)) ..."

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
  echo "[kfold_v2] Fold $fold_idx -> GPU $fold_idx  (pid ${pids[-1]})  log: $log_file"
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[kfold_v2] Fold $i complete."
  else
    echo "[kfold_v2] Fold $i FAILED." >&2
    status=1
  fi
done

[[ "$status" -ne 0 ]] && { echo "[kfold_v2] One or more folds failed." >&2; exit "$status"; }

echo "[kfold_v2] All folds done. Aggregating ..."
micromamba run -n "$MAMBA_ENV" python "$KFOLD_SCRIPT" \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --k "$K" \
  --aggregate-only \
  "${GRAPH_CACHE_ARGS[@]}"

SUMMARY="$OUTPUTS_DIR/models/$RUN_NAME/kfold/kfold_summary.json"
echo "[kfold_v2] Done. Summary -> $SUMMARY"
