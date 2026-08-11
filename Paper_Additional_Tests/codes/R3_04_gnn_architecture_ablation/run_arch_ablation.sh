#!/usr/bin/env bash
# R3-04: GNN architecture ablation on the paper's own graph, splits and trainer.
#
# Runs each architecture through the unmodified k-fold harness, one fold per GPU.
# The paper's HGT run is NOT re-run -- its recorded numbers are the reference row.
#
#   bash run_arch_ablation.sh                      # all architectures, GPUs 1,2,4,5,6
#   bash run_arch_ablation.sh --only gcn,sage      # a subset
#   bash run_arch_ablation.sh --folds 0            # smoke test on one fold
#   bash run_arch_ablation.sh --gpus 2,4 --wide    # include width-matched controls
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SECTION_GNN="$REPO/section_GNN"
GRAPH_CACHE="$SECTION_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/graph_cache/section/case_star_entity_resolved_cross_bucket_section_sep_lr_decay.reasoning_focused.pt"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="1,2,4,5,6"
FOLDS="0,1,2,3,4"
ONLY=""
WIDE=0
SKIP_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --env) MAMBA_ENV="$2"; shift 2 ;;
    --wide) WIDE=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a FOLD_ARR <<< "$FOLDS"

# Call the interpreter directly: five concurrent `micromamba run` invocations
# contend on the mamba process lock and stall.
PYTHON="$(micromamba run -n "$MAMBA_ENV" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)"
[[ -x "$PYTHON" ]] || { echo "Could not resolve interpreter for env '$MAMBA_ENV'" >&2; exit 1; }

LOG_DIR="$HERE/outputs/run_logs"
mkdir -p "$LOG_DIR"

echo "=== R3-04 GNN architecture ablation ==="
echo "env:         $MAMBA_ENV"
echo "gpus:        ${GPU_ARR[*]}"
echo "folds:       ${FOLD_ARR[*]}"
echo "graph cache: $GRAPH_CACHE"
[[ -f "$GRAPH_CACHE" ]] || { echo "Graph cache not found" >&2; exit 1; }

# Regenerate configs so they cannot drift from the paper config.
"$PYTHON" "$HERE/make_configs.py" $([[ $WIDE -eq 1 ]] && echo --wide)

if [[ $SKIP_CHECK -eq 0 ]]; then
  echo
  "$PYTHON" "$HERE/check_harness_equivalence.py"
fi

if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a ARCH_ARR <<< "$ONLY"
else
  # Ordered from no-graph upward, so the ladder fills in a readable order.
  ARCH_ARR=(mlp gcn sage gat rgcn hgat)
  [[ $WIDE -eq 1 ]] && ARCH_ARR+=(gcn_wide sage_wide)
fi

for arch in "${ARCH_ARR[@]}"; do
  CONFIG="$HERE/configs/arch_${arch}.yaml"
  RUN_NAME="arch_${arch}_kfold"
  [[ -f "$CONFIG" ]] || { echo "No config for architecture '$arch'" >&2; exit 1; }

  echo
  echo "--- $arch --------------------------------------------------------------"
  pids=()
  for i in "${!FOLD_ARR[@]}"; do
    fold="${FOLD_ARR[$i]}"
    gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
    log="$LOG_DIR/${arch}_fold_${fold}.log"
    echo "  fold $fold -> GPU $gpu  ($log)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HERE/kfold_arch_cv.py" \
      --config "$CONFIG" --graph-cache "$GRAPH_CACHE" --run-name "$RUN_NAME" \
      --k 5 --val-fraction 0.1 --fold "$fold" > "$log" 2>&1 &
    pids+=($!)
    # Stagger starts so five processes do not read the 6.3 GB cache simultaneously.
    sleep 20
  done

  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if [[ $status -ne 0 ]]; then
    echo "  !! at least one fold failed for $arch -- see $LOG_DIR/${arch}_fold_*.log" >&2
    grep -hE "Error|Traceback|CUDA out of memory" "$LOG_DIR/${arch}"_fold_*.log | head -5 >&2 || true
    exit 1
  fi

  "$PYTHON" "$HERE/kfold_arch_cv.py" \
    --config "$CONFIG" --run-name "$RUN_NAME" --k 5 --aggregate-only \
    > "$LOG_DIR/${arch}_aggregate.log" 2>&1
  "$PYTHON" - "$HERE/outputs/models/$RUN_NAME/kfold/kfold_summary.json" <<'PY'
import json, sys
agg = json.loads(open(sys.argv[1]).read())["aggregate"]
print(f"  => acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}  "
      f"macro_f1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}  "
      f"roc_auc={agg.get('roc_auc_mean', float('nan')):.4f}")
PY
done

echo
echo "Done. Summaries under $HERE/outputs/models/*/kfold/kfold_summary.json"
