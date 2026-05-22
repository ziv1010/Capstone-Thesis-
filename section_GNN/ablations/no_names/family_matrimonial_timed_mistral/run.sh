#!/usr/bin/env bash
# Ablation: no-names — preamble section dropped; judge, court, party, lawyer nodes removed.
# Cross-case sharing is ON (statute/provision/precedent nodes shared across cases).
# Usage: bash run.sh [--skip-build] [--run-name NAME] [--gpus LIST] [--k K] [--env ENV]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
RUN_NAME="ablation_no_names_family_matrimonial_kfold"
K=5
GPUS="0,1,2,3,4,5,6,7"
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --run-name)   RUN_NAME="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    --k)          K="$2"; shift 2 ;;
    --env)        MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY="micromamba run -n $MAMBA_ENV python"

if [[ "$SKIP_BUILD" == false ]]; then
  echo ">>> Building ablation graph (no-names, cross-case sharing ON) ..."
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    $PY "$SECTION_GNN/src/scripts/build_graph.py" --config "$CONFIG"
  echo ">>> Graph build done."
fi

GRAPH_CACHE_DIR=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['graph_cache_dir'])")
GRAPH_CACHE_NAME=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['graph']['cache_name'])")
GRAPH_CACHE="$GRAPH_CACHE_DIR/$GRAPH_CACHE_NAME"
OUTPUTS_DIR=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['outputs_dir'])")
KFOLD_SCRIPT="$SECTION_GNN/src/scripts/kfold_cv.py"
LOG_DIR="$OUTPUTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo ">>> K-fold: run=$RUN_NAME  k=$K  graph=$GRAPH_CACHE"
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

echo ">>> Aggregating ..."
$PY "$KFOLD_SCRIPT" --config "$CONFIG" --run-name "$RUN_NAME" --k "$K" \
  --aggregate-only --graph-cache "$GRAPH_CACHE"
echo "Done. Summary: $OUTPUTS_DIR/models/$RUN_NAME/kfold/kfold_summary.json"
