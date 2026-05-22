#!/usr/bin/env bash
# Multi-hearing stage test pipeline.
#
# Steps:
#   01 prepare flat input dir + manifest         (python, no env required)
#   02 preprocess (cleaned cases)                (env: thesis_work)
#   03 build self-contained reasoning graph      (env: thesis_work, needs GPU)
#   04 run inference w/ trained kfold model      (env: thesis_work, needs GPU)
#   05 analyse stage/model/raw transitions       (env: thesis_work)
#   06 (optional, --explain) Graph_Analyser      (env: graph_explainer, GPU)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="${GPUS:-0}"
EXPLAIN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --env)  MAMBA_ENV="$2"; shift 2 ;;
    --explain) EXPLAIN="true"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$EXP_ROOT"
echo "[run_all] EXP_ROOT=$EXP_ROOT  env=$MAMBA_ENV  gpus=$GPUS  explain=$EXPLAIN"

echo "[run_all] === 01 prepare input ==="
python3 scripts/01_prepare_input.py

echo "[run_all] === 02 preprocess ==="
bash scripts/02_preprocess.sh --env "$MAMBA_ENV"

echo "[run_all] === 03 build graph ==="
bash scripts/03_build_graph.sh --env "$MAMBA_ENV" --gpus "$GPUS"

echo "[run_all] === 04 run inference ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
  micromamba run -n "$MAMBA_ENV" python scripts/04_run_inference.py

echo "[run_all] === 05 analyze transitions ==="
micromamba run -n "$MAMBA_ENV" python scripts/05_analyze_transitions.py
micromamba run -n "$MAMBA_ENV" python scripts/05b_aggregate_transitions.py
micromamba run -n "$MAMBA_ENV" python scripts/05c_per_case_factors.py --changed-only
micromamba run -n "$MAMBA_ENV" python scripts/05d_raw_outcome_factors.py
micromamba run -n "$MAMBA_ENV" python scripts/05e_early_signal_test.py

if [[ "$EXPLAIN" == "true" ]]; then
  echo "[run_all] === 06 explain transitions (Graph_Analyser) ==="
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
    micromamba run -n "$MAMBA_ENV" python scripts/06_explain_transitions.py --mamba-env graph_explainer
fi

echo "[run_all] DONE"
