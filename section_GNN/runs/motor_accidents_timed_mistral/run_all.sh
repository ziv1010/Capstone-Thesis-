#!/usr/bin/env bash
# Full pipeline: preprocess -> build graph -> K-fold CV on 8 GPUs.
#
# Usage:
#   bash run_all.sh                                         # full run
#   bash run_all.sh --limit 200                             # smoke-test
#   bash run_all.sh --skip-preprocess                       # skip step 1
#   bash run_all.sh --skip-preprocess --skip-build          # kfold only
#   bash run_all.sh --skip-preprocess --run-name my_run_v2  # custom run name
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
RUN_NAME="${RUN_NAME:-$(basename "$SCRIPT_DIR")_kfold}"
K="${K:-5}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
LIMIT=""
SKIP_PREPROCESS=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)             MAMBA_ENV="$2";   shift 2 ;;
    --run-name)        RUN_NAME="$2";    shift 2 ;;
    --k)               K="$2";           shift 2 ;;
    --limit)           LIMIT="$2";       shift 2 ;;
    --gpus-build)      GPUS_BUILD="$2";  shift 2 ;;
    --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
    --skip-build)      SKIP_BUILD=true;  shift ;;
    --help|-h) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS+=(--limit "$LIMIT")

echo "============================================================"
echo "Pipeline: $(basename "$SCRIPT_DIR")"
echo "  env=$MAMBA_ENV  run=$RUN_NAME  k=$K  build_gpus=$GPUS_BUILD"
[[ -n "$LIMIT" ]] && echo "  limit=$LIMIT"
echo "  skip_preprocess=$SKIP_PREPROCESS  skip_build=$SKIP_BUILD"
echo "============================================================"

if [[ "$SKIP_PREPROCESS" == false ]]; then
  echo ""
  echo ">>> Step 1/3: Preprocessing"
  bash "$SCRIPT_DIR/01_preprocess.sh" --env "$MAMBA_ENV" "${LIMIT_ARGS[@]}"
fi

if [[ "$SKIP_BUILD" == false ]]; then
  echo ""
  echo ">>> Step 2/3: Building graph"
  bash "$SCRIPT_DIR/02_build_graph.sh" --env "$MAMBA_ENV" --gpus "$GPUS_BUILD" "${LIMIT_ARGS[@]}"
fi

echo ""
echo ">>> Step 3/3: K-fold cross-validation"
bash "$SCRIPT_DIR/03_kfold_8gpu.sh" \
  --env "$MAMBA_ENV" \
  --run-name "$RUN_NAME" \
  --k "$K"

echo ""
echo "Pipeline complete."
