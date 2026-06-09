#!/usr/bin/env bash
# Step 02 — Run preprocess_fixed_open.py on the prepared stage-tagged inputs.
# Produces cleaned cases under data/processed/cleaned_cases/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECTION_GNN="$(cd "$EXP_ROOT/.." && pwd)"
cd "$SECTION_GNN"
CONFIG="$EXP_ROOT/config.yaml"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"

LIMIT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARGS+=(--limit "$2"); shift 2 ;;
    --env)   MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "[02_preprocess] env=$MAMBA_ENV config=$CONFIG"
echo "[02_preprocess] clearing generated multistage processed data"
rm -rf "$EXP_ROOT/data/processed" "$EXP_ROOT/data/audits"
micromamba run -n "$MAMBA_ENV" python \
  "$SECTION_GNN/experiments/fixed_open_pipeline/preprocess_fixed_open.py" \
  --config "$CONFIG" \
  "${LIMIT_ARGS[@]}"
echo "[02_preprocess] Done."
