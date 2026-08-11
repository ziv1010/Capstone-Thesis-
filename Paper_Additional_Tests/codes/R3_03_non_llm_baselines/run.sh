#!/usr/bin/env bash
# R3-03: non-LLM baselines on the HGT's own five folds.
#
#   bash run.sh                    # full run: build features, 7 valid rows, tables
#   bash run.sh --limit 2000       # smoke test on the first 2,000 cases
#   bash run.sh --skip-build       # reuse the cached features
#
# Uses the `llm` micromamba env, which already has scikit-learn and xgboost, so
# nothing is installed and the `thesis_work` env that produced the paper numbers
# is left untouched. CPU only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_ENV="${MAMBA_ENV:-llm}"
PYTHON="$(micromamba run -n "$MAMBA_ENV" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)"
[[ -x "$PYTHON" ]] || { echo "Could not resolve interpreter for env '$MAMBA_ENV'" >&2; exit 1; }

LIMIT=""
SKIP_BUILD=0
WORKERS="${WORKERS:-48}"
FOLDS="0,1,2,3,4"
MODELS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; FOLDS="0"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --env) MAMBA_ENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$HERE/outputs"

if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "=== building features from the leakage-controlled cleaned cases ==="
  "$PYTHON" "$HERE/build_features.py" --workers "$WORKERS" \
    ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee "$HERE/outputs/build_features.log"
fi

echo
echo "=== auditing TF-IDF/HGT provenance and leakage guards ==="
"$PYTHON" "$HERE/audit_tfidf_inputs.py" 2>&1 | tee "$HERE/outputs/audit_tfidf_inputs.log"

echo
echo "=== running baselines on the HGT's own folds ==="
"$PYTHON" "$HERE/run_baselines.py" --n-jobs "$WORKERS" --folds "$FOLDS" \
  ${MODELS:+--models "$MODELS"} 2>&1 | tee "$HERE/outputs/run_baselines.log"

echo
echo "=== leakage diagnostic on the strongest text model (fold 0) ==="
"$PYTHON" "$HERE/diagnose_leakage.py" 2>&1 | tee "$HERE/outputs/diagnostic_tfidf_leakage.log"

echo
echo "=== generating LaTeX table and report ==="
"$PYTHON" "$HERE/make_tables.py"
