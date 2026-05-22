#!/usr/bin/env bash
# Full-graph Leiden community pipeline.
#
#   1. full_graph_community_detection.py     Leiden at multiple resolutions
#   2. community_hierarchy_analysis.py       broad → specific lineage tables
#   3. full_graph_community_profiling.py     per-community profile + accuracy
#   4. bridge_hub_authority_analysis.py      core / bridge / hub classification
#
# Override defaults with environment variables:
#   EXPLANATION_DIR  source explanations (run_summary.json + predictions.csv)
#   OUTPUT_DIR       where to write community + analysis CSVs
#   RESOLUTIONS      comma-separated Leiden resolutions for the sweep
#   PROFILE_RES      comma-separated resolutions to profile + classify
#   NODE_TYPES       comma-separated node types to include in the full graph
#   CASE_LIMIT       optional smoke-test cap on cases (random sample)
#   EDGE_WEIGHTING   binary | log_inverse_degree
#   EXTRA_DETECT_ARGS extra flags forwarded to detection step
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

EXPLANATION_DIR="${EXPLANATION_DIR:-${APP_DIR}/outputs/target_fold_00_8gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${APP_DIR}/outputs/pattern_why_full_graph}"
RESOLUTIONS="${RESOLUTIONS:-0.4,0.7,1.0,1.4,2.0}"
PROFILE_RES="${PROFILE_RES:-1.0}"
NODE_TYPES="${NODE_TYPES:-case,statute,provision,precedent}"
EDGE_WEIGHTING="${EDGE_WEIGHTING:-binary}"
EXTRA_DETECT_ARGS="${EXTRA_DETECT_ARGS:-}"

cd "$APP_DIR"

DETECT_CMD=(
    "${PYTHON[@]}" full_graph_community_detection.py
    --explanation-dir "$EXPLANATION_DIR"
    --output-dir "$OUTPUT_DIR"
    --resolutions "$RESOLUTIONS"
    --node-types "$NODE_TYPES"
    --edge-weighting "$EDGE_WEIGHTING"
)
if [[ -n "${CASE_LIMIT:-}" ]]; then
    DETECT_CMD+=(--case-limit "$CASE_LIMIT")
fi
if [[ -n "$EXTRA_DETECT_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA=( $EXTRA_DETECT_ARGS )
    DETECT_CMD+=("${EXTRA[@]}")
fi

echo "[run] ${DETECT_CMD[*]}"
"${DETECT_CMD[@]}"

"${PYTHON[@]}" community_hierarchy_analysis.py \
    --communities-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

IFS=',' read -ra PROFILE_RES_LIST <<< "$PROFILE_RES"
for res in "${PROFILE_RES_LIST[@]}"; do
    res_trim="$(echo "$res" | xargs)"
    [[ -z "$res_trim" ]] && continue
    "${PYTHON[@]}" full_graph_community_profiling.py \
        --explanation-dir "$EXPLANATION_DIR" \
        --communities-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --resolution "$res_trim"

    "${PYTHON[@]}" bridge_hub_authority_analysis.py \
        --explanation-dir "$EXPLANATION_DIR" \
        --communities-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --resolution "$res_trim"
done

echo "[done] outputs in $OUTPUT_DIR"
