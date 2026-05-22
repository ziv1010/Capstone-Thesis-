#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

"${PYTHON[@]}" "$SCRIPT_DIR/explain_hgt.py" \
  --model-path "$SECTION_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset/models/ablation_section_sep_enc_cross_bucket_kfold/kfold/fold_00/model.pt" \
  --graph-cache "$SECTION_GNN/data/timed_bucket_runs/cross_bucket_total_dataset/graph_cache/case_star_cross_bucket_section_sep_enc.reasoning_focused.pt" \
  --config "$SECTION_GNN/ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml" \
  --output-dir "$SCRIPT_DIR/outputs/target_fold_00" \
  --split test \
  --overwrite \
  "$@"
