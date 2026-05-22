#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:$SCRIPT_DIR:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

EXPLANATION_DIR="${EXPLANATION_DIR:-$SCRIPT_DIR/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPLANATION_DIR}"
GRAPH="${GRAPH:-$SECTION_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/graph_cache/section/case_star_entity_resolved_cross_bucket_section_sep_lr_decay.reasoning_focused.pt}"
PRED="${PRED:-$SECTION_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_00/predictions.csv}"

"${PYTHON[@]}" "$SCRIPT_DIR/identity_shortcut_audit.py" \
  --explanation-dir "$EXPLANATION_DIR" \
  --graph-cache "$GRAPH" \
  --predictions-csv "$PRED" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
