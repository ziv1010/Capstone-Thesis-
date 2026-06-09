#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:$APP_ROOT:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

EXPLANATION_DIR="${EXPLANATION_DIR:-$APP_ROOT/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00}"
FULL_GRAPH_DIR="${FULL_GRAPH_DIR:-$APP_ROOT/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPLANATION_DIR}"
MODEL="${MODEL:-$SECTION_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_00/model.pt}"
PRED="${PRED:-$SECTION_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_00/predictions.csv}"
GRAPH="${GRAPH:-$SECTION_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/graph_cache/section/case_star_entity_resolved_cross_bucket_section_sep_lr_decay.reasoning_focused.pt}"
CONFIG="${CONFIG:-$SECTION_GNN/ablations/entity_resolved_data/configs/section/cross_bucket_total_dataset/config.yaml}"
DEVICE="${MASK_AUDIT_DEVICE:-auto}"

"${PYTHON[@]}" "$APP_ROOT/mask_sensitivity_audit.py" \
  --explanation-dir "$EXPLANATION_DIR" \
  --full-graph-dir "$FULL_GRAPH_DIR" \
  --graph-cache "$GRAPH" \
  --predictions-csv "$PRED" \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "$@"
