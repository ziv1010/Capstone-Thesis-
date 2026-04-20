#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full overnight pipeline: hierarchical encoding — all 5 domain buckets + cross-bucket
#
# For each bucket (single domains first, cross-bucket last):
#   1. Build graph with hierarchical BGE-M3 encoding (new cache, old cache untouched)
#   2. Run 5-fold CV (5 folds in parallel across GPUs 0-4)
#   3. Run embedding analysis (extract → t-SNE → probing → attention → error → shap)
#
# All outputs go under:
#   section_GNN/outputs/timed_bucket_runs/<bucket>/models/ablation_hierarchical_enc_*
#   section_GNN/outputs/timed_bucket_runs/<bucket>/embedding_analysis/hier_*
#
# Old results are never touched — all new data is in distinctly named dirs.
#
# Usage:
#   nohup bash section_GNN/runs/run_hierarchical_enc_all_buckets.sh \
#     > section_GNN/runs/hier_overnight.log 2>&1 &
#   tail -f section_GNN/runs/hier_overnight.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON="micromamba run -n $MAMBA_ENV python"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

ABLATION_DIR="$SECTION_GNN/ablations/hierarchical_enc"
ANALYSIS_DIR="$SECTION_GNN/embedding_analysis"
BUILD_SCRIPT="$SECTION_GNN/final_graph/build_graph.py"
KFOLD_SCRIPT="$SECTION_GNN/dump2/scripts/kfold_cv.py"

# ── Single-domain buckets first, cross-bucket last ────────────────────────────
BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

RUN_NAMES=(
  ablation_hierarchical_enc_family_matrimonial_kfold
  ablation_hierarchical_enc_fin_fraud_kfold
  ablation_hierarchical_enc_land_property_kfold
  ablation_hierarchical_enc_motor_accidents_kfold
  ablation_hierarchical_enc_sexual_offences_kfold
  ablation_hierarchical_enc_cross_bucket_kfold
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_kfold_parallel() {
  local cfg="$1" run_name="$2" outputs_dir="$3"
  local k=5
  local log_dir="$outputs_dir/logs"
  mkdir -p "$log_dir"
  local graph_cache
  graph_cache="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')/$(grep 'cache_name:' "$cfg" | awk '{print $2}')"

  log "  Launching $k folds in parallel (GPUs 0-$((k-1))) ..."
  local pids=()
  for fold_idx in $(seq 0 $((k-1))); do
    local log_file="$log_dir/${run_name}_fold_${fold_idx}.log"
    (
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$fold_idx" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON "$KFOLD_SCRIPT" \
          --config "$cfg" \
          --run-name "$run_name" \
          --k "$k" \
          --fold "$fold_idx" \
          --val-fraction 0.1
    ) > "$log_file" 2>&1 &
    pids+=("$!")
    log "  Fold $fold_idx → GPU $fold_idx  pid=${pids[-1]}"
  done

  local status=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "  Fold $i done."
    else
      log "  Fold $i FAILED." >&2
      status=1
    fi
  done
  [[ "$status" -ne 0 ]] && { log "  One or more folds failed." >&2; return 1; }

  log "  Aggregating folds ..."
  $PYTHON "$KFOLD_SCRIPT" \
    --config "$cfg" \
    --run-name "$run_name" \
    --k "$k" \
    --aggregate-only
}

run_embedding_analysis() {
  local cfg="$1" run_name="$2" bucket="$3"
  local outputs_dir
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  local model_path="$outputs_dir/models/$run_name/kfold/fold_00/model.pt"
  local out_dir="$outputs_dir/embedding_analysis/hier_${run_name}"
  local graph_cache_dir
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  local metadata_path
  metadata_path="$(find "$graph_cache_dir" -name 'graph_metadata*.json' | head -1)"
  local predictions_path="$outputs_dir/models/$run_name/kfold/fold_00/predictions.csv"

  mkdir -p "$out_dir"
  log "  Embedding analysis output dir: $out_dir"

  log "  [1/6] Extracting embeddings ..."
  $PYTHON "$ANALYSIS_DIR/extract_embeddings.py" \
    --config   "$cfg" \
    --model    "$model_path" \
    --output   "$out_dir" \
    --run-name "$run_name"

  local npz="$out_dir/${run_name}_embeddings.npz"

  log "  [2/6] t-SNE ..."
  $PYTHON "$ANALYSIS_DIR/tsne_visualise.py" \
    --embeddings "$npz" \
    --output-dir "$out_dir"

  log "  [3/6] Probing classifier ..."
  $PYTHON "$ANALYSIS_DIR/probing_classifier.py" \
    --embeddings "$npz" \
    --output-dir "$out_dir"

  log "  [4/6] Attention / gradient saliency ..."
  $PYTHON "$ANALYSIS_DIR/attention_analysis.py" \
    --config     "$cfg" \
    --model      "$model_path" \
    --run-name   "$run_name" \
    --output-dir "$out_dir"

  log "  [5/6] Error analysis ..."
  if [[ -f "$predictions_path" && -n "$metadata_path" ]]; then
    $PYTHON "$ANALYSIS_DIR/error_analysis.py" \
      --predictions "$predictions_path" \
      --metadata    "$metadata_path" \
      --output-dir  "$out_dir" \
      --run-name    "$run_name"
  else
    log "  [5/6] Skipped — predictions or metadata not found."
  fi

  log "  [6/6] SHAP analysis ..."
  $PYTHON "$ANALYSIS_DIR/shap_analysis.py" \
    --embeddings "$npz" \
    --output-dir "$out_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

OVERALL_START=$(date +%s)
log "═══════════════════════════════════════════════════════"
log "Starting hierarchical encoding pipeline — all buckets"
log "═══════════════════════════════════════════════════════"

for i in "${!BUCKETS[@]}"; do
  bucket="${BUCKETS[$i]}"
  run_name="${RUN_NAMES[$i]}"
  cfg="$ABLATION_DIR/$bucket/config.yaml"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "BUCKET: $bucket  ($((i+1))/${#BUCKETS[@]})"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Skip graph build if cache already exists ─────────────────────────────
  cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  local_graph_pt="$graph_cache_dir/$cache_name"

  if [[ -f "$local_graph_pt" ]]; then
    log "  [build] Graph cache already exists — skipping build."
  else
    log "  [build] Building hierarchical graph ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      $PYTHON "$BUILD_SCRIPT" --config "$cfg"
    log "  [build] Done."
  fi

  # ── kfold ─────────────────────────────────────────────────────────────────
  SUMMARY="$outputs_dir/models/$run_name/kfold/kfold_summary.json"
  if [[ -f "$SUMMARY" ]]; then
    log "  [kfold] Summary already exists — skipping training."
  else
    log "  [kfold] Starting 5-fold CV ..."
    run_kfold_parallel "$cfg" "$run_name" "$outputs_dir"
    log "  [kfold] Done. Summary → $SUMMARY"
  fi

  # ── embedding analysis ────────────────────────────────────────────────────
  log "  [analysis] Running embedding analysis ..."
  run_embedding_analysis "$cfg" "$run_name" "$bucket"
  log "  [analysis] Done."

done

OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════"
log "All buckets complete in ${ELAPSED} minutes."
log "Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/embedding_analysis/hier_*"
log "═══════════════════════════════════════════════════════"
