#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: section-separated encoding ablation — all 5 domain buckets
#
# For each bucket:
#   1. Build graph with section-separated BGE-M3 encoding (new cache, old untouched)
#      Case node feature = [preamble_emb(1024) | facts_emb(1024) | args_emb(1024) | scalars]
#   2. Run 5-fold CV (5 folds in parallel across GPUs 0-4)
#
# All outputs go under:
#   section_GNN/outputs/timed_bucket_runs/<bucket>/models/ablation_section_sep_enc_*
#
# Old results are never touched — new cache files have distinct names.
#
# Usage:
#   nohup bash section_GNN/runs/run_section_sep_enc_all_buckets.sh \
#     > section_GNN/runs/section_sep_enc_overnight.log 2>&1 &
#   tail -f section_GNN/runs/section_sep_enc_overnight.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON="micromamba run -n $MAMBA_ENV python"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

ABLATION_DIR="$SECTION_GNN/ablations/section_sep_enc"
BUILD_SCRIPT="$SECTION_GNN/final_graph/build_graph_section_sep.py"
KFOLD_SCRIPT="$SECTION_GNN/src/scripts/kfold_cv.py"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

RUN_NAMES=(
  ablation_section_sep_enc_family_matrimonial_kfold
  ablation_section_sep_enc_fin_fraud_kfold
  ablation_section_sep_enc_land_property_kfold
  ablation_section_sep_enc_motor_accidents_kfold
  ablation_section_sep_enc_sexual_offences_kfold
  ablation_section_sep_enc_cross_bucket_kfold
)

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

# ─────────────────────────────────────────────────────────────────────────────
OVERALL_START=$(date +%s)
log "═══════════════════════════════════════════════════════"
log "Starting section-separated encoding pipeline — all buckets"
log "  Case feature = [preamble_emb | facts_emb | args_emb | scalars]"
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

  # ── Build graph if cache missing ──────────────────────────────────────────
  cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  graph_pt="$graph_cache_dir/$cache_name"

  if [[ -f "$graph_pt" ]]; then
    log "  [build] Graph cache exists — skipping: $graph_pt"
  else
    log "  [build] Building section-separated graph ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      $PYTHON "$BUILD_SCRIPT" --config "$cfg"
    log "  [build] Done → $graph_pt"
  fi

  # ── K-fold CV ─────────────────────────────────────────────────────────────
  SUMMARY="$outputs_dir/models/$run_name/kfold/kfold_summary.json"
  if [[ -f "$SUMMARY" ]]; then
    log "  [kfold] Summary exists — skipping: $SUMMARY"
  else
    log "  [kfold] Starting 5-fold CV ..."
    run_kfold_parallel "$cfg" "$run_name" "$outputs_dir"
    log "  [kfold] Done. Summary → $SUMMARY"
  fi

done

OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════"
log "All buckets complete in ${ELAPSED} minutes."
log "Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/models/ablation_section_sep_enc_*"
log "═══════════════════════════════════════════════════════"
