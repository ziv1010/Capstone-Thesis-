#!/usr/bin/env bash
# Run party_args_lr_decay experiment on all 6 buckets sequentially.
# For each bucket:
#   1. Build graph with party-argument case-node text (build_graph_v2.py)
#   2. Run 5-fold CV with LR decay (kfold_cv_v2.py)
#
# Usage:
#   nohup bash runs_v2/party_args_lr_decay/run_all_buckets.sh \
#     > runs_v2/party_args_lr_decay/run_all.log 2>&1 &
#   tail -f runs_v2/party_args_lr_decay/run_all.log
#
# Skip graph build if cache already exists:
#   SKIP_BUILD=1 bash run_all_buckets.sh
set -euo pipefail

V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$V2_DIR/../.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-party_args_lr_decay_kfold}"

BUILD_SCRIPT="$V2_DIR/graph/build_graph_v2.py"
KFOLD_SCRIPT="$V2_DIR/03_kfold_v2.sh"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

OVERALL_START=$(date +%s)
log "═══════════════════════════════════════════════════════"
log "party_args_lr_decay — all buckets"
log "  env=$MAMBA_ENV  skip_build=$SKIP_BUILD"
log "═══════════════════════════════════════════════════════"

for bucket in "${BUCKETS[@]}"; do
  cfg="$V2_DIR/$bucket/config.yaml"
  run_name="${bucket//_timed_mistral/}_${RUN_NAME_SUFFIX}"
  # cross_bucket special case
  [[ "$bucket" == "cross_bucket_total_dataset" ]] && run_name="cross_bucket_${RUN_NAME_SUFFIX}"

  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache_pt="$graph_cache_dir/$cache_name"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "BUCKET: $bucket  →  run_name: $run_name"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Step 1: Build graph ──────────────────────────────────────────────────
  if [[ "$SKIP_BUILD" == "1" ]] || [[ -f "$graph_cache_pt" ]]; then
    log "  [build] Graph cache exists or SKIP_BUILD=1 — skipping."
  else
    log "  [build] Building party-args graph ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS_BUILD" \
      micromamba run -n "$MAMBA_ENV" python "$BUILD_SCRIPT" --config "$cfg"
    log "  [build] Done → $graph_cache_pt"
  fi

  # ── Step 2: 5-fold CV ────────────────────────────────────────────────────
  SUMMARY="$outputs_dir/models/$run_name/kfold/kfold_summary.json"
  if [[ -f "$SUMMARY" ]]; then
    log "  [kfold] Summary already exists — skipping. ($SUMMARY)"
  else
    log "  [kfold] Starting 5-fold CV ..."
    CONFIG="$cfg" RUN_NAME="$run_name" \
      bash "$KFOLD_SCRIPT" --config "$cfg" --run-name "$run_name"
    log "  [kfold] Done → $SUMMARY"
  fi
done

OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════"
log "All buckets complete in ${ELAPSED} minutes."
log "Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/models/*_party_args_lr_decay_kfold/"
log "═══════════════════════════════════════════════════════"
