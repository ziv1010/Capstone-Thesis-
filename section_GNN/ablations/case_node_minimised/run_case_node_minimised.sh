#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Case-node-minimised ablation
# Case node: 12 scalars only — no BGE-M3 text embedding.
# Section nodes (preamble, facts, arguments, ...) stay in the graph with
# their own BGE-M3 embeddings; the GNN must aggregate text signal from them.
#
# Runs: family_matrimonial (h128, l3) + cross_bucket (h64, l2)
#
# Usage:
#   nohup bash section_GNN/ablations/case_node_minimised/run_case_node_minimised.sh \
#     > section_GNN/ablations/case_node_minimised/run.log 2>&1 &
#   tail -f section_GNN/ablations/case_node_minimised/run.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON="micromamba run -n $MAMBA_ENV python"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

BUILD_SCRIPT="$SECTION_GNN/final_graph/build_graph.py"
KFOLD_SCRIPT="$SECTION_GNN/dump2/scripts/kfold_cv.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_kfold_parallel() {
  local cfg="$1" run_name="$2"
  local outputs_dir
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  local log_dir="$outputs_dir/logs"
  mkdir -p "$log_dir"
  local k=5

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

run_bucket() {
  local bucket="$1" run_name="$2"
  local cfg="$SCRIPT_DIR/$bucket/config.yaml"
  local outputs_dir
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "BUCKET: $bucket"
  log "RUN:    $run_name"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Build graph (skip if cache exists) ──────────────────────────────────────
  cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  graph_pt="$graph_cache_dir/$cache_name"

  if [[ -f "$graph_pt" ]]; then
    log "  [build] Cache exists — skipping."
  else
    log "  [build] Building graph (case_text_sections=[]) ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      $PYTHON "$BUILD_SCRIPT" --config "$cfg"
    log "  [build] Done."
  fi

  # ── 5-fold CV (skip if summary exists) ──────────────────────────────────────
  SUMMARY="$outputs_dir/models/$run_name/kfold/kfold_summary.json"
  if [[ -f "$SUMMARY" ]]; then
    log "  [kfold] Summary exists — skipping."
  else
    log "  [kfold] Starting 5-fold CV ..."
    run_kfold_parallel "$cfg" "$run_name"
    log "  [kfold] Done → $SUMMARY"
  fi
}

OVERALL_START=$(date +%s)
log "═══════════════════════════════════════════════════════"
log "Case-node-minimised ablation — fam_mat + cross_bucket"
log "═══════════════════════════════════════════════════════"

run_bucket "family_matrimonial_timed_mistral" "ablation_case_node_minimised_family_matrimonial_kfold"
run_bucket "cross_bucket_total_dataset"       "ablation_case_node_minimised_cross_bucket_kfold"

OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════"
log "All done in ${ELAPSED} minutes."
log "Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/models/ablation_case_node_minimised_*"
log "═══════════════════════════════════════════════════════"
