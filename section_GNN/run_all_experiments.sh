#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_experiments.sh
#
# Orchestrates all 8 experiments across all 6 buckets (food_safety excluded).
# Each step is idempotent — skips if kfold_summary.json already exists.
# Preprocessing is skipped per-bucket if cleaned_cases/ already has content.
#
# Experiments:
#   1. baseline            runs/<bucket>/run_all.sh
#   2. no_names            ablations/no_names/<bucket>/run.sh
#   3. text_only           ablations/text_only/<bucket>/run.sh (or inline)
#   4. no_cross_case       ablations/no_cross_case/<bucket>/run.sh (or inline)
#   5. hierarchical_enc    runs/run_hierarchical_enc_all_buckets.sh
#   6. section_sep_enc     runs/run_section_sep_enc_all_buckets.sh
#   7. case_node_minimised inline via run_bucket() (no run.sh exists)
#   8. party_args_lr_decay runs_v2/party_args_lr_decay/run_all_buckets.sh
#
# Preprocessing (step 0) uses:
#   experiments/fixed_open_pipeline/preprocess_fixed_open.py
#
# Usage:
#   nohup bash section_GNN/run_all_experiments.sh \
#     > section_GNN/run_all_experiments.log 2>&1 &
#   tail -f section_GNN/run_all_experiments.log
#
# Env-var overrides:
#   MAMBA_ENV=thesis_work     micromamba environment
#   GPUS=0,1,2,3,4,5,6,7     GPU list for graph builds
#   SKIP_PREPROCESS=0         set to 1 to skip step 0
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$SCRIPT_DIR"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
K=5

PYTHON="micromamba run -n $MAMBA_ENV python"
BUILD_SCRIPT="$SECTION_GNN/final_graph/build_graph.py"
KFOLD_SCRIPT="$SECTION_GNN/dump2/scripts/kfold_cv.py"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

OVERALL_START=$(date +%s)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Run K folds in parallel across GPUs 0..(K-1), then aggregate.
run_kfold_parallel() {
  local cfg="$1" run_name="$2"
  local outputs_dir graph_cache_dir graph_cache_name graph_cache log_dir
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  graph_cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache="$graph_cache_dir/$graph_cache_name"
  log_dir="$outputs_dir/logs"
  mkdir -p "$log_dir"

  log "    [kfold] Launching $K folds in parallel ..."
  local pids=()
  for fold_idx in $(seq 0 $((K - 1))); do
    local log_file="$log_dir/${run_name}_fold_${fold_idx}.log"
    (
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$fold_idx" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON "$KFOLD_SCRIPT" \
          --config "$cfg" \
          --run-name "$run_name" \
          --k "$K" \
          --fold "$fold_idx" \
          --val-fraction 0.1 \
          --graph-cache "$graph_cache"
    ) > "$log_file" 2>&1 &
    pids+=("$!")
    log "    [kfold] Fold $fold_idx → GPU $fold_idx  pid=${pids[-1]}"
  done

  local status=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" \
      && log "    [kfold] Fold $i done." \
      || { log "    [kfold] Fold $i FAILED." >&2; status=1; }
  done
  [[ "$status" -ne 0 ]] && { log "    [kfold] One or more folds FAILED." >&2; return 1; }

  log "    [kfold] Aggregating ..."
  $PYTHON "$KFOLD_SCRIPT" \
    --config "$cfg" --run-name "$run_name" \
    --k "$K" --aggregate-only \
    --graph-cache "$graph_cache"
}

# Build graph if cache missing, then run kfold if summary missing.
run_bucket() {
  local cfg="$1" run_name="$2"
  local graph_cache_dir graph_cache_name graph_cache outputs_dir summary
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  graph_cache_name="$(grep 'cache_name:' "$cfg" | awk '{print $2}')"
  graph_cache="$graph_cache_dir/$graph_cache_name"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  summary="$outputs_dir/models/$run_name/kfold/kfold_summary.json"

  if [[ -f "$graph_cache" ]]; then
    log "    [build] Cache exists — skipping."
  else
    log "    [build] Building graph ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      $PYTHON "$BUILD_SCRIPT" --config "$cfg"
    log "    [build] Done → $graph_cache"
  fi

  if [[ -f "$summary" ]]; then
    log "    [kfold] Summary exists — skipping."
  else
    log "    [kfold] Starting 5-fold CV ..."
    run_kfold_parallel "$cfg" "$run_name"
    log "    [kfold] Done → $summary"
  fi
}

# Check kfold summary and call a run.sh only if not already done.
# Usage: run_via_script <summary_path> <label> <script> [args...]
run_via_script() {
  local summary="$1" label="$2"; shift 2
  if [[ -f "$summary" ]]; then
    log "    [$label] Summary exists — skipping."
  else
    log "    [$label] Running ..."
    "$@"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════════════════"
log "run_all_experiments — 8 experiments × 6 buckets  (food_safety excluded)"
log "  env=$MAMBA_ENV  gpus=$GPUS  skip_preprocess=$SKIP_PREPROCESS"
log "═══════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# 0. PREPROCESSING
#    Uses runs/<bucket>/01_preprocess.sh which calls preprocess_fixed_open.py.
#    Skipped per-bucket if cleaned_cases/ already has content.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  log ""
  log "━━━━ [0/8] PREPROCESSING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  for bucket in "${BUCKETS[@]}"; do
    log ""
    log "  BUCKET: $bucket"
    cleaned_dir="$SECTION_GNN/data/timed_bucket_runs/$bucket/processed/cleaned_cases"
    if [[ -d "$cleaned_dir" && -n "$(ls -A "$cleaned_dir" 2>/dev/null)" ]]; then
      log "    Already preprocessed — skipping."
    else
      log "    Running preprocess ..."
      MAMBA_ENV="$MAMBA_ENV" \
        bash "$SECTION_GNN/runs/$bucket/01_preprocess.sh"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE
#    runs/<bucket>/run_all.sh --skip-preprocess
#    run_name = <bucket>_kfold  (set by run_all.sh from dirname)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [1/8] BASELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for bucket in "${BUCKETS[@]}"; do
  log ""
  log "  BUCKET: $bucket"
  cfg="$SECTION_GNN/runs/$bucket/config.yaml"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  summary="$outputs_dir/models/${bucket}_kfold/kfold/kfold_summary.json"
  run_via_script "$summary" "baseline" \
    env MAMBA_ENV="$MAMBA_ENV" GPUS_BUILD="$GPUS" \
    bash "$SECTION_GNN/runs/$bucket/run_all.sh" --skip-preprocess
done

# ─────────────────────────────────────────────────────────────────────────────
# 2. NO_NAMES
#    All 6 buckets have run.sh.
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [2/8] NO_NAMES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A NO_NAMES_RUN_NAME=(
  [family_matrimonial_timed_mistral]="ablation_no_names_family_matrimonial_kfold"
  [fin_fraud_timed_mistral]="ablation_no_names_fin_fraud_kfold"
  [land_property_timed_mistral]="ablation_no_names_land_property_kfold"
  [motor_accidents_timed_mistral]="ablation_no_names_motor_accidents_kfold"
  [sexual_offences_timed_mistral]="ablation_no_names_sexual_offences_kfold"
  [cross_bucket_total_dataset]="ablation_no_names_cross_bucket_kfold"
)

for bucket in "${BUCKETS[@]}"; do
  log ""
  log "  BUCKET: $bucket"
  run_name="${NO_NAMES_RUN_NAME[$bucket]}"
  cfg="$SECTION_GNN/ablations/no_names/$bucket/config.yaml"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  summary="$outputs_dir/models/$run_name/kfold/kfold_summary.json"
  run_via_script "$summary" "no_names" \
    env MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
    bash "$SECTION_GNN/ablations/no_names/$bucket/run.sh"
done

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT_ONLY
#    family_matrimonial, fin_fraud, cross_bucket → run.sh
#    land_property, motor_accidents, sexual_offences → inline (config exists)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [3/8] TEXT_ONLY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A TEXT_ONLY_RUN_NAME=(
  [family_matrimonial_timed_mistral]="ablation_text_only_family_matrimonial_kfold"
  [fin_fraud_timed_mistral]="ablation_text_only_fin_fraud_kfold"
  [land_property_timed_mistral]="ablation_text_only_land_property_kfold"
  [motor_accidents_timed_mistral]="ablation_text_only_motor_accidents_kfold"
  [sexual_offences_timed_mistral]="ablation_text_only_sexual_offences_kfold"
  [cross_bucket_total_dataset]="ablation_text_only_cross_bucket_kfold"
)

for bucket in "${BUCKETS[@]}"; do
  log ""
  log "  BUCKET: $bucket"
  run_name="${TEXT_ONLY_RUN_NAME[$bucket]}"
  cfg="$SECTION_GNN/ablations/text_only/$bucket/config.yaml"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  summary="$outputs_dir/models/$run_name/kfold/kfold_summary.json"

  if [[ -f "$summary" ]]; then
    log "    [text_only] Summary exists — skipping."
  elif [[ -f "$SECTION_GNN/ablations/text_only/$bucket/run.sh" ]]; then
    log "    [text_only] Delegating to run.sh ..."
    MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
      bash "$SECTION_GNN/ablations/text_only/$bucket/run.sh"
  else
    log "    [text_only] Inline run ..."
    run_bucket "$cfg" "$run_name"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 4. NO_CROSS_CASE
#    family_matrimonial, fin_fraud, cross_bucket → run.sh
#    land_property, motor_accidents, sexual_offences → inline (config exists)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [4/8] NO_CROSS_CASE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A NO_CROSS_RUN_NAME=(
  [family_matrimonial_timed_mistral]="ablation_no_cross_case_family_matrimonial_kfold"
  [fin_fraud_timed_mistral]="ablation_no_cross_case_fin_fraud_kfold"
  [land_property_timed_mistral]="ablation_no_cross_case_land_property_kfold"
  [motor_accidents_timed_mistral]="ablation_no_cross_case_motor_accidents_kfold"
  [sexual_offences_timed_mistral]="ablation_no_cross_case_sexual_offences_kfold"
  [cross_bucket_total_dataset]="ablation_no_cross_case_cross_bucket_kfold"
)

for bucket in "${BUCKETS[@]}"; do
  log ""
  log "  BUCKET: $bucket"
  run_name="${NO_CROSS_RUN_NAME[$bucket]}"
  cfg="$SECTION_GNN/ablations/no_cross_case/$bucket/config.yaml"
  outputs_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')"
  summary="$outputs_dir/models/$run_name/kfold/kfold_summary.json"

  if [[ -f "$summary" ]]; then
    log "    [no_cross_case] Summary exists — skipping."
  elif [[ -f "$SECTION_GNN/ablations/no_cross_case/$bucket/run.sh" ]]; then
    log "    [no_cross_case] Delegating to run.sh ..."
    MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
      bash "$SECTION_GNN/ablations/no_cross_case/$bucket/run.sh"
  else
    log "    [no_cross_case] Inline run ..."
    run_bucket "$cfg" "$run_name"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 5. HIERARCHICAL_ENC
#    Delegates to existing all-buckets runner (has its own skip-if-done logic).
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [5/8] HIERARCHICAL_ENC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
  bash "$SECTION_GNN/runs/run_hierarchical_enc_all_buckets.sh"

# ─────────────────────────────────────────────────────────────────────────────
# 6. SECTION_SEP_ENC
#    Delegates to existing all-buckets runner (has its own skip-if-done logic).
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [6/8] SECTION_SEP_ENC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
  bash "$SECTION_GNN/runs/run_section_sep_enc_all_buckets.sh"

# ─────────────────────────────────────────────────────────────────────────────
# 7. CASE_NODE_MINIMISED
#    No per-bucket run.sh exists for any bucket; all configs exist.
#    Inline via run_bucket() for all 6 buckets.
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [7/8] CASE_NODE_MINIMISED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A CASE_MIN_RUN_NAME=(
  [family_matrimonial_timed_mistral]="ablation_case_node_minimised_family_matrimonial_kfold"
  [fin_fraud_timed_mistral]="ablation_case_node_minimised_fin_fraud_kfold"
  [land_property_timed_mistral]="ablation_case_node_minimised_land_property_kfold"
  [motor_accidents_timed_mistral]="ablation_case_node_minimised_motor_accidents_kfold"
  [sexual_offences_timed_mistral]="ablation_case_node_minimised_sexual_offences_kfold"
  [cross_bucket_total_dataset]="ablation_case_node_minimised_cross_bucket_kfold"
)

for bucket in "${BUCKETS[@]}"; do
  log ""
  log "  BUCKET: $bucket"
  run_name="${CASE_MIN_RUN_NAME[$bucket]}"
  cfg="$SECTION_GNN/ablations/case_node_minimised/$bucket/config.yaml"
  run_bucket "$cfg" "$run_name"
done

# ─────────────────────────────────────────────────────────────────────────────
# 8. PARTY_ARGS_LR_DECAY
#    Delegates to existing all-buckets runner (has its own skip-if-done logic).
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [8/8] PARTY_ARGS_LR_DECAY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MAMBA_ENV="$MAMBA_ENV" GPUS_BUILD="$GPUS" \
  bash "$SECTION_GNN/runs_v2/party_args_lr_decay/run_all_buckets.sh"

# ─────────────────────────────────────────────────────────────────────────────
OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════════════════════"
log "All experiments complete in ${ELAPSED} minutes."
log "Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/models/"
log "═══════════════════════════════════════════════════════════════════════"
