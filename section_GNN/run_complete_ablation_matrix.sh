#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_complete_ablation_matrix.sh
#
# Fills every missing cell in the 7-experiment ablation matrix.
# Each experiment type gets its own section — none are mixed.
#
#   #  Experiment              Datasets done          Missing (this script runs)
#   ──────────────────────────────────────────────────────────────────────────
#   1  baseline                all 6                  — (skip)
#   2  text_only               cross_bucket, fam_mat  fin, land, motor, sexual
#   3  no_cross_case           fam_mat                cross_bucket + fin, land, motor, sexual
#   4  section_sep_enc         cross_bucket, fin       fam_mat, land, motor, sexual
#   5  hierarchical_enc        all 6                  — (skip)
#   6  case_node_minimised     cross_bucket, fam_mat  fin, land, motor, sexual
#   7  runs_v2 party_args      fam_mat, fin           cross_bucket, land, motor, sexual
#
# Every step checks for an existing kfold_summary.json before running.
#
# Usage:
#   nohup bash section_GNN/run_complete_ablation_matrix.sh \
#     > section_GNN/run_complete_ablation_matrix.log 2>&1 &
#   tail -f section_GNN/run_complete_ablation_matrix.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$SCRIPT_DIR"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON="micromamba run -n $MAMBA_ENV python"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
K=5

BUILD_SCRIPT="$SECTION_GNN/src/scripts/build_graph.py"
KFOLD_SCRIPT="$SECTION_GNN/src/scripts/kfold_cv.py"

SMALL_BUCKETS=(
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

yaml_query() {
  local cfg="$1" path="$2"
  CONFIG_PATH="$cfg" YAML_PATH="$path" $PYTHON - <<'PY'
import os, yaml
v = yaml.safe_load(open(os.environ["CONFIG_PATH"]))
for k in os.environ["YAML_PATH"].split("."): v = v[k]
print(v)
PY
}

graph_cache_path() {
  local cfg="$1"
  echo "$(yaml_query "$cfg" "paths.graph_cache_dir")/$(yaml_query "$cfg" "graph.cache_name")"
}

summary_path() {
  local cfg="$1" run_name="$2"
  echo "$(yaml_query "$cfg" "paths.outputs_dir")/models/$run_name/kfold/kfold_summary.json"
}

run_kfold_parallel() {
  local cfg="$1" run_name="$2"
  local outputs_dir log_dir graph_cache
  outputs_dir="$(yaml_query "$cfg" "paths.outputs_dir")"
  log_dir="$outputs_dir/logs"
  mkdir -p "$log_dir"
  graph_cache="$(graph_cache_path "$cfg")"

  log "  [kfold] Launching $K folds in parallel (GPUs 0-$((K-1))) ..."
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
    log "  [kfold] Fold $fold_idx → GPU $fold_idx  pid=${pids[-1]}"
  done

  local status=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" && log "  [kfold] Fold $i done." || { log "  [kfold] Fold $i FAILED." >&2; status=1; }
  done
  [[ "$status" -ne 0 ]] && { log "  [kfold] One or more folds failed." >&2; return 1; }

  log "  [kfold] Aggregating ..."
  $PYTHON "$KFOLD_SCRIPT" \
    --config "$cfg" --run-name "$run_name" \
    --k "$K" --aggregate-only \
    --graph-cache "$graph_cache"
}

# Build graph if cache missing, then run 5-fold CV if summary missing.
run_bucket() {
  local cfg="$1" run_name="$2"
  local graph_pt summary
  graph_pt="$(graph_cache_path "$cfg")"
  summary="$(summary_path "$cfg" "$run_name")"

  if [[ -f "$graph_pt" ]]; then
    log "  [build] Cache exists — skipping."
  else
    log "  [build] Building graph ..."
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      $PYTHON "$BUILD_SCRIPT" --config "$cfg"
    log "  [build] Done → $graph_pt"
  fi

  if [[ -f "$summary" ]]; then
    log "  [kfold] Summary exists — skipping."
  else
    run_kfold_parallel "$cfg" "$run_name"
    log "  [kfold] Done → $summary"
  fi
}

# Sync a config from a template, patching project.name, paths, and cache_name.
sync_config_from_template() {
  local template_cfg="$1" base_cfg="$2" target_cfg="$3"
  local project_name="$4" cache_name="$5"

  if [[ -f "$target_cfg" ]]; then
    log "  [config] Already exists — skipping sync."
    return
  fi
  log "  [config] Generating from template ..."
  TMPL="$template_cfg" BASE="$base_cfg" TARGET="$target_cfg" \
  PROJ_NAME="$project_name" CACHE_NAME="$cache_name" \
    $PYTHON - <<'PY'
import copy, os
from pathlib import Path
import yaml

def load(p): return yaml.safe_load(open(p))
def save(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: yaml.safe_dump(d, f, sort_keys=False)

tmpl = load(os.environ["TMPL"])
base = load(os.environ["BASE"])

cfg = copy.deepcopy(tmpl)
cfg["project"]["name"] = os.environ["PROJ_NAME"]
cfg["paths"] = copy.deepcopy(base["paths"])
cfg["graph"]["cache_name"] = os.environ["CACHE_NAME"]
save(os.environ["TARGET"], cfg)
print(f"Saved {os.environ['TARGET']}")
PY
  log "  [config] Done → $target_cfg"
}

# ─────────────────────────────────────────────────────────────────────────────

OVERALL_START=$(date +%s)
log "═══════════════════════════════════════════════════════════════════════"
log "run_complete_ablation_matrix — 7 experiment types, no mixing"
log "  env=$MAMBA_ENV  gpus=$GPUS"
log "═══════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — all 6 done, nothing to run
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [1/7] BASELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  All 6 datasets complete — skipping."

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT ONLY
#    Done:    cross_bucket_total_dataset, family_matrimonial
#    Missing: fin_fraud, land_property, motor_accidents, sexual_offences
#    Template: ablations/text_only/family_matrimonial_timed_mistral/config.yaml
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [2/7] TEXT ONLY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TEXT_ONLY_TEMPLATE="$SECTION_GNN/ablations/text_only/family_matrimonial_timed_mistral/config.yaml"

for bucket in "${SMALL_BUCKETS[@]}"; do
  short="${bucket%_timed_mistral}"
  cfg="$SECTION_GNN/ablations/text_only/$bucket/config.yaml"
  run_name="ablation_text_only_${short}_kfold"
  log ""
  log "  BUCKET: $bucket"
  sync_config_from_template \
    "$TEXT_ONLY_TEMPLATE" \
    "$SECTION_GNN/runs/$bucket/config.yaml" \
    "$cfg" \
    "${short}_text_only" \
    "case_star_${short}_text_only.reasoning_focused.pt"
  run_bucket "$cfg" "$run_name"
done

# ─────────────────────────────────────────────────────────────────────────────
# 3. NO CROSS-CASE SHARING
#    Done:    family_matrimonial
#    Missing: cross_bucket_total_dataset, fin_fraud, land_property,
#             motor_accidents, sexual_offences
#    Template: ablations/no_cross_case/family_matrimonial_timed_mistral/config.yaml
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [3/7] NO CROSS-CASE SHARING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
NO_CROSS_TEMPLATE="$SECTION_GNN/ablations/no_cross_case/family_matrimonial_timed_mistral/config.yaml"

# cross_bucket has its own existing config — just run it
log ""
log "  BUCKET: cross_bucket_total_dataset"
run_bucket \
  "$SECTION_GNN/ablations/no_cross_case/cross_bucket_total_dataset/config.yaml" \
  "ablation_no_cross_case_cross_bucket_kfold"

# 4 small buckets need config sync
for bucket in "${SMALL_BUCKETS[@]}"; do
  short="${bucket%_timed_mistral}"
  cfg="$SECTION_GNN/ablations/no_cross_case/$bucket/config.yaml"
  run_name="ablation_no_cross_case_${short}_kfold"
  log ""
  log "  BUCKET: $bucket"
  sync_config_from_template \
    "$NO_CROSS_TEMPLATE" \
    "$SECTION_GNN/runs/$bucket/config.yaml" \
    "$cfg" \
    "${short}_no_cross_case" \
    "case_star_${short}_no_cross_case.reasoning_focused.pt"
  run_bucket "$cfg" "$run_name"
done

# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTION-SEPARATED ENCODING
#    Done:    cross_bucket_total_dataset, fin_fraud
#    Missing: family_matrimonial, land_property, motor_accidents, sexual_offences
#    The existing run_section_sep_enc_all_buckets.sh covers all 6 with skip logic.
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [4/7] SECTION-SEPARATED ENCODING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MAMBA_ENV="$MAMBA_ENV" GPUS="$GPUS" \
  bash "$SECTION_GNN/runs/run_section_sep_enc_all_buckets.sh"

# ─────────────────────────────────────────────────────────────────────────────
# 5. HIERARCHICAL ENCODING — all 6 done, nothing to run
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [5/7] HIERARCHICAL ENCODING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  All 6 datasets complete — skipping."

# ─────────────────────────────────────────────────────────────────────────────
# 6. CASE NODE MINIMISED
#    Done:    cross_bucket_total_dataset, family_matrimonial
#    Missing: fin_fraud, land_property, motor_accidents, sexual_offences
#    Template: ablations/case_node_minimised/family_matrimonial_timed_mistral/config.yaml
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [6/7] CASE NODE MINIMISED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CASE_MIN_TEMPLATE="$SECTION_GNN/ablations/case_node_minimised/family_matrimonial_timed_mistral/config.yaml"

for bucket in "${SMALL_BUCKETS[@]}"; do
  short="${bucket%_timed_mistral}"
  cfg="$SECTION_GNN/ablations/case_node_minimised/$bucket/config.yaml"
  run_name="ablation_case_node_minimised_${short}_kfold"
  log ""
  log "  BUCKET: $bucket"
  sync_config_from_template \
    "$CASE_MIN_TEMPLATE" \
    "$SECTION_GNN/runs/$bucket/config.yaml" \
    "$cfg" \
    "${short}_case_node_minimised" \
    "case_star_${short}_case_node_minimised.reasoning_focused.pt"
  run_bucket "$cfg" "$run_name"
done

# ─────────────────────────────────────────────────────────────────────────────
# 7. RUNS_V2 PARTY_ARGS_LR_DECAY
#    Done:    family_matrimonial, fin_fraud
#    Missing: cross_bucket_total_dataset, land_property, motor_accidents,
#             sexual_offences
#    The existing run_all_buckets.sh covers all 6 with skip logic.
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "━━━━ [7/7] RUNS_V2 PARTY_ARGS_LR_DECAY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MAMBA_ENV="$MAMBA_ENV" GPUS_BUILD="$GPUS" \
  bash "$SECTION_GNN/runs_v2/party_args_lr_decay/run_all_buckets.sh"

# ─────────────────────────────────────────────────────────────────────────────
OVERALL_END=$(date +%s)
ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))
log ""
log "═══════════════════════════════════════════════════════════════════════"
log "Complete in ${ELAPSED} minutes."
log "  Results under: section_GNN/outputs/timed_bucket_runs/<bucket>/models/"
log "═══════════════════════════════════════════════════════════════════════"
