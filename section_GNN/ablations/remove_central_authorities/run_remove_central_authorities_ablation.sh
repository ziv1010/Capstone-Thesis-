#!/usr/bin/env bash
# Run the central-authority removal ablation on top of entity-resolved data.
#
# The runner first computes central authority candidates from resolved cleaned
# cases, then removes only the selected broad/trivial statute nodes into an
# isolated cleaned-case tree before building/training graphs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$SECTION_GNN:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4}"
ONLY="party"
LR_MODE="${LR_MODE:-decay}"
STATUS_ONLY=false
SYNC_ONLY=false
ANALYZE_ONLY=false
FILTER_ONLY=false
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_SOURCE_PREPROCESS="${SKIP_SOURCE_PREPROCESS:-0}"
FORCE_SOURCE_PREPROCESS="${FORCE_SOURCE_PREPROCESS:-0}"
FORCE_FILTER="${FORCE_FILTER:-0}"
LIMIT="${LIMIT:-}"
MIN_GLOBAL_CASE_FREQUENCY="${MIN_GLOBAL_CASE_FREQUENCY:-3000}"
REMOVAL_SET="${REMOVAL_SET:-}"
DROP_LINKED_PROVISIONS=false
EXTRA_STATUTES=()

CONFIG_ROOT="$SCRIPT_DIR/configs"
ENTITY_CONFIG_ROOT="$SECTION_GNN/ablations/entity_resolved_data/configs"
CENTRALITY_DIR="$SECTION_GNN/outputs/ablations/remove_central_authorities/centrality_analysis"
ENTITY_PREPARE_CONFIGS="$SECTION_GNN/ablations/entity_resolved_data/prepare_configs.py"
ENTITY_PREPROCESS="$SECTION_GNN/ablations/entity_resolved_data/preprocess_fixed_open_resolved.py"
PREPARE_CONFIGS="$SCRIPT_DIR/prepare_configs.py"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_central_authorities.py"
FILTER_SCRIPT="$SCRIPT_DIR/filter_cleaned_cases.py"
BUILD_PARTY="$SECTION_GNN/runs_v2/party_args_lr_decay/graph/build_graph_v2.py"
BUILD_SECTION="$SECTION_GNN/final_graph/build_graph_section_sep.py"
KFOLD_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

usage() {
  cat <<EOF
Usage:
  bash ablations/remove_central_authorities/run_remove_central_authorities_ablation.sh
  bash ablations/remove_central_authorities/run_remove_central_authorities_ablation.sh --analyze-only
  bash ablations/remove_central_authorities/run_remove_central_authorities_ablation.sh --status-only

Options:
  --only party|section|both       Variant(s) to run. Default: party
  --lr-mode decay|none            Use LR decay or no LR scheduler. Default: decay
  --no-lr-decay                   Shortcut for --lr-mode none
  --buckets "fin_fraud_timed_mistral motor_accidents_timed_mistral"
  --min-global-case-frequency N   Auto-remove broad trivial statutes above this global case count. Default: 3000
  --extra-statute NAME            Add a normalized statute canonical_name to the auto-remove allow-list.
  --removal-set PATH              Reuse an existing central_authority_removal_set.json.
  --drop-linked-provisions        Also remove provisions linked to removed statutes.
  --force-source-preprocess       Recreate entity-resolved source cleaned cases first.
  --force-filter                  Recreate filtered cleaned cases.
  --skip-source-preprocess        Require existing entity-resolved source cleaned cases.
  --skip-build                    Require existing graph caches.
  --limit N                       Preprocess only the first N resolved files per bucket.
EOF
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) MAMBA_ENV="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --gpus-build) GPUS_BUILD="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --lr-mode) LR_MODE="$2"; shift 2 ;;
    --no-lr-decay) LR_MODE="none"; shift ;;
    --buckets) IFS=' ' read -r -a BUCKETS <<< "$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --min-global-case-frequency) MIN_GLOBAL_CASE_FREQUENCY="$2"; shift 2 ;;
    --extra-statute) EXTRA_STATUTES+=("$2"); shift 2 ;;
    --removal-set) REMOVAL_SET="$2"; shift 2 ;;
    --drop-linked-provisions) DROP_LINKED_PROVISIONS=true; shift ;;
    --status-only) STATUS_ONLY=true; shift ;;
    --sync-only) SYNC_ONLY=true; shift ;;
    --analyze-only) ANALYZE_ONLY=true; shift ;;
    --filter-only) FILTER_ONLY=true; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-source-preprocess) SKIP_SOURCE_PREPROCESS=1; shift ;;
    --force-source-preprocess) FORCE_SOURCE_PREPROCESS=1; shift ;;
    --force-filter) FORCE_FILTER=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$ONLY" in
  party|section|both) ;;
  *) echo "Unknown --only value: $ONLY" >&2; exit 1 ;;
esac
case "$LR_MODE" in
  decay|none) ;;
  *) echo "Unknown --lr-mode value: $LR_MODE" >&2; exit 1 ;;
esac

if [[ "$LR_MODE" == "none" ]]; then
  CONFIG_ROOT="$SCRIPT_DIR/configs_no_lr"
fi

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -lt "$K" ]]; then
  echo "Need at least K=$K train GPUs in TRAIN_GPUS, got: $TRAIN_GPUS" >&2
  exit 1
fi

PYTHON=(micromamba run -n "$MAMBA_ENV" python)
for required in \
  "$ENTITY_PREPARE_CONFIGS" "$ENTITY_PREPROCESS" "$PREPARE_CONFIGS" \
  "$ANALYZE_SCRIPT" "$FILTER_SCRIPT" "$BUILD_PARTY" "$BUILD_SECTION" "$KFOLD_V2"; do
  [[ -f "$required" ]] || { echo "Required file missing: $required" >&2; exit 1; }
done

short_bucket_name() {
  local bucket="$1"
  if [[ "$bucket" == "cross_bucket_total_dataset" ]]; then
    echo "cross_bucket"
  else
    echo "${bucket%_timed_mistral}"
  fi
}

yaml_query() {
  local cfg="$1" path="$2"
  CONFIG_PATH="$cfg" YAML_PATH="$path" "${PYTHON[@]}" - <<'PY'
import os
import yaml

value = yaml.safe_load(open(os.environ["CONFIG_PATH"], "r", encoding="utf-8"))
for part in os.environ["YAML_PATH"].split("."):
    value = value[part]
print(value)
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

run_dir_path() {
  local cfg="$1" run_name="$2"
  echo "$(yaml_query "$cfg" "paths.outputs_dir")/models/$run_name/kfold"
}

is_summary_complete() {
  local summary="$1"
  [[ -f "$summary" ]] || return 1
  SUMMARY="$summary" EXPECTED_K="$K" "${PYTHON[@]}" - <<'PY'
import json
import os
import sys

try:
    data = json.load(open(os.environ["SUMMARY"], "r", encoding="utf-8"))
except Exception:
    sys.exit(1)
expected = int(os.environ["EXPECTED_K"])
sys.exit(0 if data.get("k") == expected and data.get("n_folds_completed") == expected else 1)
PY
}

cleaned_count_for_cfg() {
  local cfg="$1"
  local cleaned_dir
  cleaned_dir="$(yaml_query "$cfg" "paths.cleaned_case_dir")"
  if [[ -d "$cleaned_dir" ]]; then
    find "$cleaned_dir" -maxdepth 1 -type f -name '*.json' | wc -l
  else
    echo 0
  fi
}

variant_enabled() {
  local variant="$1"
  [[ "$ONLY" == "both" || "$ONLY" == "$variant" ]]
}

entity_cfg() {
  local bucket="$1"
  echo "$ENTITY_CONFIG_ROOT/party/$bucket/config.yaml"
}

variant_config() {
  local variant="$1" bucket="$2"
  echo "$CONFIG_ROOT/$variant/$bucket/config.yaml"
}

run_name_for() {
  local variant="$1" bucket="$2" short
  short="$(short_bucket_name "$bucket")"
  if [[ "$variant" == "party" && "$LR_MODE" == "decay" ]]; then
    echo "${short}_central_authorities_removed_party_args_preamble_lr_decay_kfold"
  elif [[ "$variant" == "party" ]]; then
    echo "${short}_central_authorities_removed_party_args_preamble_no_lr_kfold"
  elif [[ "$LR_MODE" == "decay" ]]; then
    echo "ablation_central_authorities_removed_section_sep_lr_decay_${short}_kfold"
  else
    echo "ablation_central_authorities_removed_section_sep_no_lr_${short}_kfold"
  fi
}

sync_configs() {
  log "Syncing entity-resolved source configs"
  "${PYTHON[@]}" "$ENTITY_PREPARE_CONFIGS" --only both
  log "Syncing central-authority removal configs"
  "${PYTHON[@]}" "$PREPARE_CONFIGS" --only both --config-root "$CONFIG_ROOT" --lr-mode "$LR_MODE"
}

ensure_source_cleaned() {
  local bucket="$1" cfg count cleaned_dir entity_dir audits_dir limit_args=()
  cfg="$(entity_cfg "$bucket")"
  count="$(cleaned_count_for_cfg "$cfg")"
  if [[ "$FORCE_SOURCE_PREPROCESS" == "1" ]]; then
    cleaned_dir="$(yaml_query "$cfg" "paths.cleaned_case_dir")"
    entity_dir="$(yaml_query "$cfg" "paths.normalized_entity_dir")"
    audits_dir="$(yaml_query "$cfg" "paths.audits_dir")"
    rm -rf "$cleaned_dir" "$entity_dir" "$audits_dir"
    count=0
  fi
  if [[ "$count" -gt 0 ]]; then
    log "  [source-preprocess:$bucket] Entity-resolved cleaned cases exist ($count); skipping"
    return
  fi
  if [[ "$SKIP_SOURCE_PREPROCESS" == "1" ]]; then
    echo "Missing entity-resolved source cleaned cases for $bucket and SKIP_SOURCE_PREPROCESS=1" >&2
    exit 1
  fi
  if [[ -n "$LIMIT" ]]; then
    limit_args=(--limit "$LIMIT")
  fi
  log "  [source-preprocess:$bucket] Building entity-resolved cleaned cases"
  "${PYTHON[@]}" "$ENTITY_PREPROCESS" --config "$cfg" "${limit_args[@]}"
}

run_centrality_analysis() {
  local extra_args=()
  for statute in "${EXTRA_STATUTES[@]}"; do
    extra_args+=(--extra-statute "$statute")
  done
  log "[centrality] Computing central authority candidates"
  "${PYTHON[@]}" "$ANALYZE_SCRIPT" \
    --config-root "$ENTITY_CONFIG_ROOT/party" \
    --output-dir "$CENTRALITY_DIR" \
    --min-global-case-frequency "$MIN_GLOBAL_CASE_FREQUENCY" \
    --buckets "${BUCKETS[@]}" \
    "${extra_args[@]}"
  if [[ -z "$REMOVAL_SET" ]]; then
    REMOVAL_SET="$CENTRALITY_DIR/central_authority_removal_set.json"
  fi
}

ensure_removal_set() {
  if [[ -n "$REMOVAL_SET" && -f "$REMOVAL_SET" ]]; then
    log "[centrality] Reusing removal set -> $REMOVAL_SET"
    return
  fi
  run_centrality_analysis
}

filter_bucket_if_needed() {
  local bucket="$1" src_cfg dst_cfg input_cleaned output_cleaned output_entities output_audits summary count filter_args=()
  src_cfg="$(entity_cfg "$bucket")"
  dst_cfg="$(variant_config party "$bucket")"
  input_cleaned="$(yaml_query "$src_cfg" "paths.cleaned_case_dir")"
  output_cleaned="$(yaml_query "$dst_cfg" "paths.cleaned_case_dir")"
  output_entities="$(yaml_query "$dst_cfg" "paths.normalized_entity_dir")"
  output_audits="$(yaml_query "$dst_cfg" "paths.audits_dir")"
  summary="$(yaml_query "$dst_cfg" "paths.processed_dir")/central_authority_filter_summary.json"

  count=0
  if [[ -d "$output_cleaned" ]]; then
    count="$(find "$output_cleaned" -maxdepth 1 -type f -name '*.json' | wc -l)"
  fi
  if [[ "$FORCE_FILTER" == "1" ]]; then
    rm -rf "$output_cleaned" "$output_entities" "$output_audits"
    count=0
  fi
  if [[ "$count" -gt 0 && -f "$summary" ]]; then
    log "  [filter:$bucket] Filtered cleaned cases exist ($count); skipping"
    return
  fi
  if [[ "$DROP_LINKED_PROVISIONS" == true ]]; then
    filter_args+=(--drop-provisions-linked-to-removed-statutes)
  fi
  log "  [filter:$bucket] Removing selected central authorities"
  "${PYTHON[@]}" "$FILTER_SCRIPT" \
    --input-cleaned-dir "$input_cleaned" \
    --output-cleaned-dir "$output_cleaned" \
    --output-entity-dir "$output_entities" \
    --output-audits-dir "$output_audits" \
    --summary-path "$summary" \
    --removal-set "$REMOVAL_SET" \
    "${filter_args[@]}"
}

build_if_needed() {
  local cfg="$1" build_script="$2" label="$3"
  local graph_cache
  graph_cache="$(graph_cache_path "$cfg")"
  if [[ -f "$graph_cache" ]]; then
    log "  [build:$label] Cache exists -> $graph_cache"
    return
  fi
  if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "Missing graph cache for $label and SKIP_BUILD=1: $graph_cache" >&2
    exit 1
  fi
  log "  [build:$label] Building graph -> $graph_cache"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS_BUILD" \
    "${PYTHON[@]}" "$build_script" --config "$cfg"
}

run_missing_folds() {
  local cfg="$1" run_name="$2" graph_cache="$3" label="$4"
  local run_dir outputs_dir log_dir summary
  run_dir="$(run_dir_path "$cfg" "$run_name")"
  outputs_dir="$(yaml_query "$cfg" "paths.outputs_dir")"
  log_dir="$outputs_dir/logs"
  summary="$run_dir/kfold_summary.json"
  mkdir -p "$run_dir" "$log_dir"

  if is_summary_complete "$summary"; then
    log "  [kfold:$label] Complete summary exists -> $summary"
    return
  fi

  local missing=()
  local fold_idx
  for fold_idx in $(seq 0 $((K - 1))); do
    if [[ ! -f "$run_dir/fold_$(printf '%02d' "$fold_idx")/fold_summary.json" ]]; then
      missing+=("$fold_idx")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "  [kfold:$label] Running missing folds: ${missing[*]}"
    local pids=()
    local status=0
    for fold_idx in "${missing[@]}"; do
      local gpu="${TRAIN_GPU_ARRAY[$fold_idx]}"
      local log_file="$log_dir/${run_name}_fold_${fold_idx}.log"
      (
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "${PYTHON[@]}" "$KFOLD_V2" \
            --config "$cfg" \
            --run-name "$run_name" \
            --k "$K" \
            --fold "$fold_idx" \
            --val-fraction "$VAL_FRACTION" \
            --graph-cache "$graph_cache"
      ) > "$log_file" 2>&1 &
      pids+=("$!")
      log "  [kfold:$label] Fold $fold_idx -> GPU $gpu pid=${pids[-1]}"
    done

    local i
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        log "  [kfold:$label] Fold ${missing[$i]} done"
      else
        log "  [kfold:$label] Fold ${missing[$i]} FAILED" >&2
        status=1
      fi
    done
    [[ "$status" -ne 0 ]] && exit "$status"
  else
    log "  [kfold:$label] All fold summaries exist; aggregating"
  fi

  log "  [kfold:$label] Aggregating"
  "${PYTHON[@]}" "$KFOLD_V2" \
    --config "$cfg" \
    --run-name "$run_name" \
    --k "$K" \
    --aggregate-only \
    --graph-cache "$graph_cache"
}

status_line() {
  local variant="$1" bucket="$2" src_cfg cfg run_name label graph_cache summary run_dir fold_count src_count filtered_count complete_flag
  src_cfg="$(entity_cfg "$bucket")"
  cfg="$(variant_config "$variant" "$bucket")"
  run_name="$(run_name_for "$variant" "$bucket")"
  label="$bucket/central_authorities_removed_$variant"
  graph_cache="$(graph_cache_path "$cfg")"
  summary="$(summary_path "$cfg" "$run_name")"
  run_dir="$(run_dir_path "$cfg" "$run_name")"
  src_count="$(cleaned_count_for_cfg "$src_cfg")"
  filtered_count="$(cleaned_count_for_cfg "$cfg")"
  fold_count=0
  if [[ -d "$run_dir" ]]; then
    fold_count="$(find "$run_dir" -path '*/fold_summary.json' -type f | wc -l)"
  fi
  if is_summary_complete "$summary"; then
    complete_flag="DONE"
  else
    complete_flag="TODO"
  fi
  printf "%-5s %-66s src=%s filtered=%s folds=%s/%s graph=%s summary=%s\n" \
    "$complete_flag" "$label" "$src_count" "$filtered_count" "$fold_count" "$K" \
    "$([[ -f "$graph_cache" ]] && echo yes || echo no)" \
    "$([[ -f "$summary" ]] && echo yes || echo no)"
}

run_variant() {
  local variant="$1" bucket="$2" cfg run_name graph_cache label build_script
  cfg="$(variant_config "$variant" "$bucket")"
  run_name="$(run_name_for "$variant" "$bucket")"
  graph_cache="$(graph_cache_path "$cfg")"
  label="$bucket/central_authorities_removed_$variant"

  if [[ "$STATUS_ONLY" == true ]]; then
    status_line "$variant" "$bucket"
    return
  fi

  log "$label"
  if is_summary_complete "$(summary_path "$cfg" "$run_name")"; then
    log "  [skip] Complete"
    return
  fi
  if [[ "$variant" == "party" ]]; then
    build_script="$BUILD_PARTY"
  else
    build_script="$BUILD_SECTION"
  fi
  build_if_needed "$cfg" "$build_script" "$label"
  run_missing_folds "$cfg" "$run_name" "$graph_cache" "$label"
}

sync_configs
if [[ "$SYNC_ONLY" == true ]]; then
  log "Sync-only complete."
  exit 0
fi

log "Requested central-authority removal ablation"
log "env=$MAMBA_ENV k=$K val_fraction=$VAL_FRACTION train_gpus=$TRAIN_GPUS build_gpus=$GPUS_BUILD only=$ONLY lr_mode=$LR_MODE"

if [[ "$STATUS_ONLY" == true ]]; then
  for bucket in "${BUCKETS[@]}"; do
    log "Bucket: $bucket"
    if variant_enabled party; then
      status_line party "$bucket"
    fi
    if variant_enabled section; then
      status_line section "$bucket"
    fi
  done
  log "Done."
  exit 0
fi

for bucket in "${BUCKETS[@]}"; do
  ensure_source_cleaned "$bucket"
done

ensure_removal_set

if [[ "$ANALYZE_ONLY" == true ]]; then
  log "Analyze-only complete."
  exit 0
fi

for bucket in "${BUCKETS[@]}"; do
  filter_bucket_if_needed "$bucket"
done

if [[ "$FILTER_ONLY" == true ]]; then
  log "Filter-only complete."
  exit 0
fi

for bucket in "${BUCKETS[@]}"; do
  log "Bucket: $bucket"
  if variant_enabled party; then
    run_variant party "$bucket"
  fi
  if variant_enabled section; then
    run_variant section "$bucket"
  fi
done

log "Done."
