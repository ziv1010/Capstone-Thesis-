#!/usr/bin/env bash
# Sync non-cross-bucket ablation configs from the family_matrimonial template,
# then run only the remaining missing ablations.
#
# Excludes cross_bucket_total_dataset by design.
#
# Usage:
#   bash run_remaining_non_cross_bucket_ablations.sh
#   bash run_remaining_non_cross_bucket_ablations.sh --sync-only
#   bash run_remaining_non_cross_bucket_ablations.sh --buckets "fin_fraud_timed_mistral land_property_timed_mistral"
#   bash run_remaining_non_cross_bucket_ablations.sh --env thesis_work --k 5 --gpus-build 0,1,2,3,4,5,6,7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$SCRIPT_DIR"
FAMILY_BUCKET="family_matrimonial_timed_mistral"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K=5
GPUS_BUILD="0,1,2,3,4,5,6,7"
SYNC_ONLY=false
BUCKETS=(
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
)

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)        MAMBA_ENV="$2"; shift 2 ;;
    --k)          K="$2"; shift 2 ;;
    --gpus-build) GPUS_BUILD="$2"; shift 2 ;;
    --buckets)    IFS=' ' read -r -a BUCKETS <<< "$2"; shift 2 ;;
    --sync-only)  SYNC_ONLY=true; shift ;;
    --help|-h)    usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

yaml_query() {
  local config_path="$1"
  local yaml_query_path="$2"
  CONFIG_PATH="$config_path" YAML_QUERY_PATH="$yaml_query_path" \
    micromamba run -n "$MAMBA_ENV" python - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)

for part in os.environ["YAML_QUERY_PATH"].split("."):
    value = value[part]

print(value)
PY
}

graph_cache_path() {
  local config_path="$1"
  local graph_cache_dir
  local cache_name
  graph_cache_dir="$(yaml_query "$config_path" "paths.graph_cache_dir")"
  cache_name="$(yaml_query "$config_path" "graph.cache_name")"
  echo "$graph_cache_dir/$cache_name"
}

summary_path() {
  local config_path="$1"
  local run_name="$2"
  local outputs_dir
  outputs_dir="$(yaml_query "$config_path" "paths.outputs_dir")"
  echo "$outputs_dir/models/$run_name/kfold/kfold_summary.json"
}

sync_bucket_from_family() {
  local bucket="$1"
  local short_name="${bucket%_timed_mistral}"

  log "Syncing ablation configs for $bucket from $FAMILY_BUCKET"
  SECTION_GNN="$SECTION_GNN" \
  FAMILY_BUCKET="$FAMILY_BUCKET" \
  TARGET_BUCKET="$bucket" \
  TARGET_SHORT_NAME="$short_name" \
  micromamba run -n "$MAMBA_ENV" python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["SECTION_GNN"])
family_bucket = os.environ["FAMILY_BUCKET"]
target_bucket = os.environ["TARGET_BUCKET"]
target_short_name = os.environ["TARGET_SHORT_NAME"]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


base_config = load_yaml(root / "runs" / target_bucket / "config.yaml")
family_no_cross = load_yaml(root / "ablations" / "no_cross_case" / family_bucket / "config.yaml")
family_text_only = load_yaml(root / "ablations" / "text_only" / family_bucket / "config.yaml")
family_depth_configs = {
    depth: load_yaml(root / "ablations" / "depth" / family_bucket / f"config_depth{depth}.yaml")
    for depth in (1, 2, 3)
}

no_cross_config = copy.deepcopy(family_no_cross)
no_cross_config["project"]["name"] = f"{target_short_name}_no_cross_case"
no_cross_config["paths"] = copy.deepcopy(base_config["paths"])
no_cross_config["graph"]["cache_name"] = f"case_star_{target_short_name}_no_cross_case.reasoning_focused.pt"
save_yaml(root / "ablations" / "no_cross_case" / target_bucket / "config.yaml", no_cross_config)

text_only_config = copy.deepcopy(family_text_only)
text_only_config["project"]["name"] = f"{target_short_name}_text_only"
text_only_config["paths"] = copy.deepcopy(base_config["paths"])
text_only_config["graph"]["cache_name"] = f"case_star_{target_short_name}_text_only.reasoning_focused.pt"
save_yaml(root / "ablations" / "text_only" / target_bucket / "config.yaml", text_only_config)

for depth, family_depth_config in family_depth_configs.items():
    depth_config = copy.deepcopy(family_depth_config)
    depth_config["project"] = copy.deepcopy(base_config["project"])
    depth_config["paths"] = copy.deepcopy(base_config["paths"])
    depth_config["graph"]["cache_name"] = base_config["graph"]["cache_name"]
    save_yaml(
        root / "ablations" / "depth" / target_bucket / f"config_depth{depth}.yaml",
        depth_config,
    )
PY
}

run_kfold() {
  local config_path="$1"
  local run_name="$2"
  local graph_cache="$3"
  local outputs_dir
  local log_dir
  local aggregate_log
  local fold_idx
  local log_file
  local pids=()
  local status=0

  outputs_dir="$(yaml_query "$config_path" "paths.outputs_dir")"
  log_dir="$outputs_dir/logs"
  aggregate_log="$log_dir/kfold_cv_${run_name}.log"
  mkdir -p "$log_dir"

  log "Launching $run_name with k=$K using graph cache $graph_cache"
  for fold_idx in $(seq 0 $((K - 1))); do
    log_file="$log_dir/${run_name}_fold_${fold_idx}.log"
    (
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$fold_idx" \
        micromamba run -n "$MAMBA_ENV" python \
        "$SECTION_GNN/dump2/scripts/kfold_cv.py" \
        --config "$config_path" \
        --run-name "$run_name" \
        --k "$K" \
        --fold "$fold_idx" \
        --graph-cache "$graph_cache"
    ) 2>&1 | tee -a "$log_file" &
    pids+=("$!")
    log "Fold $fold_idx -> GPU $fold_idx (pid ${pids[-1]})"
  done

  for fold_idx in "${!pids[@]}"; do
    if wait "${pids[$fold_idx]}"; then
      log "Fold $fold_idx complete for $run_name"
    else
      log "Fold $fold_idx failed for $run_name"
      status=1
    fi
  done

  [[ "$status" -ne 0 ]] && exit "$status"

  log "Aggregating $run_name"
  micromamba run -n "$MAMBA_ENV" python \
    "$SECTION_GNN/dump2/scripts/kfold_cv.py" \
    --config "$config_path" \
    --run-name "$run_name" \
    --k "$K" \
    --aggregate-only \
    --graph-cache "$graph_cache" 2>&1 | tee -a "$aggregate_log"

  log "Summary written to $(summary_path "$config_path" "$run_name")"
}

build_graph_for_ablation() {
  local config_path="$1"
  local run_name="$2"
  local outputs_dir
  local log_dir
  local build_log

  outputs_dir="$(yaml_query "$config_path" "paths.outputs_dir")"
  log_dir="$outputs_dir/logs"
  build_log="$log_dir/build_${run_name}.log"
  mkdir -p "$log_dir"

  log "Building graph for $run_name"
  (
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS_BUILD" \
      micromamba run -n "$MAMBA_ENV" python \
      "$SECTION_GNN/final_graph/build_graph.py" \
      --config "$config_path"
  ) 2>&1 | tee -a "$build_log"
}

run_if_missing() {
  local config_path="$1"
  local run_name="$2"
  local graph_cache="$3"
  local should_build="${4:-false}"
  local summary

  summary="$(summary_path "$config_path" "$run_name")"
  if [[ -f "$summary" ]]; then
    log "Skipping $run_name because $summary already exists"
    return
  fi

  if [[ "$should_build" == true ]]; then
    build_graph_for_ablation "$config_path" "$run_name"
    graph_cache="$(graph_cache_path "$config_path")"
  fi

  if [[ ! -f "$graph_cache" ]]; then
    echo "Missing graph cache for $run_name: $graph_cache" >&2
    exit 1
  fi

  run_kfold "$config_path" "$run_name" "$graph_cache"
}

run_bucket_remaining() {
  local bucket="$1"
  local short_name="${bucket%_timed_mistral}"
  local base_config="$SECTION_GNN/runs/$bucket/config.yaml"
  local base_graph_cache

  base_graph_cache="$(graph_cache_path "$base_config")"

  log "============================================================"
  log "Bucket: $bucket"
  log "============================================================"

  run_if_missing \
    "$SECTION_GNN/ablations/no_cross_case/$bucket/config.yaml" \
    "ablation_no_cross_case_${short_name}_kfold" \
    "" \
    true

  run_if_missing \
    "$SECTION_GNN/ablations/text_only/$bucket/config.yaml" \
    "ablation_text_only_${short_name}_kfold" \
    "" \
    true

  run_if_missing \
    "$SECTION_GNN/ablations/depth/$bucket/config_depth1.yaml" \
    "ablation_depth1_${short_name}_kfold" \
    "$base_graph_cache"

  run_if_missing \
    "$SECTION_GNN/ablations/depth/$bucket/config_depth2.yaml" \
    "ablation_depth2_${short_name}_kfold" \
    "$base_graph_cache"

  run_if_missing \
    "$SECTION_GNN/ablations/depth/$bucket/config_depth3.yaml" \
    "ablation_depth3_${short_name}_kfold" \
    "$base_graph_cache"
}

log "Syncing remaining non-cross-bucket ablation configs from family template"
for bucket in "${BUCKETS[@]}"; do
  sync_bucket_from_family "$bucket"
done

if [[ "$SYNC_ONLY" == true ]]; then
  log "Sync complete. Exiting because --sync-only was set."
  exit 0
fi

log "Starting remaining ablation runs"
for bucket in "${BUCKETS[@]}"; do
  run_bucket_remaining "$bucket"
done

log "All remaining non-cross-bucket ablations are complete."
