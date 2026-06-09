#!/usr/bin/env bash
# Controlled LR-decay comparison for baseline and party-argument graphs.
#
# Matrix per bucket:
#   1. baseline_no_lr       existing runs/<bucket>/config.yaml + kfold_cv.py
#   2. baseline_lr_decay    generated config + kfold_cv_v2.py
#   3. party_args_no_lr     generated config + build_graph_v2.py + kfold_cv.py
#   4. party_args_lr_decay  existing runs_v2/party_args_lr_decay config + kfold_cv_v2.py
#
# This script is idempotent. It skips complete summaries, reuses graph caches,
# and reruns only missing fold_<idx>/fold_summary.json files before aggregating.
#
# Usage:
#   bash run_scripts/run_baseline_party_args_lr_control.sh --status-only
#   nohup bash run_scripts/run_baseline_party_args_lr_control.sh \
#     > run_logs/run_baseline_party_args_lr_control.log 2>&1 &
#   tail -f run_logs/run_baseline_party_args_lr_control.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SECTION_GNN"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4}"
STATUS_ONLY=false
SYNC_ONLY=false
SKIP_BUILD="${SKIP_BUILD:-0}"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)          MAMBA_ENV="$2"; shift 2 ;;
    --k)            K="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --gpus-build)   GPUS_BUILD="$2"; shift 2 ;;
    --train-gpus)   TRAIN_GPUS="$2"; shift 2 ;;
    --buckets)      IFS=' ' read -r -a BUCKETS <<< "$2"; shift 2 ;;
    --status-only)  STATUS_ONLY=true; shift ;;
    --sync-only)    SYNC_ONLY=true; shift ;;
    --skip-build)   SKIP_BUILD=1; shift ;;
    --help|-h)      usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -lt "$K" ]]; then
  echo "Need at least K=$K train GPUs in TRAIN_GPUS, got: $TRAIN_GPUS" >&2
  exit 1
fi

PYTHON="micromamba run -n $MAMBA_ENV python"
BUILD_GRAPH="$SECTION_GNN/src/scripts/build_graph.py"
BUILD_GRAPH_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/graph/build_graph_v2.py"
KFOLD="$SECTION_GNN/src/scripts/kfold_cv.py"
KFOLD_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py"
BASELINE_LR_ROOT="$SECTION_GNN/runs_v2/baseline_lr_decay"
PARTY_ARGS_NO_LR_ROOT="$SECTION_GNN/runs_v2/party_args_no_lr"

for required in "$BUILD_GRAPH" "$BUILD_GRAPH_V2" "$KFOLD" "$KFOLD_V2"; do
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
  CONFIG_PATH="$cfg" YAML_PATH="$path" $PYTHON - <<'PY'
import os, yaml
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
  SUMMARY="$summary" EXPECTED_K="$K" $PYTHON - <<'PY'
import json, os, sys
try:
    data = json.load(open(os.environ["SUMMARY"], "r", encoding="utf-8"))
except Exception:
    sys.exit(1)
expected = int(os.environ["EXPECTED_K"])
sys.exit(0 if data.get("k") == expected and data.get("n_folds_completed") == expected else 1)
PY
}

sync_configs() {
  log "Syncing generated configs"
  SECTION_GNN="$SECTION_GNN" \
  BASELINE_LR_ROOT="$BASELINE_LR_ROOT" \
  PARTY_ARGS_NO_LR_ROOT="$PARTY_ARGS_NO_LR_ROOT" \
  $PYTHON - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["SECTION_GNN"])
baseline_lr_root = Path(os.environ["BASELINE_LR_ROOT"])
party_args_no_lr_root = Path(os.environ["PARTY_ARGS_NO_LR_ROOT"])

buckets = [
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
    "cross_bucket_total_dataset",
]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def short_name(bucket: str) -> str:
    return "cross_bucket" if bucket == "cross_bucket_total_dataset" else bucket.removesuffix("_timed_mistral")


for bucket in buckets:
    short = short_name(bucket)

    base_cfg = load_yaml(root / "runs" / bucket / "config.yaml")
    baseline_lr = copy.deepcopy(base_cfg)
    baseline_lr["project"]["name"] = f"{short}_baseline_lr_decay"
    baseline_lr_training = baseline_lr.setdefault("training", {})
    baseline_lr_training.update(
        {
            "epochs": 90,
            "use_early_stopping": True,
            "early_stopping_patience": 20,
            "lr_scheduler": "reduce_on_plateau",
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 8,
            "lr_min": 0.000001,
        }
    )
    save_yaml(baseline_lr_root / bucket / "config.yaml", baseline_lr)

    party_lr_cfg = load_yaml(root / "runs_v2" / "party_args_lr_decay" / bucket / "config.yaml")
    party_no_lr = copy.deepcopy(party_lr_cfg)
    party_no_lr["project"]["name"] = f"{short}_party_args_no_lr"
    party_no_lr_training = party_no_lr.setdefault("training", {})
    party_no_lr_training.update(
        {
            "epochs": 60,
            "use_early_stopping": True,
            "early_stopping_patience": 15,
        }
    )
    for key in ("lr_scheduler", "lr_scheduler_factor", "lr_scheduler_patience", "lr_min"):
        party_no_lr_training.pop(key, None)
    save_yaml(party_args_no_lr_root / bucket / "config.yaml", party_no_lr)

print(f"Wrote baseline LR configs under {baseline_lr_root}")
print(f"Wrote party-args no-LR configs under {party_args_no_lr_root}")
PY
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
    $PYTHON "$build_script" --config "$cfg"
}

run_missing_folds() {
  local cfg="$1" run_name="$2" kfold_script="$3" graph_cache="$4" label="$5"
  local run_dir log_dir outputs_dir summary
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
          $PYTHON "$kfold_script" \
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
  $PYTHON "$kfold_script" \
    --config "$cfg" \
    --run-name "$run_name" \
    --k "$K" \
    --aggregate-only \
    --graph-cache "$graph_cache"
}

status_line() {
  local cfg="$1" run_name="$2" label="$3"
  local summary run_dir graph_cache fold_count complete_flag
  summary="$(summary_path "$cfg" "$run_name")"
  run_dir="$(run_dir_path "$cfg" "$run_name")"
  graph_cache="$(graph_cache_path "$cfg")"
  fold_count=0
  if [[ -d "$run_dir" ]]; then
    fold_count="$(find "$run_dir" -path '*/fold_summary.json' -type f | wc -l)"
  fi
  if is_summary_complete "$summary"; then
    complete_flag="DONE"
  else
    complete_flag="TODO"
  fi
  printf "%-5s %-43s folds=%s/%s graph=%s summary=%s\n" \
    "$complete_flag" "$label" "$fold_count" "$K" \
    "$([[ -f "$graph_cache" ]] && echo yes || echo no)" \
    "$([[ -f "$summary" ]] && echo yes || echo no)"
}

run_cell() {
  local bucket="$1" cell="$2"
  local short cfg run_name build_script kfold_script graph_cache label
  short="$(short_bucket_name "$bucket")"

  case "$cell" in
    baseline_no_lr)
      cfg="$SECTION_GNN/runs/$bucket/config.yaml"
      run_name="${bucket}_kfold"
      build_script="$BUILD_GRAPH"
      kfold_script="$KFOLD"
      label="$bucket/baseline_no_lr"
      ;;
    baseline_lr_decay)
      cfg="$BASELINE_LR_ROOT/$bucket/config.yaml"
      run_name="${short}_baseline_lr_decay_kfold"
      build_script="$BUILD_GRAPH"
      kfold_script="$KFOLD_V2"
      label="$bucket/baseline_lr_decay"
      ;;
    party_args_no_lr)
      cfg="$PARTY_ARGS_NO_LR_ROOT/$bucket/config.yaml"
      run_name="${short}_party_args_no_lr_kfold"
      build_script="$BUILD_GRAPH_V2"
      kfold_script="$KFOLD"
      label="$bucket/party_args_no_lr"
      ;;
    party_args_lr_decay)
      cfg="$SECTION_GNN/runs_v2/party_args_lr_decay/$bucket/config.yaml"
      run_name="${short}_party_args_lr_decay_kfold"
      build_script="$BUILD_GRAPH_V2"
      kfold_script="$KFOLD_V2"
      label="$bucket/party_args_lr_decay"
      ;;
    *) echo "Unknown cell: $cell" >&2; exit 1 ;;
  esac

  graph_cache="$(graph_cache_path "$cfg")"
  if [[ "$STATUS_ONLY" == true ]]; then
    status_line "$cfg" "$run_name" "$label"
    return
  fi

  log "$label"
  if is_summary_complete "$(summary_path "$cfg" "$run_name")"; then
    log "  [skip] Complete"
    return
  fi
  build_if_needed "$cfg" "$build_script" "$label"
  run_missing_folds "$cfg" "$run_name" "$kfold_script" "$graph_cache" "$label"
}

sync_configs
if [[ "$SYNC_ONLY" == true ]]; then
  log "Sync-only complete."
  exit 0
fi

log "Controlled baseline/party-args LR matrix"
log "env=$MAMBA_ENV k=$K val_fraction=$VAL_FRACTION train_gpus=$TRAIN_GPUS build_gpus=$GPUS_BUILD"

for bucket in "${BUCKETS[@]}"; do
  log "Bucket: $bucket"
  run_cell "$bucket" baseline_no_lr
  run_cell "$bucket" baseline_lr_decay
  run_cell "$bucket" party_args_no_lr
  run_cell "$bucket" party_args_lr_decay
done

log "Done."
