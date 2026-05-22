#!/usr/bin/env bash
# Run party-args no-LR first, then the two requested LR-decay experiment variants.
#
# Variants:
#   1. party_args_preamble_no_lr
#      Runs the party-args graph/config without LR decay, with preamble included
#      in the case-node text encoding.
#
#   2. party_args_preamble_lr_decay
#      Starts from runs_v2/party_args_lr_decay configs, but changes the
#      case-node text encoding to include:
#        preamble facts arguments petitioner_arguments respondent_arguments
#      Override with:
#        PARTY_CASE_TEXT_SECTIONS="preamble facts petitioner_arguments respondent_arguments"
#
#   3. section_sep_lr_decay
#      Starts from ablations/section_sep_enc configs and trains with the v2
#      LR-decay k-fold runner.
#
# Usage:
#   bash run_party_args_preamble_and_section_sep_lr_decay.sh --status-only
#   bash run_party_args_preamble_and_section_sep_lr_decay.sh --sync-only
#   nohup bash run_party_args_preamble_and_section_sep_lr_decay.sh \
#     > run_party_args_preamble_and_section_sep_lr_decay.log 2>&1 &
#
# Useful options:
#   --only party-no-lr Run only party_args_preamble_no_lr.
#   --only party       Run only party_args_preamble_lr_decay.
#   --only section     Run only section_sep_lr_decay.
#   --skip-build      Require existing graph caches and only train/aggregate.
#   --buckets "fin_fraud_timed_mistral motor_accidents_timed_mistral"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$SCRIPT_DIR"
export PYTHONPATH="$SECTION_GNN:${PYTHONPATH:-}"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4}"
SKIP_BUILD="${SKIP_BUILD:-0}"
ONLY="both"
STATUS_ONLY=false
SYNC_ONLY=false

PARTY_NO_LR_ROOT="$SECTION_GNN/runs_v2/party_args_no_lr"
PARTY_ROOT="$SECTION_GNN/runs_v2/party_args_preamble_lr_decay"
SECTION_LR_ROOT="$SECTION_GNN/ablations/section_sep_enc_lr_decay"
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
  sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d; s/^# \{0,1\}//'
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
    --only)         ONLY="$2"; shift 2 ;;
    --status-only)  STATUS_ONLY=true; shift ;;
    --sync-only)    SYNC_ONLY=true; shift ;;
    --skip-build)   SKIP_BUILD=1; shift ;;
    --help|-h)      usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$ONLY" in
  both|all|party-no-lr|party_no_lr|party_args_no_lr|party|party_args|party_args_preamble_lr_decay|section|section_sep|section_sep_lr_decay) ;;
  *) echo "Unknown --only value: $ONLY" >&2; exit 1 ;;
esac

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -lt "$K" ]]; then
  echo "Need at least K=$K train GPUs in TRAIN_GPUS, got: $TRAIN_GPUS" >&2
  exit 1
fi

PYTHON=(micromamba run -n "$MAMBA_ENV" python)

[[ -d "$PARTY_NO_LR_ROOT" ]] || { echo "Required directory missing: $PARTY_NO_LR_ROOT" >&2; exit 1; }
for required in "$BUILD_PARTY" "$BUILD_SECTION" "$KFOLD_V2"; do
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

sync_configs() {
  log "Syncing generated configs"
  SECTION_GNN="$SECTION_GNN" \
  PARTY_NO_LR_ROOT="$PARTY_NO_LR_ROOT" \
  PARTY_ROOT="$PARTY_ROOT" \
  SECTION_LR_ROOT="$SECTION_LR_ROOT" \
  PARTY_CASE_TEXT_SECTIONS="${PARTY_CASE_TEXT_SECTIONS:-preamble facts arguments petitioner_arguments respondent_arguments}" \
  "${PYTHON[@]}" - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["SECTION_GNN"])
party_no_lr_root = Path(os.environ["PARTY_NO_LR_ROOT"])
party_root = Path(os.environ["PARTY_ROOT"])
section_lr_root = Path(os.environ["SECTION_LR_ROOT"])
party_case_text_sections = os.environ["PARTY_CASE_TEXT_SECTIONS"].split()

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
    if bucket == "cross_bucket_total_dataset":
        return "cross_bucket"
    return bucket.removesuffix("_timed_mistral")


def apply_lr_decay(cfg: dict):
    training = cfg.setdefault("training", {})
    training.update(
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


def remove_lr_decay(cfg: dict):
    training = cfg.setdefault("training", {})
    training.update(
        {
            "epochs": 60,
            "use_early_stopping": True,
            "early_stopping_patience": 15,
        }
    )
    for key in (
        "lr_scheduler",
        "lr_scheduler_factor",
        "lr_scheduler_patience",
        "lr_min",
    ):
        training.pop(key, None)


for bucket in buckets:
    short = short_name(bucket)

    party_no_lr_cfg = copy.deepcopy(load_yaml(root / "runs_v2" / "party_args_no_lr" / bucket / "config.yaml"))
    party_no_lr_cfg["project"]["name"] = f"{short}_party_args_preamble_no_lr"
    party_no_lr_cfg["graph"]["case_text_sections"] = party_case_text_sections
    party_no_lr_cfg["graph"]["cache_name"] = f"case_star_global_graph_{short}_party_args_preamble.reasoning_focused.pt"
    remove_lr_decay(party_no_lr_cfg)
    save_yaml(party_no_lr_root / bucket / "config.yaml", party_no_lr_cfg)

    party_cfg = copy.deepcopy(load_yaml(root / "runs_v2" / "party_args_lr_decay" / bucket / "config.yaml"))
    party_cfg["project"]["name"] = f"{short}_party_args_preamble_lr_decay"
    party_cfg["graph"]["case_text_sections"] = party_case_text_sections
    party_cfg["graph"]["cache_name"] = f"case_star_global_graph_{short}_party_args_preamble.reasoning_focused.pt"
    apply_lr_decay(party_cfg)
    save_yaml(party_root / bucket / "config.yaml", party_cfg)

    section_cfg = copy.deepcopy(load_yaml(root / "ablations" / "section_sep_enc" / bucket / "config.yaml"))
    section_cfg["project"]["name"] = f"{short}_section_sep_enc_lr_decay"
    section_cfg["graph"]["cache_name"] = f"case_star_{short}_section_sep_enc_lr_decay.reasoning_focused.pt"
    apply_lr_decay(section_cfg)
    save_yaml(section_lr_root / bucket / "config.yaml", section_cfg)

print(f"Wrote party-args+preamble no-LR configs under {party_no_lr_root}")
print(f"Wrote party-args+preamble LR configs under {party_root}")
print(f"Wrote section-separated LR configs under {section_lr_root}")
print(f"party case_text_sections={party_case_text_sections}")
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
  printf "%-5s %-55s folds=%s/%s graph=%s summary=%s\n" \
    "$complete_flag" "$label" "$fold_count" "$K" \
    "$([[ -f "$graph_cache" ]] && echo yes || echo no)" \
    "$([[ -f "$summary" ]] && echo yes || echo no)"
}

run_cell() {
  local bucket="$1" cell="$2"
  local short cfg run_name build_script graph_cache label
  short="$(short_bucket_name "$bucket")"

  case "$cell" in
    party_no_lr)
      cfg="$PARTY_NO_LR_ROOT/$bucket/config.yaml"
      run_name="${short}_party_args_preamble_no_lr_kfold"
      build_script="$BUILD_PARTY"
      label="$bucket/party_args_preamble_no_lr"
      ;;
    party)
      cfg="$PARTY_ROOT/$bucket/config.yaml"
      run_name="${short}_party_args_preamble_lr_decay_kfold"
      build_script="$BUILD_PARTY"
      label="$bucket/party_args_preamble_lr_decay"
      ;;
    section)
      cfg="$SECTION_LR_ROOT/$bucket/config.yaml"
      run_name="ablation_section_sep_enc_lr_decay_${short}_kfold"
      build_script="$BUILD_SECTION"
      label="$bucket/section_sep_lr_decay"
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
  run_missing_folds "$cfg" "$run_name" "$graph_cache" "$label"
}

should_run_party_no_lr() {
  [[ "$ONLY" == "both" || "$ONLY" == "all" || "$ONLY" == "party-no-lr" || "$ONLY" == "party_no_lr" || "$ONLY" == "party_args_no_lr" ]]
}

should_run_party() {
  [[ "$ONLY" == "both" || "$ONLY" == "all" || "$ONLY" == "party" || "$ONLY" == "party_args" || "$ONLY" == "party_args_preamble_lr_decay" ]]
}

should_run_section() {
  [[ "$ONLY" == "both" || "$ONLY" == "all" || "$ONLY" == "section" || "$ONLY" == "section_sep" || "$ONLY" == "section_sep_lr_decay" ]]
}

sync_configs
if [[ "$SYNC_ONLY" == true ]]; then
  log "Sync-only complete."
  exit 0
fi

log "Requested party no-LR plus LR-decay experiments"
log "env=$MAMBA_ENV k=$K val_fraction=$VAL_FRACTION train_gpus=$TRAIN_GPUS build_gpus=$GPUS_BUILD only=$ONLY"
log "Order: party_args_preamble_no_lr -> party_args_preamble_lr_decay -> section_sep_lr_decay"

for bucket in "${BUCKETS[@]}"; do
  log "Bucket: $bucket"
  if should_run_party_no_lr; then
    run_cell "$bucket" party_no_lr
  fi
  if should_run_party; then
    run_cell "$bucket" party
  fi
  if should_run_section; then
    run_cell "$bucket" section
  fi
done

log "Done."
