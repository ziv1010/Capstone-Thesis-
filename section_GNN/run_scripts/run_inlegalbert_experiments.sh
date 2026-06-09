#!/usr/bin/env bash
# InLegalBERT main-comparison matrix.
#
# Builds/runs the same 8 experiment families currently completed for BGE-M3:
#   baseline, no_names, text_only, no_cross_case, hierarchical_enc,
#   section_sep_enc, case_node_minimised, party_args_lr_decay
#
# Also supports explicit LR-control experiments via --experiments:
#   baseline_lr_decay, party_args_no_lr
#
# across all 6 buckets:
#   family_matrimonial, fin_fraud, land_property, motor_accidents,
#   sexual_offences, cross_bucket_total_dataset
#
# This script writes generated configs under:
#   section_GNN/runs_inlegalbert/<experiment>/<bucket>/config.yaml
#
# Outputs go under:
#   section_GNN/outputs/inlegalbert_runs/<bucket>/models/
#
# Data caches go under:
#   section_GNN/data/inlegalbert_runs/<bucket>/
#
# Usage:
#   bash run_scripts/run_inlegalbert_experiments.sh --status-only
#   nohup bash run_scripts/run_inlegalbert_experiments.sh > run_logs/run_inlegalbert_experiments.log 2>&1 &
#   tail -f run_logs/run_inlegalbert_experiments.log
#
# Notes:
#   - This does not touch existing BGE-M3 configs or outputs.
#   - It reuses the already-cleaned case JSONs from data/timed_bucket_runs.
#   - It creates new embeddings/graph caches because the encoder changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SECTION_GNN"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4}"
MODEL_NAME="${MODEL_NAME:-law-ai/InLegalBERT}"
POOLING="${POOLING:-mean}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
STATUS_ONLY=false
SYNC_ONLY=false
SKIP_BUILD="${SKIP_BUILD:-0}"

CONFIG_ROOT="$SECTION_GNN/runs_inlegalbert"
DATA_ROOT="$SECTION_GNN/data/inlegalbert_runs"
OUT_ROOT="$SECTION_GNN/outputs/inlegalbert_runs"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

EXPERIMENTS=(
  baseline
  no_names
  text_only
  no_cross_case
  hierarchical_enc
  section_sep_enc
  case_node_minimised
  party_args_lr_decay
)

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
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
    --model-name)   MODEL_NAME="$2"; shift 2 ;;
    --pooling)      POOLING="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --max-length)   MAX_LENGTH="$2"; shift 2 ;;
    --buckets)      IFS=' ' read -r -a BUCKETS <<< "$2"; shift 2 ;;
    --experiments)  IFS=' ' read -r -a EXPERIMENTS <<< "$2"; shift 2 ;;
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
BUILD_GRAPH_SECTION_SEP="$SECTION_GNN/final_graph/build_graph_section_sep.py"
BUILD_GRAPH_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/graph/build_graph_v2.py"
KFOLD="$SECTION_GNN/src/scripts/kfold_cv.py"
KFOLD_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py"

for required in "$BUILD_GRAPH" "$BUILD_GRAPH_SECTION_SEP" "$BUILD_GRAPH_V2" "$KFOLD" "$KFOLD_V2"; do
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

source_config_for() {
  local exp="$1" bucket="$2"
  case "$exp" in
    baseline)             echo "$SECTION_GNN/runs/$bucket/config.yaml" ;;
    no_names)             echo "$SECTION_GNN/ablations/no_names/$bucket/config.yaml" ;;
    text_only)            echo "$SECTION_GNN/ablations/text_only/$bucket/config.yaml" ;;
    no_cross_case)        echo "$SECTION_GNN/ablations/no_cross_case/$bucket/config.yaml" ;;
    hierarchical_enc)     echo "$SECTION_GNN/ablations/hierarchical_enc/$bucket/config.yaml" ;;
    section_sep_enc)      echo "$SECTION_GNN/ablations/section_sep_enc/$bucket/config.yaml" ;;
    case_node_minimised)  echo "$SECTION_GNN/ablations/case_node_minimised/$bucket/config.yaml" ;;
    baseline_lr_decay)    echo "$SECTION_GNN/runs_v2/baseline_lr_decay/$bucket/config.yaml" ;;
    party_args_no_lr)     echo "$SECTION_GNN/runs_v2/party_args_no_lr/$bucket/config.yaml" ;;
    party_args_lr_decay)  echo "$SECTION_GNN/runs_v2/party_args_lr_decay/$bucket/config.yaml" ;;
    *) echo "Unknown experiment: $exp" >&2; return 1 ;;
  esac
}

config_for() {
  local exp="$1" bucket="$2"
  echo "$CONFIG_ROOT/$exp/$bucket/config.yaml"
}

run_name_for() {
  local exp="$1" bucket="$2"
  local short
  short="$(short_bucket_name "$bucket")"
  echo "inlegalbert_${short}_${exp}_kfold"
}

build_script_for() {
  local exp="$1"
  case "$exp" in
    section_sep_enc)     echo "$BUILD_GRAPH_SECTION_SEP" ;;
    party_args_no_lr|party_args_lr_decay) echo "$BUILD_GRAPH_V2" ;;
    *)                   echo "$BUILD_GRAPH" ;;
  esac
}

kfold_script_for() {
  local exp="$1"
  case "$exp" in
    baseline_lr_decay) echo "$KFOLD_V2" ;;
    party_args_lr_decay) echo "$KFOLD_V2" ;;
    *)                   echo "$KFOLD" ;;
  esac
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
  log "Generating InLegalBERT configs under $CONFIG_ROOT"
  SECTION_GNN="$SECTION_GNN" \
  CONFIG_ROOT="$CONFIG_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  OUT_ROOT="$OUT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  POOLING="$POOLING" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  EXPERIMENTS_STR="${EXPERIMENTS[*]}" \
  BUCKETS_STR="${BUCKETS[*]}" \
  $PYTHON - <<'PY'
import copy
import os
from pathlib import Path

import yaml

root = Path(os.environ["SECTION_GNN"])
config_root = Path(os.environ["CONFIG_ROOT"])
data_root = Path(os.environ["DATA_ROOT"])
out_root = Path(os.environ["OUT_ROOT"])
model_name = os.environ["MODEL_NAME"]
pooling = os.environ["POOLING"]
batch_size = int(os.environ["BATCH_SIZE"])
max_length = int(os.environ["MAX_LENGTH"])
experiments = os.environ["EXPERIMENTS_STR"].split()
buckets = os.environ["BUCKETS_STR"].split()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def short_name(bucket: str) -> str:
    return "cross_bucket" if bucket == "cross_bucket_total_dataset" else bucket.removesuffix("_timed_mistral")


def source_config(exp: str, bucket: str) -> Path:
    if exp == "baseline":
        return root / "runs" / bucket / "config.yaml"
    if exp == "baseline_lr_decay":
        return root / "runs_v2" / "baseline_lr_decay" / bucket / "config.yaml"
    if exp == "party_args_no_lr":
        return root / "runs_v2" / "party_args_no_lr" / bucket / "config.yaml"
    if exp == "party_args_lr_decay":
        return root / "runs_v2" / "party_args_lr_decay" / bucket / "config.yaml"
    return root / "ablations" / exp / bucket / "config.yaml"


for exp in experiments:
    for bucket in buckets:
        src = source_config(exp, bucket)
        if not src.exists():
            raise FileNotFoundError(f"Missing source config for {exp}/{bucket}: {src}")

        cfg = copy.deepcopy(load_yaml(src))
        short = short_name(bucket)
        bucket_data_root = data_root / bucket
        bucket_out_root = out_root / bucket

        cfg.setdefault("project", {})["name"] = f"inlegalbert_{short}_{exp}"
        paths = cfg.setdefault("paths", {})

        # Reuse already-cleaned cases, but keep all encoder/graph/output artifacts
        # separate from the BGE-M3 run.
        paths["embeddings_cache_dir"] = str(bucket_data_root / "embeddings_cache")
        paths["graph_cache_dir"] = str(bucket_data_root / "graph_cache")
        paths["audits_dir"] = str(bucket_data_root / "audits" / exp)
        paths["outputs_dir"] = str(bucket_out_root)

        cfg.setdefault("graph", {})["cache_name"] = (
            f"case_star_{short}_{exp}_inlegalbert.reasoning_focused.pt"
        )

        features = cfg.setdefault("features", {})
        old_encoder = dict(features.get("text_encoder", {}))
        old_encoder.update(
            {
                "backend": "hf_encoder",
                "model_name": model_name,
                "pooling": pooling,
                "batch_size": batch_size,
                "max_length": max_length,
                "device": "cuda",
                "data_parallel": True,
                "show_progress_bar": True,
            }
        )
        for sentence_transformer_only in (
            "multi_process",
            "multi_process_devices",
            "chunk_size",
            "precision",
        ):
            old_encoder.pop(sentence_transformer_only, None)
        features["text_encoder"] = old_encoder

        save_yaml(config_root / exp / bucket / "config.yaml", cfg)

print(f"Wrote configs to {config_root}")
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
  printf "%-5s %-55s folds=%s/%s graph=%s summary=%s\n" \
    "$complete_flag" "$label" "$fold_count" "$K" \
    "$([[ -f "$graph_cache" ]] && echo yes || echo no)" \
    "$([[ -f "$summary" ]] && echo yes || echo no)"
}

run_cell() {
  local exp="$1" bucket="$2"
  local cfg run_name build_script kfold_script graph_cache label
  cfg="$(config_for "$exp" "$bucket")"
  run_name="$(run_name_for "$exp" "$bucket")"
  build_script="$(build_script_for "$exp")"
  kfold_script="$(kfold_script_for "$exp")"
  graph_cache="$(graph_cache_path "$cfg")"
  label="$bucket/$exp"

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

log "InLegalBERT 8x6 main-comparison matrix"
log "env=$MAMBA_ENV model=$MODEL_NAME k=$K train_gpus=$TRAIN_GPUS build_gpus=$GPUS_BUILD"

for exp in "${EXPERIMENTS[@]}"; do
  log "Experiment: $exp"
  for bucket in "${BUCKETS[@]}"; do
    run_cell "$exp" "$bucket"
  done
done

log "Done."
