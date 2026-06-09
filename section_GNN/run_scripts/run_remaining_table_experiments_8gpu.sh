#!/usr/bin/env bash
# Run the remaining thesis-table experiments in the required order.
#
# Order:
#   1. BGE-M3 remaining cells
#   2. InLegalBERT first cells: party+preamble no-LR, party+preamble LR, section-sep LR
#   3. InLegalBERT remaining entity-resolved and central-authority cells
#
# The runner builds graphs sequentially, using all build GPUs for embedding work,
# then runs one experiment cell at a time with K concurrent fold jobs.
#
# Usage:
#   bash run_scripts/run_remaining_table_experiments_8gpu.sh --status-only
#   nohup bash run_scripts/run_remaining_table_experiments_8gpu.sh \
#     > run_logs/run_remaining_table_experiments_8gpu.log 2>&1 &
#   tail -f run_logs/run_remaining_table_experiments_8gpu.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SECTION_GNN"
export PYTHONPATH="$SECTION_GNN:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
K="${K:-5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
GPUS_BUILD="${GPUS_BUILD:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4}"
MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-}"
MODEL_NAME="${MODEL_NAME:-law-ai/InLegalBERT}"
POOLING="${POOLING:-mean}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LIMIT="${LIMIT:-}"
PHASE="all"
STATUS_ONLY=false
SYNC_ONLY=false
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_FILTER="${FORCE_FILTER:-0}"
FORCE_CENTRALITY="${FORCE_CENTRALITY:-0}"
INLEGAL_FIRST_ONLY=false

MIN_GLOBAL_CASE_FREQUENCY="${MIN_GLOBAL_CASE_FREQUENCY:-3000}"
REMOVAL_SET="${REMOVAL_SET:-}"

BUCKETS=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
  cross_bucket_total_dataset
)

ENTITY_DIR="$SECTION_GNN/ablations/entity_resolved_data"
CENTRAL_DIR="$SECTION_GNN/ablations/remove_central_authorities"
INLEGAL_CONFIG_ROOT="$SECTION_GNN/runs_inlegalbert_remaining"
INLEGAL_DATA_ROOT="$SECTION_GNN/data/inlegalbert_remaining"
INLEGAL_OUTPUT_ROOT="$SECTION_GNN/outputs/inlegalbert_remaining"

ENTITY_PREPARE="$ENTITY_DIR/prepare_configs.py"
ENTITY_PREPROCESS="$ENTITY_DIR/preprocess_fixed_open_resolved.py"
CENTRAL_PREPARE="$CENTRAL_DIR/prepare_configs.py"
CENTRAL_ANALYZE="$CENTRAL_DIR/analyze_central_authorities.py"
CENTRAL_FILTER="$CENTRAL_DIR/filter_cleaned_cases.py"
PARTY_SYNC="$SECTION_GNN/run_scripts/run_party_args_preamble_and_section_sep_lr_decay.sh"
BUILD_PARTY="$SECTION_GNN/runs_v2/party_args_lr_decay/graph/build_graph_v2.py"
BUILD_SECTION="$SECTION_GNN/final_graph/build_graph_section_sep.py"
KFOLD_V2="$SECTION_GNN/runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py"

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --phase all|bge|inlegalbert
  --inlegal-first-only
  --status-only
  --sync-only
  --skip-build
  --skip-preprocess
  --force-preprocess
  --force-filter
  --buckets "bucket_a bucket_b"
  --env thesis_work
  --k 5
  --val-fraction 0.1
  --gpus-build 0,1,2,3,4,5,6,7
  --train-gpus 0,1,2,3,4
  --max-parallel-folds 5
EOF
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --inlegal-first-only) INLEGAL_FIRST_ONLY=true; shift ;;
    --status-only) STATUS_ONLY=true; shift ;;
    --sync-only) SYNC_ONLY=true; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --force-preprocess) FORCE_PREPROCESS=1; shift ;;
    --force-filter) FORCE_FILTER=1; shift ;;
    --force-centrality) FORCE_CENTRALITY=1; shift ;;
    --env) MAMBA_ENV="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --gpus-build) GPUS_BUILD="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --max-parallel-folds) MAX_PARALLEL_FOLDS="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --pooling) POOLING="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --buckets) IFS=' ' read -r -a BUCKETS <<< "$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --min-global-case-frequency) MIN_GLOBAL_CASE_FREQUENCY="$2"; shift 2 ;;
    --removal-set) REMOVAL_SET="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$PHASE" in
  all|bge|inlegalbert) ;;
  *) echo "Unknown --phase value: $PHASE" >&2; exit 1 ;;
esac

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "Need at least one train GPU, got: $TRAIN_GPUS" >&2
  exit 1
fi
if [[ -z "$MAX_PARALLEL_FOLDS" ]]; then
  MAX_PARALLEL_FOLDS="$K"
fi
if [[ "$MAX_PARALLEL_FOLDS" -gt "${#TRAIN_GPU_ARRAY[@]}" ]]; then
  echo "MAX_PARALLEL_FOLDS=$MAX_PARALLEL_FOLDS exceeds train GPU count ${#TRAIN_GPU_ARRAY[@]}" >&2
  exit 1
fi

PYTHON=(micromamba run -n "$MAMBA_ENV" python)

for required in \
  "$ENTITY_PREPARE" "$ENTITY_PREPROCESS" "$CENTRAL_PREPARE" "$CENTRAL_ANALYZE" "$CENTRAL_FILTER" \
  "$PARTY_SYNC" "$BUILD_PARTY" "$BUILD_SECTION" "$KFOLD_V2"; do
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

run_dir_path() {
  local cfg="$1" run_name="$2"
  echo "$(yaml_query "$cfg" "paths.outputs_dir")/models/$run_name/kfold"
}

summary_path() {
  local cfg="$1" run_name="$2"
  echo "$(run_dir_path "$cfg" "$run_name")/kfold_summary.json"
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
  local cfg="$1" cleaned_dir
  cleaned_dir="$(yaml_query "$cfg" "paths.cleaned_case_dir")"
  if [[ -d "$cleaned_dir" ]]; then
    find "$cleaned_dir" -maxdepth 1 -type f -name '*.json' | wc -l
  else
    echo 0
  fi
}

sync_bge_configs() {
  log "[sync] Party+preamble and section-LR BGE configs"
  MAMBA_ENV="$MAMBA_ENV" bash "$PARTY_SYNC" --sync-only

  log "[sync] Entity-resolved BGE configs"
  "${PYTHON[@]}" "$ENTITY_PREPARE" \
    --only both \
    --config-root "$ENTITY_DIR/configs" \
    --lr-mode decay
  "${PYTHON[@]}" "$ENTITY_PREPARE" \
    --only both \
    --config-root "$ENTITY_DIR/configs_no_lr" \
    --lr-mode none

  log "[sync] Central-authority BGE configs"
  "${PYTHON[@]}" "$CENTRAL_PREPARE" \
    --only both \
    --config-root "$CENTRAL_DIR/configs" \
    --lr-mode decay
  "${PYTHON[@]}" "$CENTRAL_PREPARE" \
    --only both \
    --config-root "$CENTRAL_DIR/configs_no_lr" \
    --lr-mode none
}

sync_inlegal_configs() {
  log "[sync] InLegalBERT remaining-cell configs"
  SECTION_GNN="$SECTION_GNN" \
  CONFIG_ROOT="$INLEGAL_CONFIG_ROOT" \
  DATA_ROOT="$INLEGAL_DATA_ROOT" \
  OUT_ROOT="$INLEGAL_OUTPUT_ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  POOLING="$POOLING" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  BUCKETS_STR="${BUCKETS[*]}" \
  "${PYTHON[@]}" - <<'PY'
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
buckets = os.environ["BUCKETS_STR"].split()

experiments = {
    "party_args_preamble_no_lr": lambda b: root / "runs_v2" / "party_args_no_lr" / b / "config.yaml",
    "party_args_preamble_lr_decay": lambda b: root / "runs_v2" / "party_args_preamble_lr_decay" / b / "config.yaml",
    "section_sep_lr_decay": lambda b: root / "ablations" / "section_sep_enc_lr_decay" / b / "config.yaml",
    "entity_section_lr_decay": lambda b: root / "ablations" / "entity_resolved_data" / "configs" / "section" / b / "config.yaml",
    "entity_section_no_lr": lambda b: root / "ablations" / "entity_resolved_data" / "configs_no_lr" / "section" / b / "config.yaml",
    "central_section_no_lr": lambda b: root / "ablations" / "remove_central_authorities" / "configs_no_lr" / "section" / b / "config.yaml",
    "central_section_lr_decay": lambda b: root / "ablations" / "remove_central_authorities" / "configs" / "section" / b / "config.yaml",
}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def short_name(bucket: str) -> str:
    if bucket == "cross_bucket_total_dataset":
        return "cross_bucket"
    return bucket.removesuffix("_timed_mistral")


for exp, source_fn in experiments.items():
    for bucket in buckets:
        src = source_fn(bucket)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source config for {exp}/{bucket}: {src}")
        cfg = copy.deepcopy(load_yaml(src))
        short = short_name(bucket)

        cfg.setdefault("project", {})["name"] = f"inlegalbert_{short}_{exp}"
        paths = cfg.setdefault("paths", {})
        bucket_data_root = data_root / exp / bucket
        bucket_out_root = out_root / exp / bucket
        paths["embeddings_cache_dir"] = str(bucket_data_root / "embeddings_cache")
        paths["graph_cache_dir"] = str(bucket_data_root / "graph_cache")
        paths["audits_dir"] = str(bucket_data_root / "audits")
        paths["outputs_dir"] = str(bucket_out_root)

        cfg.setdefault("graph", {})["cache_name"] = f"case_star_inlegalbert_{short}_{exp}.reasoning_focused.pt"

        features = cfg.setdefault("features", {})
        encoder = dict(features.get("text_encoder", {}))
        encoder.update(
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
        for key in ("multi_process", "multi_process_devices", "chunk_size", "precision"):
            encoder.pop(key, None)
        features["text_encoder"] = encoder

        save_yaml(config_root / exp / bucket / "config.yaml", cfg)

print(f"[config] wrote InLegalBERT configs under {config_root}")
PY
}

sync_all_configs() {
  sync_bge_configs
  sync_inlegal_configs
}

entity_cfg() {
  local root="$1" bucket="$2" variant="${3:-section}"
  echo "$ENTITY_DIR/$root/$variant/$bucket/config.yaml"
}

central_cfg() {
  local root="$1" bucket="$2" variant="${3:-section}"
  echo "$CENTRAL_DIR/$root/$variant/$bucket/config.yaml"
}

ensure_entity_resolved_cleaned() {
  local bucket cfg count cleaned_dir entity_dir audits_dir limit_args=()
  [[ -n "$LIMIT" ]] && limit_args=(--limit "$LIMIT")

  for bucket in "${BUCKETS[@]}"; do
    cfg="$(entity_cfg configs "$bucket" section)"
    count="$(cleaned_count_for_cfg "$cfg")"
    if [[ "$FORCE_PREPROCESS" == "1" ]]; then
      cleaned_dir="$(yaml_query "$cfg" "paths.cleaned_case_dir")"
      entity_dir="$(yaml_query "$cfg" "paths.normalized_entity_dir")"
      audits_dir="$(yaml_query "$cfg" "paths.audits_dir")"
      rm -rf "$cleaned_dir" "$entity_dir" "$audits_dir"
      count=0
    fi
    if [[ "$count" -gt 0 ]]; then
      log "[preprocess:entity] $bucket cleaned cases exist ($count); skipping"
      continue
    fi
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      echo "Missing entity-resolved cleaned cases for $bucket and SKIP_PREPROCESS=1" >&2
      exit 1
    fi
    log "[preprocess:entity] $bucket"
    "${PYTHON[@]}" "$ENTITY_PREPROCESS" --config "$cfg" "${limit_args[@]}"
  done
}

ensure_removal_set() {
  local centrality_dir="$SECTION_GNN/outputs/ablations/remove_central_authorities/centrality_analysis"
  if [[ -z "$REMOVAL_SET" ]]; then
    REMOVAL_SET="$centrality_dir/central_authority_removal_set.json"
  fi
  if [[ "$FORCE_CENTRALITY" != "1" && -f "$REMOVAL_SET" ]]; then
    log "[centrality] Reusing removal set: $REMOVAL_SET"
    return
  fi
  log "[centrality] Computing removal set"
  "${PYTHON[@]}" "$CENTRAL_ANALYZE" \
    --config-root "$ENTITY_DIR/configs/party" \
    --output-dir "$centrality_dir" \
    --min-global-case-frequency "$MIN_GLOBAL_CASE_FREQUENCY" \
    --buckets "${BUCKETS[@]}"
  REMOVAL_SET="$centrality_dir/central_authority_removal_set.json"
}

ensure_central_filtered_cleaned() {
  local bucket src_cfg dst_cfg input_cleaned output_cleaned output_entities output_audits summary count filter_args=()

  ensure_entity_resolved_cleaned
  ensure_removal_set

  for bucket in "${BUCKETS[@]}"; do
    src_cfg="$(entity_cfg configs "$bucket" party)"
    dst_cfg="$(central_cfg configs "$bucket" section)"
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
      log "[filter:central] $bucket filtered cases exist ($count); skipping"
      continue
    fi
    log "[filter:central] $bucket"
    "${PYTHON[@]}" "$CENTRAL_FILTER" \
      --input-cleaned-dir "$input_cleaned" \
      --output-cleaned-dir "$output_cleaned" \
      --output-entity-dir "$output_entities" \
      --output-audits-dir "$output_audits" \
      --summary-path "$summary" \
      --removal-set "$REMOVAL_SET" \
      "${filter_args[@]}"
  done
}

bge_cfg_for() {
  local exp="$1" bucket="$2"
  case "$exp" in
    entity_section_lr_decay) echo "$(entity_cfg configs "$bucket" section)" ;;
    entity_section_no_lr) echo "$(entity_cfg configs_no_lr "$bucket" section)" ;;
    central_section_no_lr) echo "$(central_cfg configs_no_lr "$bucket" section)" ;;
    central_section_lr_decay) echo "$(central_cfg configs "$bucket" section)" ;;
    *) echo "Unknown BGE experiment: $exp" >&2; return 1 ;;
  esac
}

bge_run_name_for() {
  local exp="$1" bucket="$2" short
  short="$(short_bucket_name "$bucket")"
  case "$exp" in
    entity_section_lr_decay) echo "ablation_entity_resolved_section_sep_lr_decay_${short}_kfold" ;;
    entity_section_no_lr) echo "ablation_entity_resolved_section_sep_no_lr_${short}_kfold" ;;
    central_section_no_lr) echo "ablation_central_authorities_removed_section_sep_no_lr_${short}_kfold" ;;
    central_section_lr_decay) echo "ablation_central_authorities_removed_section_sep_lr_decay_${short}_kfold" ;;
    *) echo "Unknown BGE experiment: $exp" >&2; return 1 ;;
  esac
}

inlegal_cfg_for() {
  local exp="$1" bucket="$2"
  echo "$INLEGAL_CONFIG_ROOT/$exp/$bucket/config.yaml"
}

inlegal_run_name_for() {
  local exp="$1" bucket="$2" short
  short="$(short_bucket_name "$bucket")"
  case "$exp" in
    party_args_preamble_no_lr) echo "inlegalbert_${short}_party_args_preamble_no_lr_kfold" ;;
    party_args_preamble_lr_decay) echo "inlegalbert_${short}_party_args_preamble_lr_decay_kfold" ;;
    section_sep_lr_decay) echo "inlegalbert_ablation_section_sep_enc_lr_decay_${short}_kfold" ;;
    entity_section_lr_decay) echo "inlegalbert_ablation_entity_resolved_section_sep_lr_decay_${short}_kfold" ;;
    entity_section_no_lr) echo "inlegalbert_ablation_entity_resolved_section_sep_no_lr_${short}_kfold" ;;
    central_section_no_lr) echo "inlegalbert_ablation_central_authorities_removed_section_sep_no_lr_${short}_kfold" ;;
    central_section_lr_decay) echo "inlegalbert_ablation_central_authorities_removed_section_sep_lr_decay_${short}_kfold" ;;
    *) echo "Unknown InLegalBERT experiment: $exp" >&2; return 1 ;;
  esac
}

build_script_for() {
  local exp="$1"
  case "$exp" in
    party_args_preamble_no_lr|party_args_preamble_lr_decay) echo "$BUILD_PARTY" ;;
    *) echo "$BUILD_SECTION" ;;
  esac
}

make_cell() {
  local cfg="$1" run_name="$2" build_script="$3" label="$4"
  echo "$cfg|$run_name|$build_script|$label"
}

make_bge_cells() {
  local exp bucket cfg run_name build label
  BGE_CELLS=()
  for exp in entity_section_lr_decay entity_section_no_lr central_section_no_lr central_section_lr_decay; do
    for bucket in "${BUCKETS[@]}"; do
      cfg="$(bge_cfg_for "$exp" "$bucket")"
      run_name="$(bge_run_name_for "$exp" "$bucket")"
      build="$BUILD_SECTION"
      label="BGE-M3/$bucket/$exp"
      BGE_CELLS+=("$(make_cell "$cfg" "$run_name" "$build" "$label")")
    done
  done
}

make_inlegal_first_cells() {
  local exp bucket cfg run_name build label
  INLEGAL_FIRST_CELLS=()
  for exp in party_args_preamble_no_lr party_args_preamble_lr_decay section_sep_lr_decay; do
    for bucket in "${BUCKETS[@]}"; do
      cfg="$(inlegal_cfg_for "$exp" "$bucket")"
      run_name="$(inlegal_run_name_for "$exp" "$bucket")"
      build="$(build_script_for "$exp")"
      label="InLegalBERT/$bucket/$exp"
      INLEGAL_FIRST_CELLS+=("$(make_cell "$cfg" "$run_name" "$build" "$label")")
    done
  done
}

make_inlegal_later_cells() {
  local exp bucket cfg run_name build label
  INLEGAL_LATER_CELLS=()
  for exp in entity_section_lr_decay entity_section_no_lr central_section_no_lr central_section_lr_decay; do
    for bucket in "${BUCKETS[@]}"; do
      cfg="$(inlegal_cfg_for "$exp" "$bucket")"
      run_name="$(inlegal_run_name_for "$exp" "$bucket")"
      build="$(build_script_for "$exp")"
      label="InLegalBERT/$bucket/$exp"
      INLEGAL_LATER_CELLS+=("$(make_cell "$cfg" "$run_name" "$build" "$label")")
    done
  done
}

status_cell() {
  local cell="$1" cfg run_name build_script label graph_cache summary run_dir fold_count flag
  IFS='|' read -r cfg run_name build_script label <<< "$cell"
  graph_cache="$(graph_cache_path "$cfg")"
  summary="$(summary_path "$cfg" "$run_name")"
  run_dir="$(run_dir_path "$cfg" "$run_name")"
  fold_count=0
  if [[ -d "$run_dir" ]]; then
    fold_count="$(find "$run_dir" -path '*/fold_summary.json' -type f | wc -l)"
  fi
  if is_summary_complete "$summary"; then
    flag="DONE"
  else
    flag="TODO"
  fi
  printf "%-5s %-78s folds=%s/%s graph=%s summary=%s\n" \
    "$flag" "$label" "$fold_count" "$K" \
    "$([[ -f "$graph_cache" ]] && echo yes || echo no)" \
    "$([[ -f "$summary" ]] && echo yes || echo no)"
}

build_cell_if_needed() {
  local cell="$1" cfg run_name build_script label graph_cache outputs_dir log_dir log_file summary
  IFS='|' read -r cfg run_name build_script label <<< "$cell"
  [[ -f "$cfg" ]] || { echo "Missing config for $label: $cfg" >&2; exit 1; }
  summary="$(summary_path "$cfg" "$run_name")"
  if is_summary_complete "$summary"; then
    log "[skip] $label complete"
    return
  fi
  graph_cache="$(graph_cache_path "$cfg")"
  if [[ -f "$graph_cache" ]]; then
    log "[build] $label graph exists"
    return
  fi
  if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "Missing graph cache for $label and SKIP_BUILD=1: $graph_cache" >&2
    exit 1
  fi
  outputs_dir="$(yaml_query "$cfg" "paths.outputs_dir")"
  log_dir="$outputs_dir/logs"
  mkdir -p "$log_dir"
  log_file="$log_dir/build_${run_name}.log"
  log "[build] $label -> $graph_cache"
  if ! (
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS_BUILD" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PYTHON[@]}" "$build_script" --config "$cfg"
  ) > "$log_file" 2>&1; then
    echo "Build failed for $label. Last log lines:" >&2
    tail -80 "$log_file" >&2 || true
    exit 1
  fi
}

collect_tasks_for_group() {
  TASKS=()
  AGG_CELLS=()
  local cell cfg run_name build_script label graph_cache run_dir outputs_dir log_dir summary fold_idx log_file
  for cell in "$@"; do
    IFS='|' read -r cfg run_name build_script label <<< "$cell"
    summary="$(summary_path "$cfg" "$run_name")"
    if is_summary_complete "$summary"; then
      continue
    fi
    graph_cache="$(graph_cache_path "$cfg")"
    if [[ ! -f "$graph_cache" ]]; then
      echo "Missing graph cache after build for $label: $graph_cache" >&2
      exit 1
    fi
    run_dir="$(run_dir_path "$cfg" "$run_name")"
    outputs_dir="$(yaml_query "$cfg" "paths.outputs_dir")"
    log_dir="$outputs_dir/logs"
    mkdir -p "$run_dir" "$log_dir"
    AGG_CELLS+=("$cell")
    for fold_idx in $(seq 0 $((K - 1))); do
      if [[ ! -f "$run_dir/fold_$(printf '%02d' "$fold_idx")/fold_summary.json" ]]; then
        log_file="$log_dir/${run_name}_fold_${fold_idx}.log"
        TASKS+=("$cfg|$run_name|$graph_cache|$fold_idx|$log_file|$label")
      fi
    done
  done
}

run_fold_tasks() {
  if [[ "${#TASKS[@]}" -eq 0 ]]; then
    log "[kfold] No missing folds in this group"
    return
  fi

  log "[kfold] Scheduling ${#TASKS[@]} folds with max_parallel=$MAX_PARALLEL_FOLDS train_gpus=$TRAIN_GPUS"
  local next=0 running=0 status=0 done_pid gpu task cfg run_name graph_cache fold_idx log_file label pid
  local -a available_gpus=("${TRAIN_GPU_ARRAY[@]:0:$MAX_PARALLEL_FOLDS}")
  declare -A pid_gpu=()
  declare -A pid_label=()

  while [[ "$next" -lt "${#TASKS[@]}" || "$running" -gt 0 ]]; do
    while [[ "$next" -lt "${#TASKS[@]}" && "${#available_gpus[@]}" -gt 0 && "$status" -eq 0 ]]; do
      gpu="${available_gpus[0]}"
      available_gpus=("${available_gpus[@]:1}")
      task="${TASKS[$next]}"
      next=$((next + 1))
      IFS='|' read -r cfg run_name graph_cache fold_idx log_file label <<< "$task"
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
      pid="$!"
      pid_gpu["$pid"]="$gpu"
      pid_label["$pid"]="$label fold=$fold_idx log=$log_file"
      running=$((running + 1))
      log "[kfold] start gpu=$gpu pid=$pid ${pid_label[$pid]}"
    done

    if [[ "$running" -gt 0 ]]; then
      done_pid=""
      if wait -n -p done_pid; then
        log "[kfold] done pid=$done_pid ${pid_label[$done_pid]}"
      else
        log "[kfold] FAILED pid=$done_pid ${pid_label[$done_pid]}" >&2
        status=1
      fi
      available_gpus+=("${pid_gpu[$done_pid]}")
      unset "pid_gpu[$done_pid]" "pid_label[$done_pid]"
      running=$((running - 1))
    fi
  done

  if [[ "$status" -ne 0 ]]; then
    echo "One or more folds failed. Check the fold log paths above." >&2
    exit "$status"
  fi
}

aggregate_group() {
  local cell cfg run_name build_script label graph_cache summary
  for cell in "${AGG_CELLS[@]}"; do
    IFS='|' read -r cfg run_name build_script label <<< "$cell"
    summary="$(summary_path "$cfg" "$run_name")"
    if is_summary_complete "$summary"; then
      log "[aggregate] $label already complete"
      continue
    fi
    graph_cache="$(graph_cache_path "$cfg")"
    log "[aggregate] $label"
    "${PYTHON[@]}" "$KFOLD_V2" \
      --config "$cfg" \
      --run-name "$run_name" \
      --k "$K" \
      --aggregate-only \
      --graph-cache "$graph_cache"
    if ! is_summary_complete "$summary"; then
      echo "Aggregate did not complete all $K folds for $label: $summary" >&2
      exit 1
    fi
  done
}

run_group() {
  local group_name="$1"
  shift
  local cells=("$@")

  log "================================================================"
  log "$group_name"
  log "================================================================"

  if [[ "$STATUS_ONLY" == true ]]; then
    local cell
    for cell in "${cells[@]}"; do
      status_cell "$cell"
    done
    return
  fi

  local cell
  for cell in "${cells[@]}"; do
    build_cell_if_needed "$cell"
    collect_tasks_for_group "$cell"
    run_fold_tasks
    aggregate_group
  done
}

START_TS="$(date +%s)"
log "Remaining table experiment runner"
log "env=$MAMBA_ENV k=$K val_fraction=$VAL_FRACTION build_gpus=$GPUS_BUILD train_gpus=$TRAIN_GPUS max_parallel=$MAX_PARALLEL_FOLDS phase=$PHASE"

sync_all_configs

make_bge_cells
make_inlegal_first_cells
make_inlegal_later_cells

if [[ "$SYNC_ONLY" == true ]]; then
  log "Sync-only complete."
  exit 0
fi

if [[ "$PHASE" == "all" || "$PHASE" == "bge" ]]; then
  if [[ "$STATUS_ONLY" != true ]]; then
    ensure_entity_resolved_cleaned
    ensure_central_filtered_cleaned
  fi
  run_group "Phase 1/3: BGE-M3 remaining cells" "${BGE_CELLS[@]}"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "inlegalbert" ]]; then
  if [[ "$STATUS_ONLY" != true ]]; then
    ensure_entity_resolved_cleaned
    ensure_central_filtered_cleaned
  fi
  run_group "Phase 2/3: InLegalBERT first cells" "${INLEGAL_FIRST_CELLS[@]}"
  if [[ "$INLEGAL_FIRST_ONLY" != true ]]; then
    run_group "Phase 3/3: InLegalBERT remaining cells" "${INLEGAL_LATER_CELLS[@]}"
  fi
fi

END_TS="$(date +%s)"
log "Done in $(((END_TS - START_TS) / 60)) minutes."
