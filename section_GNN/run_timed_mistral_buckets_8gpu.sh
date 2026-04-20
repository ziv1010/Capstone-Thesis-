#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_CONFIG="${BASE_CONFIG:-$PROJECT_ROOT/experiments/fixed_open_pipeline/cross_bucket_cases_8k_each_mistral_reasoning_config.yaml}"
CONFIG_ROOT="${CONFIG_ROOT:-$PROJECT_ROOT/outputs/timed_bucket_runs/configs}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/timed_bucket_runs}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/timed_bucket_runs}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"

BUILD_GPUS="${BUILD_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5}"
EVAL_GPUS="${EVAL_GPUS:-6,7}"
LIMIT="${LIMIT:-}"

PREPROCESS_SCRIPT="$PROJECT_ROOT/experiments/fixed_open_pipeline/preprocess_fixed_open.py"
GRAPH_BUILD_SCRIPT="$PROJECT_ROOT/final_graph/build_graph.py"
TRAIN_SCRIPT="$PROJECT_ROOT/dump2/scripts/train_gnn.py"
EVAL_SCRIPT="$PROJECT_ROOT/dump2/scripts/evaluate_saved_model.py"

BUCKET_NAMES=(
  family_matrimonial_timed_mistral
  fin_fraud_timed_mistral
  land_property_timed_mistral
  motor_accidents_timed_mistral
  sexual_offences_timed_mistral
)

BUCKET_PATHS=(
  /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/family_matrimonial_timed_mistral
  /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/fin_fraud_timed_mistral
  /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/land_property_timed_mistral
  /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/motor_accidents_timed_mistral
  /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/sexual_offences_timed_mistral
)

EXTRA_TESTSET_NAME="cross_bucket_total_dataset"
EXTRA_TESTSET_PATH="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_total_dataset"

TRAIN_DATASET_NAMES=(
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
  bash run_timed_mistral_buckets_8gpu.sh
  bash run_timed_mistral_buckets_8gpu.sh --limit 5000
  bash run_timed_mistral_buckets_8gpu.sh --env thesis_work
  bash run_timed_mistral_buckets_8gpu.sh --build-gpus 0,1,2,3,4,5,6,7 --train-gpus 0,1,2,3,4,5 --eval-gpus 6,7

What it does:
  1. Generates one config per dataset
  2. Runs preprocessing for the 5 timed buckets plus cross_bucket_total_dataset
  3. Builds graphs for all 6 datasets using all visible build GPUs
  4. Trains all 6 datasets in parallel, one dataset per train GPU
  5. Evaluates each trained checkpoint on its own graph, and also evaluates bucket checkpoints on the combined graph

Defaults:
  env         : $MAMBA_ENV
  build gpus  : $BUILD_GPUS
  train gpus  : $TRAIN_GPUS
  eval gpus   : $EVAL_GPUS
  base config : $BASE_CONFIG
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--env-name)
      MAMBA_ENV="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --build-gpus)
      BUILD_GPUS="$2"
      shift 2
      ;;
    --train-gpus)
      TRAIN_GPUS="$2"
      shift 2
      ;;
    --eval-gpus)
      EVAL_GPUS="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
  echo "micromamba not found: $MICROMAMBA_BIN" >&2
  exit 1
fi

if ! "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$MAMBA_ENV"; then
  echo "Micromamba environment not found: $MAMBA_ENV" >&2
  exit 1
fi

for path in "$BASE_CONFIG" "$PREPROCESS_SCRIPT" "$GRAPH_BUILD_SCRIPT" "$TRAIN_SCRIPT" "$EVAL_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

for bucket_path in "${BUCKET_PATHS[@]}"; do
  if [[ ! -d "$bucket_path" ]]; then
    echo "Bucket directory not found: $bucket_path" >&2
    exit 1
  fi
done

if [[ ! -d "$EXTRA_TESTSET_PATH" ]]; then
  echo "Extra test directory not found: $EXTRA_TESTSET_PATH" >&2
  exit 1
fi

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
IFS=',' read -r -a EVAL_GPU_ARRAY <<< "$EVAL_GPUS"

if [[ "${#TRAIN_GPU_ARRAY[@]}" -lt "${#TRAIN_DATASET_NAMES[@]}" ]]; then
  echo "Need at least ${#TRAIN_DATASET_NAMES[@]} train GPUs, got ${#TRAIN_GPU_ARRAY[@]}" >&2
  exit 1
fi

if [[ "${#EVAL_GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "Need at least one eval GPU" >&2
  exit 1
fi

mkdir -p "$CONFIG_ROOT" "$DATA_ROOT" "$OUT_ROOT" "$LOG_ROOT"

echo "============================================================"
echo "Timed Mistral Bucket Pipeline"
echo "Project root : $PROJECT_ROOT"
echo "Micromamba   : $MICROMAMBA_BIN"
echo "Env          : $MAMBA_ENV"
echo "Base config  : $BASE_CONFIG"
echo "Build GPUs   : $BUILD_GPUS"
echo "Train GPUs   : $TRAIN_GPUS"
echo "Eval GPUs    : $EVAL_GPUS"
echo "Extra test   : $EXTRA_TESTSET_PATH"
if [[ -n "$LIMIT" ]]; then
  echo "Limit        : $LIMIT"
fi
echo "============================================================"

echo ""
echo ">>> Step 1/4: generating per-bucket configs"
BASE_CONFIG_PY="$BASE_CONFIG" \
CONFIG_ROOT_PY="$CONFIG_ROOT" \
DATA_ROOT_PY="$DATA_ROOT" \
OUT_ROOT_PY="$OUT_ROOT" \
"$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" - <<'PY'
import os
import copy
from pathlib import Path
import yaml

base_cfg_path = Path(os.environ["BASE_CONFIG_PY"])
config_root = Path(os.environ["CONFIG_ROOT_PY"])
data_root = Path(os.environ["DATA_ROOT_PY"])
out_root = Path(os.environ["OUT_ROOT_PY"])

buckets = [
    (
        "family_matrimonial_timed_mistral",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/family_matrimonial_timed_mistral",
    ),
    (
        "fin_fraud_timed_mistral",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/fin_fraud_timed_mistral",
    ),
    (
        "land_property_timed_mistral",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/land_property_timed_mistral",
    ),
    (
        "motor_accidents_timed_mistral",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/motor_accidents_timed_mistral",
    ),
    (
        "sexual_offences_timed_mistral",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/sexual_offences_timed_mistral",
    ),
    (
        "cross_bucket_total_dataset",
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_total_dataset",
    ),
]

base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
config_root.mkdir(parents=True, exist_ok=True)
data_root.mkdir(parents=True, exist_ok=True)
out_root.mkdir(parents=True, exist_ok=True)

for bucket_name, raw_dir in buckets:
    cfg = copy.deepcopy(base_cfg)
    bucket_data_root = data_root / bucket_name
    bucket_output_root = out_root / bucket_name

    cfg["project"]["name"] = f"{bucket_name}_reasoning_graph"
    cfg["paths"]["raw_json_dir"] = raw_dir
    cfg["paths"]["processed_dir"] = str(bucket_data_root / "processed")
    cfg["paths"]["cleaned_case_dir"] = str(bucket_data_root / "processed" / "cleaned_cases")
    cfg["paths"]["normalized_entity_dir"] = str(bucket_data_root / "processed" / "normalized_entities")
    cfg["paths"]["embeddings_cache_dir"] = str(bucket_data_root / "embeddings_cache")
    cfg["paths"]["graph_cache_dir"] = str(bucket_data_root / "graph_cache")
    cfg["paths"]["audits_dir"] = str(bucket_data_root / "audits")
    cfg["paths"]["outputs_dir"] = str(bucket_output_root)
    cfg["data"]["file_glob"] = "*.json"
    cfg["graph"]["cache_name"] = f"case_star_global_graph_{bucket_name}.reasoning_focused.pt"

    config_path = config_root / f"{bucket_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[config] wrote {config_path}")
PY

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

echo ""
echo ">>> Step 2/4: preprocessing datasets"
for idx in "${!TRAIN_DATASET_NAMES[@]}"; do
  dataset_name="${TRAIN_DATASET_NAMES[$idx]}"
  config_path="$CONFIG_ROOT/$dataset_name.yaml"
  echo "[preprocess] $dataset_name"
  "$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" \
    "$PREPROCESS_SCRIPT" \
    --config "$config_path" \
    "${LIMIT_ARGS[@]}"
done
extra_test_config="$CONFIG_ROOT/$EXTRA_TESTSET_NAME.yaml"

echo ""
echo ">>> Step 3/4: building graphs"
for idx in "${!TRAIN_DATASET_NAMES[@]}"; do
  dataset_name="${TRAIN_DATASET_NAMES[$idx]}"
  config_path="$CONFIG_ROOT/$dataset_name.yaml"
  echo "[build] $dataset_name on GPUs $BUILD_GPUS"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$BUILD_GPUS" \
    "$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" \
    "$GRAPH_BUILD_SCRIPT" \
    --config "$config_path" \
    "${LIMIT_ARGS[@]}"
done

echo ""
echo ">>> Step 4/4: training and checkpoint evaluation"

pids=()
names=()

for idx in "${!TRAIN_DATASET_NAMES[@]}"; do
  dataset_name="${TRAIN_DATASET_NAMES[$idx]}"
  train_gpu="${TRAIN_GPU_ARRAY[$idx]}"
  eval_gpu="${EVAL_GPU_ARRAY[$((idx % ${#EVAL_GPU_ARRAY[@]}))]}"
  config_path="$CONFIG_ROOT/$dataset_name.yaml"
  dataset_out_dir="$OUT_ROOT/$dataset_name"
  extra_test_out_dir="$dataset_out_dir/checkpoint_eval_${EXTRA_TESTSET_NAME}"
  log_path="$LOG_ROOT/$dataset_name.log"

  (
    set -euo pipefail

    echo "[train] $dataset_name on GPU $train_gpu"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$train_gpu" \
      "$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" \
      "$TRAIN_SCRIPT" \
      --config "$config_path" \
      --run-name "$dataset_name"

    echo "[eval] $dataset_name on GPU $eval_gpu"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$eval_gpu" \
      "$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" \
      "$EVAL_SCRIPT" \
      --config "$config_path" \
      --checkpoint "$dataset_out_dir/models/$dataset_name/model.pt" \
      --output-dir "$dataset_out_dir/checkpoint_eval" \
      --title "$dataset_name checkpoint eval"

    if [[ "$dataset_name" != "$EXTRA_TESTSET_NAME" ]]; then
      echo "[eval] $dataset_name on $EXTRA_TESTSET_NAME using GPU $eval_gpu"
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$eval_gpu" \
        "$MICROMAMBA_BIN" run -n "$MAMBA_ENV" "$PYTHON_BIN" \
        "$EVAL_SCRIPT" \
        --config "$extra_test_config" \
        --checkpoint "$dataset_out_dir/models/$dataset_name/model.pt" \
        --output-dir "$extra_test_out_dir" \
        --title "$dataset_name on $EXTRA_TESTSET_NAME"
    fi
  ) > >(tee -a "$log_path") 2>&1 &

  pids+=("$!")
  names+=("$dataset_name")
done

status=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[error] pipeline failed for ${names[$idx]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more dataset runs failed." >&2
  exit "$status"
fi

echo ""
echo "All dataset runs completed successfully."
echo "Configs : $CONFIG_ROOT"
echo "Data    : $DATA_ROOT"
echo "Outputs : $OUT_ROOT"
