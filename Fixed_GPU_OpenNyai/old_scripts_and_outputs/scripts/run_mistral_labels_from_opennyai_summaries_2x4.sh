#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/add_case_outcome_labels_from_enriched.py"
ENV_NAME="${CONDA_ENV:-llm}"

BACKEND="${BACKEND:-local_vllm}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-Small-24B-Instruct-2501}"
LANE0_GPUS="${LANE0_GPUS:-0,1,2,3}"
LANE1_GPUS="${LANE1_GPUS:-4,5,6,7}"
LANE0_TENSOR_PARALLEL_SIZE="${LANE0_TENSOR_PARALLEL_SIZE:-}"
LANE1_TENSOR_PARALLEL_SIZE="${LANE1_TENSOR_PARALLEL_SIZE:-}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"
MAX_FILES=0
RESUME=true
OVERWRITE=false
TRUST_REMOTE_CODE=false
ENFORCE_EAGER=false
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/mistral_2x4_$RUN_STAMP}"

count_csv_items() {
    local csv="$1"
    local count=0
    local item
    local items=()

    if [ -z "$csv" ]; then
        echo 0
        return
    fi

    IFS=',' read -r -a items <<< "$csv"
    for item in "${items[@]}"; do
        if [ -n "$item" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

json_file_count() {
    local dir="$1"

    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi

    find "$dir" -maxdepth 1 -type f -name '*.json' | wc -l
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --model_id|--model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --lane0_gpus|--lane0-gpus)
            LANE0_GPUS="$2"
            shift 2
            ;;
        --lane1_gpus|--lane1-gpus)
            LANE1_GPUS="$2"
            shift 2
            ;;
        --lane0_tensor_parallel_size|--lane0-tensor-parallel-size)
            LANE0_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --lane1_tensor_parallel_size|--lane1-tensor-parallel-size)
            LANE1_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --generation_batch_size|--generation-batch-size)
            GENERATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu_memory_utilization|--gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max_model_len|--max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --tokenizer_mode|--tokenizer-mode)
            TOKENIZER_MODE="$2"
            shift 2
            ;;
        --max_files|--max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-resume)
            RESUME=false
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            RESUME=false
            shift
            ;;
        --trust_remote_code|--trust-remote-code)
            TRUST_REMOTE_CODE=true
            shift
            ;;
        --enforce_eager|--enforce-eager)
            ENFORCE_EAGER=true
            shift
            ;;
        --hf_token|--hf-token)
            HF_TOKEN_VALUE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$LANE0_TENSOR_PARALLEL_SIZE" ]; then
    LANE0_TENSOR_PARALLEL_SIZE="$(count_csv_items "$LANE0_GPUS")"
fi
if [ -z "$LANE1_TENSOR_PARALLEL_SIZE" ]; then
    LANE1_TENSOR_PARALLEL_SIZE="$(count_csv_items "$LANE1_GPUS")"
fi

if [ "$BACKEND" = "local_vllm" ]; then
    if [ -z "$LANE0_GPUS" ] || [ -z "$LANE1_GPUS" ]; then
        echo "Both GPU lanes must be non-empty for local_vllm." >&2
        exit 1
    fi
    if [ "$LANE0_TENSOR_PARALLEL_SIZE" -le 0 ] || [ "$LANE1_TENSOR_PARALLEL_SIZE" -le 0 ]; then
        echo "Both tensor parallel sizes must be positive for local_vllm." >&2
        exit 1
    fi
fi

if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN must be set, or pass --hf-token." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

declare -A INPUTS=(
    [family_matrimonial]="$SCRIPT_DIR/family_matrimonial_summary_opennyai/enriched_jsons"
    [fin_fraud]="$SCRIPT_DIR/fin_fraud_summary_opennyai/enriched_jsons"
    [land_property]="$SCRIPT_DIR/land_property_summary_opennyai/enriched_jsons"
    [motor_accidents]="$SCRIPT_DIR/motor_accidents_summary_opennyai/enriched_jsons"
    [sexual_offences]="$SCRIPT_DIR/sexual_offences_summary_opennyai/enriched_jsons"
)

declare -A OUTPUTS=(
    [family_matrimonial]="$SCRIPT_DIR/family_matrimonial_labelled_mistral"
    [fin_fraud]="$SCRIPT_DIR/fin_fraud_labelled_mistral"
    [land_property]="$SCRIPT_DIR/land_property_labelled_mistral"
    [motor_accidents]="$SCRIPT_DIR/motor_accidents_labelled_mistral"
    [sexual_offences]="$SCRIPT_DIR/sexual_offences_labelled_mistral"
)

CATEGORIES=(
    family_matrimonial
    fin_fraud
    land_property
    motor_accidents
    sexual_offences
)

declare -A REMAINING
lane0_total=0
lane1_total=0
lane0_categories=()
lane1_categories=()
sortable_rows=()

for category in "${CATEGORIES[@]}"; do
    input_count="$(json_file_count "${INPUTS[$category]}")"
    if $OVERWRITE || ! $RESUME; then
        remaining="$input_count"
    else
        output_count="$(json_file_count "${OUTPUTS[$category]}/labelled_jsons")"
        remaining=$((input_count - output_count))
        if [ "$remaining" -lt 0 ]; then
            remaining=0
        fi
    fi

    REMAINING[$category]="$remaining"
    sortable_rows+=("${remaining}"$'\t'"${category}")
done

while IFS=$'\t' read -r remaining category; do
    if [ -z "$category" ] || [ "$remaining" -le 0 ]; then
        continue
    fi

    if [ "$lane0_total" -le "$lane1_total" ]; then
        lane0_categories+=("$category")
        lane0_total=$((lane0_total + remaining))
    else
        lane1_categories+=("$category")
        lane1_total=$((lane1_total + remaining))
    fi
done < <(printf '%s\n' "${sortable_rows[@]}" | sort -t $'\t' -k1,1nr)

run_category() {
    local lane_name="$1"
    local gpu_ids="$2"
    local tensor_parallel_size="$3"
    local category="$4"
    local input_dir="${INPUTS[$category]}"
    local output_dir="${OUTPUTS[$category]}"
    local max_files_flag=()
    local resume_flag=()
    local overwrite_flag=()
    local trust_remote_code_flag=()
    local enforce_eager_flag=()

    if [ "$MAX_FILES" -gt 0 ]; then
        max_files_flag+=(--max_files "$MAX_FILES")
    fi
    if $RESUME; then
        resume_flag+=(--resume)
    fi
    if $OVERWRITE; then
        overwrite_flag+=(--overwrite)
    fi
    if $TRUST_REMOTE_CODE; then
        trust_remote_code_flag+=(--trust_remote_code)
    fi
    if $ENFORCE_EAGER; then
        enforce_eager_flag+=(--enforce_eager)
    fi

    echo "------------------------------------------------------------"
    echo "Lane      : $lane_name"
    echo "Category  : $category"
    echo "Remaining : ${REMAINING[$category]}"
    echo "GPUs      : $gpu_ids"
    echo "TP size   : $tensor_parallel_size"
    echo "Input     : $input_dir"
    echo "Output    : $output_dir"
    echo "------------------------------------------------------------"

    HF_TOKEN="$HF_TOKEN_VALUE" HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN_VALUE" \
        micromamba run -n "$ENV_NAME" python "$PYTHON_SCRIPT" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --backend "$BACKEND" \
        --model_id "$MODEL_ID" \
        --generation_batch_size "$GENERATION_BATCH_SIZE" \
        --tensor_parallel_size "$tensor_parallel_size" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN" \
        --dtype "$DTYPE" \
        --tokenizer_mode "$TOKENIZER_MODE" \
        --cuda_visible_devices "$gpu_ids" \
        "${max_files_flag[@]}" \
        "${resume_flag[@]}" \
        "${overwrite_flag[@]}" \
        "${trust_remote_code_flag[@]}" \
        "${enforce_eager_flag[@]}"
}

run_lane() {
    local lane_name="$1"
    local gpu_ids="$2"
    local tensor_parallel_size="$3"
    shift 3
    local categories=("$@")
    local category

    if [ "${#categories[@]}" -eq 0 ]; then
        echo "$lane_name has no remaining categories to process."
        return 0
    fi

    for category in "${categories[@]}"; do
        run_category "$lane_name" "$gpu_ids" "$tensor_parallel_size" "$category"
        echo ""
    done
}

cleanup_children() {
    local pids=("${lane_pids[@]:-}")
    local pid

    for pid in "${pids[@]}"; do
        if [ -n "$pid" ]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait >/dev/null 2>&1 || true
}

echo "============================================================"
echo "  Mistral Labels From OpenNyAI Summaries  —  Parallel 2x4"
echo "  Environment            : $ENV_NAME"
echo "  Backend                : $BACKEND"
echo "  Model                  : $MODEL_ID"
echo "  Lane 0 GPUs            : $LANE0_GPUS"
echo "  Lane 0 TP size         : $LANE0_TENSOR_PARALLEL_SIZE"
echo "  Lane 1 GPUs            : $LANE1_GPUS"
echo "  Lane 1 TP size         : $LANE1_TENSOR_PARALLEL_SIZE"
echo "  Generation batch size  : $GENERATION_BATCH_SIZE"
echo "  GPU memory util        : $GPU_MEMORY_UTILIZATION"
echo "  Max model len          : $MAX_MODEL_LEN"
echo "  Resume                 : $RESUME"
echo "  Overwrite              : $OVERWRITE"
[ -n "$LOG_DIR" ] && echo "  Log dir                : $LOG_DIR"
[ "$MAX_FILES" -gt 0 ] && echo "  Max files per category : $MAX_FILES"
echo "============================================================"
echo "Estimated remaining files by category:"
for category in "${CATEGORIES[@]}"; do
    echo "  $category : ${REMAINING[$category]}"
done
echo ""
echo "Lane assignments:"
echo "  lane0 -> ${lane0_categories[*]:-<none>}  (estimated total: $lane0_total)"
echo "  lane1 -> ${lane1_categories[*]:-<none>}  (estimated total: $lane1_total)"
echo ""

if [ "${#lane0_categories[@]}" -eq 0 ] && [ "${#lane1_categories[@]}" -eq 0 ]; then
    echo "No remaining files detected."
    exit 0
fi

lane_pids=()
trap 'cleanup_children; exit 130' INT TERM

lane0_log="$LOG_DIR/lane0.log"
lane1_log="$LOG_DIR/lane1.log"
echo "Lane logs:"
echo "  lane0 -> $lane0_log"
echo "  lane1 -> $lane1_log"
echo ""

(
    export PYTHONUNBUFFERED=1
    run_lane "lane0" "$LANE0_GPUS" "$LANE0_TENSOR_PARALLEL_SIZE" "${lane0_categories[@]}"
) >"$lane0_log" 2>&1 &
lane_pids+=("$!")
(
    export PYTHONUNBUFFERED=1
    run_lane "lane1" "$LANE1_GPUS" "$LANE1_TENSOR_PARALLEL_SIZE" "${lane1_categories[@]}"
) >"$lane1_log" 2>&1 &
lane_pids+=("$!")

status=0
for pid in "${lane_pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

trap - INT TERM

if [ "$status" -ne 0 ]; then
    echo "One or more lanes failed." >&2
    exit "$status"
fi

echo "Parallel Mistral labelling complete."
