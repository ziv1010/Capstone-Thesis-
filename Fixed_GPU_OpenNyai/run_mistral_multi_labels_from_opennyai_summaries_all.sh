#!/usr/bin/env bash
# run_mistral_multi_labels_from_opennyai_summaries_all.sh
#
# Runs add_multi_label_outcome_from_enriched.py over all 5 case categories.
#
# Produces 6 binary labels + 1 final ternary label per case:
#   appellant_relief_granted       (0/1)
#   respondent_prevailed           (0/1)
#   conviction_or_order_quashed    (0/1)
#   bail_or_custody_relief_granted (0/1)
#   matter_disposed_procedurally   (0/1)
#   partial_relief_or_modification (0/1)
#   case_outcome_label             (appellant_won / postponed_or_procedural / appellant_lost)
#   case_outcome_score             (1 / 0 / -1)
#
# The binary labels are cross-verifiable (e.g. appellant_relief_granted=1
# forces respondent_prevailed=0) to allow CSV-level consistency checks.
#
# Usage examples:
#   bash run_mistral_multi_labels_from_opennyai_summaries_all.sh
#   bash run_mistral_multi_labels_from_opennyai_summaries_all.sh --gpus 0,1 --max_files 50
#   bash run_mistral_multi_labels_from_opennyai_summaries_all.sh --overwrite --gpus 0,1,2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/add_multi_label_outcome_from_enriched.py"
ENV_NAME="${CONDA_ENV:-llm}"

BACKEND="${BACKEND:-local_vllm}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-Small-24B-Instruct-2501}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
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

count_csv_items() {
    local csv="$1"
    local count=0
    local item

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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --tensor_parallel_size|--tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
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

if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE="$(count_csv_items "$GPUS")"
fi

if [ "$BACKEND" = "local_vllm" ]; then
    if [ -z "$GPUS" ]; then
        echo "Local vLLM backend requires a non-empty GPU list." >&2
        exit 1
    fi
    if [ "$TENSOR_PARALLEL_SIZE" -le 0 ]; then
        echo "tensor_parallel_size must be positive for local_vllm." >&2
        exit 1
    fi
fi

if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN must be set, or pass --hf-token." >&2
    exit 1
fi

# Input dirs: same enriched_jsons as the single-label script
declare -A INPUTS=(
    [family_matrimonial]="$SCRIPT_DIR/final_outputs/family_matrimonial_summary_opennyai/enriched_jsons"
    [fin_fraud]="$SCRIPT_DIR/final_outputs/fin_fraud_summary_opennyai/enriched_jsons"
    [land_property]="$SCRIPT_DIR/final_outputs/land_property_summary_opennyai/enriched_jsons"
    [motor_accidents]="$SCRIPT_DIR/final_outputs/motor_accidents_summary_opennyai/enriched_jsons"
    [sexual_offences]="$SCRIPT_DIR/final_outputs/sexual_offences_summary_opennyai/enriched_jsons"
)

# Output dirs: separate from the single-label outputs
declare -A OUTPUTS=(
    [family_matrimonial]="$SCRIPT_DIR/final_outputs/family_matrimonial_multi_labelled_mistral"
    [fin_fraud]="$SCRIPT_DIR/final_outputs/fin_fraud_multi_labelled_mistral"
    [land_property]="$SCRIPT_DIR/final_outputs/land_property_multi_labelled_mistral"
    [motor_accidents]="$SCRIPT_DIR/final_outputs/motor_accidents_multi_labelled_mistral"
    [sexual_offences]="$SCRIPT_DIR/final_outputs/sexual_offences_multi_labelled_mistral"
)

CATEGORIES=(
    family_matrimonial
    fin_fraud
    land_property
    motor_accidents
    sexual_offences
)

MAX_FILES_FLAG=()
if [ "$MAX_FILES" -gt 0 ]; then
    MAX_FILES_FLAG+=(--max_files "$MAX_FILES")
fi

RESUME_FLAG=()
if $RESUME; then
    RESUME_FLAG+=(--resume)
fi

OVERWRITE_FLAG=()
if $OVERWRITE; then
    OVERWRITE_FLAG+=(--overwrite)
fi

TRUST_REMOTE_CODE_FLAG=()
if $TRUST_REMOTE_CODE; then
    TRUST_REMOTE_CODE_FLAG+=(--trust_remote_code)
fi

ENFORCE_EAGER_FLAG=()
if $ENFORCE_EAGER; then
    ENFORCE_EAGER_FLAG+=(--enforce_eager)
fi

echo "============================================================"
echo "  Mistral Multi-Label Outcomes — All Categories"
echo "  Labels produced per case:"
echo "    Binary (0/1): appeal_allowed, appeal_dismissed, bail_granted,"
echo "                  conviction_set_aside, remanded, partial_relief"
echo "    Final:        case_outcome_label / case_outcome_score (1/0/-1)"
echo "------------------------------------------------------------"
echo "  Environment            : $ENV_NAME"
echo "  Backend                : $BACKEND"
echo "  Model                  : $MODEL_ID"
echo "  GPUs                   : ${GPUS:-<none>}"
echo "  Tensor parallel size   : $TENSOR_PARALLEL_SIZE"
echo "  Generation batch size  : $GENERATION_BATCH_SIZE"
echo "  GPU memory util        : $GPU_MEMORY_UTILIZATION"
echo "  Max model len          : $MAX_MODEL_LEN"
echo "  Resume                 : $RESUME"
echo "  Overwrite              : $OVERWRITE"
[ "$MAX_FILES" -gt 0 ] && echo "  Max files per category : $MAX_FILES"
echo "============================================================"
echo ""

for category in "${CATEGORIES[@]}"; do
    input_dir="${INPUTS[$category]}"
    output_dir="${OUTPUTS[$category]}"

    echo "------------------------------------------------------------"
    echo "Category  : $category"
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
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN" \
        --dtype "$DTYPE" \
        --tokenizer_mode "$TOKENIZER_MODE" \
        --cuda_visible_devices "$GPUS" \
        "${MAX_FILES_FLAG[@]}" \
        "${RESUME_FLAG[@]}" \
        "${OVERWRITE_FLAG[@]}" \
        "${TRUST_REMOTE_CODE_FLAG[@]}" \
        "${ENFORCE_EAGER_FLAG[@]}"

    echo ""
done

echo "All multi-label Mistral labelling categories complete."
