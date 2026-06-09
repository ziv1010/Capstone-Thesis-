#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_SCRIPT="$PROJECT_ROOT/run_opennyai_summarizer_custom.py"
ENV_NAME="${CONDA_ENV:-fixed_gpu_opennyai_final}"

USE_GPU=true
MAX_DOCS=0
WORKERS=0
PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-1}"
SUMMARY_LENGTH="${SUMMARY_LENGTH:-0.0}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
OVERWRITE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)
            USE_GPU=false
            shift
            ;;
        --max_docs|--max-docs)
            MAX_DOCS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --pipeline_batch_size|--pipeline-batch-size)
            PIPELINE_BATCH_SIZE="$2"
            shift 2
            ;;
        --summary_length|--summary-length)
            SUMMARY_LENGTH="$2"
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
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

declare -A INPUTS=(
    [family_matrimonial]="$PROJECT_ROOT/final_outputs/family_matrimonial_extract/annotations"
    [financial_fraud]="$PROJECT_ROOT/final_outputs/fin_fraud_extract/annotations"
    [land_property]="$PROJECT_ROOT/final_outputs/land_property_extract/annotations"
    [motor_accidents]="$PROJECT_ROOT/final_outputs/motor_accidents_extract/annotations"
    [sexual_offences]="$PROJECT_ROOT/final_outputs/sexual_offences_extract/annotations"
)

declare -A OUTPUTS=(
    [family_matrimonial]="$PROJECT_ROOT/final_outputs/family_matrimonial_summary_opennyai"
    [financial_fraud]="$PROJECT_ROOT/final_outputs/fin_fraud_summary_opennyai"
    [land_property]="$PROJECT_ROOT/final_outputs/land_property_summary_opennyai"
    [motor_accidents]="$PROJECT_ROOT/final_outputs/motor_accidents_summary_opennyai"
    [sexual_offences]="$PROJECT_ROOT/final_outputs/sexual_offences_summary_opennyai"
)

CATEGORIES=(
    family_matrimonial
    financial_fraud
    land_property
    motor_accidents
    sexual_offences
)

GPU_FLAG=()
if $USE_GPU; then
    GPU_FLAG+=(--use_gpu)
fi

WORKERS_FLAG=()
if [ "$WORKERS" -gt 0 ]; then
    WORKERS_FLAG+=(--workers "$WORKERS")
fi

MAX_FLAG=()
if [ "$MAX_DOCS" -gt 0 ]; then
    MAX_FLAG+=(--max_docs "$MAX_DOCS")
fi

GPUS_FLAG=()
if [ -n "$GPUS" ]; then
    GPUS_FLAG+=(--gpus "$GPUS")
fi

OVERWRITE_FLAG=()
if $OVERWRITE; then
    OVERWRITE_FLAG+=(--overwrite)
fi

echo "============================================================"
echo "  OpenNyAI Summaries  —  All Categories"
echo "  Environment         : $ENV_NAME"
echo "  GPU                 : $USE_GPU  ${GPUS:+(ids: $GPUS)}"
echo "  Workers             : $WORKERS"
echo "  Pipeline batch size : $PIPELINE_BATCH_SIZE"
echo "  Summary length      : $SUMMARY_LENGTH"
echo "  Overwrite           : $OVERWRITE"
[ "$MAX_DOCS" -gt 0 ] && echo "  Max docs            : $MAX_DOCS"
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

    micromamba run -n "$ENV_NAME" python "$PYTHON_SCRIPT" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        "${GPU_FLAG[@]}" \
        "${WORKERS_FLAG[@]}" \
        "${MAX_FLAG[@]}" \
        "${GPUS_FLAG[@]}" \
        --pipeline_batch_size "$PIPELINE_BATCH_SIZE" \
        --summary_length "$SUMMARY_LENGTH" \
        "${OVERWRITE_FLAG[@]}"

    echo ""
done

echo "All summary categories complete."
