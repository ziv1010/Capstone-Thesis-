#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_ner_rr_custom.py"
ENV_NAME="fixed_gpu_opennyai_final"

USE_GPU=true
FRESH=false
MAX_DOCS=0
WORKERS=0
PIPELINE_BATCH_SIZE=1
FREEZE_TIMEOUT_S=900
GPUS=""
RETRY_DEFERRED=false

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)
            USE_GPU=false
            shift
            ;;
        --fresh)
            FRESH=true
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
        --freeze_timeout_s|--freeze-timeout-s)
            FREEZE_TIMEOUT_S="$2"
            shift 2
            ;;
        --freeze_timeout_s=*|--freeze-timeout-s=*)
            FREEZE_TIMEOUT_S="${1#*=}"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --retry_deferred|--retry-deferred)
            RETRY_DEFERRED=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

declare -A INPUTS=(
    [family_matrimonial]="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/family_matrimonial_text"
    [financial_fraud]="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/financial_fraud_text"
    [land_property]="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/land_property_text"
    [motor_accidents]="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/motor_accidents_text"
    [sexual_offences]="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/sexual_offences_text"
)

declare -A OUTPUTS=(
    [family_matrimonial]="$SCRIPT_DIR/final_outputs/family_matrimonial_extract"
    [financial_fraud]="$SCRIPT_DIR/final_outputs/fin_fraud_extract"
    [land_property]="$SCRIPT_DIR/final_outputs/land_property_extract"
    [motor_accidents]="$SCRIPT_DIR/final_outputs/motor_accidents_extract"
    [sexual_offences]="$SCRIPT_DIR/final_outputs/sexual_offences_extract"
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

PIPELINE_FLAG=(--pipeline_batch_size "$PIPELINE_BATCH_SIZE")
FREEZE_FLAG=(--freeze_timeout_s "$FREEZE_TIMEOUT_S")
RETRY_FLAG=()
if $RETRY_DEFERRED; then
    RETRY_FLAG+=(--retry_deferred)
fi

echo "============================================================"
echo "  OpenNyAI NER + RR  —  All Input Categories"
echo "  Environment : $ENV_NAME"
echo "  GPU         : $USE_GPU  ${GPUS:+(ids: $GPUS)}"
echo "  Fresh run   : $FRESH"
echo "  Workers     : $WORKERS"
echo "  Batch size  : $PIPELINE_BATCH_SIZE"
echo "  Freeze sec  : $FREEZE_TIMEOUT_S"
[ "$RETRY_DEFERRED" = true ] && echo "  Deferred    : retrying deferred docs"
[ "$MAX_DOCS" -gt 0 ] && echo "  Max docs    : $MAX_DOCS"
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

    if $FRESH; then
        rm -rf "$output_dir"
    fi

    micromamba run -n "$ENV_NAME" python "$PYTHON_SCRIPT" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        "${GPU_FLAG[@]}" \
        "${WORKERS_FLAG[@]}" \
        "${MAX_FLAG[@]}" \
        "${GPUS_FLAG[@]}" \
        "${PIPELINE_FLAG[@]}" \
        "${FREEZE_FLAG[@]}" \
        "${RETRY_FLAG[@]}"

    echo ""
done

echo "All categories complete."
