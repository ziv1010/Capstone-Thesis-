#!/usr/bin/env bash
# =============================================================================
#  run_financial_fraud_custom.sh
#  Run the custom NER + RR extractor on INPUT_DATA/financial_fraud_text.
#
#  Usage:
#    bash Fixed_GPU_OpenNyai/run_financial_fraud_custom.sh
#    bash Fixed_GPU_OpenNyai/run_financial_fraud_custom.sh --gpus 0,1,2,3 --workers 4
#    bash Fixed_GPU_OpenNyai/run_financial_fraud_custom.sh --pipeline_batch_size 4
#    bash Fixed_GPU_OpenNyai/run_financial_fraud_custom.sh --quick
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-fixed_gpu_opennyai_final}"

INPUT_DIR="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/INPUT_DATA/financial_fraud_text"
OUTPUT_DIR="$SCRIPT_DIR/fin_fraud_extract"

USE_GPU=true
MAX_DOCS=0
GPUS=""
WORKERS=0
PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-1}"

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --cpu) USE_GPU=false ;;
        --quick) MAX_DOCS=20 ;;
        --gpus) i=$((i+1)); GPUS="${!i}" ;;
        --gpus=*) GPUS="${arg#*=}" ;;
        --workers) i=$((i+1)); WORKERS="${!i}" ;;
        --workers=*) WORKERS="${arg#*=}" ;;
        --pipeline_batch_size|--pipeline-batch-size) i=$((i+1)); PIPELINE_BATCH_SIZE="${!i}" ;;
        --pipeline_batch_size=*|--pipeline-batch-size=*) PIPELINE_BATCH_SIZE="${arg#*=}" ;;
    esac
    i=$((i+1))
done

CMD=(
    micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/run_ner_rr_custom.py"
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --pipeline_batch_size "$PIPELINE_BATCH_SIZE"
)

if $USE_GPU; then
    CMD+=(--use_gpu)
fi
if [ "$MAX_DOCS" -gt 0 ]; then
    CMD+=(--max_docs "$MAX_DOCS")
fi
if [ -n "$GPUS" ]; then
    CMD+=(--gpus "$GPUS")
fi
if [ "$WORKERS" -gt 0 ]; then
    CMD+=(--workers "$WORKERS")
fi

echo "============================================================"
echo "  OpenNyAI NER + RR  —  financial_fraud_text"
echo "  Environment         : $ENV_NAME"
echo "  GPU                 : $USE_GPU  ${GPUS:+(ids: $GPUS)}"
echo "  Workers             : ${WORKERS:-0}"
echo "  Pipeline batch size : $PIPELINE_BATCH_SIZE"
echo "  Input dir           : $INPUT_DIR"
echo "  Output dir          : $OUTPUT_DIR"
[ "$MAX_DOCS" -gt 0 ] && echo "  Max docs            : $MAX_DOCS  (--quick mode)"
echo "============================================================"
echo ""

"${CMD[@]}"

echo ""
echo "All done. Results in: $OUTPUT_DIR"
