#!/usr/bin/env bash
# =============================================================================
#  run_ner_rr_custom.sh
#  Run NER + Rhetorical Role on the narcotics_special_acts dataset.
#
#  Usage:
#    bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh                     # all 8 GPUs
#    bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --gpus 0,1,2,3      # specific GPUs
#    bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --quick             # 20 docs, all GPUs
#    bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --quick --gpus 0    # 20 docs, 1 GPU
#    bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --cpu               # force CPU
#
#  Output:
#    Fixed_GPU_OpenNyai/ner_rr_output/annotations/<file_id>.json  — one per doc
#    Fixed_GPU_OpenNyai/ner_rr_output/summary.json                — aggregate counts
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="fixed_gpu_opennyai_final"

INPUT_DIR="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Bucket_Maker_Nyaya/opennyai_txt_export/output/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev/narcotics_special_acts"
OUTPUT_DIR="$SCRIPT_DIR/ner_rr_output"

USE_GPU=true
MAX_DOCS=0
GPUS=""        # empty = all available

# ── Parse flags ───────────────────────────────────────────────────────────────
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --cpu)    USE_GPU=false ;;
        --quick)  MAX_DOCS=20   ;;
        --gpus)   i=$((i+1)); GPUS="${!i}" ;;
        --gpus=*) GPUS="${arg#*=}" ;;
    esac
    i=$((i+1))
done

GPU_FLAG="";      $USE_GPU  && GPU_FLAG="--use_gpu"
MAX_FLAG="";      [ "$MAX_DOCS" -gt 0 ] && MAX_FLAG="--max_docs $MAX_DOCS"
GPUS_FLAG="";     [ -n "$GPUS" ] && GPUS_FLAG="--gpus $GPUS"

echo "============================================================"
echo "  OpenNyAI NER + RR  —  Custom Dataset"
echo "  Environment : $ENV_NAME"
echo "  GPU         : $USE_GPU  ${GPUS:+(ids: $GPUS)}"
echo "  Input dir   : $INPUT_DIR"
echo "  Output dir  : $OUTPUT_DIR"
[ "$MAX_DOCS" -gt 0 ] && echo "  Max docs    : $MAX_DOCS  (--quick mode)"
echo "============================================================"
echo ""

micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/run_ner_rr_custom.py" \
    --input_dir  "$INPUT_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    $GPU_FLAG $MAX_FLAG $GPUS_FLAG

echo ""
echo "All done. Results in: $OUTPUT_DIR"
