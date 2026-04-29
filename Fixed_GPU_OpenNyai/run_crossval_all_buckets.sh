#!/usr/bin/env bash
# run_crossval_all_buckets.sh
# Runs cross-validated case outcome labeling on all 6 buckets sequentially.
# Safe to leave overnight. Uses --resume so a re-run skips already-done files.

set -euo pipefail

SCRIPT=/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/add_case_outcome_labels_crossval_mistral.py
MODEL=/scratch/ziv_baretto/Thesis_Ziv/hf_cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724
BASE_INPUT=/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/final_outputs
BASE_OUTPUT=/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/cross_validated_outputs
LOG_DIR="$BASE_OUTPUT/logs"

BUCKETS=(
    fin_fraud
    family_matrimonial
    food_safety
    land_property
    motor_accidents
    sexual_offences
)

mkdir -p "$BASE_OUTPUT" "$LOG_DIR"

START_ALL=$(date +%s)
echo "============================================================"
echo "  Cross-val labeling — all buckets"
echo "  Started : $(date)"
echo "  Model   : $MODEL"
echo "  Output  : $BASE_OUTPUT"
echo "============================================================"

for BUCKET in "${BUCKETS[@]}"; do
    INPUT_DIR="$BASE_INPUT/${BUCKET}_summary_opennyai/enriched_jsons"
    OUTPUT_DIR="$BASE_OUTPUT/$BUCKET"
    LOG_FILE="$LOG_DIR/${BUCKET}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Bucket  : $BUCKET"
    echo "  Input   : $INPUT_DIR"
    echo "  Output  : $OUTPUT_DIR"
    echo "  Log     : $LOG_FILE"
    echo "  Started : $(date)"
    echo "------------------------------------------------------------"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "  [SKIP] Input directory not found: $INPUT_DIR"
        continue
    fi

    START_BUCKET=$(date +%s)

    micromamba run -n llm python "$SCRIPT" \
        --input_dir              "$INPUT_DIR"  \
        --output_dir             "$OUTPUT_DIR" \
        --model_id               "$MODEL"      \
        --backend                local_vllm    \
        --tensor_parallel_size   8             \
        --gpu_memory_utilization 0.90          \
        --max_model_len          4096          \
        --generation_batch_size  64            \
        --max_output_tokens      150           \
        --dtype                  auto          \
        --resume                               \
        2>&1 | tee "$LOG_FILE"

    END_BUCKET=$(date +%s)
    ELAPSED=$(( END_BUCKET - START_BUCKET ))
    echo ""
    echo "  [DONE] $BUCKET — elapsed: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
done

END_ALL=$(date +%s)
TOTAL=$(( END_ALL - START_ALL ))
echo ""
echo "============================================================"
echo "  ALL BUCKETS COMPLETE"
echo "  Finished : $(date)"
echo "  Total    : $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"
echo "  Outputs  : $BASE_OUTPUT"
echo "  Logs     : $LOG_DIR"
echo "============================================================"
