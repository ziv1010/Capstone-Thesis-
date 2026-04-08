#!/bin/bash
# Parallel pipeline: family_matrimonial on GPU 6
# Runs all 12 chunks with max 6 concurrent processes (6 x ~6 GiB = 36 GiB VRAM, safe on 49 GiB GPU)

set -euo pipefail

INPUT_BASE="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Bucket_Maker_Nyaya/opennyai_txt_export/output/CJPE_ext_SCI_HCs_Tribunals_daily_orders_test/family_matrimonial"
OUTPUT_DIR="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/outputs/family_matrimonial"
SCRIPT="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/run_pipeline.py"
LOG_DIR="${LOG_DIR:-/tmp/fm_chunk_logs}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
PIPELINE_BATCH_SIZE="${PIPELINE_BATCH_SIZE:-4}"
GPU_ID="${GPU_ID:-6}"
CONDA_ENV="${CONDA_ENV:-fixed_gpu_opennyai_final}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

eval "$(micromamba shell hook -s bash)"
micromamba activate "$CONDA_ENV"

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting family_matrimonial parallel run on GPU $GPU_ID"
echo "[$(date)] Max $MAX_PARALLEL concurrent chunk processes"
echo "[$(date)] pipeline_batch_size=$PIPELINE_BATCH_SIZE"

running=0
for chunk_dir in "$INPUT_BASE"/chunk_*; do
    chunk_name=$(basename "$chunk_dir")
    log_file="$LOG_DIR/${chunk_name}.log"

    echo "[$(date)] Launching $chunk_name -> $log_file"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python -u "$SCRIPT" \
        --input_dir "$chunk_dir" \
        --output_dir "$OUTPUT_DIR" \
        --glob_pattern "*.txt" \
        --use_gpu \
        --gpu_devices 0 \
        --pipeline_batch_size "$PIPELINE_BATCH_SIZE" \
        > "$log_file" 2>&1 &

    running=$((running + 1))

    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        echo "[$(date)] Waiting for a slot to free (running=$running)..."
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done

echo "[$(date)] All chunks launched, waiting for remaining to finish..."
wait
echo "[$(date)] family_matrimonial DONE. Chunk logs in $LOG_DIR"
echo "[$(date)] Check output: ls $OUTPUT_DIR/combined/ | wc -l"
