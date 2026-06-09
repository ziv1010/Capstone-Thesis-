#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:$APP_ROOT:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
EXPLANATION_DIR="$APP_ROOT/outputs/target_fold_00_8gpu"
OUTPUT_DIR="$EXPLANATION_DIR"
RUN_MERGE=1

if [[ -z "${GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

usage() {
  cat <<'EOF'
Usage:
  FINAL_EXPLANATION/run_scripts/run_validation_multi_gpu.sh [options] [-- extra validate_explanations.py args]

Options:
  --gpus "0,1,2,3"       Comma-separated GPU IDs. Defaults to all visible GPUs.
  --explanation-dir PATH  Directory containing merged explanation CSVs.
  --output-dir PATH       Directory to receive merged validation CSVs. Defaults to explanation dir.
  --env NAME              micromamba env name. Default: thesis_work.
  --no-merge              Leave shard outputs unmerged.

Examples:
  # Tiny smoke: 2 cases per shard, 1 random trial.
  FINAL_EXPLANATION/run_scripts/run_validation_multi_gpu.sh --gpus 0,1 --output-dir FINAL_EXPLANATION/outputs/validation_smoke -- \
    --case-limit 2 --k-values 0,1,2 --random-trials 1 --progress-every 1

  # Full validation on 8 GPUs, merged into the same output dir used by the visualizer.
  FINAL_EXPLANATION/run_scripts/run_validation_multi_gpu.sh --gpus 0,1,2,3,4,5,6,7 \
    --explanation-dir FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
    --output-dir FINAL_EXPLANATION/outputs/target_fold_00_8gpu -- \
    --k-values 0,1,2,3,5,10,20 --random-trials 3
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --explanation-dir) EXPLANATION_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --env) MAMBA_ENV="$2"; shift 2 ;;
    --no-merge) RUN_MERGE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
N_SHARDS="${#GPU_ARRAY[@]}"
if [[ "$N_SHARDS" -lt 1 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

PYTHON=(micromamba run -n "$MAMBA_ENV" python)
SHARDS_DIR="$OUTPUT_DIR/validation_shards"
LOG_DIR="$OUTPUT_DIR/validation_logs"
mkdir -p "$SHARDS_DIR" "$LOG_DIR"

echo "[validation multi-gpu] gpus=$GPUS shards=$N_SHARDS explanations=$EXPLANATION_DIR out=$OUTPUT_DIR"

pids=()
for shard_idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard_idx]}"
  shard_name="$(printf 'shard_%02d' "$shard_idx")"
  shard_dir="$SHARDS_DIR/$shard_name"
  log_file="$LOG_DIR/$shard_name.log"
  mkdir -p "$shard_dir"
  (
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      "${PYTHON[@]}" "$APP_ROOT/validate_explanations.py" \
        --model-path "$SECTION_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset/models/ablation_section_sep_enc_cross_bucket_kfold/kfold/fold_00/model.pt" \
        --graph-cache "$SECTION_GNN/data/timed_bucket_runs/cross_bucket_total_dataset/graph_cache/case_star_cross_bucket_section_sep_enc.reasoning_focused.pt" \
        --config "$SECTION_GNN/ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml" \
        --explanation-dir "$EXPLANATION_DIR" \
        --output-dir "$shard_dir" \
        --num-shards "$N_SHARDS" \
        --shard-index "$shard_idx" \
        --overwrite-validation \
        "${EXTRA_ARGS[@]}"
  ) > "$log_file" 2>&1 &
  pids+=("$!")
  echo "[validation multi-gpu] shard=$shard_idx gpu=$gpu pid=${pids[-1]} log=$log_file"
done

status=0
for shard_idx in "${!pids[@]}"; do
  if wait "${pids[$shard_idx]}"; then
    echo "[validation multi-gpu] shard=$shard_idx done"
  else
    echo "[validation multi-gpu] shard=$shard_idx failed; see $LOG_DIR/$(printf 'shard_%02d' "$shard_idx").log" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

if [[ "$RUN_MERGE" -eq 1 ]]; then
  "${PYTHON[@]}" "$APP_ROOT/merge_validation_outputs.py" \
    --shards-dir "$SHARDS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --explanation-dir "$EXPLANATION_DIR"
fi

echo "[validation multi-gpu] done"
