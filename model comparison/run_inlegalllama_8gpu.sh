#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-model_comparison_inlegalllama}"
MODEL_NAME="${MODEL_NAME:-L-NLProc/InLegalLlama}"
MODEL_SUBFOLDER="${MODEL_SUBFOLDER:-INLegalLlama/CPT/llama2_cpt_checkpoint_3000_seq_2048}"
ADAPTER_MODE="${ADAPTER_MODE:-peft}"
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/data/motor_accidents_fold_00_test_cases}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/lnlproc_inlegalllama_motor_accidents_fold00}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export HF_HOME="${HF_HOME:-/scratch/ziv_baretto/Thesis_Ziv/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$OUTPUT_DIR/logs"

if [[ "${PREFETCH_MODEL:-1}" == "1" ]]; then
  echo "[prefetch] Download/check cache for $MODEL_NAME under HF_HOME=$HF_HOME"
  micromamba run -n "$ENV_NAME" python - "$MODEL_NAME" "$MODEL_SUBFOLDER" "$ADAPTER_MODE" <<'PY'
import json
import sys
from huggingface_hub import hf_hub_download, snapshot_download

repo_id, subfolder, adapter_mode = sys.argv[1:4]
subfolder = subfolder.strip("/")
ignore_patterns = [
    f"{subfolder}/optimizer.pt",
    f"{subfolder}/scheduler.pt",
    f"{subfolder}/rng_state.pth",
    f"{subfolder}/training_args.bin",
    f"{subfolder}/trainer_state.json",
]
snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{subfolder}/*"] if subfolder else None,
    ignore_patterns=ignore_patterns if subfolder else None,
)
if adapter_mode in {"auto", "peft"}:
    adapter_config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/adapter_config.json" if subfolder else "adapter_config.json",
    )
    with open(adapter_config_path, encoding="utf-8") as handle:
        base_model = json.load(handle)["base_model_name_or_path"]
    print(f"[prefetch] Adapter base model: {base_model}")
    snapshot_download(repo_id=base_model)
PY
fi

echo "[run] model=$MODEL_NAME"
echo "[run] model_subfolder=$MODEL_SUBFOLDER"
echo "[run] adapter_mode=$ADAPTER_MODE"
echo "[run] input=$INPUT_DIR"
echo "[run] output=$OUTPUT_DIR"
echo "[run] batch_size=$BATCH_SIZE"

if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
else
  GPU_LIST=()
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_LIST+=("$gpu")
  done
fi

NUM_SHARDS="${#GPU_LIST[@]}"
echo "[run] launching $NUM_SHARDS shards on GPU IDs: ${GPU_LIST[*]}"

pids=()
for shard_index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard_index]}"
  shard="$(printf "%02d" "$shard_index")"
  log_file="$OUTPUT_DIR/logs/shard_${shard}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/run_inlegalllama.py" \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --model-name "$MODEL_NAME" \
      --model-subfolder "$MODEL_SUBFOLDER" \
      --adapter-mode "$ADAPTER_MODE" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index" \
      --batch-size "$BATCH_SIZE" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --dtype "$DTYPE" \
      --skip-existing
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  pid_index=$((${#pids[@]} - 1))
  echo "[run] shard $shard -> GPU $gpu pid=${pids[$pid_index]} log=$log_file"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '[run] shard %02d complete\n' "$index"
  else
    printf '[run] shard %02d failed\n' "$index" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[run] one or more shards failed; inspect $OUTPUT_DIR/logs" >&2
  exit "$status"
fi

micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/summarize_inlegalllama.py" \
  --predictions-dir "$OUTPUT_DIR"

echo "[run] done: $OUTPUT_DIR/metrics.json"
