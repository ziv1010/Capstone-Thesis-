#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-model_comparison_inlegalllama}"
CHECKPOINT="${CHECKPOINT:-nyaya_facts_single}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$SCRIPT_DIR/models/factlegalllama}"
PROMPT_PROFILE="${PROMPT_PROFILE:-factlegal_facts}"
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/data/motor_accidents_fold_00_test_cases}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/factlegalllama_${CHECKPOINT}_facts_original_prompt_motor_accidents_fold00}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DTYPE="${DTYPE:-float16}"

export HF_HOME="${HF_HOME:-/scratch/ziv_baretto/Thesis_Ziv/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$OUTPUT_DIR/logs"

echo "[prepare] checkpoint=$CHECKPOINT"
ADAPTER_DIR="$(
  micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/prepare_factlegalllama_adapter.py" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$ADAPTER_ROOT" \
    --cache-dir "$HF_HOME" \
    --print-path
)"
BASE_MODEL="$(
  micromamba run -n "$ENV_NAME" python - "$ADAPTER_DIR" <<'PY'
import json
import sys
from pathlib import Path

adapter_dir = Path(sys.argv[1])
with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as handle:
    print(json.load(handle)["base_model_name_or_path"])
PY
)"

echo "[prepare] adapter_dir=$ADAPTER_DIR"
echo "[prepare] base_model=$BASE_MODEL"
echo "[prefetch] base model into HF_HOME=$HF_HOME"
micromamba run -n "$ENV_NAME" python - "$BASE_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1])
PY

echo "[run] checkpoint=$CHECKPOINT"
echo "[run] prompt_profile=$PROMPT_PROFILE"
echo "[run] input=$INPUT_DIR"
echo "[run] output=$OUTPUT_DIR"
echo "[run] launching $NUM_GPUS shards"

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  shard="$(printf "%02d" "$gpu")"
  log_file="$OUTPUT_DIR/logs/shard_${shard}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/run_original_prompt_legal_llm.py" \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --model-name "$ADAPTER_DIR" \
      --model-subfolder "" \
      --adapter-mode peft \
      --prompt-profile "$PROMPT_PROFILE" \
      --sections facts \
      --num-shards "$NUM_GPUS" \
      --shard-index "$gpu" \
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
