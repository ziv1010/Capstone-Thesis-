#!/usr/bin/env bash
# End-to-end driver for the Graph_Analyser explainability pipeline.
#
# Phase 1-4 run in the `graph_explainer` micromamba env (torch+pyg+faiss).
# Phase 5 runs in the `llm` env (vLLM).
#
# Phase 4 is parallelised across N GPUs (--p4-gpus, default 8).
# Phase 5 uses tensor-parallel across 2 GPUs (--tp, default 2).
# Phases are sequential so all GPUs are free for each phase.
#
# Usage:
#   bash scripts/run_all.sh [--config path/to/cfg.yaml] [--skip-llm] [--limit N] [--tp 2] [--p4-gpus 8]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/default.yaml"
SKIP_LLM=0
LLM_LIMIT=0
LLM_TP_SIZE=2
P4_GPUS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG="$2";     shift 2;;
    --skip-llm) SKIP_LLM=1;      shift;;
    --limit)    LLM_LIMIT="$2";  shift 2;;
    --tp)       LLM_TP_SIZE="$2"; shift 2;;
    --p4-gpus)  P4_GPUS="$2";    shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

echo "[run_all] using config: $CONFIG"
cd "$REPO_ROOT"

GRAPH_ENV="graph_explainer"
LLM_ENV="llm"

echo "[run_all] ----- Phase 1-2: inference + FAISS index -----"
micromamba run -n "$GRAPH_ENV" python scripts/phase1_2_inference_and_index.py --config "$CONFIG"

echo "[run_all] ----- Phase 3: train PGExplainer -----"
micromamba run -n "$GRAPH_ENV" python scripts/phase3_train_explainer.py --config "$CONFIG"

echo "[run_all] ----- Phase 4: extract importance (${P4_GPUS} GPUs) -----"
pids=()
for i in $(seq 0 $(( P4_GPUS - 1 ))); do
  CUDA_VISIBLE_DEVICES=$i micromamba run -n "$GRAPH_ENV" python scripts/phase4_extract_importance.py \
    --config "$CONFIG" \
    --rank "$i" \
    --world-size "$P4_GPUS" \
    --device cuda &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[run_all] phase4 worker failed (pid=$pid)"; exit 1; }
done
micromamba run -n "$GRAPH_ENV" python scripts/phase4_extract_importance.py \
  --config "$CONFIG" --merge-only --world-size "$P4_GPUS"

if [[ "$SKIP_LLM" -eq 1 ]]; then
  echo "[run_all] --skip-llm set, stopping before Phase 5"
  exit 0
fi

echo "[run_all] ----- Phase 5: Mistral explanation via vLLM (tp=${LLM_TP_SIZE}) -----"
micromamba run -n "$LLM_ENV" python scripts/phase5_llm_translate.py \
  --config "$CONFIG" \
  --limit "$LLM_LIMIT" \
  --tensor-parallel-size "$LLM_TP_SIZE"

echo "[run_all] all phases complete."
