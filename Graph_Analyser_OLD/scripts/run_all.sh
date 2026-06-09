#!/usr/bin/env bash
# End-to-end driver for the Graph_Analyser explainability pipeline.
#
# All phases run in the `graph_explainer` micromamba env (torch+pyg).
#
# Phase 4 is parallelised across N GPUs (--p4-gpus, default 8).
# Phases are sequential so all GPUs are free for each phase.
#
# Usage:
#   bash scripts/run_all.sh [--config path/to/cfg.yaml] [--skip-diagnostic] [--p4-gpus 8] [--p4-force] [--phase6-plots] [--phase7] [--phase7-gpus 8] [--phase7-limit N] [--phase7-only-untraceable]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/default.yaml"
SKIP_DIAGNOSTIC=0
P4_GPUS=8
P4_FORCE=0
PHASE6_PLOTS=0
RUN_PHASE7=0
PHASE7_GPUS=1
PHASE7_LIMIT=0
PHASE7_ONLY_UNTRACEABLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)          CONFIG="$2";       shift 2;;
    --skip-diagnostic) SKIP_DIAGNOSTIC=1; shift;;
    --p4-gpus)         P4_GPUS="$2";      shift 2;;
    --p4-force)        P4_FORCE=1;        shift;;
    --phase6-plots)    PHASE6_PLOTS=1;    shift;;
    --phase7)          RUN_PHASE7=1;      shift;;
    --phase7-gpus)     PHASE7_GPUS="$2";  shift 2;;
    --phase7-limit)    PHASE7_LIMIT="$2"; shift 2;;
    --phase7-only-untraceable) PHASE7_ONLY_UNTRACEABLE=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

echo "[run_all] using config: $CONFIG"
cd "$REPO_ROOT"

GRAPH_ENV="graph_explainer"

echo "[run_all] ----- Phase 1-2: inference -----"
micromamba run -n "$GRAPH_ENV" python scripts/phase1_2_inference_and_index.py --config "$CONFIG"

echo "[run_all] ----- Phase 3: train PGExplainer -----"
micromamba run -n "$GRAPH_ENV" python scripts/phase3_train_explainer.py --config "$CONFIG"

echo "[run_all] ----- Phase 4: extract importance (${P4_GPUS} GPUs) -----"
p4_extra_args=()
if [[ "$P4_FORCE" -eq 1 ]]; then
  p4_extra_args=(--force)
fi
pids=()
for i in $(seq 0 $(( P4_GPUS - 1 ))); do
  CUDA_VISIBLE_DEVICES=$i micromamba run -n "$GRAPH_ENV" python scripts/phase4_extract_importance.py \
    --config "$CONFIG" \
    --rank "$i" \
    --world-size "$P4_GPUS" \
    --device cuda \
    "${p4_extra_args[@]}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[run_all] phase4 worker failed (pid=$pid)"; exit 1; }
done
micromamba run -n "$GRAPH_ENV" python scripts/phase4_extract_importance.py \
  --config "$CONFIG" --merge-only --world-size "$P4_GPUS"

if [[ "$SKIP_DIAGNOSTIC" -eq 0 ]]; then
  echo "[run_all] ----- Phase 6: evidence diagnostic for all explained cases -----"
  phase6_args=(--config "$CONFIG" --all --skip-plots)
  if [[ "$PHASE6_PLOTS" -eq 1 ]]; then
    phase6_args=(--config "$CONFIG" --all)
  fi
  micromamba run -n "$GRAPH_ENV" python scripts/phase6_misclass_diagnostic.py "${phase6_args[@]}"
fi

if [[ "$RUN_PHASE7" -eq 1 ]]; then
  echo "[run_all] ----- Phase 7: embedding nearest-neighbour diagnostics (${PHASE7_GPUS} GPUs) -----"
  phase7_args=(--config "$CONFIG" --all)
  if [[ "$PHASE7_ONLY_UNTRACEABLE" -eq 1 ]]; then
    phase7_args+=(--only-untraceable)
  fi
  if [[ "$PHASE7_LIMIT" -gt 0 ]]; then
    phase7_args+=(--limit "$PHASE7_LIMIT")
  fi
  phase7_pids=()
  for i in $(seq 0 $(( PHASE7_GPUS - 1 ))); do
    CUDA_VISIBLE_DEVICES=$i micromamba run -n "$GRAPH_ENV" python scripts/phase7_topk_embedding.py \
      "${phase7_args[@]}" \
      --rank "$i" \
      --world-size "$PHASE7_GPUS" &
    phase7_pids+=($!)
  done
  for pid in "${phase7_pids[@]}"; do
    wait "$pid" || { echo "[run_all] phase7 worker failed (pid=$pid)"; exit 1; }
  done
  micromamba run -n "$GRAPH_ENV" python scripts/phase7_topk_embedding.py \
    --config "$CONFIG" \
    --merge-only
fi

echo "[run_all] all phases complete."
