#!/usr/bin/env bash
# Embedding-space analysis pipeline
# Runs: extract → t-SNE visualise → probing classifier
# for one or more (config, model) pairs.
#
# Usage:
#   bash run_analysis.sh                          # uses defaults below
#   bash run_analysis.sh --config X.yaml --model fold_00/model.pt --run-name my_run
#   bash run_analysis.sh --all-cross-bucket       # run for cross_bucket fold_00 (main) + text_only
#
# All outputs land in <outputs_dir>/embedding_analysis/ from the config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"

# ── Defaults — main cross_bucket model, fold 0 ───────────────
DEFAULT_CONFIG="$SECTION_GNN/runs/cross_bucket_total_dataset/config.yaml"
DEFAULT_MODEL="$SECTION_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset/models/cross_bucket_full_kfold/kfold/fold_00/model.pt"
DEFAULT_RUN_NAME="cross_bucket_fold00"

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
DEVICE="${DEVICE:-cuda}"
TSNE_SAMPLE="${TSNE_SAMPLE:-5000}"
ALL_CROSS_BUCKET=0

# ── CLI ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)             CONFIG="$2";      shift 2 ;;
    --model)              MODEL="$2";       shift 2 ;;
    --run-name)           RUN_NAME="$2";    shift 2 ;;
    --device)             DEVICE="$2";      shift 2 ;;
    --tsne-sample)        TSNE_SAMPLE="$2"; shift 2 ;;
    --env)                MAMBA_ENV="$2";   shift 2 ;;
    --all-cross-bucket)   ALL_CROSS_BUCKET=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PYTHON="micromamba run -n $MAMBA_ENV python"
ANALYSIS_DIR="$SCRIPT_DIR"

run_one() {
  local cfg="$1" mdl="$2" rname="$3"
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo " RUN: $rname"
  echo " config : $cfg"
  echo " model  : $mdl"
  echo "════════════════════════════════════════════════════════"

  # Derive output dir from config
  local out_dir
  out_dir="$(grep 'outputs_dir:' "$cfg" | awk '{print $2}')/embedding_analysis"
  echo "[run] output dir: $out_dir"

  # ── Step 1: Extract embeddings ────────────────────────────
  echo ""
  echo "[1/6] Extracting embeddings ..."
  $PYTHON "$ANALYSIS_DIR/extract_embeddings.py" \
    --config     "$cfg" \
    --model      "$mdl" \
    --output     "$out_dir" \
    --run-name   "$rname" \
    --device     "$DEVICE"

  local npz_path="$out_dir/${rname}_embeddings.npz"
  local graph_cache_dir
  graph_cache_dir="$(grep 'graph_cache_dir:' "$cfg" | awk '{print $2}')"
  local metadata_path
  metadata_path="$(find "$graph_cache_dir" -name 'graph_metadata*.json' | head -1)"
  local fold_dir
  fold_dir="$(dirname "$mdl")"
  local predictions_path="$fold_dir/predictions.csv"

  # ── Step 2: t-SNE ─────────────────────────────────────────
  echo ""
  echo "[2/6] Running t-SNE ..."
  $PYTHON "$ANALYSIS_DIR/tsne_visualise.py" \
    --embeddings "$npz_path" \
    --sample     "$TSNE_SAMPLE" \
    --output-dir "$out_dir"

  # ── Step 3: Probing classifier ────────────────────────────
  echo ""
  echo "[3/6] Running probing classifier ..."
  $PYTHON "$ANALYSIS_DIR/probing_classifier.py" \
    --embeddings "$npz_path" \
    --output-dir "$out_dir"

  # ── Step 4: Attention / gradient saliency ─────────────────
  echo ""
  echo "[4/6] Running attention analysis ..."
  $PYTHON "$ANALYSIS_DIR/attention_analysis.py" \
    --config     "$cfg" \
    --model      "$mdl" \
    --output-dir "$out_dir" \
    --run-name   "$rname" \
    --device     "$DEVICE"

  # ── Step 5: Error analysis ────────────────────────────────
  echo ""
  echo "[5/6] Running error analysis ..."
  if [[ -f "$predictions_path" && -n "$metadata_path" ]]; then
    $PYTHON "$ANALYSIS_DIR/error_analysis.py" \
      --predictions "$predictions_path" \
      --metadata    "$metadata_path" \
      --output-dir  "$out_dir" \
      --run-name    "$rname"
  else
    echo "[skip] predictions.csv or metadata not found — skipping error analysis"
  fi

  # ── Step 6: SHAP ──────────────────────────────────────────
  echo ""
  echo "[6/6] Running SHAP analysis ..."
  $PYTHON "$ANALYSIS_DIR/shap_analysis.py" \
    --embeddings "$npz_path" \
    --output-dir "$out_dir"

  echo ""
  echo "[done] $rname → $out_dir"
}

# ── Multi-run mode (cross_bucket full, text_only, no_cross_case) ──
if [[ "$ALL_CROSS_BUCKET" -eq 1 ]]; then
  echo "[run_analysis] Running all cross_bucket variants ..."

  BASE_OUT="$SECTION_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset"

  # Main full model
  run_one \
    "$SECTION_GNN/runs/cross_bucket_total_dataset/config.yaml" \
    "$BASE_OUT/models/cross_bucket_full_kfold/kfold/fold_00/model.pt" \
    "cross_bucket_full_fold00"

  # text_only v2 (config-aligned with main)
  TEXTONLY_MODEL="$BASE_OUT/models/ablation_text_only_cross_bucket_v2_kfold/kfold/fold_00/model.pt"
  if [[ -f "$TEXTONLY_MODEL" ]]; then
    run_one \
      "$SECTION_GNN/ablations/text_only/cross_bucket_total_dataset/config.yaml" \
      "$TEXTONLY_MODEL" \
      "cross_bucket_text_only_fold00"
  else
    echo "[skip] text_only model not found: $TEXTONLY_MODEL"
  fi

  # no_cross_case
  NOCROSS_MODEL="$BASE_OUT/models/ablation_no_cross_case_cross_bucket_kfold/kfold/fold_00/model.pt"
  if [[ -f "$NOCROSS_MODEL" ]]; then
    run_one \
      "$SECTION_GNN/ablations/no_cross_case/cross_bucket_total_dataset/config.yaml" \
      "$NOCROSS_MODEL" \
      "cross_bucket_no_cross_case_fold00"
  else
    echo "[skip] no_cross_case model not found: $NOCROSS_MODEL"
  fi

  echo ""
  echo "[run_analysis] All done."
  exit 0
fi

# ── Single run ─────────────────────────────────────────────────
run_one "$CONFIG" "$MODEL" "$RUN_NAME"
echo "[run_analysis] Finished."
