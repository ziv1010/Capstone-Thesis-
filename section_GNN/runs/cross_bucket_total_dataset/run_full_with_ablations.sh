#!/usr/bin/env bash
# Full run: preprocess -> build graph -> kfold -> 3 ablations (no_cross_case, text_only, depth).
# Start it and go offline — everything runs sequentially, logged to a single file.
#
# Usage:
#   nohup bash run_full_with_ablations.sh > run_full.log 2>&1 &
#   tail -f run_full.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_GNN="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SECTION_GNN"
ABLATIONS="$SECTION_GNN/ablations"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
GPUS="0,1,2,3,4,5,6,7"
K=5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "========================================================"
log "cross_bucket_total_dataset — full run + 3 ablations"
log "  env=$MAMBA_ENV  gpus=$GPUS  k=$K"
log "========================================================"

# ── Step 1: Preprocess ────────────────────────────────────
log "STEP 1/7  Preprocessing"
bash "$SCRIPT_DIR/01_preprocess.sh" --env "$MAMBA_ENV"
log "STEP 1/7  Done."

# ── Step 2: Build main graph ──────────────────────────────
log "STEP 2/7  Building main graph (shared statute/provision/precedent)"
bash "$SCRIPT_DIR/02_build_graph.sh" --env "$MAMBA_ENV" --gpus "$GPUS"
log "STEP 2/7  Done."

# ── Step 3: Main kfold ────────────────────────────────────
log "STEP 3/7  K-fold (main run)"
bash "$SCRIPT_DIR/03_kfold_8gpu.sh" \
  --env "$MAMBA_ENV" \
  --run-name "cross_bucket_full_kfold" \
  --k "$K"
log "STEP 3/7  Done."

# ── Step 4: Ablation — no cross-case graph build ──────────
log "STEP 4/7  Building no_cross_case ablation graph"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
  micromamba run -n "$MAMBA_ENV" python \
  "$SECTION_GNN/src/scripts/build_graph.py" \
  --config "$ABLATIONS/no_cross_case/cross_bucket_total_dataset/config.yaml"
log "STEP 4/7  Done."

# ── Step 5: Ablation — no cross-case kfold ───────────────
log "STEP 5/7  K-fold (no_cross_case ablation)"
bash "$ABLATIONS/no_cross_case/cross_bucket_total_dataset/run.sh" \
  --env "$MAMBA_ENV" \
  --skip-build \
  --k "$K"
log "STEP 5/7  Done."

# ── Step 6: Ablation — text_only ─────────────────────────
log "STEP 6/7  text_only ablation (build graph + kfold)"
bash "$ABLATIONS/text_only/cross_bucket_total_dataset/run.sh" \
  --env "$MAMBA_ENV" \
  --k "$K"
log "STEP 6/7  Done."

# ── Step 7: Ablation — depth 1/2/3 ───────────────────────
log "STEP 7/7  Depth ablation (layers 1, 2, 3 — reuses main graph)"
bash "$ABLATIONS/depth/cross_bucket_total_dataset/run.sh" \
  --env "$MAMBA_ENV" \
  --k "$K"
log "STEP 7/7  Done."

log "========================================================"
log "ALL DONE. Summaries:"
OUTPUTS="outputs/timed_bucket_runs/cross_bucket_total_dataset"
log "  Main:          $OUTPUTS/models/cross_bucket_full_kfold/kfold/kfold_summary.json"
log "  No cross-case: $OUTPUTS/models/ablation_no_cross_case_cross_bucket_kfold/kfold/kfold_summary.json"
log "  Text-only:     $OUTPUTS/models/ablation_text_only_cross_bucket_kfold/kfold/kfold_summary.json"
log "  Depth-1:       $OUTPUTS/models/ablation_depth1_cross_bucket_kfold/kfold/kfold_summary.json"
log "  Depth-2:       $OUTPUTS/models/ablation_depth2_cross_bucket_kfold/kfold/kfold_summary.json"
log "  Depth-3:       $OUTPUTS/models/ablation_depth3_cross_bucket_kfold/kfold/kfold_summary.json"
log "========================================================"
