#!/usr/bin/env bash
set -euo pipefail

# Full explanation pipeline for:
#   Entity-resolved + section-separated + LR decay + cross-bucket + fold_00
#
# Runs:
#   1. Typed counterfactual explanations
#   2. Faithfulness / prediction-bucket validation
#   3. Identity shortcut / leakage audit
#   4. Pattern-level structural analyses
#   5. HGT embedding extraction
#   6. Nearest opposite-label counterfactual neighborhoods
#   7. HDBSCAN embedding-cluster characterization
#   8. New full-graph community / bridge / hub analyses
#   9. Post-hoc identity and hub-removal inference masking audit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECTION_GNN="$REPO_ROOT/section_GNN"
export PYTHONPATH="$SECTION_GNN:$SCRIPT_DIR:${PYTHONPATH:-}"

MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

GPUS="${GPUS:-}"
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

SPLIT="${SPLIT:-test}"
RUN_DEP_CHECK="${RUN_DEP_CHECK:-1}"
RUN_EXPLANATIONS="${RUN_EXPLANATIONS:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_IDENTITY_AUDIT="${RUN_IDENTITY_AUDIT:-1}"
RUN_MASK_SENSITIVITY_AUDIT="${RUN_MASK_SENSITIVITY_AUDIT:-1}"
RUN_PATTERNS="${RUN_PATTERNS:-1}"
RUN_FULL_GRAPH="${RUN_FULL_GRAPH:-1}"
RUN_VISUALIZER="${RUN_VISUALIZER:-0}"
IDENTITY_AUDIT_PERMUTATIONS="${IDENTITY_AUDIT_PERMUTATIONS:-100}"
MASK_AUDIT_DEVICE="${MASK_AUDIT_DEVICE:-cuda:0}"
MASK_AUDIT_HUB_KS="${MASK_AUDIT_HUB_KS:-10,25,50}"

OUTPUT_BASE="${OUTPUT_BASE:-$SCRIPT_DIR/outputs/entity_resolved_section_sep_lr_decay_cross_bucket}"
EXPLAIN_DIR="${EXPLAIN_DIR:-${OUTPUT_BASE}_fold00}"
PATTERN_DIR="${PATTERN_DIR:-${OUTPUT_BASE}_pattern_why}"
FULL_GRAPH_DIR="${FULL_GRAPH_DIR:-${OUTPUT_BASE}_full_graph}"

MODEL="${MODEL:-$SECTION_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_00/model.pt}"
PRED="${PRED:-$SECTION_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_00/predictions.csv}"
GRAPH="${GRAPH:-$SECTION_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/graph_cache/section/case_star_entity_resolved_cross_bucket_section_sep_lr_decay.reasoning_focused.pt}"
CONFIG="${CONFIG:-$SECTION_GNN/ablations/entity_resolved_data/configs/section/cross_bucket_total_dataset/config.yaml}"

usage() {
  cat <<'EOF'
Usage:
  FINAL_EXPLANATION/run_entity_resolved_section_sep_lr_decay_cross_bucket_all.sh [options]

Options:
  --gpus "0,1,2,3"        GPUs for sharded explain/validate runs. Defaults to all visible GPUs.
  --split test|all|train|val
                           Split to explain. Default: test.
  --output-base PATH       Prefix for output dirs.
  --env NAME               Micromamba env. Default: thesis_work.
  --skip-dep-check         Do not check/install igraph/leidenalg/hdbscan.
  --skip-explanations      Skip typed counterfactual explanations.
  --skip-validation        Skip faithfulness / bucket validation.
  --skip-identity-audit    Skip identity shortcut / leakage audit.
  --skip-mask-sensitivity  Skip post-hoc identity/hub masking audit.
  --skip-patterns          Skip pattern-level analyses.
  --skip-full-graph        Skip full-graph community analyses.
  --visualizer             Start visualizer after all jobs finish.

Environment overrides:
  GPUS, SPLIT, MAMBA_ENV, OUTPUT_BASE, EXPLAIN_DIR, PATTERN_DIR, FULL_GRAPH_DIR
  MODEL, PRED, GRAPH, CONFIG
  RUN_DEP_CHECK, RUN_EXPLANATIONS, RUN_VALIDATION, RUN_IDENTITY_AUDIT, RUN_MASK_SENSITIVITY_AUDIT, RUN_PATTERNS, RUN_FULL_GRAPH, RUN_VISUALIZER
  IDENTITY_AUDIT_PERMUTATIONS, MASK_AUDIT_DEVICE, MASK_AUDIT_HUB_KS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; EXPLAIN_DIR="${2}_fold00"; PATTERN_DIR="${2}_pattern_why"; FULL_GRAPH_DIR="${2}_full_graph"; shift 2 ;;
    --env) MAMBA_ENV="$2"; PYTHON=(micromamba run -n "$MAMBA_ENV" python); shift 2 ;;
    --skip-dep-check) RUN_DEP_CHECK=0; shift ;;
    --skip-explanations) RUN_EXPLANATIONS=0; shift ;;
    --skip-validation) RUN_VALIDATION=0; shift ;;
    --skip-identity-audit) RUN_IDENTITY_AUDIT=0; shift ;;
    --skip-mask-sensitivity) RUN_MASK_SENSITIVITY_AUDIT=0; shift ;;
    --skip-patterns) RUN_PATTERNS=0; shift ;;
    --skip-full-graph) RUN_FULL_GRAPH=0; shift ;;
    --visualizer) RUN_VISUALIZER=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_file "$MODEL" "model checkpoint"
require_file "$PRED" "predictions CSV"
require_file "$GRAPH" "graph cache"
require_file "$CONFIG" "config"

echo "[target] model=$MODEL"
echo "[target] graph=$GRAPH"
echo "[target] config=$CONFIG"
echo "[target] predictions=$PRED"
echo "[target] gpus=$GPUS split=$SPLIT"
echo "[target] explanations=$EXPLAIN_DIR"
echo "[target] patterns=$PATTERN_DIR"
echo "[target] full_graph=$FULL_GRAPH_DIR"
echo "[target] identity_audit_permutations=$IDENTITY_AUDIT_PERMUTATIONS"
echo "[target] mask_sensitivity_device=$MASK_AUDIT_DEVICE hub_ks=$MASK_AUDIT_HUB_KS"

if [[ "$RUN_DEP_CHECK" -eq 1 ]]; then
  echo "[deps] checking igraph/leidenalg/hdbscan"
  if ! "${PYTHON[@]}" - <<'PY'
import importlib.util
missing = [name for name in ("igraph", "leidenalg", "hdbscan") if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
  then
    echo "[deps] installing igraph leidenalg hdbscan into $MAMBA_ENV"
    "${PYTHON[@]}" -m pip install igraph leidenalg hdbscan
  fi
fi

if [[ "$RUN_EXPLANATIONS" -eq 1 ]]; then
  echo "[1/9] typed counterfactual explanations"
  "$SCRIPT_DIR/run_multi_gpu.sh" \
    --gpus "$GPUS" \
    --output-dir "$EXPLAIN_DIR" \
    --split "$SPLIT" \
    --env "$MAMBA_ENV" \
    -- \
    --model-path "$MODEL" \
    --graph-cache "$GRAPH" \
    --config "$CONFIG" \
    --predictions-csv "$PRED" \
    --progress-every 25
fi

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  echo "[2/9] faithfulness and prediction-bucket validation"
  "$SCRIPT_DIR/run_validation_multi_gpu.sh" \
    --gpus "$GPUS" \
    --explanation-dir "$EXPLAIN_DIR" \
    --output-dir "$EXPLAIN_DIR" \
    --env "$MAMBA_ENV" \
    -- \
    --model-path "$MODEL" \
    --graph-cache "$GRAPH" \
    --config "$CONFIG" \
    --predictions-csv "$PRED" \
    --k-values 0,1,2,3,5,10,20 \
    --random-trials 3 \
    --progress-every 25
fi

if [[ "$RUN_IDENTITY_AUDIT" -eq 1 ]]; then
  echo "[3/9] identity shortcut / leakage audit"
  "${PYTHON[@]}" "$SCRIPT_DIR/identity_shortcut_audit.py" \
    --explanation-dir "$EXPLAIN_DIR" \
    --graph-cache "$GRAPH" \
    --predictions-csv "$PRED" \
    --output-dir "$EXPLAIN_DIR" \
    --permutations "$IDENTITY_AUDIT_PERMUTATIONS"
fi

if [[ "$RUN_PATTERNS" -eq 1 ]]; then
  mkdir -p "$PATTERN_DIR"

  echo "[4/9] structural why analysis"
  "${PYTHON[@]}" "$SCRIPT_DIR/structural_why_analysis.py" \
    --explanation-dir "$EXPLAIN_DIR" \
    --graph-cache "$GRAPH" \
    --predictions-csv "$PRED" \
    --output-dir "$PATTERN_DIR" \
    --split all \
    --max-feature-cases 0 \
    --top-k-neighbors 80

  echo "[5/9] HGT case embedding extraction"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${PYTHON[@]}" "$SCRIPT_DIR/extract_hgt_case_embeddings.py" \
    --explanation-dir "$EXPLAIN_DIR" \
    --graph-cache "$GRAPH" \
    --predictions-csv "$PRED" \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --output-dir "$PATTERN_DIR" \
    --split all \
    --device cuda:0

  echo "[6/9] nearest opposite-label counterfactual neighborhoods"
  "${PYTHON[@]}" "$SCRIPT_DIR/counterfactual_neighborhoods.py" \
    --pattern-dir "$PATTERN_DIR" \
    --output-dir "$PATTERN_DIR" \
    --query-split test \
    --candidate-split train \
    --gpus all

  echo "[7/9] embedding cluster characterization"
  "${PYTHON[@]}" "$SCRIPT_DIR/embedding_cluster_characterization.py" \
    --pattern-dir "$PATTERN_DIR" \
    --output-dir "$PATTERN_DIR" \
    --min-cluster-size 80 \
    --n-jobs -1
fi

if [[ "$RUN_FULL_GRAPH" -eq 1 ]]; then
  echo "[8/9] full-graph communities, hierarchy, profiles, bridge/hub analysis"
  EXPLANATION_DIR="$EXPLAIN_DIR" \
  OUTPUT_DIR="$FULL_GRAPH_DIR" \
  EDGE_WEIGHTING="${EDGE_WEIGHTING:-log_inverse_degree}" \
  PROFILE_RES="${PROFILE_RES:-1.0}" \
  "$SCRIPT_DIR/run_full_graph_communities.sh"
fi

if [[ "$RUN_MASK_SENSITIVITY_AUDIT" -eq 1 ]]; then
  echo "[9/9] post-hoc identity and hub-removal inference masking audit"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${PYTHON[@]}" "$SCRIPT_DIR/mask_sensitivity_audit.py" \
    --explanation-dir "$EXPLAIN_DIR" \
    --full-graph-dir "$FULL_GRAPH_DIR" \
    --graph-cache "$GRAPH" \
    --predictions-csv "$PRED" \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --output-dir "$EXPLAIN_DIR" \
    --split "$SPLIT" \
    --device "$MASK_AUDIT_DEVICE" \
    --hub-ks "$MASK_AUDIT_HUB_KS" \
    --hub-resolution "${PROFILE_RES:-1.0}"
fi

if [[ "$RUN_VISUALIZER" -eq 1 ]]; then
  "$SCRIPT_DIR/run_visualizer.sh" \
    --output-dir "$EXPLAIN_DIR" \
    --pattern-dir "$PATTERN_DIR" \
    --full-graph-dir "$FULL_GRAPH_DIR" \
    --host "${VISUALIZER_HOST:-127.0.0.1}" \
    --port "${VISUALIZER_PORT:-8899}"
fi

echo "[done] explanation outputs: $EXPLAIN_DIR"
echo "[done] pattern outputs: $PATTERN_DIR"
echo "[done] full-graph outputs: $FULL_GRAPH_DIR"
