#!/usr/bin/env bash
# =============================================================================
#  run_accuracy_check.sh
#  Run both NER and RR accuracy evaluations using OpenNyAI's published datasets.
#
#  Usage:
#    bash Accuracy_Checking/run_accuracy_check.sh              # CPU, full test set
#    bash Accuracy_Checking/run_accuracy_check.sh --gpu        # GPU, full test set
#    bash Accuracy_Checking/run_accuracy_check.sh --gpu --quick  # GPU, 10 docs (fast sanity check)
#
#  Outputs are saved to:
#    Accuracy_Checking/results/ner_accuracy_results.json
#    Accuracy_Checking/results/rr_accuracy_results.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
ENV_NAME="fixed_gpu_opennyai_final"

USE_GPU=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --gpu)   USE_GPU=true ;;
        --quick) ;;   # no longer used — both docs are always evaluated
    esac
done

mkdir -p "$RESULTS_DIR"

GPU_FLAG=""
if $USE_GPU; then
    GPU_FLAG="--use_gpu"
fi

echo "============================================================"
echo "  OpenNyAI Model Accuracy Check"
echo "  Environment : $ENV_NAME"
echo "  GPU         : $USE_GPU"
echo "  Reference   : OpenNyAI GitHub tests/examples/"
echo "  Results dir : $RESULTS_DIR"
echo "============================================================"
echo ""

# ── NER Evaluation ───────────────────────────────────────────────────────────
echo ">>> STAGE 1/2: NER Accuracy (en_legal_ner_trf)"
echo "    Reference: OpenNyAI GitHub tests/examples/ner_output_files/"
echo ""

micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/check_ner_accuracy.py" \
    --output_file "$RESULTS_DIR/ner_accuracy_results.json" \
    $GPU_FLAG

echo ""
echo "NER results saved to: $RESULTS_DIR/ner_accuracy_results.json"
echo ""

# ── RR Evaluation ────────────────────────────────────────────────────────────
echo ">>> STAGE 2/2: Rhetorical Role Accuracy (InRhetoricalRoles)"
echo "    Reference: OpenNyAI GitHub tests/examples/rr_output_files/"
echo ""

micromamba run -n "$ENV_NAME" python "$SCRIPT_DIR/check_rr_accuracy.py" \
    --output_file "$RESULTS_DIR/rr_accuracy_results.json" \
    $GPU_FLAG

echo ""
echo "RR results saved to: $RESULTS_DIR/rr_accuracy_results.json"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ACCURACY CHECK COMPLETE"
echo "============================================================"

# Print key numbers if jq is available
if command -v jq &>/dev/null; then
    echo ""
    echo "Quick summary:"
    echo ""
    echo "  NER (en_legal_ner_trf):"
    jq -r '"    Micro F1 : \(.micro.f1)  |  Macro F1 : \(.macro.f1)"' \
        "$RESULTS_DIR/ner_accuracy_results.json" 2>/dev/null || true
    echo ""
    echo "  RR (InRhetoricalRoles):"
    jq -r '.results | "    Accuracy : \(.accuracy)  |  Weighted F1 : \(.weighted_f1)  |  Macro F1 : \(.macro.f1)"' \
        "$RESULTS_DIR/rr_accuracy_results.json" 2>/dev/null || true
    echo ""
fi

echo "All results in: $RESULTS_DIR/"
