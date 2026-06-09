#!/usr/bin/env bash
# run_build.sh — Build the legal case graph
# Usage:
#   bash run_build.sh              # full build (all cases per bucket)
#   bash run_build.sh --limit 500  # quick smoke-test with 500 cases
#
# No GPU needed. All computation is CPU.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Legal Case Graph Builder ==="
echo "Config : config.yaml"
echo "Output : $(grep output_dir config.yaml | awk '{print $2}')"
echo ""

MAMBA_ENV="${MAMBA_ENV:-graph_vis}"
micromamba run -n "${MAMBA_ENV}" python build_graph.py \
    --config config.yaml \
    "$@"

echo ""
echo "=== Build complete. Run the app with: ==="
echo "    bash run_app.sh"
