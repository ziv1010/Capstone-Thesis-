#!/usr/bin/env bash
# run_app.sh — Launch the interactive Dash visualiser
#
# After starting, open an SSH tunnel from your LOCAL machine:
#   ssh -L 8050:localhost:8050 <your-server>
# Then open http://localhost:8050 in your browser.
#
# No GPU needed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${1:-8050}"
MAMBA_ENV="${MAMBA_ENV:-graph_vis}"

echo "=== Legal Case Graph Visualiser ==="
echo "Serving on port ${PORT}"
echo ""
echo "SSH tunnel (from local machine):"
echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname)"
echo ""

micromamba run -n "${MAMBA_ENV}" python app.py \
    --config config.yaml \
    --port "${PORT}"
