#!/usr/bin/env bash
# run_app.sh — Launch the stage-by-stage pipeline visualiser
#
# After starting, open an SSH tunnel from your LOCAL machine:
#   ssh -L 8053:localhost:8053 <your-server>
# Then open http://localhost:8053 in your browser.
#
# Reuses the existing 'graph_vis' micromamba env (created by GRAPH_VISUALISER/setup_env.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${1:-8053}"

echo "=== Pipeline Stage Visualiser ==="
echo "Serving on port ${PORT}"
echo ""
echo "SSH tunnel (from local machine):"
echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname)"
echo ""

micromamba run -n graph_vis python app.py --port "${PORT}"
