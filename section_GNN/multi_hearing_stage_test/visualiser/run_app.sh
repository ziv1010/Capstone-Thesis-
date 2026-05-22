#!/usr/bin/env bash
# Launch the multi-hearing stage test Dash visualiser.
#
# After starting, open an SSH tunnel from your local machine:
#   ssh -L 8050:localhost:8050 <your-server>
# Then open http://localhost:8050 in your browser.
#
# No GPU needed; reads only from ../outputs/.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${1:-8050}"
ENV_NAME="${MAMBA_ENV:-graph_vis}"

echo "=== Multi-hearing stage test visualiser ==="
echo "env=${ENV_NAME}  port=${PORT}"
echo ""
echo "SSH tunnel (from local machine):"
echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname)"
echo ""

micromamba run -n "${ENV_NAME}" python app.py --port "${PORT}"
