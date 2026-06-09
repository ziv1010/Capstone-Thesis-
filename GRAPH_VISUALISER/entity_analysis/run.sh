#!/usr/bin/env bash
# Usage:
#   ./run.sh                        # analyse both + start visualizer (port 8052)
#   ./run.sh within                 # only within-bucket analysis + viz
#   ./run.sh cross                  # only cross-bucket analysis + viz
#   ./run.sh both 8055              # both analyses, visualizer on port 8055
#   ./run.sh analyse-only           # run analysis only, no viz (for long jobs)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAMBA_ENV="${MAMBA_ENV:-graph_vis}"
PYTHON=(micromamba run -n "$MAMBA_ENV" python)

MODE="${1:-both}"
PORT="${2:-8052}"

if [ "$MODE" = "analyse-only" ]; then
    echo "═══ Running full analysis (both within + cross) — no viz ═══"
    "${PYTHON[@]}" analyse.py both
    echo "Done. Start viz separately: micromamba run -n $MAMBA_ENV python app.py --port $PORT"
    exit 0
fi

echo "═══ Step 1: Running analysis (mode=$MODE) ═══"
"${PYTHON[@]}" analyse.py "$MODE"

echo ""
echo "═══ Step 2: Starting visualizer on port $PORT ═══"
"${PYTHON[@]}" app.py --port "$PORT"
