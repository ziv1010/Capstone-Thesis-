#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAMBA_ENV="${MAMBA_ENV:-thesis_work}"
OUTPUT_DIR="${OUTPUT_DIR:-$APP_ROOT/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00}"
PATTERN_DIR="${PATTERN_DIR:-$APP_ROOT/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why}"
FULL_GRAPH_DIR="${FULL_GRAPH_DIR:-$APP_ROOT/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --pattern-dir) PATTERN_DIR="$2"; shift 2 ;;
    --full-graph-dir) FULL_GRAPH_DIR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --env) MAMBA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

exec micromamba run -n "$MAMBA_ENV" python "$APP_ROOT/visualizer.py" \
  --output-dir "$OUTPUT_DIR" \
  --pattern-dir "$PATTERN_DIR" \
  --full-graph-dir "$FULL_GRAPH_DIR" \
  --host "$HOST" \
  --port "$PORT"
