#!/usr/bin/env bash
# Run the entity-resolved section-sep + no-names ablation with and without LR decay.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_entity_resolved_data_ablation.sh"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      echo "This wrapper always runs --only section_no_names." >&2
      exit 1
      ;;
    --lr-mode|--no-lr-decay)
      echo "This wrapper runs both LR modes; do not pass $1." >&2
      exit 1
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

echo ">>> section_no_names with LR decay"
bash "$RUNNER" --only section_no_names --lr-mode decay "${FORWARD_ARGS[@]}"

echo ">>> section_no_names without LR decay"
bash "$RUNNER" --only section_no_names --lr-mode none "${FORWARD_ARGS[@]}"
