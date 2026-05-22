#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star"
SCRIPT_PATH="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/cluster_case_buckets.py"

if [[ ! -d "$ENV_PATH" ]]; then
  echo "Micromamba env not found: $ENV_PATH" >&2
  exit 1
fi

if [[ -n "${PYTHONWARNINGS:-}" ]]; then
  export PYTHONWARNINGS="${PYTHONWARNINGS},ignore:resource_tracker:UserWarning"
else
  export PYTHONWARNINGS="ignore:resource_tracker:UserWarning"
fi

exec /usr/local/bin/micromamba run -p "$ENV_PATH" python "$SCRIPT_PATH" "$@"
