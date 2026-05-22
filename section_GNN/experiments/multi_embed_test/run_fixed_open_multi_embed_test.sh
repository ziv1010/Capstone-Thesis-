#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="4,5,6,7"

micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star \
  python "$SCRIPT_DIR/run_fixed_open_multi_embed_test.py" "$@"
