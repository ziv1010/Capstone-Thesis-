#!/usr/bin/env bash
# Creates the 'graph_vis' micromamba environment for the legal case graph visualiser.
# No GPU required — all computation runs on CPU.
# Run once: bash setup_env.sh

set -euo pipefail

ENV_NAME="graph_vis"
PYTHON_VERSION="3.10"

echo "=== Creating micromamba environment: ${ENV_NAME} ==="
micromamba create -n "${ENV_NAME}" -y \
    python="${PYTHON_VERSION}" \
    -c conda-forge

echo "=== Installing packages ==="
micromamba run -n "${ENV_NAME}" pip install --no-cache-dir \
    networkx==3.3 \
    plotly==5.22.0 \
    dash==2.17.1 \
    dash-bootstrap-components==1.6.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    pyvis==0.3.2 \
    kaleido==0.2.1 \
    scipy==1.13.1

echo ""
echo "=== Done. Activate with: ==="
echo "    micromamba activate ${ENV_NAME}"
echo ""
echo "=== Or run directly with: ==="
echo "    micromamba run -n ${ENV_NAME} python build_graph.py"
