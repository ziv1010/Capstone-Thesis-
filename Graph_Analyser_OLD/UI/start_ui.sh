#!/bin/bash
# Script to launch the GNN Explainer UI

# Move to the root directory so the HTTP server can serve both UI and outputs/
cd /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Graph_Analyser

PORT=8080

echo "=========================================="
echo "🚀 Starting GNN Explainer UI Server"
echo "=========================================="
echo "The UI will be available at: http://localhost:$PORT/UI/"
echo "Press Ctrl+C to stop the server."
echo "=========================================="

# Start python HTTP server
python3 -m http.server $PORT
