#!/usr/bin/env bash
# run_pipeline.sh — Run the full Bucket Maker pipeline
# Usage:
#   bash run_pipeline.sh              # run all 3 steps
#   bash run_pipeline.sh --step 1    # run only step 1
#   bash run_pipeline.sh --step 2    # run only step 2
#   bash run_pipeline.sh --step 3    # run only step 3

set -e

ENV_NAME="bucket_maker"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP=0  # 0 = all steps

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# Export HF token so scripts can read it
export HF_TOKEN="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"

cd "${SCRIPT_DIR}"

run_step() {
    local n="$1"
    local script="$2"
    local desc="$3"
    echo ""
    echo "════════════════════════════════════════════════"
    echo " Step ${n}: ${desc}"
    echo "════════════════════════════════════════════════"
    micromamba run -n "${ENV_NAME}" python "${script}"
}

if [[ "$STEP" == 0 || "$STEP" == 1 ]]; then
    run_step 1 "01_csv_to_json.py" "CSV → Individual JSON files"
fi

if [[ "$STEP" == 0 || "$STEP" == 2 ]]; then
    run_step 2 "02_bucket_classifier.py" "Classify cases into buckets"
fi

if [[ "$STEP" == 0 || "$STEP" == 3 ]]; then
    run_step 3 "03_build_bucket_datasets.py" "Build per-bucket dataset directories"
fi

echo ""
echo "════════════════════════════════════════════════"
echo " Pipeline complete! Output is in: output/"
echo "════════════════════════════════════════════════"
