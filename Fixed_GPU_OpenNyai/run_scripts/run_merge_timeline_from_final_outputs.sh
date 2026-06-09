#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

FINAL_OUTPUTS_DIR="${FINAL_OUTPUTS_DIR:-$PROJECT_ROOT/final_outputs}"
TIMELINE_ROOT="${TIMELINE_ROOT:-$REPO_ROOT/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$TIMELINE_ROOT/merge_cases_v2.py}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
ENV_NAME="${CONDA_ENV:-case_merge}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLEAN_OUTPUTS=true

usage() {
    cat <<'EOF'
Usage:
  bash run_merge_timeline_from_final_outputs.sh
  bash run_merge_timeline_from_final_outputs.sh --no-clean
  bash run_merge_timeline_from_final_outputs.sh --env llm
  bash run_merge_timeline_from_final_outputs.sh --category fin_fraud --category land_property
  CONDA_ENV=case_merge bash run_merge_timeline_from_final_outputs.sh

What it does:
  - Reads labelled Mistral JSONs from:
      final_outputs/<category>_labelled_mistral/labelled_jsons
  - Runs Timeline_Maker/merge_cases_v2.py
  - Writes into the existing Timeline Maker output folders:
      *_timed_mistral

Options:
  --env NAME        Micromamba environment to use. Default: case_merge.
  --category NAME   Run only one category. Can be passed multiple times.
  --no-clean        Keep existing JSON files in target output folders.
  --help            Show this message.

Valid categories:
  family_matrimonial
  fin_fraud
  land_property
  motor_accidents
  sexual_offences
EOF
}

selected_categories=()
while [ $# -gt 0 ]; do
    case "$1" in
        --env|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --category)
            selected_categories+=("$2")
            shift 2
            ;;
        --no-clean)
            CLEAN_OUTPUTS=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

declare -A INPUTS=(
    [family_matrimonial]="$FINAL_OUTPUTS_DIR/family_matrimonial_labelled_mistral/labelled_jsons"
    [fin_fraud]="$FINAL_OUTPUTS_DIR/fin_fraud_labelled_mistral/labelled_jsons"
    [land_property]="$FINAL_OUTPUTS_DIR/land_property_labelled_mistral/labelled_jsons"
    [motor_accidents]="$FINAL_OUTPUTS_DIR/motor_accidents_labelled_mistral/labelled_jsons"
    [sexual_offences]="$FINAL_OUTPUTS_DIR/sexual_offences_labelled_mistral/labelled_jsons"
)

declare -A OUTPUTS=(
    [family_matrimonial]="$TIMELINE_ROOT/family_matrimonial_timed_mistral"
    [fin_fraud]="$TIMELINE_ROOT/fin_fraud_timed_mistral"
    [land_property]="$TIMELINE_ROOT/land_property_timed_mistral"
    [motor_accidents]="$TIMELINE_ROOT/motor_accidents_timed_mistral"
    [sexual_offences]="$TIMELINE_ROOT/sexual_offences_timed_mistral"
)

all_categories=(
    family_matrimonial
    fin_fraud
    land_property
    motor_accidents
    sexual_offences
)

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "merge_cases_v2.py not found: $MERGE_SCRIPT" >&2
    exit 1
fi
if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
    echo "micromamba not found: $MICROMAMBA_BIN" >&2
    exit 1
fi
if ! "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Micromamba environment not found: $ENV_NAME" >&2
    exit 1
fi

if [ ${#selected_categories[@]} -eq 0 ]; then
    selected_categories=("${all_categories[@]}")
fi

echo "============================================================"
echo "  Merge Timeline Cases From Final Outputs"
echo "  Final outputs root : $FINAL_OUTPUTS_DIR"
echo "  Timeline root      : $TIMELINE_ROOT"
echo "  Merge script       : $MERGE_SCRIPT"
echo "  Micromamba         : $MICROMAMBA_BIN"
echo "  Environment        : $ENV_NAME"
echo "  Python             : $PYTHON_BIN"
echo "  Clean outputs      : $CLEAN_OUTPUTS"
echo "============================================================"
echo ""

for category in "${selected_categories[@]}"; do
    input_dir="${INPUTS[$category]:-}"
    output_dir="${OUTPUTS[$category]:-}"

    if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
        echo "Invalid category: $category" >&2
        exit 1
    fi
    if [ ! -d "$input_dir" ]; then
        echo "Input directory missing for $category: $input_dir" >&2
        exit 1
    fi

    mkdir -p "$output_dir"

    if $CLEAN_OUTPUTS; then
        find "$output_dir" -maxdepth 1 -type f -name '*.json' -delete
    fi

    echo "------------------------------------------------------------"
    echo "Category : $category"
    echo "Input    : $input_dir"
    echo "Output   : $output_dir"
    echo "------------------------------------------------------------"

    "$MICROMAMBA_BIN" run -n "$ENV_NAME" "$PYTHON_BIN" "$MERGE_SCRIPT" \
        --input "$input_dir" \
        --output "$output_dir" \
        --skip-hidden

    file_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.json' | wc -l)"
    echo "Wrote $file_count JSON file(s) to $output_dir"
    echo ""
done

echo "All requested categories merged."
