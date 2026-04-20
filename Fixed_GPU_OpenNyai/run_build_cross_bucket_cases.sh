#!/usr/bin/env bash

set -euo pipefail

TIMELINE_ROOT="${TIMELINE_ROOT:-/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker}"
OUTPUT_DIR="${OUTPUT_DIR:-$TIMELINE_ROOT/cross_bucket_cases_8k_each_mistral}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-8000}"
CLEAN_OUTPUTS=true

usage() {
    cat <<'EOF'
Usage:
  bash run_build_cross_bucket_cases.sh
  bash run_build_cross_bucket_cases.sh --limit-per-category 8000
  bash run_build_cross_bucket_cases.sh --category fin_fraud --category land_property
  bash run_build_cross_bucket_cases.sh --no-clean

What it does:
  - Reads merged case JSONs from Timeline_Maker/*_timed_mistral
  - Excludes each folder's report.json
  - Copies up to N case JSONs per category into one cross-bucket folder
  - Prefixes filenames with the category to avoid collisions

Options:
  --limit-per-category N   Max JSON case files to take from each bucket. Default: 8000.
  --output-dir PATH        Destination folder. Default: Timeline_Maker/cross_bucket_cases_8k_each_mistral
  --category NAME          Run only one category. Can be passed multiple times.
  --no-clean               Keep existing JSON files in the output folder.
  --help                   Show this message.

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
        --limit-per-category|--limit)
            LIMIT_PER_CATEGORY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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

if ! [[ "$LIMIT_PER_CATEGORY" =~ ^[0-9]+$ ]] || [ "$LIMIT_PER_CATEGORY" -le 0 ]; then
    echo "limit-per-category must be a positive integer." >&2
    exit 1
fi

declare -A SOURCES=(
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

if [ ${#selected_categories[@]} -eq 0 ]; then
    selected_categories=("${all_categories[@]}")
fi

mkdir -p "$OUTPUT_DIR"

if $CLEAN_OUTPUTS; then
    find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' -delete
fi

declare -A copied_counts=()
declare -A available_counts=()
total_copied=0

echo "============================================================"
echo "  Build Cross-Bucket Case Folder"
echo "  Timeline root       : $TIMELINE_ROOT"
echo "  Output dir          : $OUTPUT_DIR"
echo "  Limit per category  : $LIMIT_PER_CATEGORY"
echo "  Clean outputs       : $CLEAN_OUTPUTS"
echo "============================================================"
echo ""

for category in "${selected_categories[@]}"; do
    source_dir="${SOURCES[$category]:-}"
    if [ -z "$source_dir" ]; then
        echo "Invalid category: $category" >&2
        exit 1
    fi
    if [ ! -d "$source_dir" ]; then
        echo "Source directory missing for $category: $source_dir" >&2
        exit 1
    fi

    mapfile -d '' files < <(
        find "$source_dir" -maxdepth 1 -type f -name '*.json' ! -name 'report.json' -print0 | sort -z
    )

    available="${#files[@]}"
    take="$LIMIT_PER_CATEGORY"
    if [ "$available" -lt "$take" ]; then
        take="$available"
    fi

    available_counts["$category"]="$available"
    copied_counts["$category"]="$take"

    echo "------------------------------------------------------------"
    echo "Category : $category"
    echo "Source   : $source_dir"
    echo "Available: $available"
    echo "Copying  : $take"
    echo "------------------------------------------------------------"

    for ((i = 0; i < take; i++)); do
        src_file="${files[$i]}"
        base_name="$(basename "$src_file")"
        cp -f -- "$src_file" "$OUTPUT_DIR/${category}__${base_name}"
    done

    total_copied=$((total_copied + take))
    echo ""
done

report_path="$OUTPUT_DIR/report.json"
{
    echo "{"
    echo "  \"timeline_root\": \"${TIMELINE_ROOT}\","
    echo "  \"output_dir\": \"${OUTPUT_DIR}\","
    echo "  \"limit_per_category\": ${LIMIT_PER_CATEGORY},"
    echo "  \"total_copied\": ${total_copied},"
    echo "  \"categories\": {"
    last_index=$((${#selected_categories[@]} - 1))
    for i in "${!selected_categories[@]}"; do
        category="${selected_categories[$i]}"
        comma=","
        if [ "$i" -eq "$last_index" ]; then
            comma=""
        fi
        echo "    \"${category}\": {\"available\": ${available_counts[$category]}, \"copied\": ${copied_counts[$category]}}${comma}"
    done
    echo "  }"
    echo "}"
} > "$report_path"

echo "Cross-bucket folder ready."
echo "Copied ${total_copied} case JSON file(s)."
echo "Report: $report_path"
