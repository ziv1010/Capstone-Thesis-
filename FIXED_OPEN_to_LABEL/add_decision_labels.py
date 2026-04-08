#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_BUCKET_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Bucket_Maker_Nyaya/bucketed"
)
DEFAULT_ANNOTATIONS_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/ner_rr_output/annotations"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/FIXED_OPEN_to_LABEL"
)


def maximize_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def coerce_label(value: str):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


def load_decision_labels(bucket_dir: Path, key_column: str, label_column: str) -> dict[str, object]:
    csv_files = sorted(bucket_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {bucket_dir}")

    decision_labels: dict[str, object] = {}
    conflicts: list[tuple[str, object, object, str]] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} is missing a header row")
            if key_column not in reader.fieldnames:
                raise ValueError(f"{csv_path} is missing the '{key_column}' column")
            if label_column not in reader.fieldnames:
                raise ValueError(f"{csv_path} is missing the '{label_column}' column")

            for row in reader:
                key = row[key_column].strip()
                label = coerce_label(row[label_column])
                existing = decision_labels.get(key)
                if existing is not None and existing != label:
                    conflicts.append((key, existing, label, csv_path.name))
                decision_labels[key] = label

    if conflicts:
        preview = "\n".join(
            f"  {key}: existing={existing!r}, new={new!r}, file={filename}"
            for key, existing, new, filename in conflicts[:10]
        )
        raise ValueError(f"Conflicting labels found across CSV files:\n{preview}")

    return decision_labels


def write_labeled_annotations(
    annotations_dir: Path,
    output_dir: Path,
    decision_labels: dict[str, object],
    output_field: str,
) -> tuple[int, list[str]]:
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {annotations_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing: list[str] = []

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        file_id = data.get("file_id") or json_path.stem
        if file_id not in decision_labels:
            missing.append(file_id)
            continue

        data[output_field] = decision_labels[file_id]

        output_path = output_dir / json_path.name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        written += 1

    return written, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read bucketed CSV files, copy annotation JSON files, and add the CSV decision label "
            "as a new top-level field in each output JSON."
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help=f"Directory containing the bucketed CSV files. Default: {DEFAULT_BUCKET_DIR}",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help=f"Directory containing the source annotation JSON files. Default: {DEFAULT_ANNOTATIONS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where labeled JSON files will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--key-column",
        default="filename",
        help="CSV column used to match annotation file IDs. Default: filename",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column containing the decision label. Default: label",
    )
    parser.add_argument(
        "--output-field",
        default="decision_label",
        help="Top-level JSON field to write. Default: decision_label",
    )
    return parser


def main() -> int:
    maximize_csv_field_limit()
    args = build_parser().parse_args()

    decision_labels = load_decision_labels(
        bucket_dir=args.bucket_dir,
        key_column=args.key_column,
        label_column=args.label_column,
    )

    written, missing = write_labeled_annotations(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        decision_labels=decision_labels,
        output_field=args.output_field,
    )

    print(f"Loaded {len(decision_labels)} decision labels from {args.bucket_dir}")
    print(f"Wrote {written} labeled JSON files to {args.output_dir}")

    if missing:
        preview = ", ".join(missing[:20])
        print(f"Missing labels for {len(missing)} annotation files: {preview}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
