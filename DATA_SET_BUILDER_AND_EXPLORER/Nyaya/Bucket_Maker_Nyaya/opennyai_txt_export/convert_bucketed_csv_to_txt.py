#!/usr/bin/env python3
"""Convert a bucketed CSV into bucket/chunk folders of OpenNyAI-ready .txt files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TextIO, Tuple


csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_CHUNK_SIZE = 1000

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MULTISPACE_RE = re.compile(r"[ \t]+")
EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class ChunkManifest:
    handle: TextIO
    writer: csv.DictWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one bucketed CSV into bucketwise chunk folders of .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        "--input-csv",
        required=True,
        help="Path to a bucketed CSV with filename,text,label,bucket columns.",
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where the generated folders will be written.",
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of documents per bucket chunk directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing dataset export folder for this input CSV.",
    )
    return parser.parse_args()


def sanitize_name(value: str, *, fallback: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def normalize_text(value: str) -> str:
    text = value.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = MULTISPACE_RE.sub(" ", text)
    text = EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def infer_dataset_name(input_csv: Path) -> str:
    stem = input_csv.stem
    if stem.endswith("_bucketed"):
        stem = stem[: -len("_bucketed")]
    return sanitize_name(stem, fallback="dataset")


def open_chunk_manifest(
    dataset_dir: Path,
    bucket_name: str,
    chunk_index: int,
    manifests: Dict[Tuple[str, int], ChunkManifest],
) -> ChunkManifest:
    key = (bucket_name, chunk_index)
    if key in manifests:
        return manifests[key]

    chunk_dir = dataset_dir / bucket_name / f"chunk_{chunk_index:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_dir / "manifest.csv"
    handle = manifest_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "row_number",
            "filename",
            "txt_filename",
            "label",
            "bucket",
            "chunk_index",
            "char_count",
            "word_count",
            "relative_txt_path",
        ],
    )
    writer.writeheader()
    manifest = ChunkManifest(handle=handle, writer=writer)
    manifests[key] = manifest
    return manifest


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer.")

    dataset_name = infer_dataset_name(input_csv)
    dataset_dir = output_root / dataset_name

    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset export directory already exists: {dataset_dir}\n"
                "Re-run with --overwrite to replace it."
            )
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifests: Dict[Tuple[str, int], ChunkManifest] = {}
    bucket_counts: Counter[str] = Counter()
    bucket_chunk_counts: Counter[str] = Counter()
    bucket_char_totals: Counter[str] = Counter()
    bucket_word_totals: Counter[str] = Counter()
    file_name_counts: defaultdict[str, int] = defaultdict(int)

    total_rows = 0
    written_rows = 0
    skipped_rows = 0

    try:
        with input_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"filename", "text", "label", "bucket"}
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Input CSV is missing required columns: {sorted(missing_columns)!r}. "
                    f"Found: {reader.fieldnames!r}"
                )

            for row_number, row in enumerate(reader, start=2):
                total_rows += 1

                bucket_name = sanitize_name(str(row.get("bucket", "")), fallback="unknown_bucket")
                source_filename = sanitize_name(str(row.get("filename", "")), fallback=f"row_{row_number:07d}")
                label = str(row.get("label", "")).strip()
                text = normalize_text(str(row.get("text", "")))

                if not text:
                    skipped_rows += 1
                    continue

                file_name_counts[source_filename] += 1
                file_suffix = ""
                if file_name_counts[source_filename] > 1:
                    file_suffix = f"__dup{file_name_counts[source_filename]:03d}"
                txt_filename = f"{source_filename}{file_suffix}.txt"

                doc_index_within_bucket = bucket_counts[bucket_name]
                chunk_index = (doc_index_within_bucket // args.chunk_size) + 1
                chunk_manifest = open_chunk_manifest(dataset_dir, bucket_name, chunk_index, manifests)
                chunk_dir = dataset_dir / bucket_name / f"chunk_{chunk_index:04d}"
                txt_path = chunk_dir / txt_filename

                txt_path.write_text(text, encoding="utf-8")

                char_count = len(text)
                word_count = len(text.split())
                relative_txt_path = txt_path.relative_to(dataset_dir)

                chunk_manifest.writer.writerow(
                    {
                        "row_number": row_number,
                        "filename": source_filename,
                        "txt_filename": txt_filename,
                        "label": label,
                        "bucket": bucket_name,
                        "chunk_index": chunk_index,
                        "char_count": char_count,
                        "word_count": word_count,
                        "relative_txt_path": str(relative_txt_path),
                    }
                )

                bucket_counts[bucket_name] += 1
                bucket_chunk_counts[bucket_name] = max(bucket_chunk_counts[bucket_name], chunk_index)
                bucket_char_totals[bucket_name] += char_count
                bucket_word_totals[bucket_name] += word_count
                written_rows += 1
    finally:
        for manifest in manifests.values():
            manifest.handle.close()

    summary = {
        "input_csv": str(input_csv),
        "dataset_dir": str(dataset_dir),
        "chunk_size": args.chunk_size,
        "total_rows_seen": total_rows,
        "written_rows": written_rows,
        "skipped_empty_rows": skipped_rows,
        "bucket_stats": {
            bucket_name: {
                "documents": bucket_counts[bucket_name],
                "chunks": bucket_chunk_counts[bucket_name],
                "total_chars": bucket_char_totals[bucket_name],
                "total_words": bucket_word_totals[bucket_name],
            }
            for bucket_name in sorted(bucket_counts)
        },
    }

    summary_path = dataset_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input CSV:   {input_csv}")
    print(f"Output root: {dataset_dir}")
    print(f"Chunk size:  {args.chunk_size}")
    print(f"Rows seen:   {total_rows}")
    print(f"Rows written:{written_rows}")
    print(f"Rows skipped:{skipped_rows}")
    print("\nBuckets:")
    for bucket_name in sorted(bucket_counts):
        print(
            f"  {bucket_name:<25} docs={bucket_counts[bucket_name]:>6} "
            f"chunks={bucket_chunk_counts[bucket_name]:>3}"
        )
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
