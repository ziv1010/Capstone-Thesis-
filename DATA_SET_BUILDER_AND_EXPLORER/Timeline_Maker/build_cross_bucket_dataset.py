#!/usr/bin/env python3
"""
Build one flat cross-bucket directory containing all JSON files.

Default output:
  - cross_bucket_total_dataset/          # flat directory of JSON files
  - cross_bucket_total_dataset_report.json

Because filenames collide across buckets, output filenames are prefixed with the
bucket name:
  <bucket_name>__<original_file_name>.json

By default the script creates hard links, so it is fast and does not duplicate
the JSON file contents on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BUCKETS = [
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
]

DEFAULT_OUTPUT_DIR_NAME = "cross_bucket_total_dataset"
DEFAULT_REPORT_NAME = "cross_bucket_total_dataset_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all bucket JSON files into one flat directory."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory that contains the bucket folders.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=DEFAULT_BUCKETS,
        help="Bucket folder names or absolute paths to bucket folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory path. Defaults to <base-dir>/cross_bucket_total_dataset",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSON path. Defaults to <base-dir>/cross_bucket_total_dataset_report.json",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place files in the output directory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after every N linked or copied files.",
    )
    return parser.parse_args()


def resolve_bucket_dir(base_dir: Path, bucket_arg: str) -> Path:
    bucket_path = Path(bucket_arg)
    if not bucket_path.is_absolute():
        bucket_path = base_dir / bucket_path
    return bucket_path.resolve()


def list_bucket_files(bucket_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in bucket_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_report(report_path: Path, report_data: dict) -> None:
    tmp_path = report_path.with_name(f"{report_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, report_path)


def destination_name(bucket_name: str, source_file: Path) -> str:
    return f"{bucket_name}__{source_file.name}"


def materialize_file(source_path: Path, dest_path: Path, mode: str) -> None:
    if mode == "hardlink":
        os.link(source_path, dest_path)
        return
    if mode == "copy":
        shutil.copy2(source_path, dest_path)
        return
    if mode == "symlink":
        os.symlink(source_path, dest_path)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def build_directory(
    bucket_files: list[tuple[str, Path]],
    output_dir: Path,
    mode: str,
    progress_every: int,
) -> tuple[int, list[dict], list[dict], float]:
    linked = 0
    errors: list[dict] = []
    manifest: list[dict] = []
    started = time.time()

    for bucket_name, file_path in bucket_files:
        dest_name = destination_name(bucket_name, file_path)
        dest_path = output_dir / dest_name

        try:
            materialize_file(file_path, dest_path, mode)
        except Exception as exc:  # pragma: no cover - error path
            errors.append(
                {
                    "bucket": bucket_name,
                    "file": file_path.name,
                    "path": str(file_path),
                    "error": str(exc),
                }
            )
            continue

        manifest.append(
            {
                "bucket": bucket_name,
                "source_file": file_path.name,
                "source_path": str(file_path),
                "output_file": dest_name,
                "output_path": str(dest_path),
            }
        )
        linked += 1

        if progress_every > 0 and linked % progress_every == 0:
            elapsed = time.time() - started
            print(
                f"Added {linked} files in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    return linked, errors, manifest, time.time() - started


def main() -> int:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = (args.output_dir or (base_dir / DEFAULT_OUTPUT_DIR_NAME)).resolve()
    report_path = (args.report or (base_dir / DEFAULT_REPORT_NAME)).resolve()

    bucket_dirs: list[tuple[str, Path]] = []
    for bucket_arg in args.buckets:
        bucket_dir = resolve_bucket_dir(base_dir, bucket_arg)
        if not bucket_dir.exists():
            print(f"Missing bucket directory: {bucket_dir}", file=sys.stderr)
            return 1
        if not bucket_dir.is_dir():
            print(f"Not a directory: {bucket_dir}", file=sys.stderr)
            return 1
        bucket_dirs.append((bucket_dir.name, bucket_dir))

    bucket_counts: dict[str, int] = {}
    bucket_files: list[tuple[str, Path]] = []

    for bucket_name, bucket_dir in bucket_dirs:
        files = list_bucket_files(bucket_dir)
        bucket_counts[bucket_name] = len(files)
        bucket_files.extend((bucket_name, file_path) for file_path in files)

    prepare_output_dir(output_dir)

    linked, errors, manifest, elapsed_seconds = build_directory(
        bucket_files=bucket_files,
        output_dir=output_dir,
        mode=args.mode,
        progress_every=args.progress_every,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "source_buckets": [bucket_name for bucket_name, _ in bucket_dirs],
        "files_found_per_bucket": bucket_counts,
        "total_files_found": sum(bucket_counts.values()),
        "files_materialized": linked,
        "files_failed": len(errors),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "manifest_preview": manifest[:20],
        "errors": errors,
    }
    write_report(report_path, report)

    print(f"Cross-bucket directory written to: {output_dir}")
    print(f"Report written to: {report_path}")
    print(f"Files materialized: {linked}")
    print(f"Files failed: {len(errors)}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
