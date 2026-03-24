#!/usr/bin/env python3
"""CLI entrypoint for the local OpenNyAI document pipeline."""

from __future__ import annotations

import argparse
import shlex
import sys

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GLOB_PATTERN,
    DEFAULT_PIPELINE_BATCH_SIZE,
    DEFAULT_PREPROCESSING_MODEL,
    DEFAULT_SUMMARY_LENGTH,
    DEFAULT_WORKER_PROCESSES,
    PipelineConfig,
    build_output_layout,
    configure_local_runtime,
    ensure_output_layout,
    load_project_environment,
)
from src.pipeline_runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenNyAI NER + Rhetorical Role + Summarizer over a folder of .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", "--input-dir", default=None, help="Directory containing input .txt files.")
    parser.add_argument("--output_dir", "--output-dir", default=None, help="Directory where outputs will be written.")
    parser.add_argument(
        "--use_gpu",
        "--use-gpu",
        action="store_true",
        default=None,
        help="Request GPU inference when CUDA/cupy are available.",
    )
    parser.add_argument("--glob_pattern", "--glob-pattern", default=DEFAULT_GLOB_PATTERN, help="Glob pattern for input text files.")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing outputs.")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="NER mini-batch size.")
    parser.add_argument(
        "--pipeline_batch_size",
        "--pipeline-batch-size",
        type=int,
        default=DEFAULT_PIPELINE_BATCH_SIZE,
        help="Number of processed documents to send through OpenNyAI in one pipeline call.",
    )
    parser.add_argument(
        "--worker_processes",
        "--worker-processes",
        type=int,
        default=DEFAULT_WORKER_PROCESSES,
        help="Number of independent worker processes to run. Use 2 with --gpu_devices 0,1 for two GPUs.",
    )
    parser.add_argument(
        "--gpu_devices",
        "--gpu-devices",
        default="",
        help="Comma-separated GPU ids to pin workers to, for example '0,1'.",
    )
    parser.add_argument(
        "--summary_length",
        "--summary-length",
        type=float,
        default=DEFAULT_SUMMARY_LENGTH,
        help="Summary length fraction in the 0.0-1.0 range. Use 0.0 for adaptive mode.",
    )
    parser.add_argument(
        "--preprocessing_model",
        "--preprocessing-model",
        default=DEFAULT_PREPROCESSING_MODEL,
        help="spaCy preprocessing model passed into OpenNyAI Data(...).",
    )
    return parser


def main() -> int:
    load_project_environment()
    configure_local_runtime()

    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig.from_args(args)
    output_layout = build_output_layout(config.output_dir)
    ensure_output_layout(output_layout)

    cli_command = shlex.join(["python", "run_pipeline.py", *sys.argv[1:]])
    run_result = run_pipeline(config=config, layout=output_layout, cli_command=cli_command)
    report = run_result["run_report"]

    print("\nGenerated files")
    print(run_result["tree_report"])
    print("\nCLI command")
    print(cli_command)
    print(
        "\nRun summary\n"
        f"{report['succeeded']} succeeded, {report['failed']} failed, "
        f"{report['skipped']} skipped out of {report['total_files']} discovered document(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
