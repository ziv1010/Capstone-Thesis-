#!/usr/bin/env python3
"""Run a clean one-document OpenNyai vs replication comparison."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


DEFAULT_DOC_ID = "Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc_id", default=DEFAULT_DOC_ID)
    parser.add_argument("--workspace_root")
    parser.add_argument("--project_root")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    log_text = []
    log_text.append(f"$ {shlex.join(command)}")
    log_text.append("")
    log_text.append("STDOUT")
    log_text.append(result.stdout)
    log_text.append("")
    log_text.append("STDERR")
    log_text.append(result.stderr)
    write_text(log_path, "\n".join(log_text))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {shlex.join(command)}")


def copy_existing_open_nyai_outputs(*, open_nyai_root: Path, open_output_dir: Path, doc_id: str, log_path: Path) -> None:
    copied = []
    for folder_name in ["combined", "rhetorical_roles", "ner", "annotations", "summaries"]:
        source_dir = open_nyai_root / folder_name
        target_dir = open_output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".json", ".txt"]:
            source_file = source_dir / f"{doc_id}{suffix}"
            if source_file.exists():
                shutil.copy2(source_file, target_dir / source_file.name)
                copied.append(str(target_dir / source_file.name))
    write_text(
        log_path,
        "OpenNyai single-document rerun did not produce the expected output files.\n"
        "Copied existing successful OpenNyai outputs for the same document into this workspace instead.\n\n"
        + "\n".join(copied)
        + "\n",
    )


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parents[1]
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else script_root
    comparison_root = workspace_root
    project_root = Path(args.project_root).resolve() if args.project_root else script_root.parent
    open_nyai_root = project_root / "OpenNyai"
    replication_root = project_root / "REPLICATION_OpenNyai"
    source_txt = open_nyai_root / "input_txt" / f"{args.doc_id}.txt"

    if not source_txt.exists():
        raise FileNotFoundError(f"Missing source text file: {source_txt}")

    inputs_dir = comparison_root / "inputs" / "source_txt"
    outputs_dir = comparison_root / "outputs"
    logs_dir = outputs_dir / "logs"
    open_output_dir = outputs_dir / "open_nyai"
    replication_ner_dir = outputs_dir / "replication_ner"
    replication_rr_dir = outputs_dir / "replication_rr"
    reports_dir = outputs_dir / "reports"

    if args.overwrite:
        for path in [open_output_dir, replication_ner_dir, replication_rr_dir, reports_dir, logs_dir]:
            if path.exists():
                shutil.rmtree(path)

    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    replication_ner_dir.mkdir(parents=True, exist_ok=True)
    replication_rr_dir.mkdir(parents=True, exist_ok=True)

    comparison_text_path = inputs_dir / source_txt.name
    shutil.copy2(source_txt, comparison_text_path)

    commands = {}

    open_command = [
        "micromamba",
        "run",
        "-n",
        "opennyai_py38",
        "python",
        str(open_nyai_root / "run_pipeline.py"),
        "--input_dir",
        str(inputs_dir),
        "--output_dir",
        str(open_output_dir),
        "--glob_pattern",
        "*.txt",
        "--batch_size",
        "40000",
        "--summary_length",
        "0.0",
        "--preprocessing_model",
        "en_core_web_trf",
        "--overwrite",
    ]
    commands["open_nyai"] = shlex.join(open_command)
    run_command(open_command, cwd=open_nyai_root, log_path=logs_dir / "open_nyai.log")
    expected_open_combined = open_output_dir / "combined" / f"{args.doc_id}.json"
    if not expected_open_combined.exists():
        copy_existing_open_nyai_outputs(
            open_nyai_root=open_nyai_root,
            open_output_dir=open_output_dir,
            doc_id=args.doc_id,
            log_path=logs_dir / "open_nyai_fallback.log",
        )

    replication_ner_command = [
        "micromamba",
        "run",
        "-n",
        "opennyai_py38",
        "python",
        str(script_root / "scripts" / "run_replication_ner_single.py"),
        "--repo_root",
        str(replication_root / "external" / "legal_NER"),
        "--text_path",
        str(comparison_text_path),
        "--output_path",
        str(replication_ner_dir / f"{args.doc_id}.json"),
        "--model_name",
        "en_legal_ner_trf",
        "--preamble_model_name",
        "en_core_web_sm",
        "--run_type",
        "sent",
    ]
    commands["replication_ner"] = shlex.join(replication_ner_command)
    run_command(replication_ner_command, cwd=comparison_root, log_path=logs_dir / "replication_ner.log")

    rr_input_json = replication_rr_dir / f"{args.doc_id}.input.json"
    rr_processed_json = replication_rr_dir / f"{args.doc_id}.processed.json"
    rr_predictions_json = replication_rr_dir / f"{args.doc_id}.predictions.json"
    rr_runtime_dir = replication_rr_dir / "runtime"
    rr_runtime_dir.mkdir(parents=True, exist_ok=True)
    rr_input_payload = [{"id": args.doc_id, "data": {"text": comparison_text_path.read_text(encoding="utf-8")}}]
    rr_input_json.write_text(json.dumps(rr_input_payload, indent=2), encoding="utf-8")

    rr_prep_command = [
        "micromamba",
        "run",
        "-n",
        "opennyai_py38",
        "python",
        str(replication_root / "external" / "rhetorical-role-baseline" / "infer_data_prep.py"),
        str(rr_input_json),
        str(rr_processed_json),
    ]
    commands["replication_rr_prep"] = shlex.join(rr_prep_command)
    run_command(rr_prep_command, cwd=rr_runtime_dir, log_path=logs_dir / "replication_rr_prep.log")

    rr_infer_command = [
        "micromamba",
        "run",
        "-n",
        "opennyai_rr_paper_py38",
        "python",
        str(replication_root / "external" / "rhetorical-role-baseline" / "infer_new.py"),
        str(rr_processed_json),
        str(rr_predictions_json),
        str(replication_root / "models" / "rr" / "model.pt"),
    ]
    commands["replication_rr_infer"] = shlex.join(rr_infer_command)
    run_command(rr_infer_command, cwd=rr_runtime_dir, log_path=logs_dir / "replication_rr_infer.log")

    summarize_command = [
        "python3",
        str(script_root / "scripts" / "summarize_comparison.py"),
        "--doc_id",
        args.doc_id,
        "--workspace_root",
        str(comparison_root),
    ]
    commands["summarize"] = shlex.join(summarize_command)
    run_command(summarize_command, cwd=comparison_root, log_path=logs_dir / "summarize.log")

    write_text(reports_dir / "commands.txt", "\n".join(f"{name}: {value}" for name, value in commands.items()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
