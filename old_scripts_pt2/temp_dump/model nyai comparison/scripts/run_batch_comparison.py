#!/usr/bin/env python3
"""Run the isolated OpenNyai vs replication comparison for multiple documents."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


DEFAULT_DOC_IDS = [
    "Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022",
    "Dharampal_Satyapal_Ltd_vs_The_State_Of_Maharashtra_And_3_Ors_on_30_March_2022",
    "Pravin_Prakash_Kadam_vs_State_Of_Maharashtra_on_28_February_2022",
    "Swami_Muktanand_Anr_vs_State_Of_Nct_Of_Delhi_on_21_March_2022",
    "Jagdish_Dhakad_vs_The_State_Of_Madhya_Pradesh_on_21_February_2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc_ids", nargs="*", default=DEFAULT_DOC_IDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_markdown(summary: dict) -> str:
    aggregate = summary["aggregate"]
    lines = []
    lines.append("# Batch Comparison Summary")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for doc_id in summary["doc_ids"]:
        lines.append(f"- `{doc_id}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Documents run: {aggregate['doc_count']}")
    lines.append(
        f"- Average exact-span NER jaccard: {pct(aggregate['average_ner_exact_jaccard'])}"
    )
    lines.append(
        f"- Average normalized NER overlap vs OpenNyai unique entities: {pct(aggregate['average_ner_normalized_overlap_open'])}"
    )
    lines.append(
        f"- Average normalized NER overlap vs replication unique entities: {pct(aggregate['average_ner_normalized_overlap_replication'])}"
    )
    lines.append(
        f"- Average aligned RR label agreement: {pct(aggregate['average_rr_label_agreement'])}"
    )
    lines.append(
        f"- Average RR alignment coverage: {pct(aggregate['average_rr_alignment_coverage'])}"
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "- The pretrained NER and rhetorical-role model weights appear to be the same, but the two pipelines are not output-equivalent."
    )
    lines.append(
        "- If you need paper-faithful reproduction or want outputs closest to the original repos, use `REPLICATION_OpenNyai`."
    )
    lines.append(
        "- If you want the integrated library pipeline and summarizer, use `OpenNyai`, but do not treat its JSON as interchangeable with the replication path."
    )
    lines.append("")
    lines.append("## Per Document")
    lines.append("")
    for item in summary["documents"]:
        lines.append(f"### {item['doc_id']}")
        lines.append("")
        lines.append(
            f"- NER exact shared: {item['ner']['shared_count']} / {item['ner']['open_nyai_count']} OpenNyai, {item['ner']['replication_count']} replication"
        )
        lines.append(
            f"- NER normalized overlap: {item['ner']['normalized_shared_count']} shared, {item['ner']['normalized_open_only_count']} OpenNyai-only, {item['ner']['normalized_replication_only_count']} replication-only"
        )
        lines.append(
            f"- RR aligned agreement: {item['rr']['matching_label_count']} / {item['rr']['comparable_sentence_count']}"
        )
        lines.append(
            f"- RR alignment coverage: {pct(item['metrics']['rr_alignment_coverage'])}"
        )
        lines.append(f"- Report: {item['report_path']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parents[1]
    project_root = script_root.parent
    runs_root = script_root / "runs"
    batch_reports_dir = script_root / "batch_reports"
    batch_reports_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    document_summaries = []

    for doc_id in args.doc_ids:
        workspace_root = runs_root / doc_id
        command = [
            "python3",
            str(script_root / "scripts" / "run_comparison.py"),
            "--doc_id",
            doc_id,
            "--workspace_root",
            str(workspace_root),
            "--project_root",
            str(project_root),
        ]
        if args.overwrite:
            command.append("--overwrite")
        commands.append(shlex.join(command))
        run_command(command, cwd=script_root, log_path=workspace_root / "batch_run.log")

        summary_path = workspace_root / "outputs" / "reports" / "comparison_summary.json"
        summary = load_json(summary_path)
        ner = summary["ner"]
        rr = summary["rr"]
        open_normalized_unique = ner["normalized_shared_count"] + ner["normalized_open_only_count"]
        replication_normalized_unique = (
            ner["normalized_shared_count"] + ner["normalized_replication_only_count"]
        )
        exact_union = ner["open_nyai_count"] + ner["replication_count"] - ner["shared_count"]
        rr_alignment_base = max(rr["open_nyai_sentence_count"], rr["replication_sentence_count"])
        summary["metrics"] = {
            "ner_exact_jaccard": safe_div(ner["shared_count"], exact_union),
            "ner_normalized_overlap_open": safe_div(
                ner["normalized_shared_count"], open_normalized_unique
            ),
            "ner_normalized_overlap_replication": safe_div(
                ner["normalized_shared_count"], replication_normalized_unique
            ),
            "rr_label_agreement": safe_div(
                rr["matching_label_count"], rr["comparable_sentence_count"]
            ),
            "rr_alignment_coverage": safe_div(rr["comparable_sentence_count"], rr_alignment_base),
        }
        summary["report_path"] = str(workspace_root / "outputs" / "reports" / "comparison_summary.md")
        document_summaries.append(summary)

    aggregate = {
        "doc_count": len(document_summaries),
        "average_ner_exact_jaccard": safe_div(
            sum(item["metrics"]["ner_exact_jaccard"] for item in document_summaries),
            len(document_summaries),
        ),
        "average_ner_normalized_overlap_open": safe_div(
            sum(item["metrics"]["ner_normalized_overlap_open"] for item in document_summaries),
            len(document_summaries),
        ),
        "average_ner_normalized_overlap_replication": safe_div(
            sum(item["metrics"]["ner_normalized_overlap_replication"] for item in document_summaries),
            len(document_summaries),
        ),
        "average_rr_label_agreement": safe_div(
            sum(item["metrics"]["rr_label_agreement"] for item in document_summaries),
            len(document_summaries),
        ),
        "average_rr_alignment_coverage": safe_div(
            sum(item["metrics"]["rr_alignment_coverage"] for item in document_summaries),
            len(document_summaries),
        ),
    }

    batch_summary = {
        "doc_ids": args.doc_ids,
        "paths": {
            "comparison_root": str(script_root),
            "runs_root": str(runs_root),
        },
        "aggregate": aggregate,
        "documents": document_summaries,
    }

    write_text(batch_reports_dir / "batch_commands.txt", "\n".join(commands) + "\n")
    write_text(
        batch_reports_dir / "batch_summary.json",
        json.dumps(batch_summary, indent=2),
    )
    write_text(batch_reports_dir / "batch_summary.md", build_markdown(batch_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
