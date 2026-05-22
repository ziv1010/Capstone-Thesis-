#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("-1", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine InLegalLLaMA shard predictions and compute metrics.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def read_rows(predictions_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(predictions_dir.glob("predictions*.jsonl")):
        if path.name == "combined_predictions.jsonl":
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    rows.append(
                        {
                            "status": "error",
                            "source_file": str(path),
                            "line_number": line_number,
                            "error": f"JSONDecodeError: {exc}",
                        }
                    )
                    continue
                row["source_file"] = str(path)
                rows.append(row)
    rows.sort(key=lambda item: (str(item.get("case_id", "")), int(item.get("shard_index") or 0)))
    return rows


def f1_for_label(rows: list[dict[str, Any]], label: str) -> dict[str, float | int]:
    tp = sum(1 for row in rows if row.get("true_label") == label and row.get("predicted_label") == label)
    fp = sum(1 for row in rows if row.get("true_label") != label and row.get("predicted_label") == label)
    fn = sum(1 for row in rows if row.get("true_label") == label and row.get("predicted_label") != label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scorable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("true_label") in LABELS
        and row.get("predicted_label") in LABELS
    ]
    correct = sum(1 for row in scorable if row.get("true_label") == row.get("predicted_label"))
    per_label = {label: f1_for_label(scorable, label) for label in LABELS}
    macro_f1 = sum(per_label[label]["f1"] for label in LABELS) / len(LABELS) if LABELS else 0.0
    return {
        "total_rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "error_rows": sum(1 for row in rows if row.get("status") == "error"),
        "parsed_rows": len(scorable),
        "unparsed_ok_rows": sum(
            1
            for row in rows
            if row.get("status") == "ok" and row.get("predicted_label") not in LABELS
        ),
        "accuracy": correct / len(scorable) if scorable else None,
        "macro_f1": macro_f1 if scorable else None,
        "true_label_distribution": dict(Counter(str(row.get("true_label")) for row in rows)),
        "predicted_label_distribution": dict(Counter(str(row.get("predicted_label")) for row in rows)),
        "per_label": per_label,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    combined_jsonl = args.combined_jsonl or args.predictions_dir / "combined_predictions.jsonl"
    csv_path = args.csv or args.predictions_dir / "combined_predictions.csv"
    metrics_path = args.metrics or args.predictions_dir / "metrics.json"

    with combined_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "case_id",
        "file_name",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "status",
        "shard_index",
        "elapsed_seconds",
        "raw_output",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            projected = {field: row.get(field) for field in fields}
            projected["correct"] = (
                row.get("true_label") == row.get("predicted_label")
                if row.get("true_label") in LABELS and row.get("predicted_label") in LABELS
                else ""
            )
            writer.writerow(projected)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({"metrics": str(metrics_path), **metrics}, indent=2))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.predictions_dir)
    metrics = build_metrics(rows)
    write_outputs(args, rows, metrics)


if __name__ == "__main__":
    main()
