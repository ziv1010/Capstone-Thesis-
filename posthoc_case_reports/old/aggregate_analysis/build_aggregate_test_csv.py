#!/usr/bin/env python3
"""Build the aggregate test-set CSV requested for posthoc analysis.

Reads existing Graph_Analyser outputs only:
  - phase1_2_inference/predictions.csv
  - phase4_explanations/cases/case_<node_index>.json

The LLM stage is intentionally ignored. The generated CSV has one row per test
case with bucket, prediction, actual label, and important graph nodes grouped
by legal node type.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_ANALYSER_ROOT = PROJECT_ROOT / "Graph_Analyser"
DEFAULT_PREDICTIONS = GRAPH_ANALYSER_ROOT / "outputs/phase1_2_inference/predictions.csv"
DEFAULT_CASES_DIR = GRAPH_ANALYSER_ROOT / "outputs/phase4_explanations/cases"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "aggregate_test_set_analysis.csv"

BUCKET_ALIASES = {
    "land_property": "land",
    "fin_fraud": "financial",
    "family_matrimonial": "family_matrimonial",
    "motor_accidents": "motor_accidents",
    "sexual_offences": "sexual_offences",
    "food_safety": "food_safety",
}

NODE_GROUPS = {
    "statutes_significant": ("statute",),
    "provisions_significant": ("provision",),
    "precedents_significant": ("precedent",),
    "case_ids_significant": ("case",),
    "judges_significant": ("judge",),
    "courts_significant": ("court",),
    "lawyers_significant": ("lawyer", "petitioner_lawyer", "defence_lawyer"),
    "petitioners_significant": ("petitioner",),
    "respondents_significant": ("respondent",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", default="test", help="Use 'all' to disable split filtering.")
    parser.add_argument("--top-k-per-type", type=int, default=5)
    parser.add_argument("--top-k-all", type=int, default=30)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def pretty_label(value: object) -> str:
    text = clean_text(value)
    if text in {"1", "1.0", "WIN"}:
        return "WIN (1)"
    if text in {"-1", "-1.0", "LOSE", "LOSS"}:
        return "LOSE (-1)"
    if text in {"0", "0.0", "POSTPONED"}:
        return "POSTPONED (0)"
    return text


def bucket_raw(case_id: str, file_name: str = "") -> str:
    source = clean_text(case_id) or clean_text(file_name)
    if source.startswith("STAGE") and "__" in source:
        source = source.split("__", 1)[1]
    if "__" in source:
        return source.split("__", 1)[0]
    return source.split("_", 1)[0] if source else ""


def bucket_display(raw_bucket: str) -> str:
    return BUCKET_ALIASES.get(raw_bucket, raw_bucket)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def node_score(node: dict) -> float:
    for key in ("importance", "score", "discriminative_score"):
        value = node.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def format_node(node: dict) -> str:
    text = clean_text(node.get("text") or node.get("entity") or node.get("case_id"))
    score = node_score(node)
    idx = node.get("node_index")
    suffix = f"score={score:.4f}"
    if idx is not None:
        suffix += f"; idx={idx}"
    return f"{text} [{suffix}]" if text else ""


def raw_nodes(explanation: dict, node_type: str) -> list[dict]:
    top_nodes = explanation.get("top_nodes", {}) or {}
    top_graph_nodes = explanation.get("top_graph_nodes", {}) or {}
    nodes = top_nodes.get(node_type)
    if nodes:
        return list(nodes)
    return list(top_graph_nodes.get(node_type, []) or [])


def collect_nodes(explanation: dict, node_types: Iterable[str], top_k: int) -> list[str]:
    collected: list[dict] = []
    for node_type in node_types:
        collected.extend(raw_nodes(explanation, node_type))
    collected.sort(key=node_score, reverse=True)

    seen: set[str] = set()
    formatted: list[str] = []
    for node in collected:
        text = clean_text(node.get("text") or node.get("entity") or node.get("case_id"))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        formatted_node = format_node(node)
        if formatted_node:
            formatted.append(formatted_node)
        if len(formatted) >= top_k:
            break
    return formatted


def collect_all_significant(explanation: dict, top_k_all: int) -> str:
    entries: list[tuple[float, str, str, str]] = []
    for column, node_types in NODE_GROUPS.items():
        label = column.replace("_significant", "")
        for node_type in node_types:
            for node in raw_nodes(explanation, node_type):
                raw_text = clean_text(node.get("text") or node.get("entity") or node.get("case_id"))
                formatted = format_node(node)
                if formatted and raw_text:
                    entries.append((node_score(node), label, raw_text.lower(), f"{label}: {formatted}"))
    entries.sort(key=lambda item: item[0], reverse=True)

    seen: set[str] = set()
    out: list[str] = []
    for _, label, raw_text, text in entries:
        key = f"{label}:{raw_text}"
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= top_k_all:
            break
    return "; ".join(out)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    cases_dir = Path(args.cases_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv(predictions_path)
    if args.split.lower() != "all" and rows and "split" in rows[0]:
        rows = [row for row in rows if clean_text(row.get("split")) == args.split]

    output_rows: list[dict[str, object]] = []
    for row in rows:
        node_index = clean_text(row.get("node_index"))
        explanation_path = cases_dir / f"case_{node_index}.json"
        explanation = load_json(explanation_path)
        raw_bucket = bucket_raw(row.get("case_id", ""), row.get("file_name", ""))
        prediction = pretty_label(row.get("pred_label"))
        actual = pretty_label(row.get("target_label") or row.get("raw_label"))

        out = {
            "case_node_index": node_index,
            "case_id": clean_text(row.get("case_id")),
            "file_name": clean_text(row.get("file_name")),
            "bucket": bucket_display(raw_bucket),
            "bucket_raw": raw_bucket,
            "prediction": prediction,
            "actual": actual,
            "pred_label": clean_text(row.get("pred_label")),
            "target_label": clean_text(row.get("target_label")),
            "raw_label": clean_text(row.get("raw_label")),
            "confidence": clean_text(row.get("confidence")),
            "correct": clean_text(row.get("pred_label")) == clean_text(row.get("target_label")),
            "prob_class_0": clean_text(row.get("prob_class_0")),
            "prob_class_1": clean_text(row.get("prob_class_1")),
            "explanation_json": str(explanation_path) if explanation else "",
            "explanation_missing": not bool(explanation),
        }
        for column, node_types in NODE_GROUPS.items():
            out[column] = "; ".join(collect_nodes(explanation, node_types, args.top_k_per_type)) if explanation else ""
        out["nodes_found_significant_for_prediction"] = collect_all_significant(explanation, args.top_k_all) if explanation else ""
        output_rows.append(out)

    fieldnames = [
        "case_node_index",
        "case_id",
        "file_name",
        "bucket",
        "bucket_raw",
        "prediction",
        "actual",
        "pred_label",
        "target_label",
        "raw_label",
        "confidence",
        "correct",
        "prob_class_0",
        "prob_class_1",
        "statutes_significant",
        "provisions_significant",
        "precedents_significant",
        "case_ids_significant",
        "judges_significant",
        "courts_significant",
        "lawyers_significant",
        "petitioners_significant",
        "respondents_significant",
        "nodes_found_significant_for_prediction",
        "explanation_json",
        "explanation_missing",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    missing = sum(1 for row in output_rows if row["explanation_missing"])
    print(f"Wrote {len(output_rows)} rows -> {output_path}")
    print(f"Missing explanation JSONs: {missing}")


if __name__ == "__main__":
    main()
