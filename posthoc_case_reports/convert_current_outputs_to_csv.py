#!/usr/bin/env python3
"""Convert current Graph_Analyser and multi-hearing outputs into CSV reports.

This script intentionally reads only the current output directories:
  - Graph_Analyser/outputs
  - section_GNN/multi_hearing_stage_test/outputs

CSV reports are written into two current-output subfolders:
  - posthoc_case_reports/analysis
  - posthoc_case_reports/timeline_merger

It does not read posthoc_case_reports/old.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-")
DEFAULT_GRAPH_ROOT = REPO_ROOT / "Graph_Analyser" / "outputs"
DEFAULT_STAGE_ROOT = REPO_ROOT / "section_GNN" / "multi_hearing_stage_test" / "outputs"
DEFAULT_REPORT_DIR = REPO_ROOT / "posthoc_case_reports"

STALE_ROOT_CSVS = {
    "aggregate_metrics.csv",
    "conversion_summary.csv",
    "graph_embedding_neighbours.csv",
    "graph_explanations.csv",
    "graph_misclassification_diagnostics.csv",
    "graph_predictions.csv",
    "output_inventory.csv",
    "overall_outputs_analysis.csv",
    "stage_case_factors.csv",
    "stage_decisive_factors_long.csv",
    "stage_predictions.csv",
    "stage_raw_outcome_factors.csv",
    "stage_transitions.csv",
}

OVERALL_COLUMNS = [
    "source_system",
    "record_type",
    "run_id",
    "case_id",
    "base_case_id",
    "node_index",
    "stage_index",
    "category",
    "split",
    "target_label",
    "predicted_label",
    "confidence",
    "correct",
    "transition",
    "changed_prediction",
    "final_pred_correct",
    "top_items",
    "summary_json",
    "source_file",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def label_to_outcome(value: Any) -> str:
    text = str(value)
    if text == "-1":
        return "LOSE"
    if text == "1":
        return "WIN"
    return text


def case_category(case_id: str) -> str:
    return case_id.split("__", 1)[0] if "__" in case_id else ""


def short_items(items: Iterable[Any], limit: int = 5) -> str:
    parts: list[str] = []
    for item in items:
        if len(parts) >= limit:
            break
        if isinstance(item, dict):
            label = item.get("label") or item.get("node_type") or item.get("rank") or ""
            text = item.get("entity") or item.get("text") or item.get("case_id") or ""
            score = (
                item.get("discriminative_score")
                if item.get("discriminative_score") is not None
                else item.get("importance", item.get("cosine_similarity", ""))
            )
            if score != "":
                parts.append(f"{label}:{text} ({score})")
            else:
                parts.append(f"{label}:{text}")
        else:
            parts.append(str(item))
    return "; ".join(parts)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def graph_run_dirs(graph_root: Path) -> list[Path]:
    if not graph_root.exists():
        return []
    return sorted(path for path in graph_root.iterdir() if path.is_dir())


def ingest_graph_predictions(run_dir: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = run_dir / "phase1_2_inference" / "predictions.csv"
    rows = read_csv_dicts(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        target = row.get("target_label", "")
        pred = row.get("pred_label", "")
        item = {
            "source_system": "Graph_Analyser",
            "record_type": "graph_prediction",
            "run_id": run_dir.name,
            "case_id": row.get("case_id", ""),
            "base_case_id": row.get("case_id", ""),
            "node_index": row.get("node_index", ""),
            "stage_index": "",
            "category": case_category(row.get("case_id", "")),
            "split": row.get("split", ""),
            "target_label": target,
            "predicted_label": pred,
            "confidence": row.get("confidence", ""),
            "correct": str(target == pred) if target and pred else "",
            "transition": "",
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": "",
            "summary_json": compact_json(
                {
                    "raw_label": row.get("raw_label"),
                    "target_index": row.get("target_index"),
                    "pred_index": row.get("pred_index"),
                    "prob_class_0": row.get("prob_class_0"),
                    "prob_class_1": row.get("prob_class_1"),
                    "file_name": row.get("file_name"),
                }
            ),
            "source_file": str(path),
        }
        overall_rows.append(item)
        out.append(item)
    return out


def ingest_graph_explanations(run_dir: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "phase4_explanations" / "cases").glob("case_*.json")):
        data = read_json(path)
        top_nodes = data.get("top_nodes", {})
        top_parts = []
        for node_type in ("statute", "provision", "precedent"):
            for item in top_nodes.get(node_type, [])[:3]:
                top_parts.append(
                    {
                        "label": node_type,
                        "entity": item.get("text", ""),
                        "importance": item.get("importance", ""),
                    }
                )
        row = {
            "source_system": "Graph_Analyser",
            "record_type": "graph_explanation",
            "run_id": run_dir.name,
            "case_id": data.get("case_id", ""),
            "base_case_id": data.get("case_id", ""),
            "node_index": data.get("case_node_index", ""),
            "stage_index": "",
            "category": case_category(data.get("case_id", "")),
            "split": "",
            "target_label": data.get("target_label", ""),
            "predicted_label": data.get("predicted_label", ""),
            "confidence": data.get("confidence", ""),
            "correct": str(data.get("target_label") == data.get("predicted_label")),
            "transition": "",
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": short_items(top_parts, limit=9),
            "summary_json": compact_json(
                {
                    "raw_label": data.get("raw_label"),
                    "class_probabilities": data.get("class_probabilities"),
                    "num_hops": data.get("num_hops"),
                    "top_nodes_count": {key: len(value) for key, value in top_nodes.items()},
                }
            ),
            "source_file": str(path),
        }
        overall_rows.append(row)
        rows.append(row)
    return rows


def ingest_graph_misclass(run_dir: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "phase6_misclass_diagnostic").glob("case_*.json")):
        data = read_json(path)
        weighted = data.get("weighted_evidence", {})
        per_node = data.get("per_node", [])
        row = {
            "source_system": "Graph_Analyser",
            "record_type": "graph_misclass_diagnostic",
            "run_id": run_dir.name,
            "case_id": data.get("case_id", ""),
            "base_case_id": data.get("case_id", ""),
            "node_index": data.get("case_node_index", ""),
            "stage_index": "",
            "category": case_category(data.get("case_id", "")),
            "split": "",
            "target_label": data.get("target_label", ""),
            "predicted_label": data.get("predicted_label", ""),
            "confidence": data.get("confidence", ""),
            "correct": str(not data.get("misclassified")) if data.get("misclassified") is not None else "",
            "transition": "",
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": short_items(per_node, limit=5),
            "summary_json": compact_json(
                {
                    "misclassified": data.get("misclassified"),
                    "diagnostic_scope": data.get("diagnostic_scope"),
                    "weighted_evidence": weighted,
                    "top_k_cutoffs": data.get("top_k_cutoffs"),
                }
            ),
            "source_file": str(path),
        }
        overall_rows.append(row)
        rows.append(row)
    return rows


def ingest_graph_topk(run_dir: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "phase7_topk_embedding").glob("case_*.json")):
        data = read_json(path)
        neighbours = data.get("embedding_neighbours", {}).get("neighbours", [])
        row = {
            "source_system": "Graph_Analyser",
            "record_type": "graph_embedding_neighbours",
            "run_id": run_dir.name,
            "case_id": data.get("case_id", ""),
            "base_case_id": data.get("case_id", ""),
            "node_index": data.get("case_node_index", ""),
            "stage_index": "",
            "category": case_category(data.get("case_id", "")),
            "split": "",
            "target_label": data.get("target_label", ""),
            "predicted_label": data.get("predicted_label", ""),
            "confidence": data.get("confidence", ""),
            "correct": str(data.get("target_label") == data.get("predicted_label")),
            "transition": "",
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": short_items(neighbours, limit=10),
            "summary_json": compact_json({"nearest_k": data.get("embedding_neighbours", {}).get("nearest_k")}),
            "source_file": str(path),
        }
        overall_rows.append(row)
        rows.append(row)
    return rows


def ingest_stage_predictions(stage_root: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = stage_root / "inference" / "predictions.csv"
    rows = read_csv_dicts(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        target = row.get("target_label", "")
        pred = row.get("pred_label", "")
        item = {
            "source_system": "multi_hearing_stage_test",
            "record_type": "stage_prediction",
            "run_id": "current",
            "case_id": row.get("case_id", ""),
            "base_case_id": row.get("base_case_id", ""),
            "node_index": row.get("node_index", ""),
            "stage_index": row.get("stage_index", ""),
            "category": case_category(row.get("base_case_id", "")),
            "split": "",
            "target_label": label_to_outcome(target),
            "predicted_label": label_to_outcome(pred),
            "confidence": row.get("confidence", ""),
            "correct": str(target == pred) if target and pred else "",
            "transition": "",
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": "",
            "summary_json": compact_json(
                {
                    "raw_label": row.get("raw_label"),
                    "target_index": row.get("target_index"),
                    "pred_index": row.get("pred_index"),
                    "prob_class_0": row.get("prob_class_0"),
                    "prob_class_1": row.get("prob_class_1"),
                }
            ),
            "source_file": str(path),
        }
        overall_rows.append(item)
        out.append(item)
    return out


def ingest_stage_transitions(stage_root: Path, overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = stage_root / "analysis" / "stage_transitions.csv"
    rows = read_csv_dicts(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        stage_summary = {key: value for key, value in row.items() if key.startswith("stage") and value}
        item = {
            "source_system": "multi_hearing_stage_test",
            "record_type": "stage_transition",
            "run_id": "current",
            "case_id": row.get("base_case_id", ""),
            "base_case_id": row.get("base_case_id", ""),
            "node_index": "",
            "stage_index": "",
            "category": row.get("category", ""),
            "split": row.get("outcome_split", ""),
            "target_label": row.get("true_label", ""),
            "predicted_label": "",
            "confidence": "",
            "correct": row.get("final_pred_correct", ""),
            "transition": row.get("transition", ""),
            "changed_prediction": row.get("changed_prediction", ""),
            "final_pred_correct": row.get("final_pred_correct", ""),
            "top_items": "",
            "summary_json": compact_json({"n_stages": row.get("n_stages"), "stages": stage_summary}),
            "source_file": str(path),
        }
        overall_rows.append(item)
        out.append(item)
    return out


def ingest_factor_dir(
    factor_dir: Path,
    record_type: str,
    overall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(factor_dir.glob("*.json")):
        data = read_json(path)
        factors = data.get("top_decisive_factors", [])
        row = {
            "source_system": "multi_hearing_stage_test",
            "record_type": record_type,
            "run_id": "current",
            "case_id": data.get("base_case_id", ""),
            "base_case_id": data.get("base_case_id", ""),
            "node_index": "",
            "stage_index": "",
            "category": data.get("category", ""),
            "split": data.get("outcome_split", ""),
            "target_label": data.get("true_label", ""),
            "predicted_label": "",
            "confidence": "",
            "correct": "",
            "transition": data.get("transition") or data.get("prediction_transition", ""),
            "changed_prediction": "",
            "final_pred_correct": "",
            "top_items": short_items(factors, limit=8),
            "summary_json": compact_json(
                {
                    "factor_basis": data.get("factor_basis"),
                    "raw_outcome_transition": data.get("raw_outcome_transition"),
                    "contrast_transition": data.get("contrast_transition"),
                    "n_transition_cases": data.get("n_transition_cases"),
                    "n_contrast_cases": data.get("n_contrast_cases"),
                    "n_stages": data.get("n_stages"),
                    "section_sentence_delta_first_to_last": data.get("section_sentence_delta_first_to_last"),
                    "n_new_decision_role_sentences": len(data.get("new_decision_role_sentences", [])),
                    "n_anchor_sentences": len(data.get("anchor_sentences", [])),
                }
            ),
            "source_file": str(path),
        }
        overall_rows.append(row)
        rows.append(row)
    return rows


def flatten_stage_factors(factor_dir: Path, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(factor_dir.glob("*.json")):
        data = read_json(path)
        base = {
            "source_name": source_name,
            "base_case_id": data.get("base_case_id", ""),
            "category": data.get("category", ""),
            "outcome_split": data.get("outcome_split", ""),
            "transition": data.get("transition") or data.get("prediction_transition", ""),
            "raw_outcome_transition": data.get("raw_outcome_transition", ""),
            "true_label": data.get("true_label", ""),
            "source_file": str(path),
        }
        for rank, factor in enumerate(data.get("top_decisive_factors", []), start=1):
            rows.append(
                {
                    **base,
                    "factor_rank": rank,
                    "factor_label": factor.get("label", ""),
                    "factor_entity": factor.get("entity", ""),
                    "discriminative_score": factor.get("discriminative_score", ""),
                }
            )
    return rows


def flatten_aggregate_json(path: Path, source_system: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, subvalue in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(next_prefix, subvalue)
        elif isinstance(value, list):
            rows.append(
                {
                    "source_system": source_system,
                    "file": str(path),
                    "metric": prefix,
                    "value": compact_json(value),
                }
            )
        else:
            rows.append(
                {
                    "source_system": source_system,
                    "file": str(path),
                    "metric": prefix,
                    "value": value,
                }
            )

    walk("", data)
    return rows


def build_inventory(paths: Iterable[Path], report_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "root": str(root),
                        "relative_path": str(path.relative_to(root)),
                        "suffix": path.suffix,
                        "size_bytes": path.stat().st_size,
                        "source_file": str(path),
                    }
                )
    for path in sorted(report_dir.rglob("*.csv")):
        rows.append(
            {
                "root": str(report_dir),
                "relative_path": str(path.relative_to(report_dir)),
                "suffix": path.suffix,
                "size_bytes": path.stat().st_size,
                "source_file": str(path),
            }
        )
    return rows


def write_summary(path: Path, overall_rows: list[dict[str, Any]]) -> None:
    counts = Counter((row["source_system"], row["record_type"]) for row in overall_rows)
    rows = [
        {"source_system": source, "record_type": record_type, "n_rows": count}
        for (source, record_type), count in sorted(counts.items())
    ]
    write_rows(path, rows, ["source_system", "record_type", "n_rows"])


def remove_stale_root_csvs(report_dir: Path) -> None:
    for name in STALE_ROOT_CSVS:
        path = report_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def convert(graph_root: Path, stage_root: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = report_dir / "analysis"
    timeline_dir = report_dir / "timeline_merger"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_root_csvs(report_dir)

    graph_overall_rows: list[dict[str, Any]] = []
    timeline_overall_rows: list[dict[str, Any]] = []
    graph_prediction_rows: list[dict[str, Any]] = []
    graph_explanation_rows: list[dict[str, Any]] = []
    graph_misclass_rows: list[dict[str, Any]] = []
    graph_topk_rows: list[dict[str, Any]] = []

    for run_dir in graph_run_dirs(graph_root):
        graph_prediction_rows.extend(ingest_graph_predictions(run_dir, graph_overall_rows))
        graph_explanation_rows.extend(ingest_graph_explanations(run_dir, graph_overall_rows))
        graph_misclass_rows.extend(ingest_graph_misclass(run_dir, graph_overall_rows))
        graph_topk_rows.extend(ingest_graph_topk(run_dir, graph_overall_rows))

    stage_prediction_rows = ingest_stage_predictions(stage_root, timeline_overall_rows)
    stage_transition_rows = ingest_stage_transitions(stage_root, timeline_overall_rows)
    stage_factor_rows = ingest_factor_dir(
        stage_root / "analysis" / "per_case_factors",
        "stage_case_factors",
        timeline_overall_rows,
    )
    raw_factor_rows = ingest_factor_dir(
        stage_root / "analysis" / "per_case_raw_outcome_factors",
        "stage_raw_outcome_factors",
        timeline_overall_rows,
    )

    write_rows(analysis_dir / "analysis_overall_outputs.csv", graph_overall_rows, OVERALL_COLUMNS)
    write_rows(analysis_dir / "graph_predictions.csv", graph_prediction_rows, OVERALL_COLUMNS)
    write_rows(analysis_dir / "graph_explanations.csv", graph_explanation_rows, OVERALL_COLUMNS)
    write_rows(analysis_dir / "graph_misclassification_diagnostics.csv", graph_misclass_rows, OVERALL_COLUMNS)
    write_rows(analysis_dir / "graph_embedding_neighbours.csv", graph_topk_rows, OVERALL_COLUMNS)

    write_rows(timeline_dir / "timeline_overall_outputs.csv", timeline_overall_rows, OVERALL_COLUMNS)
    write_rows(timeline_dir / "stage_predictions.csv", stage_prediction_rows, OVERALL_COLUMNS)
    write_rows(timeline_dir / "stage_transitions.csv", stage_transition_rows, OVERALL_COLUMNS)
    write_rows(timeline_dir / "stage_case_factors.csv", stage_factor_rows, OVERALL_COLUMNS)
    write_rows(timeline_dir / "stage_raw_outcome_factors.csv", raw_factor_rows, OVERALL_COLUMNS)

    all_factor_rows = flatten_stage_factors(stage_root / "analysis" / "per_case_factors", "prediction_factors")
    all_factor_rows.extend(
        flatten_stage_factors(stage_root / "analysis" / "per_case_raw_outcome_factors", "raw_outcome_factors")
    )
    write_rows(timeline_dir / "stage_decisive_factors_long.csv", all_factor_rows)

    timeline_aggregate_rows: list[dict[str, Any]] = []
    timeline_aggregate_rows.extend(
        flatten_aggregate_json(stage_root / "analysis" / "summary.json", "multi_hearing_stage_test")
    )
    timeline_aggregate_rows.extend(
        flatten_aggregate_json(stage_root / "analysis" / "transition_counts.json", "multi_hearing_stage_test")
    )
    timeline_aggregate_rows.extend(
        flatten_aggregate_json(stage_root / "analysis" / "transition_aggregates.json", "multi_hearing_stage_test")
    )
    timeline_aggregate_rows.extend(
        flatten_aggregate_json(
            stage_root / "analysis" / "raw_outcome_transition_aggregates.json",
            "multi_hearing_stage_test",
        )
    )
    write_rows(timeline_dir / "timeline_aggregate_metrics.csv", timeline_aggregate_rows)

    graph_aggregate_rows: list[dict[str, Any]] = []
    for run_dir in graph_run_dirs(graph_root):
        graph_aggregate_rows.extend(
            flatten_aggregate_json(run_dir / "phase1_2_inference" / "summary.json", "Graph_Analyser")
        )
    write_rows(analysis_dir / "analysis_aggregate_metrics.csv", graph_aggregate_rows)

    write_summary(analysis_dir / "analysis_conversion_summary.csv", graph_overall_rows)
    write_summary(timeline_dir / "timeline_conversion_summary.csv", timeline_overall_rows)
    write_rows(analysis_dir / "current_output_inventory.csv", build_inventory([graph_root, stage_root], report_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.graph_root, args.stage_root, args.report_dir)
    print(f"Wrote analysis CSVs to: {args.report_dir / 'analysis'}")
    print(f"Wrote timeline merger CSVs to: {args.report_dir / 'timeline_merger'}")


if __name__ == "__main__":
    main()
