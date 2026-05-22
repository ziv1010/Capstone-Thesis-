#!/usr/bin/env python3
"""Build early-detection CSVs for multi-hearing cases.

Outputs:
  - multi_hearing_case_paths.csv: one row per case with Hearing 1..N path.
  - multi_hearing_hearing_level_analysis.csv: one row per hearing using the
    aggregate-analysis structure: bucket, prediction, actual, and important
    legal nodes.
  - multi_hearing_summary.json: counts and coverage metadata.

If stage-specific Graph_Analyser Phase 4 outputs exist, this script uses those
for true prediction-level significant nodes. Otherwise it falls back to the
existing per-case raw-outcome factor reports, then stage entities if needed;
the `significance_source` column records which source was used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "section_GNN/multi_hearing_stage_test"
DEFAULT_PREDICTIONS = EXP_ROOT / "outputs/inference/predictions.csv"
DEFAULT_MANIFEST = EXP_ROOT / "outputs/stage_manifest.json"
DEFAULT_INPUT_JSONS = EXP_ROOT / "data/input_jsons"
DEFAULT_DIFFS_DIR = EXP_ROOT / "outputs/analysis/per_case_diffs"
DEFAULT_RAW_FACTORS_DIR = EXP_ROOT / "outputs/analysis/per_case_raw_outcome_factors"
DEFAULT_STAGE_EXPLAINER_PREDS = EXP_ROOT / "outputs/explainer/phase1_2_inference/predictions.csv"
DEFAULT_STAGE_EXPLAINER_CASES = EXP_ROOT / "outputs/explainer/phase4_explanations/cases"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

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

ENTITY_TO_COLUMN = {
    "STATUTE": "statutes_significant",
    "PROVISION": "provisions_significant",
    "PRECEDENT": "precedents_significant",
    "CASE_NUMBER": "case_ids_significant",
    "JUDGE": "judges_significant",
    "COURT": "courts_significant",
    "LAWYER": "lawyers_significant",
    "PETITIONER": "petitioners_significant",
    "RESPONDENT": "respondents_significant",
}

RAW_LABELS = {"1": "WIN (1)", "0": "POSTPONED (0)", "-1": "LOSS (-1)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--input-jsons", default=str(DEFAULT_INPUT_JSONS))
    parser.add_argument("--diffs-dir", default=str(DEFAULT_DIFFS_DIR))
    parser.add_argument("--raw-factors-dir", default=str(DEFAULT_RAW_FACTORS_DIR))
    parser.add_argument("--stage-explainer-predictions", default=str(DEFAULT_STAGE_EXPLAINER_PREDS))
    parser.add_argument("--stage-explainer-cases-dir", default=str(DEFAULT_STAGE_EXPLAINER_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k-per-type", type=int, default=5)
    parser.add_argument("--top-k-all", type=int, default=30)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pretty_binary(value: object) -> str:
    text = clean_text(value)
    if text in {"1", "1.0", "WIN"}:
        return "WIN (1)"
    if text in {"-1", "-1.0", "LOSE", "LOSS"}:
        return "LOSE (-1)"
    if text in {"0", "0.0", "POSTPONED"}:
        return "POSTPONED (0)"
    return text


def pretty_raw(value: object) -> str:
    text = clean_text(value)
    return RAW_LABELS.get(text, text)


def parse_stage(case_id: str, fallback: object = "") -> int:
    value = clean_text(fallback)
    if value:
        try:
            return int(float(value))
        except ValueError:
            pass
    match = re.match(r"^STAGE(\d+)__", clean_text(case_id))
    if match:
        return int(match.group(1))
    return 0


def bucket_raw(base_case_id: str) -> str:
    base = clean_text(base_case_id)
    if base.startswith("STAGE") and "__" in base:
        base = base.split("__", 1)[1]
    if "__" in base:
        return base.split("__", 1)[0]
    return base.split("_", 1)[0] if base else ""


def bucket_display(raw_bucket: str) -> str:
    return BUCKET_ALIASES.get(raw_bucket, raw_bucket)


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
    if not text:
        return ""
    score = node_score(node)
    idx = node.get("node_index")
    bits = []
    if score:
        bits.append(f"score={score:.4f}")
    if idx is not None:
        bits.append(f"idx={idx}")
    return f"{text} [{'; '.join(bits)}]" if bits else text


def graph_nodes(explanation: dict, node_type: str) -> list[dict]:
    top_nodes = explanation.get("top_nodes", {}) or {}
    top_graph_nodes = explanation.get("top_graph_nodes", {}) or {}
    nodes = top_nodes.get(node_type)
    if nodes:
        return list(nodes)
    return list(top_graph_nodes.get(node_type, []) or [])


def collect_graph_nodes(explanation: dict, node_types: Iterable[str], top_k: int) -> list[str]:
    nodes: list[dict] = []
    for node_type in node_types:
        nodes.extend(graph_nodes(explanation, node_type))
    nodes.sort(key=node_score, reverse=True)

    seen: set[str] = set()
    out: list[str] = []
    for node in nodes:
        text = clean_text(node.get("text") or node.get("entity") or node.get("case_id"))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(format_node(node))
        if len(out) >= top_k:
            break
    return out


def all_graph_nodes(explanation: dict, top_k_all: int) -> str:
    entries: list[tuple[float, str, str, str]] = []
    for column, node_types in NODE_GROUPS.items():
        label = column.replace("_significant", "")
        for node_type in node_types:
            for node in graph_nodes(explanation, node_type):
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


def empty_node_columns() -> dict[str, str]:
    out = {column: "" for column in NODE_GROUPS}
    out["nodes_found_significant_for_prediction"] = ""
    return out


def factor_nodes(raw_factor: dict, top_k: int, top_k_all: int) -> dict[str, str]:
    columns: dict[str, list[str]] = {column: [] for column in NODE_GROUPS}
    all_items: list[str] = []
    for factor in raw_factor.get("top_decisive_factors", []) or []:
        label = clean_text(factor.get("label")).upper()
        column = ENTITY_TO_COLUMN.get(label)
        if not column:
            continue
        formatted = format_node({"entity": factor.get("entity"), "discriminative_score": factor.get("discriminative_score")})
        if not formatted:
            continue
        if len(columns[column]) < top_k:
            columns[column].append(formatted)
        all_items.append(f"{column.replace('_significant', '')}: {formatted}")

    out = {column: "; ".join(values) for column, values in columns.items()}
    out["nodes_found_significant_for_prediction"] = "; ".join(all_items[:top_k_all])
    return out


def entity_nodes(case_diff: dict, stage_index: int, top_k: int, top_k_all: int) -> dict[str, str]:
    columns: dict[str, list[str]] = {column: [] for column in NODE_GROUPS}
    stages = case_diff.get("stages", []) or []
    stage = stages[stage_index - 1] if 0 < stage_index <= len(stages) else {}
    entities = stage.get("entities_by_label", {}) or {}
    all_items: list[str] = []
    for label, values in entities.items():
        column = ENTITY_TO_COLUMN.get(clean_text(label).upper())
        if not column:
            continue
        seen: set[str] = set()
        for value in values:
            text = clean_text(value)
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            if len(columns[column]) < top_k:
                columns[column].append(text)
            all_items.append(f"{column.replace('_significant', '')}: {text}")

    out = {column: "; ".join(values) for column, values in columns.items()}
    out["nodes_found_significant_for_prediction"] = "; ".join(all_items[:top_k_all])
    return out


def stage_actual(input_jsons: Path, filename: str) -> tuple[str, str]:
    doc = load_json(input_jsons / filename)
    score = clean_text(doc.get("case_outcome_score"))
    label = clean_text(doc.get("case_outcome_label"))
    if not score and not label:
        return "", ""
    return pretty_raw(score), label


def build_stage_explainer_lookup(predictions_path: Path) -> dict[str, str]:
    rows = read_csv(predictions_path)
    return {
        clean_text(row.get("case_id")): clean_text(row.get("node_index"))
        for row in rows
        if clean_text(row.get("case_id")) and clean_text(row.get("node_index"))
    }


def significant_nodes_for_stage(
    stage_case_id: str,
    base_case_id: str,
    stage_index: int,
    explainer_lookup: dict[str, str],
    explainer_cases_dir: Path,
    raw_factor: dict,
    case_diff: dict,
    top_k: int,
    top_k_all: int,
) -> tuple[dict[str, str], str, str]:
    if explainer_lookup and explainer_cases_dir.exists():
        node_index = explainer_lookup.get(stage_case_id)
        explanation_path = explainer_cases_dir / f"case_{node_index}.json" if node_index else None
        if explanation_path and explanation_path.exists():
            explanation = load_json(explanation_path)
            out = {
                column: "; ".join(collect_graph_nodes(explanation, node_types, top_k))
                for column, node_types in NODE_GROUPS.items()
            }
            out["nodes_found_significant_for_prediction"] = all_graph_nodes(explanation, top_k_all)
            return out, "stage_pgexplainer", str(explanation_path)

    if raw_factor:
        return factor_nodes(raw_factor, top_k, top_k_all), "raw_outcome_factor_report", ""
    if case_diff:
        return entity_nodes(case_diff, stage_index, top_k, top_k_all), "stage_entities_only", ""
    return empty_node_columns(), "missing", ""


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    manifest_path = Path(args.manifest)
    input_jsons = Path(args.input_jsons)
    diffs_dir = Path(args.diffs_dir)
    raw_factors_dir = Path(args.raw_factors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_csv(predictions_path)
    pred_by_case_id = {clean_text(row.get("case_id")): row for row in predictions}

    manifest = load_json(manifest_path)
    cases = manifest.get("cases", []) or []
    multi_cases = [case for case in cases if int(case.get("n_stages", 0) or 0) >= 2]
    max_stages = max((int(case.get("n_stages", 0) or 0) for case in multi_cases), default=0)

    explainer_lookup = build_stage_explainer_lookup(Path(args.stage_explainer_predictions))
    explainer_cases_dir = Path(args.stage_explainer_cases_dir)

    case_path_rows: list[dict[str, object]] = []
    hearing_rows: list[dict[str, object]] = []
    source_counts: Counter = Counter()

    for case in multi_cases:
        base_case_id = clean_text(case.get("base_case_id"))
        raw_bucket = bucket_raw(base_case_id)
        bucket = bucket_display(raw_bucket)
        stages = sorted(case.get("stages", []) or [], key=lambda s: int(s.get("stage_index", 0) or 0))
        n_stages = len(stages)
        final_score = clean_text(case.get("final_outcome_score"))
        case_actual_result = pretty_raw(final_score)
        case_actual_label = clean_text(case.get("final_outcome_label"))
        raw_factor = load_json(raw_factors_dir / f"{base_case_id}.json")
        case_diff = load_json(diffs_dir / f"{base_case_id}.json")

        path_row: dict[str, object] = {
            "base_case_id": base_case_id,
            "bucket": bucket,
            "bucket_raw": raw_bucket,
            "outcome_split": clean_text(case.get("outcome_split")),
            "n_hearings": n_stages,
            "case_actual_result": case_actual_result,
            "case_actual_label": case_actual_label,
            "case_actual_score": final_score,
        }

        pred_path_parts: list[str] = []
        raw_actual_path_parts: list[str] = []
        prediction_values: list[str] = []
        final_prediction = ""
        final_prediction_correct = ""

        for stage_meta in stages:
            stage_index = int(stage_meta.get("stage_index", 0) or 0)
            stage_case_id = clean_text(stage_meta.get("input_case_id")) or f"STAGE{stage_index}__{base_case_id}"
            pred = pred_by_case_id.get(stage_case_id, {})
            date = clean_text(stage_meta.get("date"))
            prediction = pretty_binary(pred.get("pred_label"))
            confidence = clean_text(pred.get("confidence"))
            hearing_actual, hearing_actual_label = stage_actual(input_jsons, clean_text(stage_meta.get("input_filename")))
            binary_actual = pretty_binary(pred.get("target_label"))

            prediction_values.append(prediction)
            pred_path_parts.append(f"Hearing {stage_index}: {prediction} on {date}")
            raw_actual_path_parts.append(f"Hearing {stage_index}: {hearing_actual} on {date}")

            path_row[f"hearing{stage_index}_date"] = date
            path_row[f"hearing{stage_index}_prediction"] = prediction
            path_row[f"hearing{stage_index}_confidence"] = confidence
            path_row[f"hearing{stage_index}_actual_raw"] = hearing_actual
            path_row[f"hearing{stage_index}_actual_label"] = hearing_actual_label

            nodes, source, explanation_path = significant_nodes_for_stage(
                stage_case_id=stage_case_id,
                base_case_id=base_case_id,
                stage_index=stage_index,
                explainer_lookup=explainer_lookup,
                explainer_cases_dir=explainer_cases_dir,
                raw_factor=raw_factor,
                case_diff=case_diff,
                top_k=args.top_k_per_type,
                top_k_all=args.top_k_all,
            )
            source_counts[source] += 1

            hearing_row = {
                "base_case_id": base_case_id,
                "case_id": stage_case_id,
                "hearing_index": stage_index,
                "date": date,
                "bucket": bucket,
                "bucket_raw": raw_bucket,
                "prediction": prediction,
                "actual": hearing_actual,
                "actual_binary_target": binary_actual,
                "case_actual_result": case_actual_result,
                "case_actual_label": case_actual_label,
                "confidence": confidence,
                "prob_class_0": clean_text(pred.get("prob_class_0")),
                "prob_class_1": clean_text(pred.get("prob_class_1")),
                "prediction_correct_against_binary_target": clean_text(pred.get("pred_label")) == clean_text(pred.get("target_label")),
                "significance_source": source,
                "explanation_json": explanation_path,
            }
            hearing_row.update(nodes)
            hearing_rows.append(hearing_row)

            if stage_index == n_stages:
                final_prediction = prediction
                final_prediction_correct = clean_text(pred.get("pred_label")) == clean_text(pred.get("target_label"))

        path_row["prediction_path"] = " -> ".join(pred_path_parts)
        path_row["raw_actual_path"] = " -> ".join(raw_actual_path_parts)
        path_row["changed_prediction"] = len(set(prediction_values)) > 1
        path_row["final_prediction"] = final_prediction
        path_row["final_prediction_correct"] = final_prediction_correct
        case_path_rows.append(path_row)

    case_path_fields = [
        "base_case_id",
        "bucket",
        "bucket_raw",
        "outcome_split",
        "n_hearings",
        "case_actual_result",
        "case_actual_label",
        "case_actual_score",
        "prediction_path",
        "raw_actual_path",
        "changed_prediction",
        "final_prediction",
        "final_prediction_correct",
    ]
    for i in range(1, max_stages + 1):
        case_path_fields.extend([
            f"hearing{i}_date",
            f"hearing{i}_prediction",
            f"hearing{i}_confidence",
            f"hearing{i}_actual_raw",
            f"hearing{i}_actual_label",
        ])

    hearing_fields = [
        "base_case_id",
        "case_id",
        "hearing_index",
        "date",
        "bucket",
        "bucket_raw",
        "prediction",
        "actual",
        "actual_binary_target",
        "case_actual_result",
        "case_actual_label",
        "confidence",
        "prob_class_0",
        "prob_class_1",
        "prediction_correct_against_binary_target",
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
        "significance_source",
        "explanation_json",
    ]

    paths_csv = output_dir / "multi_hearing_case_paths.csv"
    hearings_csv = output_dir / "multi_hearing_hearing_level_analysis.csv"
    summary_json = output_dir / "multi_hearing_summary.json"

    write_csv(paths_csv, case_path_rows, case_path_fields)
    write_csv(hearings_csv, hearing_rows, hearing_fields)

    summary = {
        "cases_with_multiple_hearings": len(multi_cases),
        "total_hearing_rows": len(hearing_rows),
        "max_hearings_in_case": max_stages,
        "n_hearings_distribution": dict(Counter(int(case.get("n_stages", 0) or 0) for case in multi_cases)),
        "cases_with_changed_prediction": sum(1 for row in case_path_rows if row.get("changed_prediction")),
        "significance_source_counts": dict(source_counts),
        "stage_pgexplainer_available": bool(explainer_lookup and explainer_cases_dir.exists()),
        "inputs": {
            "predictions": str(predictions_path),
            "manifest": str(manifest_path),
            "raw_factors_dir": str(raw_factors_dir),
            "diffs_dir": str(diffs_dir),
            "stage_explainer_predictions": str(Path(args.stage_explainer_predictions)),
            "stage_explainer_cases_dir": str(explainer_cases_dir),
        },
    }
    with summary_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Cases with multiple hearings: {len(multi_cases)}")
    print(f"Wrote case paths -> {paths_csv}")
    print(f"Wrote hearing-level analysis -> {hearings_csv}")
    print(f"Wrote summary -> {summary_json}")
    print(f"Significance sources: {dict(source_counts)}")


if __name__ == "__main__":
    main()
