#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.case_star_builder import build_case_star_graph
from src.graph.schema import CleanedCase, EntityMention, EntityRecord
from src.utils.io import ensure_dir, load_json, load_yaml

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gnn_case_star_food_law_final.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualiser" / "data" / "training_graph"

TEXT_LABELS = {
    "preamble": "Preamble",
    "facts": "Facts",
    "arguments": "Arguments",
    "petitioner_arguments": "Petitioner Arguments",
    "respondent_arguments": "Respondent Arguments",
    "other_lawyer_arguments": "Other Lawyer Arguments",
}

SUMMARY_KEY_BY_NODE_TYPE = {
    "petitioner": "petitioners",
    "respondent": "respondents",
    "court": "courts",
    "judge": "judges",
    "petitioner_lawyer": "petitioner_lawyers",
    "defence_lawyer": "defence_lawyers",
    "lawyer": "lawyers",
    "statute": "statutes",
    "provision": "provisions",
    "precedent": "precedents",
    "org": "orgs",
    "gpe": "gpes",
    "date": "dates",
    "case_number": "case_numbers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact per-case training graph data for the browser visualiser."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def unique_preserve(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", value).strip()
        if not clean:
            continue
        key = normalize_text(clean)
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def case_id_to_title(case_id: str) -> str:
    return re.sub(r"\s+", " ", case_id.replace("_", " ")).strip()


def extract_display_date(case_id: str) -> str:
    match = re.search(r"on_(\d{1,2})_([A-Za-z]+)_([0-9]{4})", case_id)
    if not match:
        return ""
    day, month, year = match.groups()
    return f"{int(day)} {month[:3]} {year}"


def outcome_for_display(raw_label: str | None) -> str:
    return str(raw_label or "unknown")


def node_label(node_type: str, case_title: str, text: str) -> str:
    if node_type == "case":
        return case_title
    if node_type in TEXT_LABELS:
        return TEXT_LABELS[node_type]
    return text


def detail_filename(case_id: str) -> str:
    return f"{case_id}.json"


def load_effective_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    cfg = load_yaml(config_path)
    graph_cache_dir = Path(cfg.get("paths", {}).get("graph_cache_dir", ""))
    snapshot_path = graph_cache_dir / "graph_config_snapshot.yaml"
    if snapshot_path.exists():
        return load_yaml(snapshot_path), snapshot_path
    return cfg, config_path


def cleaned_case_from_dict(payload: dict[str, Any]) -> CleanedCase:
    entities = []
    for entity in payload.get("entities", []):
        mentions = [
            EntityMention(
                entity_type=str(mention.get("entity_type")),
                raw_text=str(mention.get("raw_text", "")),
                canonical_text=str(mention.get("canonical_text", "")),
                section=mention.get("section"),
                annotation_id=str(mention.get("annotation_id", "")),
                start=mention.get("start"),
                end=mention.get("end"),
            )
            for mention in entity.get("mentions", [])
        ]
        entities.append(
            EntityRecord(
                entity_type=str(entity.get("entity_type")),
                raw_name=str(entity.get("raw_name", "")),
                canonical_name=str(entity.get("canonical_name", "")),
                mentions=mentions,
                local_case_frequency=int(entity.get("local_case_frequency", len(mentions))),
                first_seen_section=entity.get("first_seen_section"),
                seen_in_arguments=bool(entity.get("seen_in_arguments", False)),
                seen_in_preamble=bool(entity.get("seen_in_preamble", False)),
                linked_statute_canonical=entity.get("linked_statute_canonical"),
            )
        )
    return CleanedCase(
        case_id=str(payload.get("case_id")),
        file_name=str(payload.get("file_name")),
        file_id=payload.get("file_id"),
        internal_file_id=payload.get("internal_file_id"),
        source_path=payload.get("source_path"),
        raw_label=payload.get("raw_label"),
        texts=dict(payload.get("texts", {})),
        metadata=dict(payload.get("metadata", {})),
        entities=entities,
        leakage_audit=dict(payload.get("leakage_audit", {})),
    )


def load_cleaned_cases(cleaned_case_dir: Path) -> dict[str, Any]:
    cleaned_cases: dict[str, Any] = {}
    for path in sorted(cleaned_case_dir.glob("*.json")):
        cleaned_case = cleaned_case_from_dict(load_json(path))
        cleaned_cases[cleaned_case.case_id] = cleaned_case
    return cleaned_cases


def build_case_fields(case_graph: Any, cleaned_case: Any, split_assignments: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in case_graph.nodes:
        summary_key = SUMMARY_KEY_BY_NODE_TYPE.get(node.node_type)
        if summary_key is None:
            continue
        grouped[summary_key].append(node.text)

    search_parts = [
        cleaned_case.case_id,
        case_id_to_title(cleaned_case.case_id),
        str(cleaned_case.raw_label or ""),
        split_assignments.get(cleaned_case.case_id, "unknown"),
    ]
    for summary_key in SUMMARY_KEY_BY_NODE_TYPE.values():
        search_parts.extend(unique_preserve(grouped.get(summary_key, []))[:10])

    return {
        "id": cleaned_case.case_id,
        "title": case_id_to_title(cleaned_case.case_id),
        "date": extract_display_date(cleaned_case.case_id),
        "raw_label": cleaned_case.raw_label,
        "outcome": outcome_for_display(cleaned_case.raw_label),
        "split": split_assignments.get(cleaned_case.case_id, "unknown"),
        "petitioners": unique_preserve(grouped.get("petitioners", [])),
        "respondents": unique_preserve(grouped.get("respondents", [])),
        "courts": unique_preserve(grouped.get("courts", [])),
        "judges": unique_preserve(grouped.get("judges", [])),
        "petitioner_lawyers": unique_preserve(grouped.get("petitioner_lawyers", [])),
        "defence_lawyers": unique_preserve(grouped.get("defence_lawyers", [])),
        "lawyers": unique_preserve(grouped.get("lawyers", [])),
        "statutes": unique_preserve(grouped.get("statutes", [])),
        "provisions": unique_preserve(grouped.get("provisions", [])),
        "precedents": unique_preserve(grouped.get("precedents", [])),
        "orgs": unique_preserve(grouped.get("orgs", [])),
        "gpes": unique_preserve(grouped.get("gpes", [])),
        "dates": unique_preserve(grouped.get("dates", [])),
        "case_numbers": unique_preserve(grouped.get("case_numbers", [])),
        "preamble_summary": cleaned_case.texts.get("preamble", ""),
        "facts_summary": cleaned_case.texts.get("facts", ""),
        "arguments_summary": cleaned_case.texts.get("arguments", ""),
        "petitioner_arguments_summary": cleaned_case.texts.get("petitioner_arguments", ""),
        "respondent_arguments_summary": cleaned_case.texts.get("respondent_arguments", ""),
        "other_lawyer_arguments_summary": cleaned_case.texts.get("other_lawyer_arguments", ""),
        "node_count": len(case_graph.nodes),
        "edge_count": len(case_graph.edges),
        "search_text": " ".join(part for part in search_parts if part).lower(),
    }


def build_case_summary(case_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case_fields["id"],
        "title": case_fields["title"],
        "date": case_fields["date"],
        "raw_label": case_fields["raw_label"],
        "outcome": case_fields["outcome"],
        "split": case_fields["split"],
        "node_count": case_fields["node_count"],
        "edge_count": case_fields["edge_count"],
        "search_text": case_fields["search_text"],
    }


def build_case_detail(case_graph: Any, case_fields: dict[str, Any], connections: list[dict[str, Any]]) -> dict[str, Any]:
    title = case_fields["title"]
    return {
        **case_fields,
        "graph_nodes": [
            {
                "id": node.node_key,
                "type": node.node_type,
                "label": node_label(node.node_type, title, node.text),
                "text": node.text,
                "meta": node.metadata,
                "share_across_cases": node.share_across_cases,
            }
            for node in case_graph.nodes
        ],
        "graph_edges": [
            {
                "source": edge.src_key,
                "target": edge.dst_key,
                "relation": edge.relation,
                "meta": edge.metadata,
            }
            for edge in case_graph.edges
        ],
        "connections": connections,
    }


def main() -> None:
    args = parse_args()
    effective_cfg, cfg_source_path = load_effective_config(Path(args.config))
    paths_cfg = effective_cfg.get("paths", {})
    cleaned_case_dir = Path(paths_cfg.get("cleaned_case_dir"))
    graph_cache_dir = Path(paths_cfg.get("graph_cache_dir"))
    split_assignments_path = graph_cache_dir / "split_assignments.json"
    split_assignments = load_json(split_assignments_path) if split_assignments_path.exists() else {}

    cleaned_cases_by_id = load_cleaned_cases(cleaned_case_dir)
    included_case_ids = sorted(split_assignments.keys()) if split_assignments else sorted(cleaned_cases_by_id.keys())
    cleaned_cases = {case_id: cleaned_cases_by_id[case_id] for case_id in included_case_ids if case_id in cleaned_cases_by_id}

    case_graphs: dict[str, Any] = {}
    case_summaries: dict[str, dict[str, Any]] = {}
    shared_node_members: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for case_id, cleaned_case in cleaned_cases.items():
        case_graph = build_case_star_graph(cleaned_case, effective_cfg.get("graph", {}))
        case_graphs[case_id] = case_graph
        case_fields = build_case_fields(case_graph, cleaned_case, split_assignments)
        case_summaries[case_id] = build_case_summary(case_fields)
        case_summaries[case_id]["_detail_fields"] = case_fields

        for node in case_graph.nodes:
            if node.node_type == "case" or not node.share_across_cases:
                continue
            shared_node_members[node.node_key].append((case_id, node.node_type, node.text))

    output_dir = ensure_dir(Path(args.output_dir))
    case_output_dir = ensure_dir(output_dir / "cases")

    index_rows: list[dict[str, Any]] = []
    split_counts = Counter(split_assignments.get(case_id, "unknown") for case_id in cleaned_cases)

    for case_id, case_graph in case_graphs.items():
        summary = case_summaries[case_id]
        detail_fields = summary.pop("_detail_fields")
        reasons_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        for node in case_graph.nodes:
            if node.node_type == "case" or not node.share_across_cases:
                continue
            for other_case_id, node_type, node_text in shared_node_members[node.node_key]:
                if other_case_id == case_id:
                    continue
                reasons_by_case[other_case_id][node.node_key] = {
                    "node_id": node.node_key,
                    "type": node_type,
                    "label": node_text,
                }

        connections: list[dict[str, Any]] = []
        for other_case_id, reasons_map in reasons_by_case.items():
            other_summary = case_summaries[other_case_id]
            reasons = sorted(
                reasons_map.values(),
                key=lambda item: (item["type"], item["label"]),
            )
            connections.append(
                {
                    "case_id": other_case_id,
                    "title": other_summary["title"],
                    "date": other_summary["date"],
                    "outcome": other_summary["outcome"],
                    "raw_label": other_summary["raw_label"],
                    "split": other_summary["split"],
                    "reasons": reasons,
                }
            )

        connections.sort(key=lambda item: (-len(item["reasons"]), item["title"]))
        detail = build_case_detail(case_graph, detail_fields, connections)
        detail_path = case_output_dir / detail_filename(case_id)
        detail_path.write_text(json.dumps(detail, indent=2))

        index_rows.append(
            {
                **summary,
                "connection_degree": len(connections),
                "detail_path": f"data/training_graph/cases/{detail_filename(case_id)}",
            }
        )

    index_rows.sort(key=lambda item: item["title"])
    index_payload = {
        "config_source": str(cfg_source_path),
        "cleaned_case_dir": str(cleaned_case_dir),
        "graph_cache_dir": str(graph_cache_dir),
        "included_case_count": len(index_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "uses_graph_snapshot": cfg_source_path.name == "graph_config_snapshot.yaml",
        "notes": "Exact local graphs exported from filtered cleaned cases via build_case_star_graph().",
    }

    index_path = output_dir / "index.js"
    index_path.write_text(
        "window.TRAINING_GRAPH_META = "
        + json.dumps(index_payload, indent=2)
        + ";\n\nwindow.TRAINING_GRAPH_CASE_INDEX = "
        + json.dumps(index_rows, indent=2)
        + ";\n",
    )

    print(f"Wrote exact training graph index to {index_path}")
    print(f"Wrote {len(index_rows)} per-case graph JSON files to {case_output_dir}")


if __name__ == "__main__":
    main()
