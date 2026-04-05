#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "food_law_final" / "processed" / "cleaned_cases"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "visualiser" / "data" / "cases_catalog.js"

CONNECTIVITY_TYPES = {
    "court",
    "judge",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "statute",
    "provision",
    "precedent",
    "case_number",
}

GRAPH_ENTITY_LIMITS = {
    "petitioners": 3,
    "respondents": 4,
    "courts": 2,
    "judges": 3,
    "petitioner_lawyers": 3,
    "defence_lawyers": 3,
    "lawyers": 4,
    "statutes": 5,
    "provisions": 6,
    "precedents": 4,
    "case_numbers": 4,
}

FORCED_CASE_IDS = {
    "A_G_Rajendran_vs_The_Food_Safety_Officer_on_24_April_2024",
    "A_Selvam_vs_The_Food_Safety_Officer_on_24_April_2024",
    "A_Boopathiraja_vs_The_Food_Safety_Officer_on_21_November_2024",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a larger static case catalog for the section_GNN visualiser.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--filter-substring", default="food")
    parser.add_argument("--max-shared-frequency", type=int, default=6)
    parser.add_argument("--top-connected", type=int, default=18)
    parser.add_argument("--low-connected", type=int, default=10)
    parser.add_argument("--isolates", type=int, default=12)
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
    title = case_id.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def base_identity(case_id: str) -> str:
    title = case_id_to_title(case_id)
    title = re.sub(r"\s+\(\d+\)$", "", title)
    return title.lower()


def extract_display_date(case_id: str) -> str | None:
    match = re.search(r"on_(\d{1,2})_([A-Za-z]+)_([0-9]{4})", case_id)
    if not match:
        return None
    day, month, year = match.groups()
    return f"{int(day)} {month[:3]} {year}"


def truncate_text(value: str | None, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def map_outcome(raw_label: str | None) -> str:
    if raw_label == "appellant_won":
        return "appellant_won"
    if raw_label == "appellant_lost":
        return "respondent_won"
    return "pending"


def build_case_record(case_json: dict[str, Any], filtered_descriptor_keys: set[tuple[str, str]]) -> dict[str, Any]:
    entity_groups: dict[str, list[str]] = defaultdict(list)
    connectivity_groups: dict[str, list[str]] = defaultdict(list)

    for entity in case_json.get("entities", []):
        entity_type = str(entity.get("entity_type") or "")
        raw_name = str(entity.get("raw_name") or entity.get("canonical_name") or "").strip()
        canonical_name = str(entity.get("canonical_name") or raw_name).strip()
        if not raw_name:
            continue

        if entity_type == "petitioner":
            entity_groups["petitioners"].append(raw_name)
        elif entity_type == "respondent":
            entity_groups["respondents"].append(raw_name)
        elif entity_type == "court":
            entity_groups["courts"].append(raw_name)
        elif entity_type == "judge":
            entity_groups["judges"].append(raw_name)
        elif entity_type == "petitioner_lawyer":
            entity_groups["petitioner_lawyers"].append(raw_name)
        elif entity_type == "defence_lawyer":
            entity_groups["defence_lawyers"].append(raw_name)
        elif entity_type == "lawyer":
            entity_groups["lawyers"].append(raw_name)
        elif entity_type == "statute":
            entity_groups["statutes"].append(raw_name)
        elif entity_type == "provision":
            entity_groups["provisions"].append(raw_name)
        elif entity_type == "precedent":
            entity_groups["precedents"].append(raw_name)
        elif entity_type == "case_number":
            entity_groups["case_numbers"].append(raw_name)

        descriptor_key = (entity_type, normalize_text(canonical_name))
        if descriptor_key in filtered_descriptor_keys:
            connectivity_groups[entity_type].append(raw_name)

    graph_entities = {
        key: unique_preserve(values)[: GRAPH_ENTITY_LIMITS[key]]
        for key, values in entity_groups.items()
    }

    texts = case_json.get("texts", {})
    facts_summary = truncate_text(texts.get("facts") or texts.get("preamble") or "")
    arguments_summary = truncate_text(
        texts.get("arguments")
        or texts.get("petitioner_arguments")
        or texts.get("respondent_arguments")
        or ""
    )

    case_id = str(case_json.get("case_id"))
    title = case_id_to_title(case_id)
    date = extract_display_date(case_id) or ""

    return {
        "id": case_id,
        "title": title,
        "date": date,
        "outcome": map_outcome(case_json.get("raw_label")),
        "petitioners": graph_entities.get("petitioners", []),
        "respondents": graph_entities.get("respondents", []),
        "courts": graph_entities.get("courts", []),
        "judges": graph_entities.get("judges", []),
        "petitioner_lawyers": graph_entities.get("petitioner_lawyers", []),
        "defence_lawyers": graph_entities.get("defence_lawyers", []),
        "lawyers": graph_entities.get("lawyers", []),
        "statutes": graph_entities.get("statutes", []),
        "provisions": graph_entities.get("provisions", []),
        "precedents": graph_entities.get("precedents", []),
        "case_numbers": graph_entities.get("case_numbers", []),
        "facts_summary": facts_summary,
        "arguments_summary": arguments_summary,
        "connectivity": {
            entity_type: unique_preserve(values)
            for entity_type, values in connectivity_groups.items()
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    filter_substring = str(args.filter_substring).lower()

    case_jsons: dict[str, dict[str, Any]] = {}
    descriptor_members: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_descriptors: dict[str, set[tuple[str, str]]] = {}

    for path in sorted(input_dir.glob("*.json")):
        case_json = json.loads(path.read_text())
        case_id = str(case_json.get("case_id"))
        if filter_substring and filter_substring not in case_id.lower():
            continue

        descriptors: set[tuple[str, str]] = set()
        for entity in case_json.get("entities", []):
            entity_type = str(entity.get("entity_type") or "")
            canonical_name = str(entity.get("canonical_name") or entity.get("raw_name") or "").strip()
            if entity_type not in CONNECTIVITY_TYPES or not canonical_name:
                continue
            descriptors.add((entity_type, normalize_text(canonical_name)))

        case_jsons[case_id] = case_json
        case_descriptors[case_id] = descriptors
        for descriptor in descriptors:
            descriptor_members[descriptor].add(case_id)

    filtered_case_descriptors: dict[str, set[tuple[str, str]]] = {}
    degrees: dict[str, int] = {}
    for case_id, descriptors in case_descriptors.items():
        filtered = {
            descriptor
            for descriptor in descriptors
            if 1 < len(descriptor_members[descriptor]) <= args.max_shared_frequency
        }
        filtered_case_descriptors[case_id] = filtered

        neighbors: set[str] = set()
        for descriptor in filtered:
            neighbors.update(descriptor_members[descriptor])
        neighbors.discard(case_id)
        degrees[case_id] = len(neighbors)

    positive_cases = sorted(
        [case_id for case_id, degree in degrees.items() if degree > 0],
        key=lambda case_id: (-degrees[case_id], case_id),
    )
    low_connected_cases = sorted(
        [case_id for case_id, degree in degrees.items() if degree > 0],
        key=lambda case_id: (degrees[case_id], case_id),
    )
    isolated_cases = sorted(
        [case_id for case_id, degree in degrees.items() if degree == 0]
    )

    selected_case_ids: list[str] = []
    selected_seen: set[str] = set()
    selected_bases: set[str] = set()

    def try_add(case_id: str) -> bool:
        if case_id not in case_jsons:
            return False
        base_key = base_identity(case_id)
        if case_id in selected_seen or base_key in selected_bases:
            return False
        selected_case_ids.append(case_id)
        selected_seen.add(case_id)
        selected_bases.add(base_key)
        return True

    for case_id in sorted(FORCED_CASE_IDS):
        try_add(case_id)
    for case_id in positive_cases:
        if len(selected_case_ids) >= args.top_connected:
            break
        try_add(case_id)
    for case_id in low_connected_cases:
        if len(selected_case_ids) >= args.top_connected + args.low_connected:
            break
        try_add(case_id)
    for case_id in isolated_cases:
        if len(selected_case_ids) >= args.top_connected + args.low_connected + args.isolates:
            break
        try_add(case_id)

    output_cases = [
        {
            **build_case_record(case_jsons[case_id], filtered_case_descriptors[case_id]),
            "connection_degree": degrees[case_id],
        }
        for case_id in selected_case_ids
    ]

    catalog_meta = {
        "source_dir": str(input_dir),
        "filter_substring": filter_substring,
        "source_case_count": len(case_jsons),
        "selected_case_count": len(output_cases),
        "max_shared_frequency": args.max_shared_frequency,
        "selection_breakdown": {
            "forced": len([case_id for case_id in selected_case_ids if case_id in FORCED_CASE_IDS]),
            "connected_or_low_connected": len([case for case in output_cases if case["connection_degree"] > 0]),
            "isolates": len([case for case in output_cases if case["connection_degree"] == 0]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "window.VISUALISER_CATALOG_META = "
        + json.dumps(catalog_meta, indent=2)
        + ";\n\nwindow.VISUALISER_CASES = "
        + json.dumps(output_cases, indent=2)
        + ";\n",
    )

    print(f"Wrote {len(output_cases)} cases to {output_path}")
    print(f"Connected cases in catalog: {catalog_meta['selection_breakdown']['connected_or_low_connected']}")
    print(f"Isolates in catalog: {catalog_meta['selection_breakdown']['isolates']}")


if __name__ == "__main__":
    main()
