#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove selected central authority entities from cleaned cases.")
    parser.add_argument("--input-cleaned-dir", required=True)
    parser.add_argument("--output-cleaned-dir", required=True)
    parser.add_argument("--output-entity-dir", required=True)
    parser.add_argument("--output-audits-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--removal-set", required=True)
    parser.add_argument(
        "--drop-provisions-linked-to-removed-statutes",
        action="store_true",
        help="Also drop provision entities whose linked_statute_canonical is one of the removed statutes.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def recompute_metadata(metadata: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(metadata)
    counts: Counter[str] = Counter()
    names: defaultdict[str, list[str]] = defaultdict(list)
    for entity in entities:
        entity_type = str(entity.get("entity_type") or "")
        counts[entity_type] += 1
        canonical_name = str(entity.get("canonical_name") or "")
        if canonical_name:
            names[entity_type].append(canonical_name)

    out["respondent_count"] = counts["respondent"]
    out["judge_count"] = counts["judge"]
    out["lawyer_count"] = counts["lawyer"] + counts["petitioner_lawyer"] + counts["defence_lawyer"]
    out["statute_count"] = counts["statute"]
    out["provision_count"] = counts["provision"]
    out["precedent_count"] = counts["precedent"]
    out["court_count"] = counts["court"]
    out["court_names"] = sorted(set(names["court"]))
    out["judge_names"] = sorted(set(names["judge"]))
    out["statute_names"] = sorted(set(names["statute"]))
    return out


def main() -> None:
    args = parse_args()
    input_cleaned_dir = Path(args.input_cleaned_dir)
    output_cleaned_dir = Path(args.output_cleaned_dir)
    output_entity_dir = Path(args.output_entity_dir)
    output_audits_dir = Path(args.output_audits_dir)
    summary_path = Path(args.summary_path)
    removal_set = load_json(Path(args.removal_set))

    selected = removal_set.get("selected_authorities", [])
    removed_keys = {
        (
            str(item.get("entity_type") or ""),
            str(item.get("canonical_name") or "").strip().lower(),
        )
        for item in selected
    }
    removed_statutes = {
        canonical_name
        for entity_type, canonical_name in removed_keys
        if entity_type == "statute"
    }

    if not input_cleaned_dir.is_dir():
        raise FileNotFoundError(f"Input cleaned directory missing: {input_cleaned_dir}")

    output_cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_entity_dir.mkdir(parents=True, exist_ok=True)
    output_audits_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input_cleaned_dir": str(input_cleaned_dir),
        "output_cleaned_dir": str(output_cleaned_dir),
        "removal_set": str(args.removal_set),
        "num_files": 0,
        "total_entities_before": 0,
        "total_entities_after": 0,
        "removed_entity_counts": Counter(),
        "removed_by_name": Counter(),
        "cases": [],
    }

    for path in sorted(input_cleaned_dir.glob("*.json")):
        payload = load_json(path)
        before_entities = list(payload.get("entities", []) or [])
        after_entities: list[dict[str, Any]] = []
        removed_entities: list[dict[str, Any]] = []

        for entity in before_entities:
            entity_type = str(entity.get("entity_type") or "")
            canonical_name = str(entity.get("canonical_name") or "").strip().lower()
            key = (entity_type, canonical_name)
            linked_statute = str(entity.get("linked_statute_canonical") or "").strip().lower()
            remove = key in removed_keys
            if (
                args.drop_provisions_linked_to_removed_statutes
                and entity_type == "provision"
                and linked_statute in removed_statutes
            ):
                remove = True

            if remove:
                removed_entities.append(entity)
                summary["removed_entity_counts"][entity_type] += 1
                summary["removed_by_name"][f"{entity_type}|{canonical_name}"] += 1
            else:
                after_entities.append(entity)

        payload["entities"] = after_entities
        payload["metadata"] = recompute_metadata(dict(payload.get("metadata", {})), after_entities)
        audit = dict(payload.get("leakage_audit", {}))
        audit["central_authority_removal"] = {
            "removed_count": len(removed_entities),
            "removed_entities": [
                {
                    "entity_type": entity.get("entity_type"),
                    "canonical_name": entity.get("canonical_name"),
                    "raw_name": entity.get("raw_name"),
                    "local_case_frequency": entity.get("local_case_frequency"),
                }
                for entity in removed_entities
            ],
        }
        payload["leakage_audit"] = audit

        write_json(output_cleaned_dir / path.name, payload)
        write_json(
            output_entity_dir / path.name,
            {
                "case_id": payload.get("case_id"),
                "entities": after_entities,
            },
        )
        write_json(output_audits_dir / path.name, audit)

        summary["num_files"] += 1
        summary["total_entities_before"] += len(before_entities)
        summary["total_entities_after"] += len(after_entities)
        if removed_entities:
            summary["cases"].append(
                {
                    "case_id": payload.get("case_id"),
                    "file_name": payload.get("file_name"),
                    "removed_count": len(removed_entities),
                }
            )

    summary["removed_entity_counts"] = dict(sorted(summary["removed_entity_counts"].items()))
    summary["removed_by_name"] = dict(sorted(summary["removed_by_name"].items()))
    write_json(summary_path, summary)
    print(f"Wrote filtered cleaned cases to {output_cleaned_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
