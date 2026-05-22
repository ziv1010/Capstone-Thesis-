#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SECTION_GNN = Path(__file__).resolve().parents[2]
DEFAULT_ENTITY_CONFIG_ROOT = SECTION_GNN / "ablations" / "entity_resolved_data" / "configs" / "party"
DEFAULT_OUTPUT_DIR = SECTION_GNN / "outputs" / "ablations" / "remove_central_authorities" / "centrality_analysis"
BUCKETS = (
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
    "cross_bucket_total_dataset",
)

# Keep this intentionally conservative. Domain-defining statutes such as the
# Motor Vehicles Act, POCSO Act, Dowry Prohibition Act, and Land Acquisition Act
# are central but not trivial for their buckets, so they are not auto-removed.
DEFAULT_TRIVIAL_STATUTES = {
    "indian penal code 1860",
    "code of criminal procedure 1973",
    "bharatiya nagarik suraksha sanhita 2023",
    "bharatiya nyaya sanhita 2023",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find highly central authority nodes in entity-resolved cleaned cases."
    )
    parser.add_argument("--config-root", default=str(DEFAULT_ENTITY_CONFIG_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--buckets", nargs="*", default=list(BUCKETS))
    parser.add_argument("--min-global-case-frequency", type=int, default=3000)
    parser.add_argument(
        "--extra-statute",
        action="append",
        default=[],
        help="Additional normalized statute canonical_name to auto-remove.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cleaned_dir_for(config_root: Path, bucket: str) -> Path:
    cfg = load_yaml(config_root / bucket / "config.yaml")
    return Path(cfg["paths"]["cleaned_case_dir"])


def iter_cleaned_cases(cleaned_dir: Path):
    for path in sorted(cleaned_dir.glob("*.json")):
        yield path, load_json(path)


def main() -> None:
    args = parse_args()
    config_root = Path(args.config_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trivial_statutes = set(DEFAULT_TRIVIAL_STATUTES)
    trivial_statutes.update(str(item).strip().lower() for item in args.extra_statute if str(item).strip())

    global_case_sets: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    global_mentions: Counter[tuple[str, str]] = Counter()
    bucket_case_sets: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    bucket_mentions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    sample_raw_names: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for bucket in args.buckets:
        cleaned_dir = cleaned_dir_for(config_root, bucket)
        if not cleaned_dir.is_dir():
            raise FileNotFoundError(f"Cleaned case directory missing for {bucket}: {cleaned_dir}")

        for path, payload in iter_cleaned_cases(cleaned_dir):
            case_id = str(payload.get("case_id") or path.stem)
            seen_in_case: set[tuple[str, str]] = set()
            for entity in payload.get("entities", []) or []:
                entity_type = str(entity.get("entity_type") or "")
                if entity_type not in {"statute", "provision", "precedent"}:
                    continue
                canonical_name = str(entity.get("canonical_name") or "").strip().lower()
                if not canonical_name:
                    continue
                key = (entity_type, canonical_name)
                seen_in_case.add(key)
                mention_count = int(entity.get("local_case_frequency") or len(entity.get("mentions", []) or []) or 1)
                global_mentions[key] += mention_count
                bucket_mentions[bucket][key] += mention_count
                raw_name = str(entity.get("raw_name") or "")
                if raw_name:
                    sample_raw_names[key][raw_name] += mention_count
            for key in seen_in_case:
                global_case_sets[key].add((bucket, case_id))
                bucket_case_sets[bucket][key].add(case_id)

    rows: list[dict[str, Any]] = []
    for key, global_cases in global_case_sets.items():
        entity_type, canonical_name = key
        bucket_counts = {
            bucket: len(bucket_case_sets[bucket].get(key, set()))
            for bucket in args.buckets
            if len(bucket_case_sets[bucket].get(key, set())) > 0
        }
        global_case_frequency = len(global_cases)
        selected = (
            entity_type == "statute"
            and canonical_name in trivial_statutes
            and global_case_frequency >= args.min_global_case_frequency
        )
        reason = ""
        if selected:
            reason = (
                f"trivial broad statute with global_case_frequency >= "
                f"{args.min_global_case_frequency}"
            )
        rows.append(
            {
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "global_case_frequency": global_case_frequency,
                "global_mention_count": int(global_mentions[key]),
                "bucket_case_counts": json.dumps(bucket_counts, sort_keys=True),
                "selected_for_removal": "yes" if selected else "no",
                "selection_reason": reason,
                "sample_raw_names": " | ".join(
                    raw for raw, _ in sample_raw_names[key].most_common(5)
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["entity_type"],
            -int(row["global_case_frequency"]),
            -int(row["global_mention_count"]),
            row["canonical_name"],
        )
    )

    csv_path = output_dir / "central_authority_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "entity_type",
                "canonical_name",
                "global_case_frequency",
                "global_mention_count",
                "bucket_case_counts",
                "selected_for_removal",
                "selection_reason",
                "sample_raw_names",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    selected = [
        row
        for row in rows
        if row["selected_for_removal"] == "yes"
    ]
    removal_set = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config_root": str(config_root),
        "candidate_csv": str(csv_path),
        "min_global_case_frequency": args.min_global_case_frequency,
        "selection_policy": (
            "Auto-remove only broad, trivial statute nodes in the curated allow-list. "
            "Do not auto-remove domain-defining statutes or specific provisions."
        ),
        "trivial_statute_allowlist": sorted(trivial_statutes),
        "selected_authorities": [
            {
                "entity_type": row["entity_type"],
                "canonical_name": row["canonical_name"],
                "global_case_frequency": int(row["global_case_frequency"]),
                "global_mention_count": int(row["global_mention_count"]),
                "bucket_case_counts": json.loads(row["bucket_case_counts"]),
                "selection_reason": row["selection_reason"],
            }
            for row in selected
        ],
    }
    removal_path = output_dir / "central_authority_removal_set.json"
    write_json(removal_path, removal_set)

    print(f"Wrote candidates: {csv_path}")
    print(f"Wrote removal set: {removal_path}")
    print("Selected authorities:")
    for item in removal_set["selected_authorities"]:
        print(
            f"  - {item['entity_type']} | {item['canonical_name']} | "
            f"cases={item['global_case_frequency']} mentions={item['global_mention_count']}"
        )


if __name__ == "__main__":
    main()
