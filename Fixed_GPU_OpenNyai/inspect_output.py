#!/usr/bin/env python3
"""Inspect one combined OpenNyAI output JSON file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.output_formatter import entity_label, rhetorical_role, summary_sections_present, unwrap_raw_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one combined output JSON file.")
    parser.add_argument("combined_json", help="Path to outputs/combined/<file_id>.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    combined_path = Path(args.combined_json).expanduser().resolve()
    payload = json.loads(combined_path.read_text(encoding="utf-8"))
    raw_doc = unwrap_raw_result(payload)

    role_counts = Counter()
    total_entities = 0

    for annotation in raw_doc.get("annotations", []):
        role = rhetorical_role(annotation)
        if role:
            role_counts[role] += 1
        total_entities += len(annotation.get("entities") or [])

    sections = summary_sections_present(raw_doc)
    file_id = payload.get("file_id") or raw_doc.get("id") or combined_path.stem

    print(f"file_id: {file_id}")
    print(f"number_of_sentences: {len(raw_doc.get('annotations', []))}")
    print(f"rhetorical_role_counts: {dict(role_counts)}")
    print(f"total_entities: {total_entities}")
    print(f"summary_sections_present: {sections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
