#!/usr/bin/env python3
"""
Step 01 — Build the flat input directory of stage-tagged JSONs from the
multi-hearing case folders in the entity-resolved cases_by_outcome/.

Each multi-hearing case folder contains 2 JSON files (one per hearing).
We sort by hearing date, copy the earlier one as STAGE1__<id>.json and the
later one as STAGE2__<id>.json into a flat directory consumed by the existing
preprocess_fixed_open.py pipeline.

A manifest CSV/JSON is written so downstream steps can pair stages and look
up dates / categories / true outcomes.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-")
SOURCE_ROOT = REPO_ROOT / "DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/cases_by_outcome"
EXP_ROOT = REPO_ROOT / "section_GNN/multi_hearing_stage_test"
INPUT_DIR = EXP_ROOT / "data/input_jsons"
MANIFEST_DIR = EXP_ROOT / "outputs"

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

DATE_RE = re.compile(r"on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})", re.IGNORECASE)


def parse_date(filename: str) -> datetime:
    m = DATE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse date from: {filename}")
    day, month, year = m.group(1), m.group(2).lower(), m.group(3)
    return datetime(int(year), MONTH_MAP[month], int(day))


def slugify(name: str) -> str:
    """Replace path-hostile chars while keeping the case identity readable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def case_folder_to_category(folder_name: str) -> str:
    if "__" not in folder_name:
        return "unknown"
    return folder_name.split("__", 1)[0]


def main() -> None:
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    skipped: list[dict] = []

    for outcome in ("winning", "losing"):
        outcome_dir = SOURCE_ROOT / outcome
        if not outcome_dir.is_dir():
            print(f"[warn] missing dir: {outcome_dir}", file=sys.stderr)
            continue

        for case_dir in sorted(outcome_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            files = sorted(p for p in case_dir.iterdir() if p.suffix.lower() == ".json")
            if len(files) < 2:
                skipped.append({"case_dir": str(case_dir), "n_files": len(files), "reason": "expected at least 2 hearing files"})
                continue

            try:
                dated = sorted(((parse_date(f.name), f) for f in files), key=lambda t: t[0])
            except ValueError as exc:
                skipped.append({"case_dir": str(case_dir), "reason": str(exc)})
                continue

            category = case_folder_to_category(case_dir.name)
            base_id = slugify(case_dir.name)
            n_stages = len(dated)

            stage_records = []
            for i, (d, f) in enumerate(dated, start=1):
                stage_name = f"STAGE{i}__{base_id}.json"
                shutil.copy2(f, INPUT_DIR / stage_name)
                stage_records.append({
                    "stage_index": i,
                    "input_filename": stage_name,
                    "input_case_id": Path(stage_name).stem,
                    "source_filename": f.name,
                    "date": d.isoformat()[:10],
                    "is_final": i == n_stages,
                })

            with open(dated[-1][1]) as fp:
                final_doc = json.load(fp)
            final_label = final_doc.get("case_outcome_label", "")
            final_score = final_doc.get("case_outcome_score", "")

            manifest_rows.append({
                "outcome_split": outcome,
                "category": category,
                "case_dir": case_dir.name,
                "base_case_id": base_id,
                "n_stages": n_stages,
                "stages": stage_records,
                "final_outcome_label": final_label,
                "final_outcome_score": final_score,
            })

    manifest_csv = MANIFEST_DIR / "stage_manifest.csv"
    manifest_json = MANIFEST_DIR / "stage_manifest.json"

    flat_rows = []
    for row in manifest_rows:
        for s in row["stages"]:
            flat_rows.append({
                "outcome_split": row["outcome_split"],
                "category": row["category"],
                "base_case_id": row["base_case_id"],
                "n_stages": row["n_stages"],
                "stage_index": s["stage_index"],
                "is_final": s["is_final"],
                "input_filename": s["input_filename"],
                "input_case_id": s["input_case_id"],
                "source_filename": s["source_filename"],
                "date": s["date"],
                "final_outcome_label": row["final_outcome_label"],
                "final_outcome_score": row["final_outcome_score"],
            })

    if flat_rows:
        with open(manifest_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    with open(manifest_json, "w") as fp:
        json.dump({"cases": manifest_rows, "skipped": skipped}, fp, indent=2)

    print(f"Prepared cases: {len(manifest_rows)}")
    print(f"Skipped cases:  {len(skipped)}")
    print(f"Files written:  {len(list(INPUT_DIR.glob('*.json')))} -> {INPUT_DIR}")
    print(f"Manifest:       {manifest_csv}")


if __name__ == "__main__":
    main()
