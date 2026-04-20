"""
01_csv_to_json.py
=================
Converts CJPE CSV(s) into individual JSON files, one per case.

Usage:
  python 01_csv_to_json.py                         # processes ALL *.csv files in the folder
  python 01_csv_to_json.py path/to/file.csv        # processes a specific CSV

CSV columns: filename, text, label
Output: output/case_jsons/{filename}.json
"""

import csv
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

csv.field_size_limit(10**9)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "case_jsons"

# Resolve CSV paths from CLI arg or glob all *.csv in folder
if len(sys.argv) > 1:
    CSV_PATHS = [Path(sys.argv[1])]
else:
    CSV_PATHS = sorted(SCRIPT_DIR.glob("*.csv"))
    if not CSV_PATHS:
        print("ERROR: No CSV files found in script directory.", file=sys.stderr)
        sys.exit(1)


def write_json(row_data: dict) -> str:
    filename = row_data["filename"].strip()
    out_path = OUTPUT_DIR / f"{filename}.json"
    # Skip if already done
    if out_path.exists():
        return f"SKIP {filename}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row_data, f, ensure_ascii=False, indent=2)
    return f"OK {filename}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_filenames = set()

    for csv_path in CSV_PATHS:
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
            continue
        print(f"[01] Reading: {csv_path.name}  ({csv_path.stat().st_size / 1e6:.0f} MB)")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["filename"].strip()
                if fn in seen_filenames:
                    continue   # deduplicate across CSVs
                seen_filenames.add(fn)
                rows.append({
                    "filename": fn,
                    "text": row["text"],
                    "label": int(row["label"]) if row["label"].strip().isdigit() else row["label"].strip(),
                })

    print(f"[01] Total unique rows across {len(CSV_PATHS)} CSV(s): {len(rows):,}")
    print(f"[01] Writing JSONs to: {OUTPUT_DIR}")

    skipped = 0
    written = 0
    errors  = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(write_json, r): r["filename"] for r in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="CSV→JSON"):
            try:
                result = future.result()
                if result.startswith("SKIP"):
                    skipped += 1
                else:
                    written += 1
            except Exception as e:
                errors += 1
                print(f"\nERROR on {futures[future]}: {e}", file=sys.stderr)

    print(f"\n[01] Done. Written: {written:,} | Skipped (existed): {skipped:,} | Errors: {errors}")
    print(f"[01] Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
