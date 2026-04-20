"""
03_build_bucket_datasets.py
===========================
Reads output/bucket_index.json and organises all classified case JSONs
into per-bucket dataset directories.

Output structure:
  output/buckets/
    bucket_1_violent_crimes/
      cases.csv          — summary of all cases in this bucket
      <filename>.json    — symlinks to output/case_jsons/<filename>.json
    bucket_2_motor_accident/
      ...
    ...
"""

import json
import os
import shutil
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

CASE_JSONS_DIR = Path(__file__).parent / "output" / "case_jsons"
BUCKETS_DIR = Path(__file__).parent / "output" / "buckets"
INDEX_PATH = Path(__file__).parent / "output" / "bucket_index.json"

BUCKET_SLUGS = {
    1: "bucket_1_violent_crimes",
    2: "bucket_2_motor_accident",
    3: "bucket_3_financial_crimes",
    4: "bucket_4_family_law",
    5: "bucket_5_bail_custody",
    6: "bucket_6_land_property",
    7: "bucket_7_service_employment",
    8: "bucket_8_narcotics_arms",
    9: "bucket_9_other",
}

BUCKET_META = {
    1: {"name": "Violent & Serious Crimes Against Persons", "emoji": "🔴"},
    2: {"name": "Motor Accident & Negligence (Compensation)", "emoji": "🚗"},
    3: {"name": "Financial, Economic & Property Crimes", "emoji": "💰"},
    4: {"name": "Family, Matrimonial & Personal Law", "emoji": "👨‍👩‍👧"},
    5: {"name": "Bail, Custody & Procedural Matters", "emoji": "🏛️"},
    6: {"name": "Land, Property & Revenue Disputes", "emoji": "🏗️"},
    7: {"name": "Service, Employment & Constitutional Matters", "emoji": "🏢"},
    8: {"name": "Narcotics, Arms & Special Statutes", "emoji": "💊"},
    9: {"name": "Other", "emoji": "📦"},
}


def main():
    if not INDEX_PATH.exists():
        print(f"ERROR: bucket_index.json not found at {INDEX_PATH}")
        print("Run 02_bucket_classifier.py first.")
        return

    print(f"[03] Loading bucket index from: {INDEX_PATH}")
    with open(INDEX_PATH, "r") as f:
        index = json.load(f)  # {filename: bucket_id}

    print(f"[03] Index has {len(index):,} classified cases")

    # Group filenames by bucket
    buckets: dict[int, list[str]] = defaultdict(list)
    for filename, bucket_id in index.items():
        buckets[int(bucket_id)].append(filename)

    # Create bucket directories
    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)

    total_linked = 0
    total_missing = 0

    for bucket_id in sorted(BUCKET_META.keys()):
        slug = BUCKET_SLUGS[bucket_id]
        meta = BUCKET_META[bucket_id]
        bucket_dir = BUCKETS_DIR / slug
        bucket_dir.mkdir(exist_ok=True)

        cases = buckets.get(bucket_id, [])
        print(f"\n[03] {meta['emoji']} {meta['name']}: {len(cases):,} cases → {slug}/")

        csv_rows = []
        linked = 0
        missing = 0

        for filename in tqdm(cases, desc=f"  Linking {slug}", leave=False):
            src = CASE_JSONS_DIR / f"{filename}.json"
            dst = bucket_dir / f"{filename}.json"

            if not src.exists():
                missing += 1
                continue

            # Create relative symlink (portable)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # Use relative path for symlink so dir is movable
            rel_src = os.path.relpath(src, bucket_dir)
            dst.symlink_to(rel_src)
            linked += 1

            # Collect CSV row data
            try:
                with open(src, "r", encoding="utf-8") as fj:
                    case_data = json.load(fj)
                csv_rows.append({
                    "filename": filename,
                    "label": case_data.get("label", ""),
                    "bucket": bucket_id,
                    "bucket_name": meta["name"],
                    "classification_method": case_data.get("classification_method", ""),
                })
            except Exception:
                csv_rows.append({
                    "filename": filename,
                    "label": "",
                    "bucket": bucket_id,
                    "bucket_name": meta["name"],
                    "classification_method": "",
                })

        # Write cases.csv for this bucket
        csv_path = bucket_dir / "cases.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=["filename", "label", "bucket", "bucket_name", "classification_method"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        # Write README for this bucket
        readme_path = bucket_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as rf:
            rf.write(f"# {meta['emoji']} {meta['name']}\n\n")
            rf.write(f"**Bucket ID:** {bucket_id}\n\n")
            rf.write(f"**Total cases:** {linked:,}\n\n")
            rf.write(f"Each `.json` file is a symlink to the corresponding case in `../../case_jsons/`.\n\n")
            rf.write(f"`cases.csv` — summary table (filename, label, bucket, classification_method)\n")

        total_linked += linked
        total_missing += missing

        print(f"     Linked: {linked:,} | Missing source: {missing} | cases.csv written")

    # Write master summary
    summary = {
        "total_cases": len(index),
        "total_linked": total_linked,
        "total_missing_source": total_missing,
        "buckets": {
            str(bid): {
                "slug": BUCKET_SLUGS[bid],
                "name": BUCKET_META[bid]["name"],
                "emoji": BUCKET_META[bid]["emoji"],
                "count": len(buckets.get(bid, [])),
            }
            for bid in sorted(BUCKET_META.keys())
        },
    }
    summary_path = BUCKETS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*50}")
    print(f"[03] Dataset build complete!")
    print(f"     Total linked: {total_linked:,}")
    print(f"     Missing:      {total_missing}")
    print(f"     Buckets dir:  {BUCKETS_DIR}")
    print(f"     Summary:      {summary_path}")
    print(f"\n  {'Bucket':<44} {'Count':>6}")
    print(f"  {'─'*50}")
    for bid in sorted(BUCKET_META.keys()):
        b = BUCKET_META[bid]
        cnt = len(buckets.get(bid, []))
        print(f"  {b['emoji']} {b['name']:<40} {cnt:>6,}")


if __name__ == "__main__":
    main()
