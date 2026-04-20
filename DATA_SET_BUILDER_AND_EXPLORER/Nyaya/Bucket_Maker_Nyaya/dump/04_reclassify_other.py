"""
04_reclassify_other.py
======================
Re-classifies all cases currently in bucket 9 (Other) into 3 new buckets:
  10  🛒 Consumer Disputes
  11  🏦 Banking, Finance & Tax
  12  🍽️ Food Safety & Regulatory
  9   📦 Other  (cases that don't fit any new bucket)

Steps:
  1. Load all 'Other' cases from bucket_9_other/cases.csv
  2. Apply keyword rules for buckets 10, 11, 12
  3. Report overlaps (cases matching multiple new buckets)
  4. Update each JSON + bucket_index.json
  5. Rebuild all affected bucket directories (9, 10, 11, 12)

Priority when overlapping: Food (12) > Consumer (10) > Banking (11) > Other (9)
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

BUCKET_DIR   = Path(__file__).parent / "output" / "buckets"
CASE_JSONS   = Path(__file__).parent / "output" / "case_jsons"
INDEX_PATH   = Path(__file__).parent / "output" / "bucket_index.json"
STATS_PATH   = Path(__file__).parent / "output" / "bucket_stats.json"

# ── New bucket definitions ────────────────────────────────────────────────────
NEW_BUCKETS = {
    10: {
        "name":    "Consumer Disputes",
        "emoji":   "🛒",
        "slug":    "bucket_10_consumer_disputes",
        "keywords": [
            r"\bNCDRC\b", r"\bNational Consumer\b",
            r"\bconsumer forum\b", r"\bconsumer commission\b",
            r"\bdistrict forum\b", r"\bstate commission\b.*\bconsumer\b",
            r"\bConsumer Protection Act\b", r"\bConsumer Disputes Redressal\b",
            r"\bdeficiency.{0,20}service\b", r"\bservice.{0,20}deficiency\b",
            r"\bcomplainant\b.*\bopposite party\b",
            r"\brevision petition\b.*\bconsumer\b",
        ],
    },
    11: {
        "name":    "Banking, Finance & Tax",
        "emoji":   "🏦",
        "slug":    "bucket_11_banking_finance_tax",
        "keywords": [
            # Banking
            r"\bNPA\b", r"\bnon.performing asset\b", r"\bRBI\b",
            r"\bDebt Recovery Tribunal\b", r"\bDRT\b", r"\bSARFAESI\b",
            r"\bbank.{0,15}loan\b", r"\bloan.{0,15}default\b",
            r"\bmortgage\b", r"\bhypothecation\b",
            # Customs & Excise
            r"\bcustoms.{0,20}duty\b", r"\bduty.{0,20}customs\b",
            r"\bexcise duty\b", r"\bcentral excise\b", r"\bCESTAT\b",
            r"\bCustoms Act\b", r"\banti.dumping\b", r"\btariff\b",
            r"\bimport duty\b", r"\bexport duty\b",
            # Income Tax / GST
            r"\bincome tax\b", r"\bITAT\b", r"\bIT Act\b",
            r"\bassessment.{0,20}tax\b", r"\btax.{0,20}assessment\b",
            r"\bGST\b", r"\bservice tax\b", r"\bVAT\b",
            r"\bwithholding tax\b", r"\badvance tax\b",
            r"\bIncome Tax Act\b", r"\bI\.T\. Act\b",
        ],
    },
    12: {
        "name":    "Food Safety & Regulatory",
        "emoji":   "🍽️",
        "slug":    "bucket_12_food_safety",
        "keywords": [
            r"\bFSSAI\b", r"\bFood Safety and Standards\b",
            r"\bFood Safety Act\b", r"\bFSS Act\b",
            r"\bFood Business Operator\b", r"\bFBO\b",
            r"\bfood adulterat\b", r"\badulterat.{0,20}food\b",
            r"\bfood contaminat\b", r"\bcontaminat.{0,20}food\b",
            r"\bfood inspector\b", r"\bfood sample\b",
            r"\bPrevention of Food Adulteration\b", r"\bPFA Act\b",
            r"\bfood licen[sc]\b", r"\bfood standard\b",
            r"\bFood and Drug\b", r"\bFDA\b.*\bfood\b",
        ],
    },
}

# Priority for overlap resolution (highest priority first)
PRIORITY_ORDER = [12, 10, 11]  # Food > Consumer > Banking

REMAINING_OTHER = {
    "name":  "Other",
    "emoji": "📦",
    "slug":  "bucket_9_other",
}

EXISTING_BUCKETS = {
    1: {"name": "Violent & Serious Crimes Against Persons",     "emoji": "🔴", "slug": "bucket_1_violent_crimes"},
    2: {"name": "Motor Accident & Negligence (Compensation)",   "emoji": "🚗", "slug": "bucket_2_motor_accident"},
    3: {"name": "Financial, Economic & Property Crimes",        "emoji": "💰", "slug": "bucket_3_financial_crimes"},
    4: {"name": "Family, Matrimonial & Personal Law",           "emoji": "👨‍👩‍👧", "slug": "bucket_4_family_law"},
    5: {"name": "Bail, Custody & Procedural Matters",          "emoji": "🏛️", "slug": "bucket_5_bail_custody"},
    6: {"name": "Land, Property & Revenue Disputes",           "emoji": "🏗️", "slug": "bucket_6_land_property"},
    7: {"name": "Service, Employment & Constitutional Matters", "emoji": "🏢", "slug": "bucket_7_service_employment"},
    8: {"name": "Narcotics, Arms & Special Statutes",          "emoji": "💊", "slug": "bucket_8_narcotics_arms"},
    9: {"name": "Other",                                        "emoji": "📦", "slug": "bucket_9_other"},
}
ALL_BUCKETS = {**EXISTING_BUCKETS, **NEW_BUCKETS}

# Pre-compile patterns
COMPILED = {
    bid: [re.compile(kw, re.IGNORECASE) for kw in meta["keywords"]]
    for bid, meta in NEW_BUCKETS.items()
}


def keyword_match(text: str) -> list[int]:
    """Returns list of new bucket IDs that match this text (may be multiple)."""
    matches = []
    for bid, patterns in COMPILED.items():
        if any(p.search(text) for p in patterns):
            matches.append(bid)
    return matches


def resolve_overlap(matches: list[int]) -> int:
    """Pick winner based on priority order."""
    for bid in PRIORITY_ORDER:
        if bid in matches:
            return bid
    return 9


def rebuild_bucket_dir(bucket_id: int, filenames: list[str], meta: dict):
    """Create/update a bucket directory with symlinks and cases.csv."""
    slug = meta["slug"]
    bucket_dir = BUCKET_DIR / slug
    bucket_dir.mkdir(exist_ok=True)

    linked = 0
    missing = 0
    csv_rows = []

    for filename in tqdm(filenames, desc=f"  Linking {slug}", leave=False):
        src = CASE_JSONS / f"{filename}.json"
        dst = bucket_dir / f"{filename}.json"

        if not src.exists():
            missing += 1
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(os.path.relpath(src, bucket_dir))
        linked += 1

        try:
            with open(src) as f:
                data = json.load(f)
            csv_rows.append({
                "filename": filename,
                "label": data.get("label", ""),
                "bucket": bucket_id,
                "bucket_name": meta["name"],
                "classification_method": data.get("classification_method", ""),
            })
        except Exception:
            csv_rows.append({
                "filename": filename, "label": "", 
                "bucket": bucket_id, "bucket_name": meta["name"],
                "classification_method": "",
            })

    with open(bucket_dir / "cases.csv", "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["filename","label","bucket","bucket_name","classification_method"])
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(bucket_dir / "README.md", "w") as rf:
        rf.write(f"# {meta['emoji']} {meta['name']}\n\n")
        rf.write(f"**Bucket ID:** {bucket_id}\n\n")
        rf.write(f"**Total cases:** {linked:,}\n\n")
        rf.write(f"Each `.json` file is a symlink to `../../case_jsons/`.\n")

    return linked, missing


def main():
    # ── Load current Other cases ───────────────────────────────────────────
    other_csv = BUCKET_DIR / "bucket_9_other" / "cases.csv"
    if not other_csv.exists():
        print(f"ERROR: {other_csv} not found. Run 03_build_bucket_datasets.py first.")
        return

    other_filenames = []
    with open(other_csv) as f:
        for row in csv.DictReader(f):
            other_filenames.append(row["filename"])

    print(f"[04] 'Other' cases to re-classify: {len(other_filenames):,}")

    # ── Load full index ────────────────────────────────────────────────────
    with open(INDEX_PATH) as f:
        index = json.load(f)

    # ── Classify each Other case ──────────────────────────────────────────
    print(f"\n[04] Applying keyword rules for 3 new buckets...")

    new_assignments: dict[str, int] = {}   # filename → new bucket_id
    overlap_log: list[dict] = []           # cases matching multiple new buckets
    bucket_files: dict[int, list[str]] = defaultdict(list)

    for filename in tqdm(other_filenames, desc="  Classifying", unit="case"):
        src = CASE_JSONS / f"{filename}.json"
        if not src.exists():
            continue

        try:
            with open(src) as f:
                data = json.load(f)
        except Exception:
            continue

        text = data.get("text", "")
        matches = keyword_match(text)

        if len(matches) > 1:
            overlap_log.append({
                "filename": filename,
                "matches": matches,
                "resolved_to": resolve_overlap(matches),
            })

        if matches:
            bid = resolve_overlap(matches)
        else:
            bid = 9  # stays in Other

        new_assignments[filename] = bid
        bucket_files[bid].append(filename)

    # ── Overlap report ─────────────────────────────────────────────────────
    print(f"\n[04] ── Overlap Report ──────────────────────────────")
    print(f"  Cases matching multiple new buckets: {len(overlap_log)}")
    if overlap_log:
        overlap_counts = Counter(
            tuple(sorted(o["matches"])) for o in overlap_log
        )
        for combo, cnt in overlap_counts.most_common(10):
            names = " ∩ ".join(NEW_BUCKETS[b]["name"] for b in combo)
            print(f"    {cnt:>4}×  {names}")
    print(f"  Resolution priority: Food(12) > Consumer(10) > Banking(11)")

    # ── Assignment summary ─────────────────────────────────────────────────
    print(f"\n[04] ── New Assignment Summary ──────────────────────")
    new_counts = Counter(new_assignments.values())
    for bid in sorted(new_counts):
        meta = ALL_BUCKETS.get(bid, {"name": "?", "emoji": "?"})
        print(f"  {meta['emoji']} [{bid:>2}] {meta['name']:<42} {new_counts[bid]:>6,}")

    # ── Update JSONs + index ───────────────────────────────────────────────
    print(f"\n[04] Updating case JSONs...")
    for filename, bid in tqdm(new_assignments.items(), desc="  Writing JSONs", unit="case"):
        src = CASE_JSONS / f"{filename}.json"
        if not src.exists():
            continue
        try:
            with open(src) as f:
                data = json.load(f)
            meta = ALL_BUCKETS.get(bid, REMAINING_OTHER)
            data["bucket"]                  = bid
            data["bucket_name"]             = meta["name"]
            data["bucket_emoji"]            = meta["emoji"]
            data["classification_method"]   = "keyword_reclassify" if bid != 9 else "other"
            with open(src, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            index[filename] = bid
        except Exception as e:
            print(f"\n  WARN: {filename}: {e}")

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index updated ✓")

    # ── Rebuild bucket dirs for 9, 10, 11, 12 ─────────────────────────────
    print(f"\n[04] Rebuilding bucket directories...")

    # Gather all filenames still in each existing bucket (1-8) for the summary
    existing_counts = Counter(v for k, v in index.items() if int(v) <= 8)

    buckets_to_rebuild = {
        9:  (REMAINING_OTHER,  bucket_files.get(9, [])),
        10: (NEW_BUCKETS[10],  bucket_files.get(10, [])),
        11: (NEW_BUCKETS[11],  bucket_files.get(11, [])),
        12: (NEW_BUCKETS[12],  bucket_files.get(12, [])),
    }

    for bid, (meta, filenames) in buckets_to_rebuild.items():
        if bid in NEW_BUCKETS:
            full_meta = NEW_BUCKETS[bid]
        elif bid == 9:
            full_meta = REMAINING_OTHER
        else:
            full_meta = EXISTING_BUCKETS[bid]

        print(f"\n  {full_meta['emoji']} [{bid}] {full_meta['name']}: {len(filenames):,} cases")
        linked, missing = rebuild_bucket_dir(bid, filenames, full_meta)
        print(f"     Linked: {linked:,} | Missing: {missing}")

    # ── Write overlap log ──────────────────────────────────────────────────
    overlap_path = Path(__file__).parent / "output" / "overlap_report.json"
    with open(overlap_path, "w") as f:
        json.dump({
            "total_overlaps": len(overlap_log),
            "resolution_priority": [NEW_BUCKETS[b]["name"] for b in PRIORITY_ORDER],
            "cases": overlap_log[:500],  # cap at 500 for readability
        }, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    final_counts = Counter(index.values())
    print(f"\n[04] ══ Final Bucket Distribution (all 86,017 cases) ══")
    print(f"  {'Bucket':<46} {'Count':>7}")
    print(f"  {'─'*54}")
    for bid in sorted(ALL_BUCKETS):
        meta = ALL_BUCKETS[bid]
        cnt  = final_counts.get(bid, 0)
        print(f"  {meta['emoji']} [{bid:>2}] {meta['name']:<40} {cnt:>7,}")
    print(f"  {'─'*54}")
    print(f"  {'TOTAL':<46} {sum(final_counts.values()):>7,}")
    print(f"\n  Overlap report → {overlap_path}")
    print(f"  Index          → {INDEX_PATH}")


if __name__ == "__main__":
    main()
