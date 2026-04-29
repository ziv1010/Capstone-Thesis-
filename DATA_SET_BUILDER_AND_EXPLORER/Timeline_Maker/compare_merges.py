"""
compare_merges.py
=================
Compares old_outputs (v2 merge) against output_merged_v3 (v3 merge).

For each category it reports:
  - unchanged   : same filename in both, no merge status change
  - wrong_split : was _MERGED in old, now split into multiple standalones in v3
                  (these are the bug fixes — cases that were wrongly merged)
  - new_merged  : now _MERGED in v3, was standalone(s) in old
                  (genuine multi-hearing cases newly caught)
  - only_old    : file exists only in old (e.g. a wrong merge output that disappeared)
  - only_new    : file exists only in v3 (e.g. previously merged file now kept separate)

Usage:
  python3 compare_merges.py
"""

import os, re, json
from collections import defaultdict

OLD_BASE = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/old_outputs"
NEW_BASE = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3"

# Maps new category name → old directory name
CATEGORY_MAP = {
    "fin_fraud":          "fin_fraud_timed_mistral",
    "family_matrimonial": "family_matrimonial_timed_mistral",
    "land_property":      "land_property_timed_mistral",
    "motor_accidents":    "motor_accidents_timed_mistral",
    "sexual_offences":    "sexual_offences_timed_mistral",
    # food_safety had no old output
}

SUFFIX_RE = re.compile(r'_(MERGED|DEDUPED)\.json$', re.IGNORECASE)
DATE_RE   = re.compile(r'\s+on\s+\d{1,2}_\w+_\d{4}', re.IGNORECASE)


def strip_suffix(fname: str) -> str:
    """Remove _MERGED / _DEDUPED suffix, keep .json."""
    return SUFFIX_RE.sub('.json', fname)


def base_case_name(fname: str) -> str:
    """
    Extract the case name without the hearing date or merge suffix.
    e.g. "Amit Kumar vs Bihar on 7_Oct_2024_MERGED.json" → "Amit Kumar vs Bihar"
    Used to group files that refer to the same case parties.
    """
    no_suffix = strip_suffix(fname)
    no_ext    = no_suffix[:-5]          # remove .json
    # remove trailing " on DD_Month_YYYY" date slug
    no_date   = DATE_RE.sub('', no_ext)
    # also handle original "on DD Month, YYYY" style
    no_date2  = re.sub(r'\s+on\s+\d{1,2}\s+\w+,?\s*\d{4}.*', '', no_date, flags=re.IGNORECASE)
    return no_date2.strip()


def load_files(directory: str) -> dict:
    """Return {filename: True} for all .json files (excluding report.json)."""
    if not os.path.isdir(directory):
        return {}
    return {
        f: True
        for f in os.listdir(directory)
        if f.endswith('.json') and f != 'report.json'
    }


def is_merged(fname: str) -> bool:
    return bool(SUFFIX_RE.search(fname))


def compare_category(cat_new: str, cat_old: str) -> dict:
    old_dir = os.path.join(OLD_BASE, cat_old)
    new_dir = os.path.join(NEW_BASE, cat_new)

    old_files = load_files(old_dir)
    new_files = load_files(new_dir)

    old_set = set(old_files)
    new_set = set(new_files)

    # ── Exact filename matches ────────────────────────────────────────────────
    both      = old_set & new_set
    only_old  = old_set - new_set
    only_new  = new_set - old_set

    unchanged = {f for f in both if not is_merged(f)}
    same_merged = {f for f in both if is_merged(f)}

    # ── Detect wrong_split: old had X_MERGED, new has multiple standalones ───
    # Group old merged files by base case name
    old_merged_by_case = defaultdict(list)
    for f in old_set:
        if is_merged(f):
            old_merged_by_case[base_case_name(f)].append(f)

    # Group new standalone files by base case name
    new_standalone_by_case = defaultdict(list)
    for f in new_set:
        if not is_merged(f):
            new_standalone_by_case[base_case_name(f)].append(f)

    # Group new merged files by base case name
    new_merged_by_case = defaultdict(list)
    for f in new_set:
        if is_merged(f):
            new_merged_by_case[base_case_name(f)].append(f)

    wrong_split = []   # old wrongly merged, now separate
    new_merged  = []   # genuinely merged in v3, was standalone(s) in old

    for case, old_m_files in old_merged_by_case.items():
        new_m = new_merged_by_case.get(case, [])
        new_s = new_standalone_by_case.get(case, [])
        if new_s and not new_m:
            # Was merged in old, now split into standalones → wrong merge fixed
            wrong_split.append({
                "case":          case,
                "old_merged":    old_m_files,
                "new_standalone": new_s,
            })

    for case, new_m_files in new_merged_by_case.items():
        old_m = old_merged_by_case.get(case, [])
        old_s_count = len([f for f in old_set if base_case_name(f) == case and not is_merged(f)])
        if not old_m and old_s_count > 0:
            # Was standalone(s) in old, now correctly merged in v3
            new_merged.append({
                "case":         case,
                "new_merged":   new_m_files,
                "old_standalone_count": old_s_count,
            })

    return {
        "category":        cat_new,
        "old_total":       len(old_set),
        "new_total":       len(new_set),
        "unchanged":       len(unchanged),
        "same_merged":     len(same_merged),
        "wrong_split":     wrong_split,
        "new_merged":      new_merged,
        "only_old":        sorted(only_old),
        "only_new":        sorted(only_new),
    }


def main():
    grand = {
        "old_total": 0, "new_total": 0,
        "unchanged": 0, "same_merged": 0,
        "wrong_split": 0, "new_merged": 0,
        "only_old": 0, "only_new": 0,
    }

    all_results = []
    for cat_new, cat_old in sorted(CATEGORY_MAP.items()):
        r = compare_category(cat_new, cat_old)
        all_results.append(r)

        grand["old_total"]   += r["old_total"]
        grand["new_total"]   += r["new_total"]
        grand["unchanged"]   += r["unchanged"]
        grand["same_merged"] += r["same_merged"]
        grand["wrong_split"] += len(r["wrong_split"])
        grand["new_merged"]  += len(r["new_merged"])
        grand["only_old"]    += len(r["only_old"])
        grand["only_new"]    += len(r["only_new"])

    # ── Print ─────────────────────────────────────────────────────────────────
    W = 70
    print("=" * W)
    print("MERGE COMPARISON  (old_outputs  vs  output_merged_v3)")
    print("=" * W)

    for r in all_results:
        cat = r["category"]
        print(f"\n┌── {cat.upper()} {'─'*(W-6-len(cat))}")
        print(f"│  Files in old : {r['old_total']:>5}   Files in v3 : {r['new_total']:>5}")
        print(f"│  Unchanged (standalone) : {r['unchanged']:>5}")
        print(f"│  Unchanged (merged)     : {r['same_merged']:>5}")
        print(f"│  Wrong merges fixed (split in v3) : {len(r['wrong_split']):>4}")
        print(f"│  New genuine merges in v3         : {len(r['new_merged']):>4}")
        print(f"│  Only in old  : {len(r['only_old']):>4}   Only in v3 : {len(r['only_new']):>4}")

        if r["wrong_split"]:
            print(f"│")
            print(f"│  Wrong merges fixed ({len(r['wrong_split'])}):")
            for ws in r["wrong_split"][:10]:
                print(f"│    [WAS MERGED] {ws['old_merged'][0][:60]}")
                print(f"│    → now {len(ws['new_standalone'])} standalone file(s):")
                for sf in ws["new_standalone"][:4]:
                    print(f"│        {sf[:60]}")
                if len(ws["new_standalone"]) > 4:
                    print(f"│        ... and {len(ws['new_standalone'])-4} more")
            if len(r["wrong_split"]) > 10:
                print(f"│    ... and {len(r['wrong_split'])-10} more (see report JSON)")

        if r["new_merged"]:
            print(f"│")
            print(f"│  New genuine merges ({len(r['new_merged'])}):")
            for nm in r["new_merged"][:5]:
                print(f"│    [NOW MERGED] {nm['new_merged'][0][:60]}")
                print(f"│    ← was {nm['old_standalone_count']} standalone(s) in old")
            if len(r["new_merged"]) > 5:
                print(f"│    ... and {len(r['new_merged'])-5} more")

        print(f"└{'─'*(W-1)}")

    print(f"\n{'═'*W}")
    print("GRAND TOTAL")
    print(f"{'═'*W}")
    print(f"  Old total files : {grand['old_total']:>6}")
    print(f"  New total files : {grand['new_total']:>6}")
    print(f"  Unchanged       : {grand['unchanged']:>6}  (same filename, not merged)")
    print(f"  Same merged     : {grand['same_merged']:>6}  (same _MERGED file in both)")
    print(f"  Wrong merges fixed (split) : {grand['wrong_split']:>4}")
    print(f"  New genuine merges         : {grand['new_merged']:>4}")
    print(f"  Only in old     : {grand['only_old']:>6}")
    print(f"  Only in v3      : {grand['only_new']:>6}")
    print()

    # Save detailed report
    report_path = os.path.join(NEW_BASE, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2, ensure_ascii=False)
    print(f"  Full report saved → {report_path}")


if __name__ == "__main__":
    main()
