"""
Case Merger Script  (v2)
========================
Groups JSON case files by case name, identifies multi-date hearings of the same case,
checks if the latest file contains earlier hearing info, and if not, merges them.

Fixes vs v1:
  - Correct chronological ordering (oldest → latest)
  - Proper containment check (coverage ratio, not 4 lucky samples)
  - Full JSON merge: annotations (with adjusted char offsets) + summary (all sections)
  - Hearing timeline metadata embedded in output

Outputs:
  output_merged/          — one file per case group
  output_merged/report.json
"""

import os, re, json, copy
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "input_data", "augmented_jsons")
OUT_DIR  = os.path.join(SCRIPT_DIR, "output_merged")
REPORT_PATH = os.path.join(OUT_DIR, "report.json")

os.makedirs(OUT_DIR, exist_ok=True)

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}
FILE_PATTERN = re.compile(r'^(.+)_on_(\d+_\w+_\d{4})(\s*\(\d+\))?\.json$')

# Separator used in merged text — also parsed by visualiser
SEP_TEMPLATE  = "\n\n" + "="*80 + "\n[HEARING: {date} — {fname}]\n" + "="*80 + "\n\n"
SEP_FINAL_TPL = "\n\n" + "="*80 + "\n[FINAL DECISION: {date} — {fname}]\n" + "="*80 + "\n\n"


# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime:
    parts = date_str.lower().split("_")
    return datetime(int(parts[2]), MONTH_MAP.get(parts[1], 1), int(parts[0]))

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_text(doc: dict) -> str:
    return doc.get("raw_result", {}).get("data", {}).get("text", "")

def get_annotations(doc: dict) -> list:
    return doc.get("raw_result", {}).get("annotations", [])

def get_summary(doc: dict) -> dict:
    return doc.get("raw_result", {}).get("summary", {}) or {}


# ── Containment check (v2) ────────────────────────────────────────────────────

def is_genuinely_contained(older_text: str, latest_text: str,
                            n_samples: int = 20, sample_len: int = 150,
                            coverage_threshold: float = 0.60) -> bool:
    """
    Returns True only if a substantial fraction of the older text's content
    appears verbatim in the latest text.

    Logic:
      1. If older is longer than latest it CANNOT be contained → False
      2. Strip the first 10% of older text (header/party boilerplate shared by all hearings)
      3. Take n_samples evenly spread across the body
      4. Count what fraction are found verbatim in latest
      5. True only if coverage >= threshold
    """
    if not older_text or not latest_text:
        return False

    # Hard rule: if older is longer than latest it can't be fully contained
    if len(older_text) > len(latest_text) * 1.05:
        return False

    # Strip preamble boilerplate (first 10%)
    body = older_text[len(older_text) // 10:]
    if len(body) < sample_len:
        return False

    step = max(1, (len(body) - sample_len) // n_samples)
    hits = 0
    for i in range(n_samples):
        start = i * step
        chunk = body[start: start + sample_len].strip()
        if len(chunk) >= 80 and chunk in latest_text:
            hits += 1

    coverage = hits / n_samples
    return coverage >= coverage_threshold


# ── Annotation merging ────────────────────────────────────────────────────────

def shift_annotations(annotations: list, offset: int) -> list:
    """Return a deep copy of annotations with start/end shifted by offset."""
    shifted = []
    for ann in annotations:
        a = copy.deepcopy(ann)
        a["start"] = a.get("start", 0) + offset
        a["end"]   = a.get("end",   0) + offset
        # Tag which hearing this came from
        a["_hearing_offset"] = offset
        shifted.append(a)
    return shifted


# ── Summary merging ───────────────────────────────────────────────────────────

def merge_summaries(summaries_with_dates: list[tuple[str, dict]]) -> dict:
    """
    Merge summaries from all hearings.
    - Keys unique to one hearing keep their name.
    - Keys that appear in multiple hearings are prefixed with their date.
    - The final hearing's keys get a 'final_' prefix for the standard sections.
    """
    # Count key occurrences
    from collections import Counter
    key_count = Counter()
    for _, summ in summaries_with_dates:
        key_count.update(summ.keys())

    merged = {}
    for date, summ in summaries_with_dates:
        is_last = (date == summaries_with_dates[-1][0])
        date_slug = date.replace("_", "-")
        for k, v in summ.items():
            if key_count[k] == 1:
                # Only appears in one hearing — keep as-is
                merged[k] = v
            else:
                # Appears in multiple hearings — prefix with date
                new_key = f"{date_slug}__{k}" if not is_last else f"final__{k}"
                merged[new_key] = v
    return merged


# ── Full document merge ───────────────────────────────────────────────────────

def full_merge(docs_by_date: list[tuple[str, dict, str]]) -> dict:
    """
    docs_by_date: list of (date_str, doc_dict, filename) sorted oldest→latest

    Returns a single merged document with:
      - Full concatenated text (oldest→latest, each section labelled)
      - All annotations from all files, char offsets adjusted
      - Summary: all sections from all hearings merged
      - Hearing timeline metadata
      - Outcome fields from the latest hearing
    """
    n = len(docs_by_date)

    # ── Build merged text ──
    text_parts  = []   # (date, fname, text, sep_str)
    offset_map  = []   # (date, fname, text_start_in_merged, text_len)

    current_offset = 0
    full_text = ""

    for i, (date, doc, fname) in enumerate(docs_by_date):
        is_last = (i == n - 1)
        text = get_text(doc)

        if i == 0:
            # First section: no leading separator
            full_text += text
            offset_map.append((date, fname, current_offset, len(text)))
            current_offset += len(text)
        else:
            # Add separator before this section
            if is_last:
                sep = SEP_FINAL_TPL.format(date=date, fname=fname)
            else:
                sep = SEP_TEMPLATE.format(date=date, fname=fname)
            full_text += sep + text
            current_offset += len(sep)
            offset_map.append((date, fname, current_offset, len(text)))
            current_offset += len(text)

    # ── Build merged annotations ──
    merged_annotations = []
    for date, fname, text_start, _ in offset_map:
        # Find the doc for this date
        doc = next(d for (dt, d, _f) in docs_by_date if dt == date)
        anns = get_annotations(doc)
        shifted = shift_annotations(anns, text_start)
        # Tag each annotation with its hearing date
        for a in shifted:
            a["_hearing_date"] = date
            a["_hearing_file"] = fname
        merged_annotations.extend(shifted)

    # ── Build merged summary ──
    summaries_with_dates = [(date, get_summary(doc)) for date, doc, _ in docs_by_date]
    merged_summary = merge_summaries(summaries_with_dates)

    # ── Build hearing timeline ──
    hearing_timeline = []
    for date, doc, fname in docs_by_date:
        hearing_timeline.append({
            "date":          date,
            "file":          fname,
            "outcome_label": doc.get("case_outcome_label"),
            "outcome_score": doc.get("case_outcome_score"),
            "is_final":      (date == docs_by_date[-1][0]),
            "text_length":   len(get_text(doc)),
            "annotation_count": len(get_annotations(doc)),
        })

    # ── Assemble final doc (base = latest) ──
    latest_date, latest_doc, latest_fname = docs_by_date[-1]
    merged = copy.deepcopy(latest_doc)

    merged["raw_result"]["data"]["text"]              = full_text
    merged["raw_result"]["data"]["preamble_end_char_offset"] = \
        docs_by_date[0][1].get("raw_result", {}).get("data", {}).get("preamble_end_char_offset", 0)
    merged["raw_result"]["annotations"]               = merged_annotations
    merged["raw_result"]["summary"]                   = merged_summary

    # Outcome from latest
    merged["case_outcome_label"]  = latest_doc.get("case_outcome_label")
    merged["case_outcome_score"]  = latest_doc.get("case_outcome_score")
    merged["llm_case_outcome"]    = latest_doc.get("llm_case_outcome")

    # Metadata
    merged["_merged_from"]        = [fname for _, _, fname in docs_by_date]
    merged["_hearing_timeline"]   = hearing_timeline
    merged["_section_offsets"]    = [
        {"date": d, "file": f, "text_start": s, "text_length": l}
        for d, f, s, l in offset_map
    ]
    merged["_merge_note"] = (
        f"Merged {n} hearings (oldest→latest). "
        f"Text sections separated by labelled dividers. "
        f"Annotation offsets adjusted for merged text position."
    )

    return merged


def pick_best(docs: list[dict], fnames: list[str]) -> tuple[dict, str]:
    """For duplicate files (same date), keep the one with the longest text."""
    best = max(range(len(docs)), key=lambda i: len(get_text(docs[i])))
    return docs[best], fnames[best]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".json")]

    by_case: dict[str, list] = defaultdict(list)
    for f in all_files:
        m = FILE_PATTERN.match(f)
        if m:
            by_case[m.group(1)].append((m.group(2), m.group(3), f))

    report = {
        "total_files": len(all_files),
        "total_case_groups": len(by_case),
        "single_file_cases": 0,
        "same_date_duplicates": [],
        "multi_date_cases": [],
    }

    for case_name, entries in sorted(by_case.items()):
        unique_dates = sorted(set(e[0] for e in entries), key=parse_date)

        # ── Same date duplicates ──────────────────────────────────────────────
        if len(unique_dates) == 1 and len(entries) > 1:
            docs   = [load_json(os.path.join(BASE_DIR, e[2])) for e in entries]
            fnames = [e[2] for e in entries]
            best_doc, best_fname = pick_best(docs, fnames)
            best_doc["_merged_from"] = fnames
            best_doc["_merge_note"]  = f"Duplicate files for same date. Kept longest text ({best_fname})."

            out_fname = f"{case_name}_on_{unique_dates[0]}_DEDUPED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as f:
                json.dump(best_doc, f, indent=2, ensure_ascii=False)

            report["same_date_duplicates"].append({
                "case": case_name, "date": unique_dates[0],
                "files": fnames, "kept": best_fname, "output": out_fname,
            })

        # ── Multi-date hearings ───────────────────────────────────────────────
        elif len(unique_dates) > 1:
            # For each date, pick best representative
            date_to_best: dict[str, tuple[dict, str]] = {}
            for date in unique_dates:
                date_entries = [e for e in entries if e[0] == date]
                docs   = [load_json(os.path.join(BASE_DIR, e[2])) for e in date_entries]
                fnames_d = [e[2] for e in date_entries]
                date_to_best[date] = pick_best(docs, fnames_d)

            # Build list sorted oldest → latest
            docs_by_date = [
                (date, date_to_best[date][0], date_to_best[date][1])
                for date in unique_dates
            ]

            # Containment check (for report only — we always build the full merge)
            latest_text = get_text(docs_by_date[-1][1])
            containment_check = []
            for date, doc, fname in docs_by_date[:-1]:
                older_text = get_text(doc)
                contained  = is_genuinely_contained(older_text, latest_text)
                containment_check.append({
                    "file": fname, "date": date,
                    "contained_in_latest": contained,
                    "older_len": len(older_text),
                    "latest_len": len(latest_text),
                })

            all_contained = all(c["contained_in_latest"] for c in containment_check)

            # Build full merged JSON
            merged_doc = full_merge(docs_by_date)

            # Also note which sections were NOT contained (appended_from)
            appended = [c["file"] for c in containment_check if not c["contained_in_latest"]]
            merged_doc["_appended_from"] = appended

            latest_date = unique_dates[-1]
            out_fname = f"{case_name}_on_{latest_date}_MERGED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as f:
                json.dump(merged_doc, f, indent=2, ensure_ascii=False)

            report["multi_date_cases"].append({
                "case": case_name,
                "dates": unique_dates,
                "files": [date_to_best[d][1] for d in unique_dates],
                "latest_file": date_to_best[unique_dates[-1]][1],
                "all_earlier_contained_in_latest": all_contained,
                "containment_per_file": containment_check,
                "appended_from": appended,
                "output": out_fname,
            })

        # ── Single file ───────────────────────────────────────────────────────
        else:
            fname = entries[0][2]
            doc = load_json(os.path.join(BASE_DIR, fname))
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            report["single_file_cases"] += 1

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    multi  = report["multi_date_cases"]
    merged_count    = sum(1 for m in multi if m["appended_from"])
    contained_count = sum(1 for m in multi if not m["appended_from"])

    print(f"\n{'='*65}")
    print("CASE MERGER SUMMARY  (v2)")
    print(f"{'='*65}")
    print(f"Total input files        : {report['total_files']}")
    print(f"Total case groups        : {report['total_case_groups']}")
    print(f"Single-file cases        : {report['single_file_cases']}")
    print(f"Same-date duplicates     : {len(report['same_date_duplicates'])}")
    print(f"Multi-date cases         : {len(multi)}")
    print(f"  → Needed append        : {merged_count}")
    print(f"  → Already contained    : {contained_count}")
    print()
    print("Multi-date cases:")
    for mc in multi:
        status = "MERGED" if mc["appended_from"] else "CONTAINED"
        print(f"  [{status:9}] {mc['case'][:55]}")
        print(f"              dates: {mc['dates']}")
        if mc["appended_from"]:
            print(f"              appended: {[f.split('_on_')[-1] for f in mc['appended_from']]}")
        for cp in mc["containment_per_file"]:
            print(f"              {cp['date']}: contained={cp['contained_in_latest']} older_len={cp['older_len']} latest_len={cp['latest_len']}")
    print(f"\nReport : {REPORT_PATH}")
    print(f"Output : {OUT_DIR}")


if __name__ == "__main__":
    main()
