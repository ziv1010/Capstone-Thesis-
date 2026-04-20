"""
Case Merger Script  (v3 – labelled_jsons compatible)
=====================================================
Supports the flat JSON format produced by the OpenNyAI/labelled pipeline:
  {
    "sentences": [...],          # list of sentence dicts with start/end offsets
    "opennyai_summary": {...},   # section → text
    "case_outcome_label": ...,
    "case_outcome_score": ...,
    "llm_case_outcome": {...}
  }

Filename format expected:
  "Case Name on 15 July, 2024.json"
  "Case Name on 15 July, 2024 (2).json"   # duplicates

Usage:
  python merge_cases_v2.py \\
      --input  /path/to/labelled_jsons \\
      --output /path/to/output_merged   \\
      [--skip-hidden]   # skip files starting with '.' (default: True)

Outputs:
  output_dir/                — one file per case group
  output_dir/report.json
"""

import os, re, json, copy, argparse
from collections import defaultdict
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge multi-hearing legal case JSONs")
    p.add_argument(
        "--input", "-i",
        default="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/fin_fraud_labelled/labelled_jsons",
        help="Directory containing labelled JSON files",
    )
    p.add_argument(
        "--output", "-o",
        default="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v2",
        help="Output directory for merged files",
    )
    p.add_argument(
        "--skip-hidden", action="store_true", default=True,
        help="Skip files whose names start with '.' (default: True)",
    )
    return p.parse_args()


# ── Constants ─────────────────────────────────────────────────────────────────

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

# Matches:  "Case Name on 15 July, 2024.json"
#           "Case Name on 15 July, 2024 (2).json"
FILE_PATTERN = re.compile(
    r'^(.+?)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})\s*(\(\d+\))?\.json$',
    re.IGNORECASE,
)

SEP_TEMPLATE  = "\n\n" + "="*80 + "\n[HEARING: {date} — {fname}]\n" + "="*80 + "\n\n"
SEP_FINAL_TPL = "\n\n" + "="*80 + "\n[FINAL DECISION: {date} — {fname}]\n" + "="*80 + "\n\n"


# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_date(day: str, month: str, year: str) -> datetime:
    """Parse day / month-name / year into a datetime for sorting."""
    return datetime(int(year), MONTH_MAP.get(month.lower(), 1), int(day))


def date_slug(day: str, month: str, year: str) -> str:
    """Return a compact sortable slug like '15_July_2024'."""
    return f"{day}_{month.capitalize()}_{year}"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Accessors for the flat labelled_jsons format ──────────────────────────────

def get_sentences(doc: dict) -> list:
    """Return the sentences list (which carry start/end offsets)."""
    return doc.get("sentences", [])


def get_full_text(doc: dict) -> str:
    """
    Reconstruct the full document text by joining sentence texts.
    We preserve the original character offsets by padding gaps with spaces.
    """
    sentences = get_sentences(doc)
    if not sentences:
        return ""
    # Build text respecting start offsets
    max_end = max(s.get("end", 0) for s in sentences)
    buf = [" "] * max_end
    for s in sentences:
        text = s.get("text", "")
        start = s.get("start", 0)
        for i, ch in enumerate(text):
            if start + i < max_end:
                buf[start + i] = ch
    return "".join(buf)


def get_summary(doc: dict) -> dict:
    return doc.get("opennyai_summary") or {}


# ── Containment check ─────────────────────────────────────────────────────────

def is_genuinely_contained(older_text: str, latest_text: str,
                            n_samples: int = 20, sample_len: int = 150,
                            coverage_threshold: float = 0.60) -> bool:
    if not older_text or not latest_text:
        return False
    if len(older_text) > len(latest_text) * 1.05:
        return False
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
    return (hits / n_samples) >= coverage_threshold


# ── Sentence (annotation) shifting ───────────────────────────────────────────

def shift_sentences(sentences: list, offset: int, hearing_date: str, hearing_file: str) -> list:
    """
    Deep-copy sentences and shift their start/end by `offset`.
    Tags each sentence with which hearing it came from.
    """
    shifted = []
    for s in sentences:
        s2 = copy.deepcopy(s)
        s2["start"] = s2.get("start", 0) + offset
        s2["end"]   = s2.get("end",   0) + offset
        # Shift entity offsets too
        for ent in s2.get("entities", []):
            ent["start"] = ent.get("start", 0) + offset
            ent["end"]   = ent.get("end",   0) + offset
        s2["_hearing_date"] = hearing_date
        s2["_hearing_file"] = hearing_file
        s2["_hearing_offset"] = offset
        shifted.append(s2)
    return shifted


# ── Summary merging ───────────────────────────────────────────────────────────

def merge_summaries(summaries_with_dates: list) -> dict:
    """
    summaries_with_dates: list of (date_slug, summary_dict)
    Keys unique to one hearing keep their name.
    Keys shared across hearings get prefixed with date slug (final hearing → 'final__').
    """
    from collections import Counter
    key_count = Counter()
    for _, summ in summaries_with_dates:
        key_count.update(summ.keys())

    merged = {}
    for date_slug_str, summ in summaries_with_dates:
        is_last = (date_slug_str == summaries_with_dates[-1][0])
        for k, v in summ.items():
            if key_count[k] == 1:
                merged[k] = v
            else:
                new_key = f"final__{k}" if is_last else f"{date_slug_str}__{k}"
                merged[new_key] = v
    return merged


# ── Full document merge ───────────────────────────────────────────────────────

def full_merge(docs_by_date: list) -> dict:
    """
    docs_by_date: [(date_slug, doc_dict, filename), ...] sorted oldest → latest

    Returns a single merged document.
    """
    n = len(docs_by_date)

    # ── Build merged text ──
    full_text = ""
    offset_map = []   # (date_slug, fname, text_start, text_len)
    current_offset = 0

    for i, (ds, doc, fname) in enumerate(docs_by_date):
        is_last = (i == n - 1)
        text = get_full_text(doc)

        if i == 0:
            full_text += text
            offset_map.append((ds, fname, current_offset, len(text)))
            current_offset += len(text)
        else:
            sep = (SEP_FINAL_TPL if is_last else SEP_TEMPLATE).format(date=ds, fname=fname)
            full_text += sep + text
            current_offset += len(sep)
            offset_map.append((ds, fname, current_offset, len(text)))
            current_offset += len(text)

    # ── Build merged sentences (with adjusted offsets) ──
    merged_sentences = []
    new_sentence_id = 1
    for ds, fname, text_start, _ in offset_map:
        doc = next(d for (dts, d, _f) in docs_by_date if dts == ds)
        for s in shift_sentences(get_sentences(doc), text_start, ds, fname):
            s["sentence_id"] = new_sentence_id
            new_sentence_id += 1
            merged_sentences.append(s)

    # ── Build merged summary ──
    summaries_with_dates = [(ds, get_summary(doc)) for ds, doc, _ in docs_by_date]
    merged_summary = merge_summaries(summaries_with_dates)

    # ── Build hearing timeline ──
    hearing_timeline = []
    for ds, doc, fname in docs_by_date:
        hearing_timeline.append({
            "date":              ds,
            "file":              fname,
            "outcome_label":     doc.get("case_outcome_label"),
            "outcome_score":     doc.get("case_outcome_score"),
            "is_final":          (ds == docs_by_date[-1][0]),
            "text_length":       len(get_full_text(doc)),
            "sentence_count":    len(get_sentences(doc)),
        })

    # ── Assemble final doc (base = latest) ──
    latest_ds, latest_doc, latest_fname = docs_by_date[-1]
    merged = copy.deepcopy(latest_doc)

    # Overwrite sentences with merged + offset-adjusted list
    merged["sentences"] = merged_sentences

    # Overwrite summary
    merged["opennyai_summary"] = merged_summary

    # Outcome from latest
    merged["case_outcome_label"] = latest_doc.get("case_outcome_label")
    merged["case_outcome_score"] = latest_doc.get("case_outcome_score")
    merged["llm_case_outcome"]   = latest_doc.get("llm_case_outcome")

    # Metadata
    merged["_merged_from"]     = [fname for _, _, fname in docs_by_date]
    merged["_hearing_timeline"] = hearing_timeline
    merged["_section_offsets"] = [
        {"date": d, "file": f, "text_start": s, "text_length": l}
        for d, f, s, l in offset_map
    ]
    merged["_merge_note"] = (
        f"Merged {n} hearings (oldest→latest). "
        f"Text sections separated by labelled dividers. "
        f"Sentence offsets adjusted for merged text position."
    )

    return merged


def pick_best(docs: list, fnames: list):
    """For duplicate files (same date), keep the one with the most sentences."""
    best = max(range(len(docs)), key=lambda i: len(get_sentences(docs[i])))
    return docs[best], fnames[best]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    BASE_DIR    = args.input
    OUT_DIR     = args.output
    REPORT_PATH = os.path.join(OUT_DIR, "report.json")
    SKIP_HIDDEN = args.skip_hidden

    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = [
        f for f in os.listdir(BASE_DIR)
        if f.endswith(".json")
        and (not SKIP_HIDDEN or not f.startswith("."))
    ]

    by_case: dict[str, list] = defaultdict(list)
    skipped_pattern = []

    for f in all_files:
        m = FILE_PATTERN.match(f)
        if m:
            case_name = m.group(1).strip()
            day, month, year = m.group(2), m.group(3), m.group(4)
            ds = date_slug(day, month, year)
            by_case[case_name].append((ds, day, month, year, f))
        else:
            skipped_pattern.append(f)

    report = {
        "input_dir":            BASE_DIR,
        "total_files":          len(all_files),
        "skipped_pattern_mismatch": skipped_pattern,
        "total_case_groups":    len(by_case),
        "single_file_cases":    0,
        "same_date_duplicates": [],
        "multi_date_cases":     [],
    }

    for case_name, entries in sorted(by_case.items()):
        # Sort entries by date
        entries_sorted = sorted(
            entries,
            key=lambda e: parse_date(e[1], e[2], e[3])
        )
        unique_dates = sorted(set(e[0] for e in entries_sorted),
                              key=lambda ds: parse_date(*ds.split("_")))

        # ── Same-date duplicates ──────────────────────────────────────────────
        if len(unique_dates) == 1 and len(entries_sorted) > 1:
            docs   = [load_json(os.path.join(BASE_DIR, e[4])) for e in entries_sorted]
            fnames = [e[4] for e in entries_sorted]
            best_doc, best_fname = pick_best(docs, fnames)
            best_doc["_merged_from"] = fnames
            best_doc["_merge_note"]  = f"Duplicate files for same date. Kept most sentences ({best_fname})."

            ds = unique_dates[0]
            out_fname = f"{case_name} on {ds}_DEDUPED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as fp:
                json.dump(best_doc, fp, indent=2, ensure_ascii=False)

            report["same_date_duplicates"].append({
                "case": case_name, "date": ds,
                "files": fnames, "kept": best_fname, "output": out_fname,
            })

        # ── Multi-date hearings ───────────────────────────────────────────────
        elif len(unique_dates) > 1:
            date_to_best: dict = {}
            for ds in unique_dates:
                date_entries = [e for e in entries_sorted if e[0] == ds]
                docs     = [load_json(os.path.join(BASE_DIR, e[4])) for e in date_entries]
                fnames_d = [e[4] for e in date_entries]
                date_to_best[ds] = pick_best(docs, fnames_d)

            docs_by_date = [
                (ds, date_to_best[ds][0], date_to_best[ds][1])
                for ds in unique_dates
            ]

            # Containment check
            latest_text = get_full_text(docs_by_date[-1][1])
            containment_check = []
            for ds, doc, fname in docs_by_date[:-1]:
                older_text = get_full_text(doc)
                contained  = is_genuinely_contained(older_text, latest_text)
                containment_check.append({
                    "file": fname, "date": ds,
                    "contained_in_latest": contained,
                    "older_len": len(older_text),
                    "latest_len": len(latest_text),
                })

            all_contained = all(c["contained_in_latest"] for c in containment_check)
            merged_doc    = full_merge(docs_by_date)
            appended      = [c["file"] for c in containment_check if not c["contained_in_latest"]]
            merged_doc["_appended_from"] = appended

            latest_ds = unique_dates[-1]
            out_fname = f"{case_name} on {latest_ds}_MERGED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as fp:
                json.dump(merged_doc, fp, indent=2, ensure_ascii=False)

            report["multi_date_cases"].append({
                "case":   case_name,
                "dates":  unique_dates,
                "files":  [date_to_best[d][1] for d in unique_dates],
                "latest_file": date_to_best[unique_dates[-1]][1],
                "all_earlier_contained_in_latest": all_contained,
                "containment_per_file": containment_check,
                "appended_from": appended,
                "output": out_fname,
            })

        # ── Single file ───────────────────────────────────────────────────────
        else:
            fname = entries_sorted[0][4]
            doc   = load_json(os.path.join(BASE_DIR, fname))
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fp:
                json.dump(doc, fp, indent=2, ensure_ascii=False)
            report["single_file_cases"] += 1

    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    multi          = report["multi_date_cases"]
    merged_count   = sum(1 for m in multi if m["appended_from"])
    contained_count= sum(1 for m in multi if not m["appended_from"])

    print(f"\n{'='*65}")
    print("CASE MERGER SUMMARY  (v3 — labelled_jsons)")
    print(f"{'='*65}")
    print(f"Input directory          : {BASE_DIR}")
    print(f"Total input files        : {report['total_files']}")
    print(f"Pattern-skipped files    : {len(skipped_pattern)}")
    print(f"Total case groups        : {report['total_case_groups']}")
    print(f"Single-file cases        : {report['single_file_cases']}")
    print(f"Same-date duplicates     : {len(report['same_date_duplicates'])}")
    print(f"Multi-date cases         : {len(multi)}")
    print(f"  → Needed append        : {merged_count}")
    print(f"  → Already contained    : {contained_count}")
    if skipped_pattern:
        print(f"\nFiles that didn't match filename pattern ({len(skipped_pattern)}):")
        for sf in skipped_pattern[:10]:
            print(f"  {sf}")
        if len(skipped_pattern) > 10:
            print(f"  ... and {len(skipped_pattern) - 10} more (see report.json)")
    print()
    print("Multi-date cases:")
    for mc in multi:
        status = "MERGED" if mc["appended_from"] else "CONTAINED"
        print(f"  [{status:9}] {mc['case'][:55]}")
        print(f"              dates: {mc['dates']}")
        if mc["appended_from"]:
            print(f"              appended: {[f.split(' on ')[-1] for f in mc['appended_from']]}")
        for cp in mc["containment_per_file"]:
            print(f"              {cp['date']}: contained={cp['contained_in_latest']} "
                  f"older_len={cp['older_len']} latest_len={cp['latest_len']}")
    print(f"\nReport : {REPORT_PATH}")
    print(f"Output : {OUT_DIR}")


if __name__ == "__main__":
    main()
