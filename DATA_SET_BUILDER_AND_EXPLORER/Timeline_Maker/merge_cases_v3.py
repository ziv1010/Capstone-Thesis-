"""
Case Merger Script v4 — case-number-aware grouping
====================================================
Fixes the v3/v2 flaw where different cases with identical party names were merged.

The v2 script grouped solely by the case name extracted from the filename, causing
unrelated cases like "Amit Kumar vs The State Of Bihar on <date1>" and
"Amit Kumar vs The State Of Bihar on <date2>" to be merged even though they were
completely different proceedings with different court-assigned registration numbers.

Fix: extract the court-assigned case registration number from the PREAMBLE section
of each document. Group by (case_name, case_number) instead of case_name alone.
Files where no case number can be extracted are kept as standalone (never merged).

Filename format expected:
  "Case Name on 15 July, 2024.json"
  "Case Name on 15 July, 2024 (2).json"   # same-date duplicates

Usage:
  python merge_cases_v3.py \\
      --input  /path/to/augmented_jsons \\
      --output /path/to/output_merged   \\
      [--skip-hidden]

Outputs:
  output_dir/                — one file per case group
  output_dir/report.json
"""

import os, re, json, copy, argparse
from collections import defaultdict
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge multi-hearing legal case JSONs (v4 — case-number-aware)")
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing augmented/labelled JSON files",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
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

FILE_PATTERN = re.compile(
    r'^(.+?)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})\s*(\(\d+\))?\.json$',
    re.IGNORECASE,
)

SEP_TEMPLATE  = "\n\n" + "="*80 + "\n[HEARING: {date} — {fname}]\n" + "="*80 + "\n\n"
SEP_FINAL_TPL = "\n\n" + "="*80 + "\n[FINAL DECISION: {date} — {fname}]\n" + "="*80 + "\n\n"

# Regex patterns applied against text to extract a case registration number.
# Two-group patterns return (number_part, year). Single-group patterns (CNR)
# return a full alphanumeric key string.
#
# Formats handled:
#   "No. 5154/2014"              → "5154/2014"
#   "No.51986 of 2023"           → "51986/2023"
#   "No. - 5941 of 2024"         → "5941/2024"
#   "No.10341/2023"              → "10341/2023"
#   "NOs: 3054, 3056 of 2023"    → "3054/2023"
#   "Crm-M-44633-2024"           → "44633/2024"  (Punjab/Haryana 3-part)
#   "Crm-M-7993-2021 (O&M)"      → "7993/2021"
#   "CNR No.: DLNT01-003755-2016" → "DLNT01-003755-2016"
#   "+ W.P.(C) 11139/2022"       → "11139/2022"  (Delhi HC '+' prefix)
#   "902/2024"                   → "902/2024"    (broad DIGITS/YEAR fallback)
CASE_NO_PATTERNS = [
    # "No." / "Nos." then optional punctuation then digits then /YYYY
    re.compile(r'\bNo[s]?\.?\s*[-–:]?\s*(\d[\d,\s]*?)\s*/\s*(\d{4})\b', re.IGNORECASE),
    # "No." / "Nos." then optional punctuation then digits then "of YYYY"
    re.compile(r'\bNo[s]?\.?\s*[-–:]?\s*(\d[\d,\s]*?)\s+of\s+(\d{4})\b', re.IGNORECASE),
    # Punjab/Haryana dash-year: "Crm-M-44633-2024", "CRR-2345-2023"
    re.compile(r'\b[A-Za-z][A-Za-z.-]*-(\d{3,})-(\d{4})\b'),
    # Punjab/Haryana "of"-year: "Crm-M-35074 of 2023", "Crl.Rev.-178 of 2022"
    re.compile(r'\b[A-Za-z][A-Za-z.-]*-\s*(\d{3,})\s+of\s+(\d{4})\b', re.IGNORECASE),
    # CNR number (alphanumeric, explicit label): "CNR No.: DLNT01-003755-2016"
    re.compile(r'\bCNR\s+No\.?\s*:?\s*([\w][\w-]{5,})', re.IGNORECASE),
    # Delhi HC "+" prefix lines: "+ W.P.(C) 11139/2022" / "+ Bail Appln. 4825/2024"
    re.compile(r'^\+\s+\S.*?(\d{3,})/((?:19|20)\d{2})\b', re.MULTILINE),
    # Broad fallback: 3+-digit number / 4-digit year (excludes DD/MM dates which are ≤2 digits)
    re.compile(r'\b(\d{3,})/((?:19|20)\d{2})\b'),
]

# Indices of patterns that produce two capture groups (number_part, year).
# Pattern index 4 (CNR) is the only single-group outlier.
# List has indices 0-6: 0,1=No-style; 2,3=Punjab; 4=CNR; 5=Delhi+; 6=broad
_TWO_GROUP_IDXS = (0, 1, 2, 3, 5, 6)


def _match_patterns(text: str, min_number_len: int = 2) -> str | None:
    """Apply CASE_NO_PATTERNS against `text`; return normalised key or None."""
    for idx in _TWO_GROUP_IDXS:
        m = CASE_NO_PATTERNS[idx].search(text)
        if m:
            raw = re.sub(r'[\s,]+', '', m.group(1))
            year = m.group(2)
            if raw.isdigit() and len(raw) >= min_number_len:
                return f"{raw}/{year}"
    # CNR single-group (index 4)
    m = CASE_NO_PATTERNS[4].search(text)
    if m:
        return m.group(1).upper()
    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_date(day: str, month: str, year: str) -> datetime:
    return datetime(int(year), MONTH_MAP.get(month.lower(), 1), int(day))


def date_slug(day: str, month: str, year: str) -> str:
    return f"{day}_{month.capitalize()}_{year}"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Case number extraction ────────────────────────────────────────────────────

def extract_case_number(doc: dict) -> str | None:
    """
    Extract the primary court case registration number for this document.
    Returns a normalised key like '5941/2024' or 'DLNT01-003755-2016', or None.

    Three-tier search (each restricted to preamble material to avoid picking up
    case numbers from cited precedents in the body text):

    1. Regex on opennyai_summary['PREAMBLE'] text (fast, usually sufficient).
    2. CASE_NUMBER entities tagged by the OpenNyAI NER model in PREAMBLE-role
       sentences — the NER already identified candidate spans; we normalise them.
       A lower minimum digit length (1) is safe here because the NER model
       has already confirmed the span is a case number.
    3. Regex on the full concatenated text of all PREAMBLE-role sentences.
       Catches courts (e.g. Punjab & Haryana) where the preamble summary is
       truncated and the full registration number only appears in the sentence
       body.
    """
    # ── Tier 1: summary preamble ──────────────────────────────────────────────
    summary_preamble = (doc.get("opennyai_summary") or {}).get("PREAMBLE", "")
    result = _match_patterns(summary_preamble, min_number_len=2)
    if result:
        return result

    # ── Collect PREAMBLE-role sentences once (used by tiers 2 & 3) ───────────
    preamble_sents = [
        s for s in doc.get("sentences", [])
        if s.get("rhetorical_role") == "PREAMBLE"
    ]

    # ── Tier 2: CASE_NUMBER entities from PREAMBLE sentences (NER-sourced) ───
    for sent in preamble_sents:
        for ent in sent.get("entities", []):
            if ent.get("label") == "CASE_NUMBER":
                ent_text = ent.get("text", "").replace("\n", " ").strip()
                result = _match_patterns(ent_text, min_number_len=1)
                if result:
                    return result

    # ── Tier 3: regex on full PREAMBLE sentence text ──────────────────────────
    full_preamble = " ".join(s.get("text", "") for s in preamble_sents)
    result = _match_patterns(full_preamble, min_number_len=2)
    if result:
        return result

    return None


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_sentences(doc: dict) -> list:
    return doc.get("sentences", [])


def get_full_text(doc: dict) -> str:
    sentences = get_sentences(doc)
    if not sentences:
        return ""
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


# ── Sentence shifting ─────────────────────────────────────────────────────────

def shift_sentences(sentences: list, offset: int, hearing_date: str, hearing_file: str) -> list:
    shifted = []
    for s in sentences:
        s2 = copy.deepcopy(s)
        s2["start"] = s2.get("start", 0) + offset
        s2["end"]   = s2.get("end",   0) + offset
        for ent in s2.get("entities", []):
            ent["start"] = ent.get("start", 0) + offset
            ent["end"]   = ent.get("end",   0) + offset
        s2["_hearing_date"]   = hearing_date
        s2["_hearing_file"]   = hearing_file
        s2["_hearing_offset"] = offset
        shifted.append(s2)
    return shifted


# ── Summary merging ───────────────────────────────────────────────────────────

def merge_summaries(summaries_with_dates: list) -> dict:
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
    """docs_by_date: [(date_slug, doc_dict, filename), ...] sorted oldest → latest"""
    n = len(docs_by_date)

    full_text = ""
    offset_map = []
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

    merged_sentences = []
    new_sentence_id = 1
    for ds, fname, text_start, _ in offset_map:
        doc = next(d for (dts, d, _f) in docs_by_date if dts == ds)
        for s in shift_sentences(get_sentences(doc), text_start, ds, fname):
            s["sentence_id"] = new_sentence_id
            new_sentence_id += 1
            merged_sentences.append(s)

    summaries_with_dates = [(ds, get_summary(doc)) for ds, doc, _ in docs_by_date]
    merged_summary = merge_summaries(summaries_with_dates)

    hearing_timeline = []
    for ds, doc, fname in docs_by_date:
        hearing_timeline.append({
            "date":           ds,
            "file":           fname,
            "outcome_label":  doc.get("case_outcome_label"),
            "outcome_score":  doc.get("case_outcome_score"),
            "is_final":       (ds == docs_by_date[-1][0]),
            "text_length":    len(get_full_text(doc)),
            "sentence_count": len(get_sentences(doc)),
        })

    latest_ds, latest_doc, latest_fname = docs_by_date[-1]
    merged = copy.deepcopy(latest_doc)
    merged["sentences"]        = merged_sentences
    merged["opennyai_summary"] = merged_summary
    merged["case_outcome_label"] = latest_doc.get("case_outcome_label")
    merged["case_outcome_score"] = latest_doc.get("case_outcome_score")
    merged["llm_case_outcome"]   = latest_doc.get("llm_case_outcome")
    merged["_merged_from"]       = [fname for _, _, fname in docs_by_date]
    merged["_hearing_timeline"]  = hearing_timeline
    merged["_section_offsets"]   = [
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
    """For same-date duplicates, keep the one with the most sentences."""
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

    # ── Parse filenames + load docs to extract case numbers ──────────────────
    # by_group[(case_name, case_number_key)] = [(date_slug, day, month, year, fname, doc)]
    # case_number_key is "DIGITS/YEAR" or None (standalone)
    by_group: dict = defaultdict(list)
    skipped_pattern = []
    no_case_number  = []   # files that parsed OK but had no extractable case number

    for f in all_files:
        m = FILE_PATTERN.match(f)
        if not m:
            skipped_pattern.append(f)
            continue

        case_name = m.group(1).strip()
        day, month, year = m.group(2), m.group(3), m.group(4)
        ds  = date_slug(day, month, year)
        doc = load_json(os.path.join(BASE_DIR, f))

        case_number = extract_case_number(doc)

        if case_number is None:
            # Cannot verify identity — treat as standalone, never merged
            no_case_number.append(f)
            # Use a unique sentinel so it never merges with anything
            group_key = (case_name, f"__solo__{f}")
        else:
            group_key = (case_name, case_number)

        by_group[group_key].append((ds, day, month, year, f, doc))

    report = {
        "input_dir":                BASE_DIR,
        "total_files":              len(all_files),
        "skipped_pattern_mismatch": skipped_pattern,
        "no_case_number_extracted": no_case_number,
        "total_case_groups":        0,
        "single_file_cases":        0,
        "same_date_duplicates":     [],
        "multi_date_cases":         [],
    }

    # ── Process each group ────────────────────────────────────────────────────
    for (case_name, case_number_key), entries in sorted(by_group.items()):
        entries_sorted = sorted(entries, key=lambda e: parse_date(e[1], e[2], e[3]))
        unique_dates   = sorted(
            set(e[0] for e in entries_sorted),
            key=lambda d: parse_date(*d.split("_")),
        )
        report["total_case_groups"] += 1

        # ── Same-date duplicates ──────────────────────────────────────────────
        if len(unique_dates) == 1 and len(entries_sorted) > 1:
            docs   = [e[5] for e in entries_sorted]
            fnames = [e[4] for e in entries_sorted]
            best_doc, best_fname = pick_best(docs, fnames)
            best_doc["_merged_from"] = fnames
            best_doc["_merge_note"]  = f"Duplicate files for same date. Kept most sentences ({best_fname})."

            ds        = unique_dates[0]
            out_fname = f"{case_name} on {ds}_DEDUPED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as fp:
                json.dump(best_doc, fp, indent=2, ensure_ascii=False)

            report["same_date_duplicates"].append({
                "case":        case_name,
                "case_number": case_number_key,
                "date":        ds,
                "files":       fnames,
                "kept":        best_fname,
                "output":      out_fname,
            })

        # ── Multi-date hearings (same case number, different hearing dates) ──
        elif len(unique_dates) > 1:
            date_to_best: dict = {}
            for ds in unique_dates:
                date_entries = [e for e in entries_sorted if e[0] == ds]
                docs_d   = [e[5] for e in date_entries]
                fnames_d = [e[4] for e in date_entries]
                date_to_best[ds] = pick_best(docs_d, fnames_d)

            docs_by_date = [
                (ds, date_to_best[ds][0], date_to_best[ds][1])
                for ds in unique_dates
            ]

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
            merged_doc["_appended_from"]  = appended
            merged_doc["_case_number"]    = case_number_key

            latest_ds = unique_dates[-1]
            out_fname = f"{case_name} on {latest_ds}_MERGED.json"
            with open(os.path.join(OUT_DIR, out_fname), "w", encoding="utf-8") as fp:
                json.dump(merged_doc, fp, indent=2, ensure_ascii=False)

            report["multi_date_cases"].append({
                "case":        case_name,
                "case_number": case_number_key,
                "dates":       unique_dates,
                "files":       [date_to_best[d][1] for d in unique_dates],
                "latest_file": date_to_best[unique_dates[-1]][1],
                "all_earlier_contained_in_latest": all_contained,
                "containment_per_file": containment_check,
                "appended_from": appended,
                "output":      out_fname,
            })

        # ── Single file ───────────────────────────────────────────────────────
        else:
            fname = entries_sorted[0][4]
            doc   = entries_sorted[0][5]
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fp:
                json.dump(doc, fp, indent=2, ensure_ascii=False)
            report["single_file_cases"] += 1

    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    multi          = report["multi_date_cases"]
    merged_count   = sum(1 for mc in multi if mc["appended_from"])
    contained_count= sum(1 for mc in multi if not mc["appended_from"])

    print(f"\n{'='*65}")
    print("CASE MERGER SUMMARY  (v4 — case-number-aware grouping)")
    print(f"{'='*65}")
    print(f"Input directory          : {BASE_DIR}")
    print(f"Total input files        : {report['total_files']}")
    print(f"Pattern-skipped files    : {len(skipped_pattern)}")
    print(f"No case-number extracted : {len(no_case_number)} (kept as standalone)")
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

    if no_case_number:
        print(f"\nFiles kept standalone (no case number found) ({len(no_case_number)}):")
        for sf in no_case_number[:10]:
            print(f"  {sf}")
        if len(no_case_number) > 10:
            print(f"  ... and {len(no_case_number) - 10} more (see report.json)")

    if multi:
        print("\nMulti-date merged cases:")
        for mc in multi:
            status = "MERGED" if mc["appended_from"] else "CONTAINED"
            print(f"  [{status:9}] {mc['case'][:45]}  [No. {mc['case_number']}]")
            print(f"              dates: {mc['dates']}")

    print(f"\nReport : {REPORT_PATH}")
    print(f"Output : {OUT_DIR}")


if __name__ == "__main__":
    main()
