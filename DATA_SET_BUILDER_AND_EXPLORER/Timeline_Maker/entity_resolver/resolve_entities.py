"""
Three-pass entity resolution for legal documents.

Reads:   input dir of JSONs (each JSON has 'sentences' -> list[ {entities: [...]} ]).
Writes:  output dir of JSONs with the same shape, plus each STATUTE/PROVISION/
         PRECEDENT entity gets two new fields: 'canonical_id' and 'canonical_name'.

Targeted entity labels (case-insensitive in input):  STATUTE, PROVISION, PRECEDENT.
All other labels pass through unchanged.

Pass 1 -- collect: walk every JSON, gather unique (raw_text, label) tuples and
                   count their occurrences across the corpus.
Pass 2 -- build maps: deterministically resolve each unique raw text.
                   - statutes: rule-based alias dict
                   - provisions: regex parser
                   - precedents: citation extractor + transitive citation-set
                                 union-find (so {a,b} and {b,c} -> {a,b,c})
Pass 3 -- apply: walk every JSON again, attach canonical_id/canonical_name,
                 write to the output dir mirroring the input layout.

Outputs (alongside the resolved corpus):
   _entity_maps/{statutes,provisions,precedents}.json
   _audit/{statute_merges,provision_merges,precedent_merges}.csv
   _audit/unmapped_{statutes,provisions,precedents}.csv
   _audit/run_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from statute_aliases import resolve_statute, normalize_for_alias  # noqa: E402
from provision_parser import parse_provision  # noqa: E402
from precedent_resolver import resolve_precedent, _light_normalize  # noqa: E402

TARGET_LABELS = {"STATUTE", "PROVISION", "PRECEDENT"}


# ------------------------------ Union-Find for precedents ------------------------------

class UF:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ------------------------------ I/O helpers ------------------------------

def iter_json_files(root: Path):
    for p in root.rglob("*.json"):
        yield p


def load_json(path: Path) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
        return None


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def iter_entities(doc: dict):
    """Yield (sentence_idx, entity_idx, entity_dict, label_upper) for each entity in a doc."""
    sents = doc.get("sentences") or []
    for si, s in enumerate(sents):
        ents = s.get("entities") or []
        for ei, e in enumerate(ents):
            label = (e.get("label") or "").upper()
            yield si, ei, e, label


# ------------------------------ Pass 1 ------------------------------

def collect_unique_entities(input_root: Path) -> dict[str, Counter]:
    """Return {label_upper: Counter(raw_text -> count)} for STATUTE/PROVISION/PRECEDENT."""
    counts: dict[str, Counter] = {lab: Counter() for lab in TARGET_LABELS}
    n_files = 0
    n_entities = 0
    t0 = time.time()
    for path in iter_json_files(input_root):
        n_files += 1
        doc = load_json(path)
        if not isinstance(doc, dict):
            continue
        for _, _, e, label in iter_entities(doc):
            if label not in TARGET_LABELS:
                continue
            text = (e.get("text") or "").strip()
            if not text:
                continue
            counts[label][text] += 1
            n_entities += 1
        if n_files % 5000 == 0:
            print(f"  pass-1: {n_files} files, {n_entities} target entities, "
                  f"{time.time() - t0:.1f}s", flush=True)
    print(f"[pass-1 done] {n_files} files, {n_entities} target entities, "
          f"{time.time() - t0:.1f}s")
    print(f"  unique statutes:   {len(counts['STATUTE'])}")
    print(f"  unique provisions: {len(counts['PROVISION'])}")
    print(f"  unique precedents: {len(counts['PRECEDENT'])}")
    return counts


# ------------------------------ Pass 2 ------------------------------

def build_statute_map(raw_counts: Counter) -> tuple[dict, list]:
    """
    raw_text -> { canonical_id, canonical_name, normalized, count }.
    Returns (map, unmapped_rows).
    """
    out: dict[str, dict] = {}
    unmapped: list[tuple[str, int]] = []
    for raw, cnt in raw_counts.items():
        cid, disp, norm = resolve_statute(raw)
        if cid is None:
            # Keep as its own canonical, identified by normalized text (so trivial
            # spelling variants like "X" vs " X " still merge). Display = raw.
            cid = f"statute:_unmapped:{norm}" if norm else f"statute:_unmapped:{raw.lower()}"
            disp = raw.strip()
            unmapped.append((raw, cnt))
        out[raw] = {
            "canonical_id": cid,
            "canonical_name": disp,
            "normalized": norm,
            "count": cnt,
        }
    return out, unmapped


def build_provision_map(raw_counts: Counter) -> tuple[dict, list]:
    out: dict[str, dict] = {}
    unmapped: list[tuple[str, int]] = []
    for raw, cnt in raw_counts.items():
        cid, disp = parse_provision(raw)
        if cid is None:
            norm_key = raw.strip().lower()
            cid = f"provision:_unparsed:{norm_key}"
            disp = raw.strip()
            unmapped.append((raw, cnt))
        out[raw] = {
            "canonical_id": cid,
            "canonical_name": disp,
            "count": cnt,
        }
    return out, unmapped


def build_precedent_map(raw_counts: Counter) -> tuple[dict, list]:
    """
    Per-mention citation extraction. The canonical id of each precedent is
    PRIMARY citation only -- the first citation token extracted from the raw
    text. Multiple-citation strings are NOT unioned across distinct cases,
    because a single mention can list several unrelated citations and unioning
    would falsely merge them. (Example we saw in the smoke test: a text listing
    six different SCC citations across five distinct cases caused all five to
    collapse into one canonical -- bad.)

    A precedent that quotes parallel citations of the SAME case (e.g.
    "(2014) 6 SCC 466 : (2014) 3 SCC (Cri) 54") will land on the first
    extracted citation. The parallel form on its own (in another document)
    will land on a different canonical -- a missed merge, but per the
    project's high-precision rule, missed merges beat false merges.
    """
    raw_to_cites: dict[str, list[str]] = {}
    unmapped: list[tuple[str, int]] = []

    for raw, cnt in raw_counts.items():
        _, _, cites = resolve_precedent(raw)
        if not cites:
            unmapped.append((raw, cnt))
        raw_to_cites[raw] = cites

    # Pick a representative display name per primary-citation: most-frequent raw,
    # tie-broken by length (longer is usually the fuller form, e.g. with party names).
    primary_to_best_raw: dict[str, str] = {}
    primary_to_best_count: dict[str, int] = {}
    for raw, cnt in raw_counts.items():
        cites = raw_to_cites.get(raw, [])
        if not cites:
            continue
        primary = cites[0]
        prev_cnt = primary_to_best_count.get(primary, -1)
        if cnt > prev_cnt or (cnt == prev_cnt and len(raw) > len(primary_to_best_raw.get(primary, ""))):
            primary_to_best_raw[primary] = raw
            primary_to_best_count[primary] = cnt

    out: dict[str, dict] = {}
    for raw, cnt in raw_counts.items():
        cites = raw_to_cites.get(raw, [])
        if cites:
            primary = cites[0]
            cid = "precedent:" + primary
            disp = primary_to_best_raw.get(primary, raw).strip()
            out[raw] = {
                "canonical_id": cid,
                "canonical_name": disp,
                "primary_citation": primary,
                "all_citations_in_text": cites,
                "count": cnt,
            }
        else:
            norm_key = _light_normalize(raw) or raw.strip().lower()
            cid = f"precedent:_unmapped:{norm_key}"
            out[raw] = {
                "canonical_id": cid,
                "canonical_name": raw.strip(),
                "primary_citation": None,
                "all_citations_in_text": [],
                "count": cnt,
            }
    return out, unmapped


# ------------------------------ Pass 3 ------------------------------

def apply_maps(input_root: Path, output_root: Path, maps: dict[str, dict]) -> tuple[int, int]:
    """Apply canonical maps to every JSON, mirroring directory structure."""
    n_files = 0
    n_resolved_ents = 0
    t0 = time.time()
    for in_path in iter_json_files(input_root):
        n_files += 1
        doc = load_json(in_path)
        if not isinstance(doc, dict):
            continue
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        for _, _, ent, label in iter_entities(doc):
            if label not in TARGET_LABELS:
                continue
            text = (ent.get("text") or "").strip()
            if not text:
                continue
            entry = maps[label].get(text)
            if entry is None:
                continue
            ent["canonical_id"] = entry["canonical_id"]
            ent["canonical_name"] = entry["canonical_name"]
            n_resolved_ents += 1
        save_json(out_path, doc)
        if n_files % 5000 == 0:
            print(f"  pass-3: {n_files} files written, {n_resolved_ents} entities resolved, "
                  f"{time.time() - t0:.1f}s", flush=True)
    print(f"[pass-3 done] {n_files} files, {n_resolved_ents} entities resolved, "
          f"{time.time() - t0:.1f}s")
    return n_files, n_resolved_ents


# ------------------------------ Audit writers ------------------------------

def write_entity_map(path: Path, m: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def write_merges_csv(path: Path, m: dict, label: str) -> None:
    """Write a CSV grouping raw forms by canonical_id, sorted by total count desc."""
    by_canon: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for raw, entry in m.items():
        by_canon[entry["canonical_id"]].append((raw, entry.get("count", 0)))

    rows = []
    for cid, members in by_canon.items():
        members.sort(key=lambda t: -t[1])
        total = sum(c for _, c in members)
        canon_name = m[members[0][0]].get("canonical_name", "")
        rows.append((cid, canon_name, total, len(members), members))
    rows.sort(key=lambda r: -r[2])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_id", "canonical_name", "total_count", "num_aliases", "raw_aliases (count)"])
        for cid, name, total, k, members in rows:
            members_str = " | ".join(f"{r} ({c})" for r, c in members)
            w.writerow([cid, name, total, k, members_str])


def write_unmapped_csv(path: Path, unmapped: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unmapped.sort(key=lambda t: -t[1])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_text", "count"])
        for raw, cnt in unmapped:
            w.writerow([raw, cnt])


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--maps-only", action="store_true",
                    help="Build entity maps and audit CSVs only; skip writing the resolved corpus.")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="If >0, scan at most this many files in pass 1+3 (for quick tests).")
    args = ap.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.limit_files > 0:
        # Wrap iter_json_files to cap at the given count
        global iter_json_files
        original_iter = iter_json_files

        def capped(root):
            for i, p in enumerate(original_iter(root)):
                if i >= args.limit_files:
                    break
                yield p
        iter_json_files = capped  # type: ignore

    print(f"=== Pass 1: collecting unique entities from {input_root} ===")
    counts = collect_unique_entities(input_root)

    print("=== Pass 2: building canonical maps ===")
    statute_map, statute_unmapped = build_statute_map(counts["STATUTE"])
    provision_map, provision_unmapped = build_provision_map(counts["PROVISION"])
    precedent_map, precedent_unmapped = build_precedent_map(counts["PRECEDENT"])

    maps_dir = output_root / "_entity_maps"
    audit_dir = output_root / "_audit"
    write_entity_map(maps_dir / "statutes.json", statute_map)
    write_entity_map(maps_dir / "provisions.json", provision_map)
    write_entity_map(maps_dir / "precedents.json", precedent_map)
    write_merges_csv(audit_dir / "statute_merges.csv", statute_map, "STATUTE")
    write_merges_csv(audit_dir / "provision_merges.csv", provision_map, "PROVISION")
    write_merges_csv(audit_dir / "precedent_merges.csv", precedent_map, "PRECEDENT")
    write_unmapped_csv(audit_dir / "unmapped_statutes.csv", statute_unmapped)
    write_unmapped_csv(audit_dir / "unmapped_provisions.csv", provision_unmapped)
    write_unmapped_csv(audit_dir / "unmapped_precedents.csv", precedent_unmapped)

    n_files = n_resolved = 0
    if not args.maps_only:
        print("=== Pass 3: applying maps and writing resolved corpus ===")
        n_files, n_resolved = apply_maps(input_root, output_root, {
            "STATUTE": statute_map,
            "PROVISION": provision_map,
            "PRECEDENT": precedent_map,
        })

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "unique_counts": {k: len(v) for k, v in counts.items()},
        "canonical_counts": {
            "STATUTE": len({v["canonical_id"] for v in statute_map.values()}),
            "PROVISION": len({v["canonical_id"] for v in provision_map.values()}),
            "PRECEDENT": len({v["canonical_id"] for v in precedent_map.values()}),
        },
        "unmapped_counts": {
            "STATUTE": len(statute_unmapped),
            "PROVISION": len(provision_unmapped),
            "PRECEDENT": len(precedent_unmapped),
        },
        "files_written": n_files,
        "entities_resolved": n_resolved,
    }
    write_entity_map(audit_dir / "run_summary.json", summary)
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
