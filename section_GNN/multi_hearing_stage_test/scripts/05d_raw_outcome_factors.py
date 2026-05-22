#!/usr/bin/env python3
"""
Step 05d - Build per-case factor reports for raw actual-outcome transitions.

This complements 05c, which is prediction-transition based. A case can have a
stable model prediction such as LOSE -> LOSE while the actual raw outcome moves
from POSTPONED (0) to LOSS (-1). This script explains those actual raw changes.

Outputs:
  - outputs/analysis/per_case_raw_outcome_factors/<case>.json
  - outputs/analysis/per_case_raw_outcome_factors/_index.csv
  - outputs/analysis/raw_outcome_transition_aggregates.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

EXP_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/multi_hearing_stage_test")
DIFFS_DIR = EXP_ROOT / "outputs/analysis/per_case_diffs"
INPUT_JSON_DIR = EXP_ROOT / "data/input_jsons"
OUT_DIR = EXP_ROOT / "outputs/analysis/per_case_raw_outcome_factors"
AGG_OUT = EXP_ROOT / "outputs/analysis/raw_outcome_transition_aggregates.json"

RAW_NAMES = {"1": "WIN", "0": "POSTPONED", "-1": "LOSS"}
LEGAL_LABELS = ("STATUTE", "PROVISION", "PRECEDENT")
PARTY_LABELS = ("JUDGE", "LAWYER", "COURT")
DECISION_ROLES = ("RPC", "RATIO", "STA", "PRE_RELIED", "PRE_NOT_RELIED")
NOISY_LABELS = {"DATE", "CASE_NUMBER", "GPE", "ORG", "OTHER_PERSON", "PETITIONER", "RESPONDENT"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diffs-dir", default=str(DIFFS_DIR))
    p.add_argument("--input-jsons", default=str(INPUT_JSON_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--aggregate-out", default=str(AGG_OUT))
    p.add_argument("--top-k-factors", type=int, default=8)
    p.add_argument("--max-anchor-sentences", type=int, default=12)
    p.add_argument("--all", action="store_true",
                   help="Emit reports for stable raw outcomes too. Default is raw-changed only.")
    return p.parse_args()


def raw_label(value) -> str:
    s = str(value).strip()
    return f"{RAW_NAMES.get(s, s)} ({s})" if s else "UNKNOWN"


def load_stage_docs(input_dir: Path, base_case_id: str, n_stages: int) -> list[dict | None]:
    docs: list[dict | None] = []
    for i in range(1, n_stages + 1):
        path = input_dir / f"STAGE{i}__{base_case_id}.json"
        if not path.exists():
            docs.append(None)
            continue
        with open(path) as f:
            docs.append(json.load(f))
    return docs


def text_normalize(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


def first_last_added(case_diff: dict) -> dict[str, list[str]]:
    stages = case_diff.get("stages", [])
    if len(stages) < 2:
        return {}
    first, last = stages[0], stages[-1]
    a_ents = first.get("entities_by_label", {}) or {}
    b_ents = last.get("entities_by_label", {}) or {}
    added_by_label: dict[str, list[str]] = {}
    for label in set(a_ents) | set(b_ents):
        added = sorted(set(b_ents.get(label, [])) - set(a_ents.get(label, [])))
        if added:
            added_by_label[label] = added
    return added_by_label


def first_last_section_delta(case_diff: dict) -> dict[str, int]:
    stages = case_diff.get("stages", [])
    if len(stages) < 2:
        return {}
    first, last = stages[0], stages[-1]
    a_sec = first.get("section_sentence_counts", {}) or {}
    b_sec = last.get("section_sentence_counts", {}) or {}
    return {
        role: b_sec.get(role, 0) - a_sec.get(role, 0)
        for role in set(a_sec) | set(b_sec)
        if b_sec.get(role, 0) - a_sec.get(role, 0) != 0
    }


def raw_transition_for_case(case_diff: dict, input_dir: Path) -> dict | None:
    n_stages = case_diff.get("n_stages", len(case_diff.get("stages", [])))
    if n_stages < 2:
        return None
    docs = load_stage_docs(input_dir, case_diff["base_case_id"], n_stages)
    raw_values = []
    for doc in docs:
        if not doc:
            raw_values.append("")
        else:
            raw_values.append(str(doc.get("case_outcome_score", "")).strip())
    if not raw_values or not raw_values[0] or not raw_values[-1]:
        return None
    return {
        "raw_values": raw_values,
        "from_raw": raw_values[0],
        "to_raw": raw_values[-1],
        "transition": f"{raw_label(raw_values[0])} -> {raw_label(raw_values[-1])}",
        "docs": docs,
    }


def build_raw_records(cases: list[dict], input_dir: Path) -> list[dict]:
    records = []
    for case_diff in cases:
        raw = raw_transition_for_case(case_diff, input_dir)
        if raw is None:
            continue
        records.append({
            "base_case_id": case_diff["base_case_id"],
            "transition": raw["transition"],
            "from_raw": raw["from_raw"],
            "to_raw": raw["to_raw"],
            "prediction_transition": case_diff.get("transition", ""),
            "added_by_label": first_last_added(case_diff),
            "section_delta": first_last_section_delta(case_diff),
        })
    return records


def choose_contrast(target: dict, grouped: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    start = target["from_raw"]
    stable_key = f"{raw_label(start)} -> {raw_label(start)}"
    target_key = target["transition"]
    if stable_key in grouped and stable_key != target_key:
        return stable_key, grouped[stable_key]

    same_start = [
        rec
        for key, recs in grouped.items()
        if key != target_key
        for rec in recs
        if rec["from_raw"] == start
    ]
    if same_start:
        return f"{raw_label(start)} -> OTHER", same_start

    other = [rec for key, recs in grouped.items() if key != target_key for rec in recs]
    return "ALL_OTHER_RAW_TRANSITIONS", other


def entity_counts(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        seen = set()
        for label, ents in rec.get("added_by_label", {}).items():
            if label in NOISY_LABELS:
                continue
            for ent in ents:
                if ent:
                    seen.add((label, ent))
        counts.update(seen)
    return counts


def build_priors(records: list[dict]) -> tuple[dict, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["transition"]].append(rec)

    priors: dict[str, dict[str, dict[str, float]]] = {}
    contrast_meta = {}
    for transition, target_records in grouped.items():
        contrast_name, contrast_records = choose_contrast(target_records[0], grouped)
        n_t = max(1, len(target_records))
        n_c = max(1, len(contrast_records))
        target_counts = entity_counts(target_records)
        contrast_counts = entity_counts(contrast_records)

        scored: dict[str, dict[str, float]] = defaultdict(dict)
        for label, ent in set(target_counts) | set(contrast_counts):
            t_cnt = target_counts.get((label, ent), 0)
            c_cnt = contrast_counts.get((label, ent), 0)
            f_t = (t_cnt + 0.5) / (n_t + 1)
            f_c = (c_cnt + 0.5) / (n_c + 1)
            scored[label][ent] = round(f_t - f_c, 4)
        priors[transition] = scored
        contrast_meta[transition] = {
            "contrast_transition": contrast_name,
            "n_transition_cases": len(target_records),
            "n_contrast_cases": len(contrast_records),
        }
    return priors, contrast_meta


def new_decision_sentences(docs: list[dict | None], max_n: int) -> list[dict]:
    if len(docs) < 2 or not docs[0] or not docs[-1]:
        return []
    earlier_sentence_set = {
        text_normalize(s.get("text", ""))
        for s in docs[0].get("sentences", [])
        if s.get("rhetorical_role") in DECISION_ROLES
    }
    out = []
    for s in docs[-1].get("sentences", []):
        if s.get("rhetorical_role") not in DECISION_ROLES:
            continue
        tn = text_normalize(s.get("text", ""))
        if tn and tn not in earlier_sentence_set:
            out.append({
                "rhetorical_role": s.get("rhetorical_role"),
                "text": s.get("text", "").strip(),
                "entities": [{"text": e.get("text", "").strip(), "label": e.get("label", "")}
                             for e in s.get("entities", []) or []],
            })
            if len(out) >= max_n:
                break
    return out


def find_anchor_sentences(later_doc: dict | None, top_entities: list[tuple[str, str, float]],
                          max_n: int) -> list[dict]:
    if not later_doc:
        return []
    target_strings = {text_normalize(e[1]) for e in top_entities if e[1]}
    anchors = []
    for s in later_doc.get("sentences", []):
        if s.get("rhetorical_role") not in DECISION_ROLES:
            continue
        sent_text = s.get("text", "")
        sent_text_norm = text_normalize(sent_text)
        matched = []
        for ent in s.get("entities", []) or []:
            if text_normalize(ent.get("text", "")) in target_strings:
                matched.append({"text": ent.get("text", "").strip(), "label": ent.get("label", "")})
        if not matched:
            for tn in target_strings:
                if tn and tn in sent_text_norm and len(tn) >= 5:
                    matched.append({"text": tn, "label": "(substring_match)"})
                    break
        if matched:
            anchors.append({
                "rhetorical_role": s.get("rhetorical_role"),
                "text": sent_text.strip(),
                "matched_entities": matched,
            })
            if len(anchors) >= max_n:
                break
    return anchors


def build_report(case_diff: dict, priors: dict, contrast_meta: dict, input_dir: Path, args) -> dict | None:
    raw = raw_transition_for_case(case_diff, input_dir)
    if raw is None:
        return None
    if not args.all and raw["from_raw"] == raw["to_raw"]:
        return None

    transition = raw["transition"]
    added_by_label = first_last_added(case_diff)
    transition_priors = priors.get(transition, {})
    scored: list[tuple[str, str, float]] = []
    for label, ents in added_by_label.items():
        if label in NOISY_LABELS:
            continue
        label_priors = transition_priors.get(label, {})
        for ent in ents:
            scored.append((label, ent, label_priors.get(ent, 0.0)))

    label_priority = {l: i for i, l in enumerate(LEGAL_LABELS + PARTY_LABELS)}
    scored.sort(key=lambda t: (label_priority.get(t[0], 99), -t[2]))
    top_factors = scored[: args.top_k_factors]

    n_stages = case_diff.get("n_stages", len(case_diff.get("stages", [])))
    docs = raw["docs"]
    meta = contrast_meta.get(transition, {})
    return {
        "base_case_id": case_diff["base_case_id"],
        "category": case_diff.get("category", ""),
        "outcome_split": case_diff.get("outcome_split", ""),
        "factor_basis": "raw_outcome",
        "transition": transition,
        "raw_outcome_transition": transition,
        "prediction_transition": case_diff.get("transition", ""),
        "contrast_transition": meta.get("contrast_transition", ""),
        "n_transition_cases": meta.get("n_transition_cases", 0),
        "n_contrast_cases": meta.get("n_contrast_cases", 0),
        "n_stages": n_stages,
        "true_label": case_diff.get("true_label"),
        "raw_outcome_path": [raw_label(v) for v in raw["raw_values"]],
        "section_sentence_delta_first_to_last": first_last_section_delta(case_diff),
        "top_decisive_factors": [
            {"label": l, "entity": e, "discriminative_score": s}
            for (l, e, s) in top_factors
        ],
        "new_decision_role_sentences": new_decision_sentences(docs, args.max_anchor_sentences),
        "anchor_sentences": find_anchor_sentences(docs[-1] if docs else None,
                                                  top_factors,
                                                  args.max_anchor_sentences),
    }


def aggregate_summary(records: list[dict], contrast_meta: dict) -> dict:
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["transition"]].append(rec)
    return {
        "transition_counts": dict(Counter(rec["transition"] for rec in records).most_common()),
        "contrasts": contrast_meta,
        "notes": {
            "diff_mode": "first_stage vs last_stage",
            "basis": "raw case_outcome_score from input_jsons",
            "raw_labels": RAW_NAMES,
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_jsons)

    cases = []
    for diff_path in sorted(Path(args.diffs_dir).glob("*.json")):
        with open(diff_path) as f:
            cases.append(json.load(f))

    records = build_raw_records(cases, input_dir)
    priors, contrast_meta = build_priors(records)

    with open(args.aggregate_out, "w") as f:
        json.dump(aggregate_summary(records, contrast_meta), f, indent=2)

    index_rows = []
    written = 0
    skipped_stable = 0
    for case_diff in cases:
        report = build_report(case_diff, priors, contrast_meta, input_dir, args)
        if report is None:
            skipped_stable += 1
            continue
        with open(out_dir / f"{report['base_case_id']}.json", "w") as f:
            json.dump(report, f, indent=2)
        written += 1
        top_str = "; ".join(f"{f['label']}:{f['entity'][:40]}" for f in report["top_decisive_factors"][:3])
        index_rows.append({
            "base_case_id": report["base_case_id"],
            "category": report["category"],
            "raw_outcome_transition": report["raw_outcome_transition"],
            "prediction_transition": report["prediction_transition"],
            "contrast_transition": report["contrast_transition"],
            "n_transition_cases": report["n_transition_cases"],
            "n_contrast_cases": report["n_contrast_cases"],
            "n_top_factors": len(report["top_decisive_factors"]),
            "n_new_decision_sentences": len(report["new_decision_role_sentences"]),
            "n_anchor_sentences": len(report["anchor_sentences"]),
            "top_factors_preview": top_str,
        })

    if index_rows:
        with open(out_dir / "_index.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)

    print(f"[05d] wrote {written} raw-outcome factor reports -> {out_dir}")
    if not args.all:
        print(f"[05d] skipped {skipped_stable} stable raw-outcome cases")
    print(f"[05d] aggregate: {args.aggregate_out}")
    print(f"[05d] index: {out_dir / '_index.csv'}")


if __name__ == "__main__":
    main()
