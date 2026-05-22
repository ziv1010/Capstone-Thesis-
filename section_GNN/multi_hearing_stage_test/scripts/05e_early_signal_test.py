#!/usr/bin/env python3
"""
Step 05e - Early-detection signal test for multi-hearing analysis.

This script turns the stage-level transition outputs into paper-facing signal
tables:

1. first_hearing_early_signals.csv
   Features already visible in the first hearing that are associated with the
   model predicting the final outcome correctly at stage 1.

2. later_added_correction_signals.csv
   Features introduced between the first and last hearing that are associated
   with correcting an initially wrong prediction by the last stage.

The test is deliberately simple and auditable: every candidate signal is binary,
and each row reports support, target rates with/without the signal, lift,
smoothed odds ratio, and a two-proportion z-score. The goal is not causal proof;
it is to identify recurrent legal/structural cues that make early detection
possible or that explain when the model changes its mind later.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

EXP_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/multi_hearing_stage_test")
ANALYSIS_DIR = EXP_ROOT / "outputs/analysis"
DIFFS_DIR = ANALYSIS_DIR / "per_case_diffs"

NOISY_ENTITY_LABELS = {"DATE", "CASE_NUMBER", "GPE", "ORG"}
LEGAL_ENTITY_LABELS = {"STATUTE", "PROVISION", "PRECEDENT", "COURT", "JUDGE", "LAWYER"}
SECTION_ROLES = [
    "PREAMBLE",
    "FAC",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "STA",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "RPC",
    "RATIO",
    "ANALYSIS",
    "ISSUE",
    "RLC",
    "NONE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transitions", default=str(ANALYSIS_DIR / "stage_transitions.csv"))
    p.add_argument("--diffs-dir", default=str(DIFFS_DIR))
    p.add_argument("--out-dir", default=str(ANALYSIS_DIR / "early_signal_test"))
    p.add_argument("--min-support", type=int, default=10)
    p.add_argument("--confidence-threshold", type=float, default=0.80)
    p.add_argument("--top-specific-entities", type=int, default=250,
                   help="Per entity label, keep only the most common specific entity strings.")
    return p.parse_args()


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def load_cases(diffs_dir: Path) -> list[dict]:
    out = []
    for path in sorted(diffs_dir.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            case = json.load(handle)
        if len(case.get("stages", [])) >= 2:
            out.append(case)
    return out


def first_last_diff(case: dict) -> tuple[dict[str, int], dict[str, dict[str, list[str]]]]:
    stages = case.get("stages", [])
    first, last = stages[0], stages[-1]

    section_delta = {}
    first_sections = first.get("section_sentence_counts", {}) or {}
    last_sections = last.get("section_sentence_counts", {}) or {}
    for role in set(first_sections) | set(last_sections):
        delta = int(last_sections.get(role, 0)) - int(first_sections.get(role, 0))
        if delta:
            section_delta[role] = delta

    entity_diff = {}
    first_ents = first.get("entities_by_label", {}) or {}
    last_ents = last.get("entities_by_label", {}) or {}
    for label in set(first_ents) | set(last_ents):
        before = {norm_text(x) for x in first_ents.get(label, []) if str(x).strip()}
        after = {norm_text(x) for x in last_ents.get(label, []) if str(x).strip()}
        added = sorted(after - before)
        removed = sorted(before - after)
        if added or removed:
            entity_diff[label] = {"added": added, "removed": removed}
    return section_delta, entity_diff


def collect_specific_entity_vocabulary(cases: list[dict], top_k: int) -> dict[str, set[str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for case in cases:
        first = case["stages"][0]
        for label, values in (first.get("entities_by_label", {}) or {}).items():
            if label not in LEGAL_ENTITY_LABELS:
                continue
            for value in values:
                value_norm = norm_text(value)
                if value_norm:
                    counts[label][value_norm] += 1
        _, entity_diff = first_last_diff(case)
        for label, change in entity_diff.items():
            if label not in LEGAL_ENTITY_LABELS:
                continue
            for value_norm in change.get("added", []):
                counts[label][value_norm] += 1
    return {
        label: {value for value, _ in counter.most_common(top_k)}
        for label, counter in counts.items()
    }


def first_hearing_features(case: dict, specific_vocab: dict[str, set[str]]) -> set[str]:
    first = case["stages"][0]
    features = set()

    sections = first.get("section_sentence_counts", {}) or {}
    total_sentences = sum(int(v) for v in sections.values())
    if total_sentences >= 25:
        features.add("first:total_sentences_ge_25")
    if total_sentences >= 50:
        features.add("first:total_sentences_ge_50")
    for role in SECTION_ROLES:
        count = int(sections.get(role, 0))
        if count > 0:
            features.add(f"first:section_present:{role}")
        if count >= 3:
            features.add(f"first:section_ge3:{role}")
        if count >= 8:
            features.add(f"first:section_ge8:{role}")

    entities = first.get("entities_by_label", {}) or {}
    for label, values in entities.items():
        if label in NOISY_ENTITY_LABELS:
            continue
        clean_values = [norm_text(v) for v in values if str(v).strip()]
        if not clean_values:
            continue
        features.add(f"first:entity_label_present:{label}")
        if len(clean_values) >= 2:
            features.add(f"first:entity_label_ge2:{label}")
        if len(clean_values) >= 5:
            features.add(f"first:entity_label_ge5:{label}")
        if label in LEGAL_ENTITY_LABELS:
            for value_norm in clean_values:
                if value_norm in specific_vocab.get(label, set()):
                    features.add(f"first:entity:{label}:{value_norm}")
    return features


def later_added_features(case: dict, specific_vocab: dict[str, set[str]]) -> set[str]:
    section_delta, entity_diff = first_last_diff(case)
    features = set()

    for role, delta in section_delta.items():
        if delta > 0:
            features.add(f"later:section_added:{role}")
            if delta >= 3:
                features.add(f"later:section_added_ge3:{role}")
            if delta >= 8:
                features.add(f"later:section_added_ge8:{role}")

    for label, change in entity_diff.items():
        if label in NOISY_ENTITY_LABELS:
            continue
        added = change.get("added", [])
        if added:
            features.add(f"later:entity_label_added:{label}")
        if len(added) >= 2:
            features.add(f"later:entity_label_added_ge2:{label}")
        if label in LEGAL_ENTITY_LABELS:
            for value_norm in added:
                if value_norm in specific_vocab.get(label, set()):
                    features.add(f"later:entity_added:{label}:{value_norm}")
    return features


def two_prop_z(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    if total_a == 0 or total_b == 0:
        return 0.0
    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    denom = math.sqrt(max(pooled * (1 - pooled) * (1 / total_a + 1 / total_b), 1e-12))
    return (p1 - p2) / denom


def rank_binary_features(
    feature_sets: list[set[str]],
    targets: list[bool],
    min_support: int,
    feature_prefix_allow: Iterable[str] | None = None,
) -> list[dict]:
    n = len(targets)
    positives = sum(targets)
    feature_counts: Counter = Counter()
    feature_pos_counts: Counter = Counter()
    allow = tuple(feature_prefix_allow or [])

    for feats, target in zip(feature_sets, targets):
        for feat in feats:
            if allow and not feat.startswith(allow):
                continue
            feature_counts[feat] += 1
            if target:
                feature_pos_counts[feat] += 1

    rows = []
    base_rate = positives / n if n else 0.0
    for feat, present_n in feature_counts.items():
        if present_n < min_support or present_n >= n:
            continue
        present_pos = feature_pos_counts[feat]
        absent_n = n - present_n
        absent_pos = positives - present_pos
        present_rate = present_pos / present_n
        absent_rate = absent_pos / absent_n if absent_n else 0.0
        odds_ratio = ((present_pos + 0.5) / (present_n - present_pos + 0.5)) / (
            (absent_pos + 0.5) / (absent_n - absent_pos + 0.5)
        )
        z = two_prop_z(present_pos, present_n, absent_pos, absent_n)
        rows.append({
            "feature": feat,
            "support": present_n,
            "support_frac": round(present_n / n, 4),
            "target_rate_present": round(present_rate, 4),
            "target_rate_absent": round(absent_rate, 4),
            "lift_vs_base": round(present_rate - base_rate, 4),
            "rate_diff_present_minus_absent": round(present_rate - absent_rate, 4),
            "smoothed_odds_ratio": round(odds_ratio, 4),
            "z_score": round(z, 4),
            "target_count_present": present_pos,
            "n_cases": n,
        })

    return sorted(
        rows,
        key=lambda r: (abs(r["z_score"]), abs(r["rate_diff_present_minus_absent"]), r["support"]),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_csv(args.transitions)
    transition_lookup = transitions.set_index("base_case_id").to_dict("index")
    cases = load_cases(Path(args.diffs_dir))
    specific_vocab = collect_specific_entity_vocabulary(cases, args.top_specific_entities)

    first_feature_sets = []
    first_targets = []
    first_locked_targets = []
    first_case_rows = []

    correction_feature_sets = []
    correction_targets = []
    correction_case_rows = []

    for case in cases:
        base_id = case["base_case_id"]
        row = transition_lookup.get(base_id)
        if row is None:
            continue
        true_label = str(row.get("true_label"))
        stage1_pred = str(row.get("stage1_pred"))
        final_pred_correct = bool(row.get("final_pred_correct"))
        try:
            stage1_conf = float(row.get("stage1_conf"))
        except Exception:
            stage1_conf = 0.0

        early_correct = stage1_pred == true_label
        early_locked = early_correct and stage1_conf >= args.confidence_threshold
        first_feats = first_hearing_features(case, specific_vocab)
        first_feature_sets.append(first_feats)
        first_targets.append(early_correct)
        first_locked_targets.append(early_locked)
        first_case_rows.append({
            "base_case_id": base_id,
            "category": case.get("category", ""),
            "true_label": true_label,
            "stage1_pred": stage1_pred,
            "stage1_conf": round(stage1_conf, 4),
            "early_correct": early_correct,
            "early_locked_correct": early_locked,
            "transition": case.get("transition", ""),
        })

        if not early_correct:
            later_feats = later_added_features(case, specific_vocab)
            correction_feature_sets.append(later_feats)
            correction_targets.append(final_pred_correct)
            correction_case_rows.append({
                "base_case_id": base_id,
                "category": case.get("category", ""),
                "true_label": true_label,
                "stage1_pred": stage1_pred,
                "stage1_conf": round(stage1_conf, 4),
                "corrected_by_last_stage": final_pred_correct,
                "transition": case.get("transition", ""),
            })

    early_rows = rank_binary_features(first_feature_sets, first_targets, args.min_support)
    locked_rows = rank_binary_features(first_feature_sets, first_locked_targets, args.min_support)
    correction_rows = rank_binary_features(correction_feature_sets, correction_targets, args.min_support)

    write_csv(out_dir / "first_hearing_early_signals.csv", early_rows)
    write_csv(out_dir / "first_hearing_high_conf_early_signals.csv", locked_rows)
    write_csv(out_dir / "later_added_correction_signals.csv", correction_rows)
    pd.DataFrame(first_case_rows).to_csv(out_dir / "case_level_early_detection.csv", index=False)
    pd.DataFrame(correction_case_rows).to_csv(out_dir / "case_level_late_corrections.csv", index=False)

    summary = {
        "n_cases": len(first_targets),
        "early_correct_cases": sum(first_targets),
        "early_correct_rate": round(sum(first_targets) / len(first_targets), 4) if first_targets else None,
        "early_high_conf_threshold": args.confidence_threshold,
        "early_high_conf_correct_cases": sum(first_locked_targets),
        "early_high_conf_correct_rate": round(sum(first_locked_targets) / len(first_locked_targets), 4) if first_locked_targets else None,
        "n_initially_wrong_cases": len(correction_targets),
        "corrected_by_last_stage_cases": sum(correction_targets),
        "correction_rate_among_initially_wrong": round(sum(correction_targets) / len(correction_targets), 4) if correction_targets else None,
        "min_support": args.min_support,
        "top_first_hearing_signals": early_rows[:15],
        "top_high_conf_first_hearing_signals": locked_rows[:15],
        "top_later_added_correction_signals": correction_rows[:15],
        "interpretation": {
            "first_hearing_early_signals": "Features present in the first hearing whose presence changes the probability that stage-1 prediction equals the final label.",
            "later_added_correction_signals": "Features added after the first hearing among initially wrong cases, tested against whether the final-stage prediction becomes correct.",
            "z_score": "Normal-approximation two-proportion z-score; use as a ranking/audit statistic, not as causal proof.",
        },
    }
    with open(out_dir / "early_signal_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[05e] cases={summary['n_cases']} early_correct={summary['early_correct_rate']}")
    print(f"[05e] initially_wrong={summary['n_initially_wrong_cases']} correction_rate={summary['correction_rate_among_initially_wrong']}")
    print("\n[05e] Top first-hearing early-detection signals:")
    for row in early_rows[:10]:
        print(
            f"  {row['feature']} | support={row['support']} "
            f"rate={row['target_rate_present']:.3f} vs {row['target_rate_absent']:.3f} "
            f"diff={row['rate_diff_present_minus_absent']:+.3f} z={row['z_score']:+.2f}"
        )
    print("\n[05e] Top later-added correction signals:")
    for row in correction_rows[:10]:
        print(
            f"  {row['feature']} | support={row['support']} "
            f"rate={row['target_rate_present']:.3f} vs {row['target_rate_absent']:.3f} "
            f"diff={row['rate_diff_present_minus_absent']:+.3f} z={row['z_score']:+.2f}"
        )
    print(f"\n[05e] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
