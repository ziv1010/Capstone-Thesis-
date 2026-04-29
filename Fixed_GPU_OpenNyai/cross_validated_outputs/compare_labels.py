#!/usr/bin/env python3
"""Compare case_outcome_label between:
  - Mistral single-pass labels  (final_outputs/<bucket>_labelled_mistral/labelled_jsons/)
  - Cross-val labels            (cross_validated_outputs/<bucket>/augmented_jsons/)

Matched by filename. Outputs per-bucket discrepancy CSVs + one aggregate report.

Usage:
    python compare_labels.py [--output_dir ./label_comparison]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


FINAL_OUTPUTS_BASE = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/final_outputs"
)
CROSSVAL_BASE = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/cross_validated_outputs"
)

BUCKETS = [
    "fin_fraud",
    "family_matrimonial",
    "food_safety",
    "land_property",
    "motor_accidents",
    "sexual_offences",
]

# food_safety has no labelled_mistral folder — skip gracefully
MISTRAL_DIR_TEMPLATE  = "{bucket}_labelled_mistral/labelled_jsons"
CROSSVAL_DIR_TEMPLATE = "{bucket}/augmented_jsons"


def load_label(path: Path) -> str | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return str(d.get("case_outcome_label", "")).strip() or None
    except Exception:
        return None


def load_crossval_meta(path: Path) -> dict[str, Any]:
    """Pull the extra crossval detail we want to surface in the report."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        llm = d.get("llm_case_outcome") or {}
        return {
            "crossval_label":      llm.get("crossval_label"),
            "crossval_confidence": llm.get("crossval_confidence"),
            "win_score":           llm.get("win_score"),
            "loss_score":          llm.get("loss_score"),
            "neutral_score":       llm.get("neutral_score"),
        }
    except Exception:
        return {}


def compare_bucket(
    bucket: str,
    output_dir: Path,
) -> dict[str, Any]:
    mistral_dir  = FINAL_OUTPUTS_BASE / MISTRAL_DIR_TEMPLATE.format(bucket=bucket)
    crossval_dir = CROSSVAL_BASE      / CROSSVAL_DIR_TEMPLATE.format(bucket=bucket)

    if not mistral_dir.exists():
        print(f"  [SKIP] No mistral labelled dir for {bucket}: {mistral_dir}")
        return {"bucket": bucket, "status": "skipped_no_mistral_dir"}

    if not crossval_dir.exists():
        print(f"  [SKIP] No crossval augmented dir for {bucket}: {crossval_dir}")
        return {"bucket": bucket, "status": "skipped_no_crossval_dir"}

    # Index mistral files by filename (case-insensitive safe)
    mistral_files: dict[str, Path] = {
        f.name: f
        for f in mistral_dir.glob("*.json")
        if not f.name.startswith(".")
    }
    crossval_files: dict[str, Path] = {
        f.name: f
        for f in crossval_dir.glob("*.json")
        if not f.name.startswith(".")
    }

    common   = sorted(mistral_files.keys() & crossval_files.keys())
    only_m   = sorted(mistral_files.keys() - crossval_files.keys())
    only_cv  = sorted(crossval_files.keys() - mistral_files.keys())

    rows_all        = []
    rows_discrepant = []

    for fname in common:
        m_label  = load_label(mistral_files[fname])
        cv_label = load_label(crossval_files[fname])
        meta     = load_crossval_meta(crossval_files[fname])

        match = (m_label == cv_label)
        row = {
            "filename":            fname,
            "mistral_label":       m_label,
            "crossval_label_std":  cv_label,          # appellant_won / _lost / postponed etc.
            "crossval_label_raw":  meta.get("crossval_label"),  # WIN / LOSS / NEUTRAL / UNKNOWN
            "crossval_confidence": meta.get("crossval_confidence"),
            "win_score":           meta.get("win_score"),
            "loss_score":          meta.get("loss_score"),
            "neutral_score":       meta.get("neutral_score"),
            "match":               match,
        }
        rows_all.append(row)
        if not match:
            rows_discrepant.append(row)

    # ── Write per-bucket CSVs ──────────────────────────────────────────────────
    bucket_out = output_dir / bucket
    bucket_out.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename", "mistral_label", "crossval_label_std", "crossval_label_raw",
        "crossval_confidence", "win_score", "loss_score", "neutral_score", "match",
    ]

    all_csv = bucket_out / "all_matched_files.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_all)

    discrepant_csv = bucket_out / "discrepancies.csv"
    with discrepant_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_discrepant)

    # ── Label distribution breakdown ──────────────────────────────────────────
    label_vals = ["appellant_won", "postponed_or_procedural", "appellant_lost", "unknown", None]

    mistral_dist  = {lbl: sum(1 for r in rows_all if r["mistral_label"]      == lbl) for lbl in label_vals}
    crossval_dist = {lbl: sum(1 for r in rows_all if r["crossval_label_std"] == lbl) for lbl in label_vals}

    # Confusion: mistral_label → crossval_label → count
    confusion: dict[str, dict[str, int]] = {}
    for r in rows_all:
        m = str(r["mistral_label"])
        c = str(r["crossval_label_std"])
        confusion.setdefault(m, {}).setdefault(c, 0)
        confusion[m][c] += 1

    stats = {
        "bucket":               bucket,
        "status":               "ok",
        "total_mistral_files":  len(mistral_files),
        "total_crossval_files": len(crossval_files),
        "common_files":         len(common),
        "only_in_mistral":      len(only_m),
        "only_in_crossval":     len(only_cv),
        "matching_labels":      sum(1 for r in rows_all if r["match"]),
        "discrepant_labels":    len(rows_discrepant),
        "agreement_rate":       round(
            sum(1 for r in rows_all if r["match"]) / len(rows_all), 4
        ) if rows_all else None,
        "mistral_label_dist":   {str(k): v for k, v in mistral_dist.items()},
        "crossval_label_dist":  {str(k): v for k, v in crossval_dist.items()},
        "confusion_matrix":     confusion,   # rows = mistral, cols = crossval
        "discrepancy_csv":      str(discrepant_csv),
        "all_csv":              str(all_csv),
    }

    json_path = bucket_out / "stats.json"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"  {bucket:<25}  matched={len(common):>5}  "
        f"agree={stats['matching_labels']:>5}  "
        f"disagree={len(rows_discrepant):>5}  "
        f"agreement={stats['agreement_rate']:.1%}"
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default=str(CROSSVAL_BASE / "label_comparison"),
        help="Directory to write comparison outputs to.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Label comparison: Mistral single-pass  vs  Cross-validation")
    print("=" * 70)

    all_stats = []
    for bucket in BUCKETS:
        print(f"\n→ {bucket}")
        stats = compare_bucket(bucket, output_dir)
        all_stats.append(stats)

    # ── Aggregate report ──────────────────────────────────────────────────────
    ok_stats = [s for s in all_stats if s.get("status") == "ok"]

    total_common     = sum(s["common_files"]      for s in ok_stats)
    total_matching   = sum(s["matching_labels"]   for s in ok_stats)
    total_discrepant = sum(s["discrepant_labels"] for s in ok_stats)

    aggregate = {
        "buckets_processed": len(ok_stats),
        "buckets_skipped":   len(all_stats) - len(ok_stats),
        "total_common_files":     total_common,
        "total_matching_labels":  total_matching,
        "total_discrepant_labels": total_discrepant,
        "overall_agreement_rate": round(total_matching / total_common, 4) if total_common else None,
        "per_bucket": all_stats,
    }

    agg_path = output_dir / "aggregate_report.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  OVERALL  matched={total_common}  agree={total_matching}  "
          f"disagree={total_discrepant}  "
          f"rate={aggregate['overall_agreement_rate']:.1%}" if total_common else "  OVERALL  no files matched")
    print(f"  Aggregate report → {agg_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
