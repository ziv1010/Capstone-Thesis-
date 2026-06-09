#!/usr/bin/env python3
"""
Generate a single before/after rhetorical-role leakage-filtering figure.

This script is intentionally separate from generate_plots_full.py so existing
figures are not overwritten. By default it fails if any target output already
exists; pass --overwrite only when regenerating this specific new figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_utils import load_config, resolve_output_arg, resolve_path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_CLEANED_DIR = (
    ROOT.parent
    / "section_GNN/data/timed_bucket_runs/cross_bucket_total_dataset/processed/cleaned_cases"
)
DEFAULT_OUT_DIR = ROOT / "outputs/leakage_role_comparison"

BUCKETS = [
    ("fin_fraud", "fin_fraud", "Financial fraud"),
    ("family_matrimonial", "family_matrimonial", "Family/matrimonial"),
    ("land_property", "land_property", "Land/property"),
    ("motor_accidents", "motor_accidents", "Motor accidents"),
    ("sexual_offences", "sexual_offences", "Sexual offences"),
]

ROLE_ORDER = [
    "PREAMBLE",
    "FAC",
    "ISSUE",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "STA",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "RATIO",
    "RLC",
    "RPC",
    "NONE",
]

ROLE_LABELS = {
    "PREAMBLE": "Preamble",
    "FAC": "Facts",
    "ISSUE": "Issue",
    "ARG_PETITIONER": "Arg. petitioner",
    "ARG_RESPONDENT": "Arg. respondent",
    "ANALYSIS": "Analysis",
    "STA": "Statute",
    "PRE_RELIED": "Prec. relied",
    "PRE_NOT_RELIED": "Prec. not relied",
    "RATIO": "Ratio",
    "RLC": "RLC",
    "RPC": "RPC",
    "NONE": "None",
}

ROLE_COLORS = {
    "PREAMBLE": "#4C78A8",
    "FAC": "#F58518",
    "ISSUE": "#B279A2",
    "ARG_PETITIONER": "#54A24B",
    "ARG_RESPONDENT": "#E45756",
    "ANALYSIS": "#72B7B2",
    "STA": "#EECA3B",
    "PRE_RELIED": "#5F9ED1",
    "PRE_NOT_RELIED": "#9D755D",
    "RATIO": "#BAB0AC",
    "RLC": "#D37295",
    "RPC": "#8CD17D",
    "NONE": "#79706E",
}

LEAKAGE_ROLES = {"RLC", "RPC"}
DROPPED_BY_FILTER = {"ANALYSIS", "ISSUE", "NONE", "RATIO", "RLC", "RPC"}


def raw_dirs_from_config(config: dict[str, Any]) -> dict[str, Path]:
    raw_dirs: dict[str, Path] = {}
    for bucket in config.get("buckets", []):
        name = bucket.get("name")
        data_dir = bucket.get("data_dir")
        if name and data_dir:
            raw_dirs[str(name)] = Path(str(data_dir))
    return raw_dirs


def iter_json_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix == ".json" and path.name.lower() != "report.json"
    )


def average_role_proportions_from_counters(
    counters: list[Counter[str]],
) -> dict[str, float]:
    totals = {role: 0.0 for role in ROLE_ORDER}
    usable_cases = 0
    for counts in counters:
        total = sum(counts.values())
        if total <= 0:
            continue
        usable_cases += 1
        for role in ROLE_ORDER:
            totals[role] += counts.get(role, 0) / total
    if usable_cases == 0:
        return totals
    return {role: value / usable_cases for role, value in totals.items()}


def load_before_role_data(raw_dirs: dict[str, Path]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    panel_data: dict[str, dict[str, float]] = {}
    summary: dict[str, Any] = {}

    for bucket_key, _prefix, _display in BUCKETS:
        directory = raw_dirs[bucket_key]
        counters: list[Counter[str]] = []
        aggregate = Counter()
        files = iter_json_files(directory)

        for path in files:
            try:
                with path.open(encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue

            case_counts = Counter(
                str(sentence.get("rhetorical_role", "UNKNOWN") or "UNKNOWN").upper()
                for sentence in payload.get("sentences", []) or []
            )
            counters.append(case_counts)
            aggregate.update(case_counts)

        panel_data[bucket_key] = average_role_proportions_from_counters(counters)
        summary[bucket_key] = {
            "source_dir": str(directory),
            "cases_read": len(counters),
            "aggregate_role_counts": dict(sorted(aggregate.items())),
        }

    return panel_data, summary


def bucket_from_cleaned_file(path: Path) -> str | None:
    name = path.name
    for _bucket_key, prefix, _display in BUCKETS:
        if name.startswith(f"{prefix}__"):
            return _bucket_key
    return None


def load_after_role_data(cleaned_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    counters_by_bucket: dict[str, list[Counter[str]]] = defaultdict(list)
    aggregate_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    dropped_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    reconstructed_preamble_by_bucket: dict[str, int] = defaultdict(int)
    skipped = 0
    missing_source_paths = 0

    for path in iter_json_files(cleaned_dir):
        bucket_key = bucket_from_cleaned_file(path)
        if bucket_key is None:
            skipped += 1
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            skipped += 1
            continue

        audit = payload.get("leakage_audit", {}) or {}
        kept_counts = Counter(
            {
                str(role).upper(): int(count)
                for role, count in (audit.get("kept_sentence_role_counts", {}) or {}).items()
            }
        )
        retained_sources = (audit.get("retained_texts", {}) or {}).get("sources", {}) or {}
        if "PREAMBLE" not in kept_counts and retained_sources.get("preamble") == "sentence_prefix:preamble_end_char_offset":
            source_path = Path(str(payload.get("source_path") or ""))
            if source_path.exists():
                try:
                    with source_path.open(encoding="utf-8") as source_fh:
                        source_payload = json.load(source_fh)
                    preamble_count = sum(
                        1
                        for sentence in source_payload.get("sentences", []) or []
                        if str(sentence.get("rhetorical_role", "") or "").upper() == "PREAMBLE"
                    )
                    if preamble_count:
                        kept_counts["PREAMBLE"] += preamble_count
                        reconstructed_preamble_by_bucket[bucket_key] += preamble_count
                except Exception:
                    missing_source_paths += 1
            else:
                missing_source_paths += 1

        dropped_counts = Counter(
            {
                str(role).upper(): int(count)
                for role, count in (audit.get("dropped_sentence_role_counts", {}) or {}).items()
            }
        )
        counters_by_bucket[bucket_key].append(kept_counts)
        aggregate_by_bucket[bucket_key].update(kept_counts)
        dropped_by_bucket[bucket_key].update(dropped_counts)

    panel_data: dict[str, dict[str, float]] = {}
    summary: dict[str, Any] = {
        "skipped_files": skipped,
        "missing_source_paths_for_preamble_reconstruction": missing_source_paths,
        "buckets": {},
    }
    for bucket_key, _prefix, _display in BUCKETS:
        counters = counters_by_bucket.get(bucket_key, [])
        aggregate = aggregate_by_bucket.get(bucket_key, Counter())
        dropped = dropped_by_bucket.get(bucket_key, Counter())
        panel_data[bucket_key] = average_role_proportions_from_counters(counters)
        summary["buckets"][bucket_key] = {
            "source_dir": str(cleaned_dir),
            "cases_read": len(counters),
            "aggregate_kept_role_counts": dict(sorted(aggregate.items())),
            "aggregate_dropped_role_counts": dict(sorted(dropped.items())),
            "kept_leakage_role_count": sum(aggregate.get(role, 0) for role in LEAKAGE_ROLES),
            "dropped_leakage_role_count": sum(dropped.get(role, 0) for role in LEAKAGE_ROLES),
            "reconstructed_preamble_sentence_count": reconstructed_preamble_by_bucket.get(bucket_key, 0),
        }

    return panel_data, summary


def stacked_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, float]],
) -> None:
    bucket_keys = [bucket_key for bucket_key, _prefix, _display in BUCKETS]
    labels = [display for _bucket_key, _prefix, display in BUCKETS]
    x = np.arange(len(bucket_keys))
    bottom = np.zeros(len(bucket_keys))

    for role in ROLE_ORDER:
        values = np.array([data[bucket_key].get(role, 0.0) for bucket_key in bucket_keys])
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=ROLE_COLORS[role],
            width=0.78,
            edgecolor="white",
            linewidth=0.25,
            label=ROLE_LABELS[role],
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Average sentence-role proportion")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)


def ensure_outputs_are_new(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        existing_list = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing output files. "
            "Pass --overwrite to regenerate this new figure.\n"
            f"{existing_list}"
        )


def write_csv(path: Path, before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "bucket", "role", "average_sentence_role_proportion"])
        for panel_name, panel_data in (("before", before), ("after", after)):
            for bucket_key, _prefix, _display in BUCKETS:
                for role in ROLE_ORDER:
                    writer.writerow([panel_name, bucket_key, role, panel_data[bucket_key].get(role, 0.0)])


def make_figure(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    stacked_panel(axes[0], before)
    stacked_panel(axes[1], after)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    out_dir = resolve_output_arg(args.out_dir, config_path)
    cleaned_dir = resolve_path(args.cleaned_dir, Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "rhetorical_role_before_after_leakage_filtering"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    out_csv = out_dir / f"{stem}.csv"
    out_summary = out_dir / f"{stem}_summary.json"
    ensure_outputs_are_new([out_png, out_pdf, out_csv, out_summary], args.overwrite)

    raw_dirs = raw_dirs_from_config(config)
    before, before_summary = load_before_role_data(raw_dirs)
    after, after_summary = load_after_role_data(cleaned_dir)

    make_figure(before, after, out_png, out_pdf)
    write_csv(out_csv, before, after)
    with out_summary.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "before": before_summary,
                "after": after_summary,
                "role_order": ROLE_ORDER,
                "dropped_by_filter": sorted(DROPPED_BY_FILTER),
                "figure_outputs": {"png": str(out_png), "pdf": str(out_pdf), "csv": str(out_csv)},
            },
            fh,
            indent=2,
        )

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
