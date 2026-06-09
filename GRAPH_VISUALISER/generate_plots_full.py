#!/usr/bin/env python3
"""
generate_plots_full.py — Thesis plots from the FULL raw dataset (~74k cases).

Reads directly from Timeline_Maker bucket directories (no graph needed).
Outputs PDF + PNG figures to outputs/plots_full/.

Usage:
  python generate_plots_full.py [--config config.yaml] [--out outputs/plots_full]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

from path_utils import load_config, resolve_output_arg

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

BUCKET_DISPLAY = {
    "financial_fraud":    "Financial fraud",
    "fin_fraud":          "Financial fraud",
    "family_matrimonial": "Family/matrimonial",
    "land_property":      "Land/property",
    "motor_accidents":    "Motor accidents",
    "sexual_offences":    "Sexual offences",
}

ENTITY_TYPES = ["STATUTE", "PROVISION", "PRECEDENT", "COURT", "JUDGE",
                "LAWYER", "PETITIONER", "RESPONDENT"]

# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    print(f"  saved {stem}.pdf / .png")
    plt.close(fig)


def bucket_colors(cfg: dict) -> dict:
    return {b["name"]: b["color"] for b in cfg.get("buckets", [])}


def normalise_outcome(label: str | None) -> str:
    if not label:
        return "unknown"
    l = label.lower()
    if "appellant_won" in l or "petitioner_won" in l or "_won" in l:
        return "win"
    if "appellant_lost" in l or "respondent_won" in l or "_lost" in l:
        return "loss"
    return "unknown"


def load_bucket(data_dir: Path, limit: int | None = None) -> list[dict]:
    files = sorted(p for p in data_dir.iterdir()
                   if p.suffix == ".json" and p.name.lower() != "report.json")
    if limit:
        files = files[:limit]
    cases = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                cases.append(json.load(fh))
        except Exception:
            pass
    return cases


# ── per-case entity counter ────────────────────────────────────────────────────

def count_entities(case: dict) -> dict[str, int]:
    ner = case.get("ner_by_label", {})
    return {etype: (len(v) if isinstance(v, list) else v)
            for etype, v in ner.items() if etype in set(ENTITY_TYPES)}


def sentence_count_by_role(case: dict) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for s in case.get("sentences", []):
        role = s.get("rhetorical_role", "UNKNOWN")
        counts[role] += 1
    return dict(counts)


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_case_counts(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Horizontal bar: total cases per domain."""
    labels = [BUCKET_DISPLAY.get(k, k) for k in bucket_data]
    values = [len(v) for v in bucket_data.values()]
    colors = [bcolors.get(k, "#888") for k in bucket_data]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of cases")
    ax.set_title("Total cases per legal domain (full dataset)")
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, out_dir, "case_counts_full")


def plot_outcome_distribution(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Stacked bar: win / loss / unknown per domain."""
    ordered = list(bucket_data.keys())
    wins, losses, unknowns = [], [], []
    for bkt in ordered:
        w = l = u = 0
        for case in bucket_data[bkt]:
            o = normalise_outcome(case.get("case_outcome_label"))
            if o == "win":    w += 1
            elif o == "loss": l += 1
            else:             u += 1
        wins.append(w); losses.append(l); unknowns.append(u)

    x = np.arange(len(ordered))
    xlabels = [BUCKET_DISPLAY.get(b, b) for b in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, wins,    label="Win",     color="#27ae60")
    ax.bar(x, losses,  bottom=wins,     label="Loss",    color="#e74c3c")
    ax.bar(x, unknowns,
           bottom=[w + l for w, l in zip(wins, losses)],
           label="Unknown", color="#95a5a6")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("Number of cases")
    ax.set_title("Case outcome distribution — full dataset")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, "outcome_distribution_full")


def plot_outcome_rate(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Grouped bar: win-rate and loss-rate (excluding unknowns) per domain."""
    ordered = list(bucket_data.keys())
    win_rates, loss_rates = [], []
    for bkt in ordered:
        w = l = 0
        for case in bucket_data[bkt]:
            o = normalise_outcome(case.get("case_outcome_label"))
            if o == "win":    w += 1
            elif o == "loss": l += 1
        total = w + l or 1
        win_rates.append(w / total * 100)
        loss_rates.append(l / total * 100)

    x = np.arange(len(ordered))
    xlabels = [BUCKET_DISPLAY.get(b, b) for b in ordered]
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width/2, win_rates,  width, label="Win %",  color="#27ae60")
    ax.bar(x + width/2, loss_rates, width, label="Loss %", color="#e74c3c")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("% of labelled cases")
    ax.set_title("Win / loss rate per domain (labelled cases only)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, "outcome_rate_full")


def plot_entity_type_totals(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Grouped bar: total entity mentions per type, broken down by domain."""
    etype_bucket: dict[str, dict[str, int]] = {e: defaultdict(int) for e in ENTITY_TYPES}
    for bkt, cases in bucket_data.items():
        for case in cases:
            for etype, cnt in count_entities(case).items():
                etype_bucket[etype][bkt] += cnt

    ordered_bkts = list(bucket_data.keys())
    x = np.arange(len(ENTITY_TYPES))
    width = 0.15
    offsets = np.linspace(-(len(ordered_bkts)-1)/2, (len(ordered_bkts)-1)/2, len(ordered_bkts))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, bkt in enumerate(ordered_bkts):
        vals = [etype_bucket[e][bkt] for e in ENTITY_TYPES]
        ax.bar(x + offsets[i] * width, vals, width,
               label=BUCKET_DISPLAY.get(bkt, bkt), color=bcolors.get(bkt, "#888"),
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in ENTITY_TYPES], rotation=25, ha="right")
    ax.set_ylabel("Total entity mentions")
    ax.set_title("Entity mention counts by type and domain — full dataset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "entity_type_totals_full")


def plot_entity_per_case(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Box plot: total entities per case, per domain."""
    ordered = list(bucket_data.keys())
    data = []
    for bkt in ordered:
        per_case = [sum(count_entities(c).values()) for c in bucket_data[bkt]]
        data.append(per_case)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, bkt in zip(bp["boxes"], ordered):
        patch.set_facecolor(bcolors.get(bkt, "#888"))
        patch.set_alpha(0.8)
    ax.set_xticklabels([BUCKET_DISPLAY.get(b, b) for b in ordered],
                       rotation=25, ha="right")
    ax.set_ylabel("Entity mentions per case")
    ax.set_title("Entity richness per case by domain (outliers hidden)")
    fig.tight_layout()
    savefig(fig, out_dir, "entity_per_case_full")


def plot_sentence_count_distribution(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Box plot: sentences per case per domain."""
    ordered = list(bucket_data.keys())
    data = [[len(c.get("sentences", [])) for c in bucket_data[bkt]] for bkt in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, bkt in zip(bp["boxes"], ordered):
        patch.set_facecolor(bcolors.get(bkt, "#888"))
        patch.set_alpha(0.8)
    ax.set_xticklabels([BUCKET_DISPLAY.get(b, b) for b in ordered],
                       rotation=25, ha="right")
    ax.set_ylabel("Sentences per case")
    ax.set_title("Case length (sentence count) by domain (outliers hidden)")
    fig.tight_layout()
    savefig(fig, out_dir, "sentence_count_distribution_full")


def plot_rhetorical_role_breakdown(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Stacked bar: average sentence proportion by rhetorical role per domain."""
    ROLES = ["PREAMBLE", "FAC", "ISSUE", "ARG_PETITIONER", "ARG_RESPONDENT",
             "ANALYSIS", "STA", "PRE_RELIED", "PRE_NOT_RELIED", "RLC", "RPC", "NONE"]
    role_display = {
        "PREAMBLE": "Preamble", "FAC": "Facts", "ISSUE": "Issue",
        "ARG_PETITIONER": "Arg (pet.)", "ARG_RESPONDENT": "Arg (resp.)",
        "ANALYSIS": "Analysis", "STA": "Statute", "PRE_RELIED": "Prec. relied",
        "PRE_NOT_RELIED": "Prec. not relied", "RLC": "RLC", "RPC": "RPC", "NONE": "None",
    }
    role_colors = [
        "#3498db","#e67e22","#9b59b6","#27ae60","#e74c3c",
        "#1abc9c","#f39c12","#2980b9","#8e44ad","#d35400","#16a085","#95a5a6"
    ]

    ordered = list(bucket_data.keys())
    # avg proportion of each role per domain
    matrix = np.zeros((len(ordered), len(ROLES)))
    for i, bkt in enumerate(ordered):
        for case in bucket_data[bkt]:
            rc = sentence_count_by_role(case)
            total = sum(rc.values()) or 1
            for j, role in enumerate(ROLES):
                matrix[i, j] += rc.get(role, 0) / total
        matrix[i] /= max(len(bucket_data[bkt]), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(ordered))
    for j, role in enumerate(ROLES):
        ax.bar(range(len(ordered)), matrix[:, j], bottom=bottom,
               label=role_display.get(role, role), color=role_colors[j % len(role_colors)])
        bottom += matrix[:, j]
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([BUCKET_DISPLAY.get(b, b) for b in ordered], rotation=25, ha="right")
    ax.set_ylabel("Avg sentence proportion")
    ax.set_title("Rhetorical role composition by domain")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, out_dir, "rhetorical_role_breakdown_full")


def plot_labelling_coverage(bucket_data: dict[str, list], bcolors: dict, out_dir: Path) -> None:
    """Bar: % of cases with a valid outcome label per domain."""
    ordered = list(bucket_data.keys())
    rates = []
    for bkt in ordered:
        cases = bucket_data[bkt]
        labelled = sum(1 for c in cases
                       if normalise_outcome(c.get("case_outcome_label")) != "unknown")
        rates.append(labelled / max(len(cases), 1) * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [bcolors.get(b, "#888") for b in ordered]
    bars = ax.bar([BUCKET_DISPLAY.get(b, b) for b in ordered], rates,
                  color=colors, edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% labelled cases")
    ax.set_ylim(0, 115)
    ax.set_title("Outcome labelling coverage by domain")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    savefig(fig, out_dir, "labelling_coverage_full")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out",    default="outputs/plots_full")
    args = parser.parse_args()

    cfg, config_path = load_config(args.config)
    out_dir = resolve_output_arg(args.out, config_path)
    bcolors = bucket_colors(cfg)

    print("Loading full dataset …")
    bucket_data: dict[str, list] = {}
    for bkt_cfg in cfg.get("buckets", []):
        name  = bkt_cfg["name"]
        ddir  = Path(bkt_cfg["data_dir"])
        limit = bkt_cfg.get("limit")
        print(f"  {name} — reading {ddir.name} …", end=" ", flush=True)
        cases = load_bucket(ddir, limit)
        bucket_data[name] = cases
        print(f"{len(cases):,} cases")

    total = sum(len(v) for v in bucket_data.values())
    print(f"\nTotal: {total:,} cases across {len(bucket_data)} domains\n")

    plot_case_counts(bucket_data, bcolors, out_dir)
    plot_outcome_distribution(bucket_data, bcolors, out_dir)
    plot_outcome_rate(bucket_data, bcolors, out_dir)
    plot_entity_type_totals(bucket_data, bcolors, out_dir)
    plot_entity_per_case(bucket_data, bcolors, out_dir)
    plot_sentence_count_distribution(bucket_data, bcolors, out_dir)
    plot_rhetorical_role_breakdown(bucket_data, bcolors, out_dir)
    plot_labelling_coverage(bucket_data, bcolors, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
