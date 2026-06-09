#!/usr/bin/env python3
"""Render Phase 4 PGExplainer evidence panels for selected cases.

This figure helper uses only bucket-local Phase 4 bundles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import Paths, explained_case_indices, load_phase4  # noqa: E402

CATEGORY_COLORS = {
    "statute": "#2ca02c",
    "provision": "#9467bd",
    "precedent": "#ff7f0e",
}
TARGET_COLOR = "#111111"


def render_labelled_subgraph(
    paths: Paths,
    case_idx: int,
    out_dir: Path,
    max_per_cat: int = 3,
) -> Path:
    bundle = load_phase4(paths, case_idx)
    case_id = str(bundle.get("case_id", ""))[:55]
    target_lbl = str(bundle.get("target_label", "?")).strip()
    pred_lbl = str(bundle.get("predicted_label", "?")).strip()
    conf = float(bundle.get("confidence", 0.0))
    verdict = "CORRECT" if target_lbl == pred_lbl else "WRONG"

    categories: list[tuple[str, list[dict]]] = []
    top = bundle.get("top_nodes", {}) or {}
    for key in ("statute", "provision", "precedent"):
        items = [item for item in (top.get(key) or []) if item][:max_per_cat]
        if items:
            categories.append((key, items))

    total = sum(len(items) for _, items in categories) or 1
    max_imp = max(
        (float(item.get("importance", 0.0)) for _, items in categories for item in items),
        default=1.0,
    )
    max_imp = max(max_imp, 1e-6)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter([0], [0], s=1400, c=TARGET_COLOR, zorder=3)
    ax.text(
        0,
        -0.18,
        f"TARGET\ntrue={target_lbl} pred={pred_lbl} ({conf:.2f})\n{verdict}",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(0, 0.18, case_id, ha="center", va="bottom", fontsize=9, style="italic")

    angle_step = 2 * np.pi / total
    k = 0
    seen: set[str] = set()
    for category, items in categories:
        color = CATEGORY_COLORS[category]
        for item in items:
            theta = k * angle_step - np.pi / 2
            x, y = np.cos(theta), np.sin(theta)
            imp = float(item.get("importance", 0.0))
            lw = 0.8 + 5.0 * (imp / max_imp)
            ax.plot([0, x], [0, y], color=color, linewidth=lw, alpha=0.8, zorder=1)
            ax.scatter([x], [y], s=200, c=color, zorder=2, edgecolors="white", linewidths=1.2)
            text = str(item.get("text", ""))[:45]
            ha = "left" if x >= 0 else "right"
            x_text = x + 0.12 * (1 if x >= 0 else -1)
            ax.text(
                x_text,
                y,
                f"{text}\nimp {imp:.2f}",
                ha=ha,
                va="center",
                fontsize=8,
                color="#222222",
            )
            seen.add(category)
            k += 1

    legend_handles = [mpatches.Patch(color=CATEGORY_COLORS[key], label=key) for key in sorted(seen)]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Bucket-local PGExplainer top legal nodes - case {case_idx}", fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"panel_b_subgraph_case_{case_idx}_labelled.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-indices", type=int, nargs="+", default=None)
    ap.add_argument("--auto", type=int, default=2)
    args = ap.parse_args()

    paths = Paths.default()
    if args.case_indices:
        case_indices = args.case_indices
    else:
        case_indices = explained_case_indices(paths)[: args.auto]

    out_dir = paths.fig_dir / "labelled_pair"
    for idx in case_indices:
        fp = render_labelled_subgraph(paths, idx, out_dir)
        print(f"saved {fp}", flush=True)


if __name__ == "__main__":
    main()
