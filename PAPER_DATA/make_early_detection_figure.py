#!/usr/bin/env python3
"""Build the paper figure for multi-hearing early-detection signals."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


EARLY_SIGNALS = [
    (">=3 fact-role\nsentences", 0.6538, 0.3667, 52),
    ("Ratio segment\npresent", 0.6486, 0.4000, 37),
    (">=25 first-hearing\nsentences", 0.6078, 0.3884, 51),
    ("Petitioner argument\npresent", 0.5862, 0.3860, 58),
    ("Respondent argument\npresent", 0.6667, 0.4138, 27),
]

LATE_SIGNALS = [
    ("Facts added", 0.8169, 0.5217, 71),
    (">=3 fact sentences\nadded", 0.8276, 0.6111, 58),
    ("Section 482\nadded", 0.9444, 0.6974, 18),
    ("Section 438(2)\nadded", 1.0000, 0.7108, 11),
    ("Precedent\nadded", 0.7971, 0.6000, 69),
]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def panel(ax, rows, title, x_label, present_color, absent_color, baseline, baseline_label):
    labels = [r[0] for r in rows]
    present = np.array([r[1] * 100 for r in rows])
    absent = np.array([r[2] * 100 for r in rows])
    support = [r[3] for r in rows]
    y = np.arange(len(rows))
    h = 0.33

    ax.barh(y + h / 2, absent, height=h, color=absent_color, label="Signal absent")
    ax.barh(y - h / 2, present, height=h, color=present_color, label="Signal present")
    ax.axvline(baseline * 100, color="#334155", linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(
        baseline * 100 + 1.0,
        -0.48,
        baseline_label,
        ha="left",
        va="center",
        fontsize=7.2,
        color="#334155",
    )

    for i, (p, a, n) in enumerate(zip(present, absent, support)):
        ax.text(p + 1.1, i - h / 2, f"{p:.1f}%", va="center", fontsize=7.5, color="#0f172a")
        ax.text(a + 1.1, i + h / 2, f"{a:.1f}%", va="center", fontsize=7.5, color="#475569")
        ax.text(2.0, i - 0.45, f"n={n}", va="center", ha="left", fontsize=6.8, color="#64748b")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel(x_label, fontsize=8.6)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#111827")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.8, colors="#475569")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    panel(
        axes[0],
        EARLY_SIGNALS,
        "A. First-hearing signals",
        "First-stage prediction already correct",
        present_color="#2e8b57",
        absent_color="#cde8d8",
        baseline=0.4535,
        baseline_label="overall 45.4%",
    )
    panel(
        axes[1],
        LATE_SIGNALS,
        "B. Later-added correction signals",
        "Initially wrong cases corrected by final stage",
        present_color="#1f7a9a",
        absent_color="#cfeaf2",
        baseline=0.7447,
        baseline_label="overall 74.5%",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), w_pad=2.8)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"early_detection_signals.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
