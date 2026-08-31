#!/usr/bin/env python3
"""
Generate static paper figures from FINAL_EXPLANATION analysis outputs.

Figures:
  fig1_faithfulness_curves
  fig2_community_cluster_sankey
  fig3_contrastive_subgraph_51419_15962
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
EXP3 = ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
EXP5 = ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why"
OUT_DIR = ROOT / "figures"


COLORS = {
    "counterfactual": "#1f77b4",
    "attention": "#ff7f0e",
    "random": "#7f7f7f",
    "query": "#d62728",
    "opposite": "#2ca02c",
    "shared": "#6a51a3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return p.parse_args()


def savefig(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[figure] wrote {path}")


def mean_by(items: list[dict], key_fields: tuple[str, ...], value_field: str) -> dict[tuple, float]:
    vals: dict[tuple, list[float]] = defaultdict(list)
    for row in items:
        key = tuple(row[k] for k in key_fields)
        try:
            vals[key].append(float(row[value_field]))
        except (TypeError, ValueError):
            pass
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def fig1_faithfulness(out_dir: Path, formats: list[str]) -> None:
    curves_path = EXP3 / "faithfulness_curves.csv"
    auc_path = EXP3 / "faithfulness_auc_summary.csv"
    rows = list(csv.DictReader(open(curves_path, encoding="utf-8")))
    auc_rows = {r["ranker"]: r for r in csv.DictReader(open(auc_path, encoding="utf-8"))}

    suff = mean_by(rows, ("ranker", "k_requested"), "sufficiency_proba")
    comp = mean_by(rows, ("ranker", "k_requested"), "comprehensiveness_drop")

    rankers = ["counterfactual", "attention", "random"]
    labels = {"counterfactual": "Counterfactual", "attention": "Attention", "random": "Random"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)

    for ranker in rankers:
        xs = sorted({int(k[1]) for k in suff if k[0] == ranker})
        y_s = [suff[(ranker, str(x))] for x in xs]
        y_c = [comp[(ranker, str(x))] for x in xs]
        auc = auc_rows[ranker]
        axes[0].plot(
            xs,
            y_s,
            marker="o",
            linewidth=2.2,
            color=COLORS[ranker],
            label=f"{labels[ranker]}  AUC={float(auc['mean_sufficiency_auc']):.3f}",
        )
        axes[1].plot(
            xs,
            y_c,
            marker="o",
            linewidth=2.2,
            color=COLORS[ranker],
            label=f"{labels[ranker]}  AUC={float(auc['mean_comprehensiveness_auc']):.3f}",
        )

    axes[0].set_title("Sufficiency: keep top-k evidence")
    axes[0].set_ylabel("Mean predicted-class probability")
    axes[1].set_title("Comprehensiveness: remove top-k evidence")
    axes[1].set_ylabel("Mean probability drop")
    for ax in axes:
        ax.set_xlabel("Top-k evidence groups")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(pad=1.0, w_pad=2.2)
    savefig(fig, out_dir, "fig1_faithfulness_curves", formats)
    plt.close(fig)


def draw_ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.34):
    verts = [
        (x0, y0a),
        ((x0 + x1) / 2, y0a),
        ((x0 + x1) / 2, y1a),
        (x1, y1a),
        (x1, y1b),
        ((x0 + x1) / 2, y1b),
        ((x0 + x1) / 2, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig2_sankey(out_dir: Path, formats: list[str]) -> None:
    rows = list(csv.DictReader(open(EXP5 / "case_embedding_clusters.csv", encoding="utf-8")))
    align_rows = list(csv.DictReader(open(EXP5 / "structural_embedding_alignment.csv", encoding="utf-8")))
    community_counts = Counter(int(r["community_id"]) for r in rows)
    top_communities = [cid for cid, _ in community_counts.most_common(12)]

    flow = Counter()
    label_counts: dict[int, Counter] = defaultdict(Counter)
    domain_counts: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        c = int(r["community_id"])
        if c not in top_communities:
            continue
        left = f"C{c}"
        cluster = r["embedding_cluster_id"]
        right = "Noise" if cluster == "-1" else f"Cluster {cluster}"
        flow[(left, right)] += 1
        label_counts[c][r["target_label"]] += 1
        domain_counts[c][r["domain_bucket"]] += 1

    left_order = [f"C{c}" for c in top_communities]
    right_order = ["Cluster 0", "Cluster 1", "Noise"]
    total = sum(flow.values())

    left_totals = {l: sum(flow[(l, r)] for r in right_order) for l in left_order}
    right_totals = {r: sum(flow[(l, r)] for l in left_order) for r in right_order}

    def positions(order, totals, gap=0.016, min_h=0.022):
        y = 0.97
        pos = {}
        usable = 0.94 - gap * (len(order) - 1)
        weights = {item: math.sqrt(max(totals[item], 1)) for item in order}
        weight_sum = sum(weights.values())
        for item in order:
            h = max(min_h, usable * weights[item] / weight_sum)
            pos[item] = (y - h, y)
            y -= h + gap
        return pos

    left_pos = positions(left_order, left_totals)
    right_pos = positions(right_order, right_totals, gap=0.075, min_h=0.10)
    left_cursor = {k: v[0] for k, v in left_pos.items()}
    right_cursor = {k: v[0] for k, v in right_pos.items()}
    cluster_colors = {"Cluster 0": "#2563eb", "Cluster 1": "#16a34a", "Noise": "#8b8f97"}

    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    x_left, x_right = 0.24, 0.77
    bar_w = 0.055

    bg = patches.FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.90,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#f8fafc",
        edgecolor="#e2e8f0",
        linewidth=1.0,
        zorder=-5,
    )
    ax.add_patch(bg)

    for l in left_order:
        y0, y1 = left_pos[l]
        cid = int(l[1:])
        dom = domain_counts[cid].most_common(1)[0][0].replace("_", " ")
        lab = label_counts[cid].most_common(1)[0][0]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_left - bar_w, y0),
                bar_w,
                y1 - y0,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor="#1e293b",
                edgecolor="none",
                alpha=0.95,
            )
        )
        label = f"{l}  ({left_totals[l]:,})\n{dom}; label {lab}"
        ax.text(x_left - bar_w - 0.018, (y0 + y1) / 2, label, ha="right", va="center", fontsize=8.2, color="#111827")

    for r in right_order:
        y0, y1 = right_pos[r]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_right, y0),
                bar_w,
                y1 - y0,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor=cluster_colors[r],
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax.text(x_right + bar_w + 0.018, (y0 + y1) / 2, f"{r}\n{right_totals[r]:,}", ha="left", va="center", fontsize=9.4, color="#111827")

    for l in left_order:
        left_height = left_pos[l][1] - left_pos[l][0]
        total_l = max(left_totals[l], 1)
        for r in right_order:
            count = flow[(l, r)]
            if not count:
                continue
            h = left_height * count / total_l
            right_height = right_pos[r][1] - right_pos[r][0]
            h_right = right_height * count / max(right_totals[r], 1)
            y0a = left_cursor[l]
            y0b = y0a + h
            y1a = right_cursor[r]
            y1b = y1a + h_right
            alpha = 0.18 + 0.34 * (count / total_l)
            draw_ribbon(ax, x_left, y0a, y0b, x_right, y1a, y1b, cluster_colors[r], alpha=min(alpha, 0.55))
            left_cursor[l] = y0b
            right_cursor[r] = y1b

    metric_text = ""
    if align_rows:
        ar = align_rows[0]
        bits = []
        for key, label in [("nmi", "NMI"), ("ari", "ARI"), ("v_measure", "V")]:
            if key in ar and ar[key]:
                bits.append(f"{label}={float(ar[key]):.3f}")
        metric_text = "  |  ".join(bits)
    ax.text(x_left - 0.02, 1.005, "Structural communities", ha="center", va="bottom", fontsize=12, weight="bold", color="#0f172a")
    ax.text(x_right + 0.03, 1.005, "Embedding clusters", ha="center", va="bottom", fontsize=12, weight="bold", color="#0f172a")
    ax.text(0.5, 0.965, metric_text, ha="center", va="center", fontsize=10, color="#475569")
    fig.tight_layout(pad=0.4)
    savefig(fig, out_dir, "fig2_community_cluster_sankey", formats)
    plt.close(fig)


def fig3_contrastive(out_dir: Path, formats: list[str]) -> None:
    """The published contrastive subgraph, case 51419 vs training case 15962.

    Delegates to the shared renderer used by the visualiser and the slide
    figures, so all three stay identical.  The pair is pinned (rather than
    re-resolved from the test pool) because it is the figure the paper cites,
    and the evidence keeps its published IDF + label-skew ranking.  Case 15962
    is a training case, so its column carries no counterfactual badges.
    """
    from generate_presentation_figures import figure_contrast
    from presentation_graphs import CaseNeighborIndex, CounterfactualFactorIndex, contrast_graph

    graph = contrast_graph(
        CaseNeighborIndex(EXP5),
        CounterfactualFactorIndex(EXP3),
        51419,
        side="opposite",
        other_case=15962,
        limit=5,
        order="evidence",
    )
    if not graph.get("available"):
        raise RuntimeError(f"Could not build the 51419 -> 15962 contrast: {graph.get('reason')}")
    # Filename is pinned: Latex_Documentation/PAPER_DATA/main.tex includes it.
    # detail=True keeps the idf readings and Δ values a print figure can carry;
    # the slide figures drop both.
    figure_contrast(
        graph, out_dir, formats, "", rows=5,
        stem="fig3_contrastive_subgraph_51419_15962", detail=True,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig1_faithfulness(out_dir, args.formats)
    fig2_sankey(out_dir, args.formats)
    fig3_contrastive(out_dir, args.formats)


if __name__ == "__main__":
    main()
