#!/usr/bin/env python3
"""
generate_plots.py — Export thesis-ready static figures from GRAPH_VISUALISER outputs.

Reads the pre-built artefacts in outputs/ and writes PDF + PNG figures to
outputs/plots/.  Run after build_graph.py.

Usage:
  python generate_plots.py [--config config.yaml] [--out outputs/plots]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

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

ENTITY_ORDER = ["statute", "provision", "precedent", "court", "judge", "lawyer",
                "petitioner", "respondent"]


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    print(f"  saved {stem}.pdf / .png")
    plt.close(fig)


def bucket_colors(cfg: dict) -> dict:
    return {b["name"]: b["color"] for b in cfg.get("buckets", [])}


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_node_type_distribution(G: nx.Graph, out_dir: Path) -> None:
    """Bar chart: number of nodes per type."""
    counts: dict[str, int] = defaultdict(int)
    for _, d in G.nodes(data=True):
        counts[d.get("node_type", "unknown")] += 1

    order = ["case"] + ENTITY_ORDER
    labels = [t for t in order if t in counts]
    values = [counts[t] for t in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4f86c6", edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Node type")
    ax.set_ylabel("Count")
    ax.set_title("Node-type composition of the legal graph")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    savefig(fig, out_dir, "node_type_distribution")


def plot_case_distribution_by_bucket(stats: dict, bcolors: dict, out_dir: Path) -> None:
    """Horizontal bar chart: cases per legal domain."""
    bc = stats.get("bucket_counts", {})
    labels = [BUCKET_DISPLAY.get(k, k) for k in bc]
    values = list(bc.values())
    colors = [bcolors.get(k, "#888") for k in bc]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of cases")
    ax.set_title("Case distribution across legal domains")
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, out_dir, "case_distribution_by_bucket")


def plot_degree_distribution(G: nx.Graph, out_dir: Path) -> None:
    """Log-log degree distribution (all nodes + case-only)."""
    all_degrees = [d for _, d in G.degree()]
    case_degrees = [G.degree(n) for n, a in G.nodes(data=True) if a.get("node_type") == "case"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, degrees, title in [
        (axes[0], all_degrees, "All nodes"),
        (axes[1], case_degrees, "Case nodes only"),
    ]:
        deg_counts = defaultdict(int)
        for d in degrees:
            deg_counts[d] += 1
        xs = sorted(deg_counts)
        ys = [deg_counts[x] for x in xs]
        ax.scatter(xs, ys, s=10, alpha=0.7, color="#4f86c6")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (log)")
        ax.set_ylabel("Count (log)")
        ax.set_title(f"Degree distribution — {title}")

    fig.suptitle("Power-law degree distribution in the legal graph", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, "degree_distribution")


def plot_top_hub_entities(stats: dict, bcolors: dict, out_dir: Path) -> None:
    """Horizontal bar: top shared-entity hubs by degree (entities only)."""
    hubs = [h for h in stats.get("top_hubs", []) if h["type"] != "case"][:15]
    if not hubs:
        return

    labels = [h["label"][:45] for h in hubs]
    values = [h["degree"] for h in hubs]
    # colour by entity type (use a fixed palette)
    type_colors = {
        "statute": "#e67e22", "provision": "#f39c12", "precedent": "#8e44ad",
        "court": "#27ae60", "judge": "#2980b9", "lawyer": "#16a085",
        "petitioner": "#c0392b", "respondent": "#d35400",
    }
    colors = [type_colors.get(h["type"], "#888") for h in hubs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    ax.set_xlabel("Degree (number of connected cases)")
    ax.set_title("Top shared-entity hubs (by degree)")

    # legend
    seen_types = list(dict.fromkeys(h["type"] for h in hubs))
    patches = [mpatches.Patch(color=type_colors.get(t, "#888"), label=t.capitalize())
               for t in seen_types]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir, "top_hub_entities")


def plot_cross_bucket_bridges(stats: dict, bcolors: dict, out_dir: Path) -> None:
    """Bar chart: entities that bridge the most legal domains."""
    bridges = [b for b in stats.get("top_hubs", [])
               if b.get("bucket_count", 0) >= 2][:15]
    if not bridges:
        print("  (no bridge entities found — skipping)")
        return

    bridges.sort(key=lambda b: b["bucket_count"], reverse=True)
    labels = [b["label"][:40] for b in bridges]
    bucket_counts = [b["bucket_count"] for b in bridges]
    degrees = [b["degree"] for b in bridges]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, degrees,    width, label="Degree", color="#4f86c6")
    ax.bar(x + width / 2, bucket_counts, width, label="Domains bridged", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Cross-domain bridge entities")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, "cross_bucket_bridges")


def plot_connectivity_score_distributions(G: nx.Graph, bcolors: dict, out_dir: Path) -> None:
    """Box-plot of case connectivity scores, one box per domain."""
    bucket_scores: dict[str, list] = defaultdict(list)
    for _, d in G.nodes(data=True):
        if d.get("node_type") == "case" and "connectivity_score" in d:
            bucket_scores[d["bucket"]].append(d["connectivity_score"])

    if not bucket_scores:
        print("  (no connectivity scores — skipping)")
        return

    ordered = [b for b in BUCKET_DISPLAY if b in bucket_scores]
    data    = [bucket_scores[b] for b in ordered]
    xlabels = [BUCKET_DISPLAY[b] for b in ordered]
    colors  = [bcolors.get(b, "#888") for b in ordered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("Connectivity score")
    ax.set_title("Distribution of case connectivity scores by domain")
    fig.tight_layout()
    savefig(fig, out_dir, "connectivity_score_distributions")


def plot_entity_sharing_heatmap(G: nx.Graph, bcolors: dict, out_dir: Path) -> None:
    """Heatmap: how many shared entities each pair of domains has in common."""
    buckets = [b for b in BUCKET_DISPLAY if b in
               {d.get("bucket") for _, d in G.nodes(data=True)}]
    n = len(buckets)
    if n < 2:
        return

    # For each shared entity, collect which buckets it touches
    entity_buckets: dict[str, set] = defaultdict(set)
    for node, d in G.nodes(data=True):
        if d.get("node_type") == "case":
            case_bucket = d.get("bucket")
            for nbr in G.neighbors(node):
                nd = G.nodes[nbr]
                if nd.get("is_shared"):
                    entity_buckets[nbr].add(case_bucket)

    matrix = np.zeros((n, n), dtype=int)
    for entity, bset in entity_buckets.items():
        blist = [b for b in bset if b in buckets]
        for i, bi in enumerate(buckets):
            for j, bj in enumerate(buckets):
                if bi in blist and bj in blist and i != j:
                    matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    tick_labels = [BUCKET_DISPLAY.get(b, b) for b in buckets]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Shared entity count")
    ax.set_title("Cross-domain entity sharing matrix")
    fig.tight_layout()
    savefig(fig, out_dir, "entity_sharing_heatmap")


def plot_outcome_distribution(G: nx.Graph, bcolors: dict, out_dir: Path) -> None:
    """Stacked bar: win/loss/unknown per domain."""
    bucket_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, d in G.nodes(data=True):
        if d.get("node_type") == "case":
            bucket_outcomes[d["bucket"]][d.get("outcome", "unknown")] += 1

    ordered = [b for b in BUCKET_DISPLAY if b in bucket_outcomes]
    if not ordered:
        return

    wins    = [bucket_outcomes[b].get("win", 0)     for b in ordered]
    losses  = [bucket_outcomes[b].get("loss", 0)    for b in ordered]
    unknown = [bucket_outcomes[b].get("unknown", 0) for b in ordered]
    xlabels = [BUCKET_DISPLAY[b] for b in ordered]
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, wins,    label="Win",     color="#27ae60")
    ax.bar(x, losses,  bottom=wins,     label="Loss",    color="#e74c3c")
    ax.bar(x, unknown, bottom=[w+l for w,l in zip(wins, losses)],
           label="Unknown", color="#95a5a6")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("Number of cases")
    ax.set_title("Case outcome distribution by domain")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir, "outcome_distribution")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out",    default="outputs/plots")
    args = parser.parse_args()

    cfg, config_path = load_config(args.config)
    out_dir  = resolve_output_arg(args.out, config_path)
    artefact = Path(cfg.get("output_dir", cfg.get("output", {}).get("dir", "outputs")))
    bcolors  = bucket_colors(cfg)

    print("Loading artefacts …")
    with open(artefact / "graph_sample.pkl", "rb") as f:
        G: nx.Graph = pickle.load(f)
    with open(artefact / "stats.json") as f:
        stats: dict = json.load(f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Writing plots to {out_dir}/\n")

    plot_node_type_distribution(G, out_dir)
    plot_case_distribution_by_bucket(stats, bcolors, out_dir)
    plot_degree_distribution(G, out_dir)
    plot_top_hub_entities(stats, bcolors, out_dir)
    plot_cross_bucket_bridges(stats, bcolors, out_dir)
    plot_connectivity_score_distributions(G, bcolors, out_dir)
    plot_entity_sharing_heatmap(G, bcolors, out_dir)
    plot_outcome_distribution(G, bcolors, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
