#!/usr/bin/env python3
"""
graph_stats.py — Comprehensive per-bucket and cross-bucket graph statistics.

Loads graph_full.pkl and computes:
  • Per-bucket: case/entity counts, degree stats, top hubs, entity diversity
  • Cross-bucket: entity sharing matrix, bridge analysis, shared-entity overlap
  • Centrality: degree (all nodes), PageRank (all nodes)
                (approximate betweenness disabled by default — too slow on this graph)
  • Saves stats JSON + thesis-ready PDF/PNG plots to outputs/graph_stats/

Usage:
  python graph_stats.py [--config config.yaml] [--out outputs/graph_stats]
                        [--with-betweenness]        # enable approx betweenness (slow)
                        [--betweenness-k 300]       # samples for approx betweenness
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from path_utils import load_config, resolve_output_arg

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
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
ENTITY_TYPES = ["statute", "provision", "precedent", "court", "judge",
                "lawyer", "petitioner", "respondent"]

# ── helpers ────────────────────────────────────────────────────────────────────

def bucket_colors(cfg: dict) -> dict:
    return {b["name"]: b["color"] for b in cfg.get("buckets", [])}

def savefig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    print(f"    saved {stem}")
    plt.close(fig)

def tick(label: str):
    print(f"  [{time.strftime('%H:%M:%S')}] {label} ...", flush=True)

def percentile_stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
        "std": float(a.std()),
    }


# ── 1. basic per-node attributes ───────────────────────────────────────────────

def collect_node_index(G: nx.Graph) -> tuple[dict, dict, dict]:
    """
    Returns:
      case_nodes  : {bucket -> [node_key, ...]}
      entity_nodes: {node_type -> [node_key, ...]}
      degree_map  : {node_key -> degree}
    """
    tick("Indexing nodes")
    case_nodes: dict[str, list] = defaultdict(list)
    entity_nodes: dict[str, list] = defaultdict(list)
    degree_map: dict[str, int] = dict(G.degree())

    for n, d in G.nodes(data=True):
        ntype = d.get("node_type", "unknown")
        if ntype == "case":
            case_nodes[d.get("bucket", "unknown")].append(n)
        else:
            entity_nodes[ntype].append(n)

    return dict(case_nodes), dict(entity_nodes), degree_map


# ── 2. per-bucket stats ────────────────────────────────────────────────────────

def per_bucket_stats(G: nx.Graph, case_nodes: dict, entity_nodes: dict,
                     degree_map: dict) -> dict:
    tick("Computing per-bucket stats")
    stats = {}
    for bkt, cases in case_nodes.items():
        case_degrees = [degree_map[n] for n in cases]

        # entity counts reachable from this bucket
        etype_counts: dict[str, set] = defaultdict(set)
        for cn in cases:
            for nb in G.neighbors(cn):
                nd = G.nodes[nb]
                etype_counts[nd.get("node_type", "?")].add(nb)

        # shared vs local entity split
        shared_nbrs = sum(1 for nb in etype_counts.get("statute", set()))
        # unique entity neighbours
        total_unique_entities = sum(len(s) for s in etype_counts.values())

        # outcome distribution
        outcomes: dict[str, int] = defaultdict(int)
        for cn in cases:
            outcomes[G.nodes[cn].get("outcome", "unknown")] += 1

        stats[bkt] = {
            "case_count":            len(cases),
            "case_degree":           percentile_stats(case_degrees),
            "entity_type_unique":    {k: len(v) for k, v in etype_counts.items()},
            "total_unique_entities": total_unique_entities,
            "outcomes":              dict(outcomes),
        }
    return stats


# ── 3. cross-bucket entity sharing ────────────────────────────────────────────

def cross_bucket_sharing(G: nx.Graph, case_nodes: dict) -> dict:
    tick("Computing cross-bucket entity sharing")

    # for each shared entity, which buckets reference it?
    entity_buckets: dict[str, set] = defaultdict(set)
    for bkt, cases in case_nodes.items():
        for cn in cases:
            for nb in G.neighbors(cn):
                if G.nodes[nb].get("is_shared", False):
                    entity_buckets[nb].add(bkt)

    buckets = sorted(case_nodes.keys())
    n = len(buckets)
    # co-occurrence matrix
    matrix = np.zeros((n, n), dtype=int)
    for entity, bset in entity_buckets.items():
        blist = [b for b in bset if b in buckets]
        for i, bi in enumerate(buckets):
            for j, bj in enumerate(buckets):
                if bi in blist and bj in blist:
                    matrix[i, j] += 1

    # bridge entities: shared by >= 2 buckets
    bridge_info = []
    for entity, bset in entity_buckets.items():
        if len(bset) >= 2:
            bridge_info.append({
                "node":         entity,
                "label":        G.nodes[entity].get("label", entity)[:60],
                "node_type":    G.nodes[entity].get("node_type", "?"),
                "bucket_count": len(bset),
                "buckets":      sorted(bset),
                "degree":       G.degree(entity),
            })
    bridge_info.sort(key=lambda x: (x["bucket_count"], x["degree"]), reverse=True)

    # how many shared entities per type bridge >= k buckets
    bridge_type_summary: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for b in bridge_info:
        bridge_type_summary[b["node_type"]][b["bucket_count"]] += 1

    return {
        "matrix":       matrix.tolist(),
        "buckets":      buckets,
        "bridges":      bridge_info[:200],   # top 200 saved to JSON
        "bridge_type_summary": {k: dict(v) for k, v in bridge_type_summary.items()},
        "entity_buckets": entity_buckets,    # kept in-memory only
    }


# ── 4. centrality ──────────────────────────────────────────────────────────────

def compute_pagerank(G: nx.Graph) -> dict[str, float]:
    tick("Computing PageRank (full graph, may take ~1 min)")
    return nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)


def compute_approx_betweenness(G: nx.Graph, entity_nodes: dict,
                                k: int = 300) -> dict[str, float]:
    """Approximate betweenness on the entity-only subgraph (no case nodes)."""
    tick(f"Building entity subgraph for betweenness (k={k} samples)")
    all_entities = [n for nodes in entity_nodes.values() for n in nodes]
    Ge = G.subgraph(all_entities)
    # Only keep the largest connected component to avoid isolated nodes
    lcc = max(nx.connected_components(Ge), key=len)
    Ge_lcc = Ge.subgraph(lcc).copy()
    tick(f"  Entity LCC: {Ge_lcc.number_of_nodes():,} nodes, "
         f"{Ge_lcc.number_of_edges():,} edges — running approx betweenness")
    bc = nx.betweenness_centrality(Ge_lcc, k=k, normalized=True, seed=42)
    return bc


# ── 5. top-N summaries ─────────────────────────────────────────────────────────

def top_nodes(G: nx.Graph, score_map: dict[str, float],
              node_type_filter: str | None, n: int = 20) -> list[dict]:
    results = []
    for node, score in score_map.items():
        d = G.nodes[node]
        if node_type_filter and d.get("node_type") != node_type_filter:
            continue
        results.append({
            "node":      node,
            "label":     d.get("label", node)[:60],
            "node_type": d.get("node_type", "?"),
            "bucket":    d.get("bucket", "entity"),
            "score":     score,
            "degree":    G.degree(node),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_bucket_entity_composition(bkt_stats: dict, bcolors: dict, out_dir: Path):
    """Grouped bar: unique entity types per bucket."""
    ordered = [b for b in BUCKET_DISPLAY if b in bkt_stats]
    xtypes  = [t for t in ENTITY_TYPES if t not in ("petitioner", "respondent")]
    x = np.arange(len(ordered))
    width = 0.11
    n = len(xtypes)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    ec = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
          "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, etype in enumerate(xtypes):
        vals = [bkt_stats[b]["entity_type_unique"].get(etype, 0) for b in ordered]
        ax.bar(x + offsets[i], vals, width,
               label=etype.capitalize(), color=ec.get(etype, "#aaa"),
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=20, ha="right")
    ax.set_ylabel("Unique entity nodes")
    ax.set_title("Unique shared-entity nodes per type and domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "per_bucket_entity_composition")


def plot_case_degree_boxplot(bkt_stats: dict, bcolors: dict, out_dir: Path):
    """Reconstructed box plot from percentile stats."""
    ordered = [b for b in BUCKET_DISPLAY if b in bkt_stats]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, bkt in enumerate(ordered):
        s = bkt_stats[bkt]["case_degree"]
        bp_data = {
            "med":    s["median"],
            "q1":     s["p25"],
            "q3":     s["p75"],
            "whislo": s["min"],
            "whishi": min(s["p99"], s["p75"] + 1.5*(s["p75"]-s["p25"])),
            "fliers": [],
        }
        ax.bxp([bp_data], positions=[i], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=bcolors.get(bkt,"#888"), alpha=0.8),
               medianprops=dict(color="white", linewidth=2),
               whiskerprops=dict(color="#555"),
               capprops=dict(color="#555"))
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=20, ha="right")
    ax.set_ylabel("Case node degree")
    ax.set_title("Case node degree distribution by domain (whiskers = p99 cap)")
    fig.tight_layout()
    savefig(fig, out_dir, "case_degree_boxplot")


def plot_degree_distribution_by_type(G: nx.Graph, out_dir: Path):
    """Log-log degree distribution per entity type (one subplot each)."""
    types = ["case"] + ENTITY_TYPES
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for ax, ntype in zip(axes, types):
        degs = [G.degree(n) for n, d in G.nodes(data=True)
                if d.get("node_type") == ntype]
        if not degs:
            ax.set_visible(False); continue
        counts = defaultdict(int)
        for d in degs: counts[d] += 1
        xs, ys = zip(*sorted(counts.items()))
        ax.scatter(xs, ys, s=6, alpha=0.6, color="#4f86c6")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ntype.capitalize(), fontsize=10)
        ax.set_xlabel("Degree", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(types):]:
        ax.set_visible(False)
    fig.suptitle("Degree distributions per node type (log-log)", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, "degree_dist_by_type")


def plot_cross_bucket_matrix(sharing: dict, bcolors: dict, out_dir: Path):
    """Heatmap of shared-entity co-occurrence between buckets."""
    matrix = np.array(sharing["matrix"])
    buckets = sharing["buckets"]
    labels = [BUCKET_DISPLAY.get(b, b) for b in buckets]
    n = len(buckets)

    fig, ax = plt.subplots(figsize=(7, 6))
    # off-diagonal only to highlight cross-bucket
    display = matrix.copy().astype(float)
    np.fill_diagonal(display, 0)
    im = ax.imshow(display, cmap="YlOrRd")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=8,
                    color="black" if display[i, j] < display.max()*0.6 else "white")
    plt.colorbar(im, ax=ax, label="Shared entity count")
    ax.set_title("Cross-domain entity sharing\n(diagonal = self-count, shown as 0)")
    fig.tight_layout()
    savefig(fig, out_dir, "cross_bucket_sharing_matrix")


def plot_bridge_entity_distribution(sharing: dict, out_dir: Path):
    """Bar: how many entities bridge exactly k domains, by entity type."""
    type_summary = sharing["bridge_type_summary"]
    all_k = sorted({k for v in type_summary.values() for k in v.keys()})
    types_in_plot = [t for t in ENTITY_TYPES if t in type_summary]

    type_colors = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
                   "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D",
                   "petitioner":"#16A085","respondent":"#D35400"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(all_k))
    width = 0.8 / max(len(types_in_plot), 1)
    n = len(types_in_plot)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    for i, etype in enumerate(types_in_plot):
        vals = [type_summary[etype].get(k, 0) for k in all_k]
        ax.bar(x + offsets[i], vals, width,
               label=etype.capitalize(), color=type_colors.get(etype, "#aaa"),
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} domain{'s' if k>1 else ''}" for k in all_k])
    ax.set_ylabel("Number of entities")
    ax.set_title("Bridge entities: how many domains each entity type spans")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    savefig(fig, out_dir, "bridge_entity_distribution")


def plot_top_pagerank(G: nx.Graph, pr: dict, out_dir: Path):
    """Top 20 highest PageRank nodes across all types."""
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = [G.nodes[n].get("label", n)[:35] for n, _ in top]
    scores = [v for _, v in top]
    types  = [G.nodes[n].get("node_type", "?") for n, _ in top]
    type_colors = {"case":"#4f86c6","statute":"#F1C40F","provision":"#E67E22",
                   "precedent":"#C0392B","court":"#8E44AD","judge":"#2980B9",
                   "lawyer":"#7F8C8D","petitioner":"#16A085","respondent":"#D35400"}
    colors = [type_colors.get(t, "#aaa") for t in types]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("PageRank score")
    ax.set_title("Top 20 nodes by PageRank")
    seen = {}
    patches = []
    for t, c in zip(types, colors):
        if t not in seen:
            seen[t] = True
            patches.append(mpatches.Patch(color=c, label=t.capitalize()))
    ax.legend(handles=patches, fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "top_pagerank_nodes")


def plot_top_pagerank_by_type(G: nx.Graph, pr: dict, out_dir: Path):
    """Top 10 PageRank nodes for each entity type (3×3 grid)."""
    types = ["statute", "provision", "precedent", "court", "judge", "lawyer"]
    type_colors = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
                   "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, ntype in zip(axes, types):
        top = [(n, v) for n, v in pr.items()
               if G.nodes[n].get("node_type") == ntype]
        top.sort(key=lambda x: x[1], reverse=True)
        top = top[:10]
        if not top: ax.set_visible(False); continue
        labels = [G.nodes[n].get("label", n)[:40] for n, _ in top]
        scores = [v for _, v in top]
        ax.barh(labels[::-1], scores[::-1],
                color=type_colors.get(ntype, "#aaa"), edgecolor="white", linewidth=0.3)
        ax.set_title(ntype.capitalize(), fontsize=10)
        ax.set_xlabel("PageRank", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Top 10 nodes by PageRank per entity type", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, "top_pagerank_by_type")


def plot_top_betweenness(G: nx.Graph, bc: dict, out_dir: Path):
    """Top 20 nodes by approximate betweenness centrality."""
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = [G.nodes[n].get("label", n)[:40] for n, _ in top]
    scores = [v for _, v in top]
    types  = [G.nodes[n].get("node_type", "?") for n, _ in top]
    type_colors = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
                   "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D",
                   "petitioner":"#16A085","respondent":"#D35400"}
    colors = [type_colors.get(t, "#aaa") for t in types]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Approx. betweenness centrality (entity subgraph)")
    ax.set_title("Top 20 entity nodes by betweenness centrality\n"
                 "(approximated on entity-only subgraph)")
    seen = {}
    patches = []
    for t, c in zip(types, colors):
        if t not in seen:
            seen[t] = True
            patches.append(mpatches.Patch(color=c, label=t.capitalize()))
    ax.legend(handles=patches, fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "top_betweenness_nodes")


def plot_top_hubs_by_degree(G: nx.Graph, entity_nodes: dict, out_dir: Path):
    """Top 15 entities per type by degree (2×3 grid for shared types)."""
    types = ["statute", "provision", "precedent", "court", "judge", "lawyer"]
    type_colors = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
                   "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, ntype in zip(axes, types):
        nodes = entity_nodes.get(ntype, [])
        top = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[:15]
        if not top: ax.set_visible(False); continue
        labels = [G.nodes[n].get("label", n)[:40] for n in top]
        degrees = [G.degree(n) for n in top]
        ax.barh(labels[::-1], degrees[::-1],
                color=type_colors.get(ntype, "#aaa"), edgecolor="white", linewidth=0.3)
        ax.set_title(ntype.capitalize(), fontsize=10)
        ax.set_xlabel("Degree", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Top 15 entities by degree per shared-entity type", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir, "top_hubs_by_degree_per_type")


def plot_entity_bucket_span(sharing: dict, out_dir: Path):
    """Stacked bar: for each entity type, how many entities span 1 / 2 / 3+ buckets."""
    type_summary = sharing["bridge_type_summary"]
    types = [t for t in ENTITY_TYPES if t in type_summary]
    all_k = sorted({k for v in type_summary.values() for k in v.keys()})

    cmap = plt.cm.get_cmap("Blues", len(all_k)+2)
    colors = [cmap(i+2) for i in range(len(all_k))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(types))
    for ki, k in enumerate(all_k):
        vals = [type_summary[t].get(k, 0) for t in types]
        ax.bar(types, vals, bottom=bottom,
               label=f"{k} domain{'s' if k>1 else ''}",
               color=colors[ki], edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)
    ax.set_ylabel("Number of entity nodes")
    ax.set_title("Entity nodes by domain-span count per type")
    ax.set_xticklabels([t.capitalize() for t in types], rotation=20, ha="right")
    ax.legend(title="Domains spanned", fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "entity_bucket_span")


def plot_pagerank_vs_degree(G: nx.Graph, pr: dict, entity_nodes: dict, out_dir: Path):
    """Scatter: PageRank vs degree for shared entity nodes (coloured by type)."""
    type_colors = {"statute":"#F1C40F","provision":"#E67E22","precedent":"#C0392B",
                   "court":"#8E44AD","judge":"#2980B9","lawyer":"#7F8C8D"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for ntype, color in type_colors.items():
        nodes = entity_nodes.get(ntype, [])
        if not nodes: continue
        xs = [G.degree(n) for n in nodes]
        ys = [pr.get(n, 0) for n in nodes]
        ax.scatter(xs, ys, s=4, alpha=0.4, color=color,
                   label=ntype.capitalize(), rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("PageRank")
    ax.set_title("PageRank vs degree for shared entity nodes")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    savefig(fig, out_dir, "pagerank_vs_degree")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--out",           default="outputs/graph_stats")
    parser.add_argument("--with-betweenness", action="store_true",
                        help="Compute approximate betweenness (slow on large graphs)")
    parser.add_argument("--betweenness-k", type=int, default=300,
                        help="Number of pivot samples for approx betweenness")
    args = parser.parse_args()

    cfg, config_path = load_config(args.config)
    out_dir = resolve_output_arg(args.out, config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcolors = bucket_colors(cfg)

    artefact = Path(cfg.get("output_dir", "outputs"))

    # ── load ──────────────────────────────────────────────────────────────────
    tick("Loading graph_full.pkl")
    with open(artefact / "graph_full.pkl", "rb") as f:
        G: nx.Graph = pickle.load(f)
    print(f"    {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges")

    # ── index ─────────────────────────────────────────────────────────────────
    case_nodes, entity_nodes, degree_map = collect_node_index(G)

    # ── per-bucket ────────────────────────────────────────────────────────────
    bkt_stats = per_bucket_stats(G, case_nodes, entity_nodes, degree_map)

    # ── cross-bucket ──────────────────────────────────────────────────────────
    sharing = cross_bucket_sharing(G, case_nodes)

    # ── centrality ────────────────────────────────────────────────────────────
    pr = compute_pagerank(G)
    if args.with_betweenness:
        bc = compute_approx_betweenness(G, entity_nodes, k=args.betweenness_k)
    else:
        tick("Skipping approx betweenness (pass --with-betweenness to enable)")
        bc = {}

    # ── save summary JSON ────────────────────────────────────────────────────
    tick("Saving stats JSON")
    summary = {
        "graph": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "avg_degree":  round(2*G.number_of_edges()/G.number_of_nodes(), 3),
        },
        "per_bucket": bkt_stats,
        "cross_bucket": {
            "sharing_matrix": sharing["matrix"],
            "buckets_order":  sharing["buckets"],
            "bridge_type_summary": sharing["bridge_type_summary"],
            "top_bridges": sharing["bridges"][:50],
        },
        "top_pagerank_global":    top_nodes(G, pr,  None, 30),
        "top_pagerank_statute":   top_nodes(G, pr,  "statute", 20),
        "top_pagerank_precedent": top_nodes(G, pr,  "precedent", 20),
        "top_pagerank_judge":     top_nodes(G, pr,  "judge", 20),
        "top_betweenness":        top_nodes(G, bc,  None, 30) if bc else [],
        "top_degree_entities": {
            ntype: sorted(
                [{"node": n, "label": G.nodes[n].get("label","")[:60],
                  "degree": G.degree(n)} for n in nodes],
                key=lambda x: x["degree"], reverse=True
            )[:20]
            for ntype, nodes in entity_nodes.items()
        },
    }
    with open(out_dir / "graph_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    saved graph_stats.json")

    # ── plots ─────────────────────────────────────────────────────────────────
    tick("Generating plots")
    plot_bucket_entity_composition(bkt_stats, bcolors, out_dir)
    plot_case_degree_boxplot(bkt_stats, bcolors, out_dir)
    plot_degree_distribution_by_type(G, out_dir)
    plot_cross_bucket_matrix(sharing, bcolors, out_dir)
    plot_bridge_entity_distribution(sharing, out_dir)
    plot_top_hubs_by_degree(G, entity_nodes, out_dir)
    plot_entity_bucket_span(sharing, out_dir)
    plot_top_pagerank(G, pr, out_dir)
    plot_top_pagerank_by_type(G, pr, out_dir)
    plot_pagerank_vs_degree(G, pr, entity_nodes, out_dir)
    if bc:
        plot_top_betweenness(G, bc, out_dir)

    print(f"\nDone — outputs in {out_dir}/")


if __name__ == "__main__":
    main()
