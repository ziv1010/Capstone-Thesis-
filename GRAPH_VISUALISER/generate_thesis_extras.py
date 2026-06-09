#!/usr/bin/env python3
"""
generate_thesis_extras.py — Additional thesis-ready figures for Section 5.

Reads:
  entity_analysis/outputs/within_bucket/*.json  (new v3 data)
  outputs/graph_stats/graph_stats.json           (full-graph stats)
  outputs/stats.json                             (hub/bridge stats)
  outputs/graph_full.pkl                         (full graph — for CCDF)

Writes to: outputs/thesis_extras/
"""
from __future__ import annotations

import json
import pickle
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ROOT     = Path(__file__).resolve().parent
WITHIN   = ROOT / "entity_analysis" / "outputs" / "within_bucket"
STATS_J  = ROOT / "outputs" / "graph_stats" / "graph_stats.json"
HUBS_J   = ROOT / "outputs" / "stats.json"
FULL_PKL = ROOT / "outputs" / "graph_full.pkl"
OUT      = ROOT / "outputs" / "thesis_extras"
OUT.mkdir(parents=True, exist_ok=True)

BUCKET_DISPLAY = {
    "financial_fraud":    "Financial Fraud",
    "family_matrimonial": "Family/Matrimonial",
    "land_property":      "Land/Property",
    "motor_accidents":    "Motor Accidents",
    "sexual_offences":    "Sexual Offences",
    "fin_fraud":          "Financial Fraud",
}
BUCKET_COLORS = {
    "financial_fraud":    "#E74C3C",
    "family_matrimonial": "#9B59B6",
    "land_property":      "#1ABC9C",
    "motor_accidents":    "#F39C12",
    "sexual_offences":    "#3498DB",
    "fin_fraud":          "#E74C3C",
}
TYPE_COLORS = {
    "STATUTE":   "#e63946",
    "PROVISION": "#f4a261",
    "JUDGE":     "#2a9d8f",
    "COURT":     "#457b9d",
    "LAWYER":    "#a8dadc",
    "PRECEDENT": "#8338ec",
    "GPE":       "#fb8500",
}

def savefig(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.35)
    print(f"  saved {stem}")
    plt.close(fig)


def _wrap_network_label(text: str, line_width: int = 18, max_chars: int = 48) -> str:
    """Wrap graph labels so larger fonts remain readable without running off-canvas."""
    label = " ".join(str(text).split())
    if len(label) > max_chars:
        label = textwrap.shorten(label, width=max_chars, placeholder="...")
    return "\n".join(textwrap.wrap(label, width=line_width, break_long_words=False)) or label


# ─────────────────────────────────────────────────────────────────────────────
# 1. WITHIN-BUCKET NETWORK FINGERPRINTS — composite 2×2
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_label_overlaps(label_positions: dict, pos: dict, ax,
                             iterations: int = 40, push_strength: float = 0.04) -> dict:
    """Iteratively push label offsets apart when bounding boxes overlap."""
    keys = list(label_positions.keys())
    offsets = {k: list(label_positions[k]) for k in keys}

    # approximate bbox half-size in data coords (rough but effective)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]; yspan = ylim[1] - ylim[0]
    bw = xspan * 0.13
    bh = yspan * 0.09

    for _ in range(iterations):
        moved = False
        for i, ki in enumerate(keys):
            xi, yi = pos[ki][0] + offsets[ki][0], pos[ki][1] + offsets[ki][1]
            fx, fy = 0.0, 0.0
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                xj, yj = pos[kj][0] + offsets[kj][0], pos[kj][1] + offsets[kj][1]
                dx, dy = xi - xj, yi - yj
                if abs(dx) < bw and abs(dy) < bh:
                    # push away
                    fx += push_strength * (bw - abs(dx)) * (1 if dx >= 0 else -1)
                    fy += push_strength * (bh - abs(dy)) * (1 if dy >= 0 else -1)
                    moved = True
            offsets[ki][0] += fx
            offsets[ki][1] += fy
        if not moved:
            break
    return offsets


def within_bucket_network_fingerprints(top_n: int = 12, max_edges: int = 28) -> None:
    """
    Show only the top-`top_n` nodes and the `max_edges` strongest co-occurrence
    edges per bucket.  Fewer nodes + pruned edges + kamada-kawai layout = readable.
    """
    buckets = ["family_matrimonial", "land_property", "motor_accidents", "sexual_offences"]
    titles  = [BUCKET_DISPLAY[b] for b in buckets]
    panel_bg = ["#f9f0ff", "#f0fff4", "#fff8f0", "#f0f6ff"]   # subtle tint per panel

    fig = plt.figure(figsize=(30, 22))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.20)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.88, bottom=0.085)

    # collect which TYPE_COLORS actually appear so the legend stays accurate
    seen_types: set[str] = set()

    for idx, bucket in enumerate(buckets):
        path = WITHIN / f"{bucket}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        top_nodes = sorted(nodes, key=lambda n: n.get("pagerank", 0), reverse=True)[:top_n]
        node_set  = {n["entity"] for n in top_nodes}

        # keep only edges between top nodes, sorted by weight descending
        candidate_edges = sorted(
            [e for e in edges if e["source"] in node_set and e["target"] in node_set],
            key=lambda e: e["weight"], reverse=True,
        )
        # guarantee each node gets at least its strongest edge, then fill to max_edges
        included, node_covered = [], set()
        for e in candidate_edges:
            included.append(e)
            node_covered.update([e["source"], e["target"]])
            if len(node_covered) == len(node_set):
                break
        seen_pairs = {(e["source"], e["target"]) for e in included}
        for e in candidate_edges:
            if len(included) >= max_edges:
                break
            pair = (e["source"], e["target"])
            rpair = (e["target"], e["source"])
            if pair not in seen_pairs and rpair not in seen_pairs:
                included.append(e)
                seen_pairs.add(pair)
        sub_edges = included

        G_full = nx.Graph()
        for n in top_nodes:
            G_full.add_node(n["entity"], **n)
        for e in sub_edges:
            G_full.add_edge(e["source"], e["target"], weight=e["weight"],
                            dist=1.0 / e["weight"])

        # 2-core: keep only nodes with ≥2 connections — removes peripheral satellites
        core = nx.k_core(G_full, k=2)
        # fall back to 1-core if 2-core is too small
        if len(core) < 5:
            core = G_full.subgraph(max(nx.connected_components(G_full), key=len)).copy()
        G = core

        node_info = {entity: G_full.nodes[entity] for entity in G.nodes()}
        for entity in G.nodes():
            seen_types.add(G_full.nodes[entity].get("type", "?"))

        # kamada_kawai with 'dist': high co-occurrence weight → short distance → close placement
        try:
            pos = nx.kamada_kawai_layout(G, weight="dist")
        except Exception:
            pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5, iterations=300)

        # normalise positions to [-1, 1] so padding is predictable
        xs_raw = np.array([v[0] for v in pos.values()])
        ys_raw = np.array([v[1] for v in pos.values()])
        xr = max(xs_raw.max() - xs_raw.min(), 1e-6)
        yr = max(ys_raw.max() - ys_raw.min(), 1e-6)
        pos = {e: ((v[0] - xs_raw.mean()) / xr * 2,
                   (v[1] - ys_raw.mean()) / yr * 2) for e, v in pos.items()}

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor(panel_bg[idx])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # generous fixed limits — labels are placed in offset-points so they
        # can go slightly outside the data range; clip_on=False keeps them visible
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")

        # --- edges: width and opacity scaled by normalised weight ---
        g_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
        if g_edges:
            weights = [w for _, _, w in g_edges]
            max_w, min_w = max(weights), min(weights)
            span = max(max_w - min_w, 1)
            for u, v, w in g_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                t     = (w - min_w) / span
                lw    = 1.0 + 4.0 * t
                alpha = 0.28 + 0.62 * t
                color = plt.cm.Greys(0.35 + 0.50 * t)
                ax.plot([x0, x1], [y0, y1], color=color,
                        linewidth=lw, zorder=1, alpha=alpha, solid_capstyle="round")

        # --- nodes: size ∝ PageRank, colour by entity type ---
        pr_vals = [node_info[e].get("pagerank", 0) for e in G.nodes()]
        pr_min, pr_max = min(pr_vals), max(pr_vals)
        pr_span = max(pr_max - pr_min, 1e-9)

        for entity, (x, y) in pos.items():
            n     = node_info[entity]
            ntype = n.get("type", "?")
            color = TYPE_COLORS.get(ntype, "#999")
            t_pr  = (n.get("pagerank", 0) - pr_min) / pr_span
            size  = 350 + 1200 * t_pr
            ax.scatter(x, y, s=size, c=color, zorder=3,
                       edgecolors="white", linewidths=1.8, alpha=0.95,
                       clip_on=False)

        # --- labels: radiate from centroid, then iteratively de-overlap ---
        xs = np.array([v[0] for v in pos.values()])
        ys = np.array([v[1] for v in pos.values()])
        cx, cy = xs.mean(), ys.mean()

        init_offsets: dict = {}
        for entity, (x, y) in pos.items():
            dx, dy = x - cx, y - cy
            dist   = (dx**2 + dy**2) ** 0.5 or 1.0
            init_offsets[entity] = [20 * dx / dist, 20 * dy / dist]

        final_offsets = _resolve_label_overlaps(init_offsets, pos, ax,
                                                iterations=60, push_strength=0.06)

        for entity, (x, y) in pos.items():
            n     = node_info[entity]
            label = _wrap_network_label(n["text"], line_width=18, max_chars=42)
            ox, oy = final_offsets[entity]
            ha = "left" if ox >= 0 else "right"
            va = "bottom" if oy >= 0 else "top"
            # Keep enlarged labels away from panel titles and image edges.
            if y > 0.78:
                oy = -abs(oy)
                va = "top"
            elif y < -0.78:
                oy = abs(oy)
                va = "bottom"
            if x > 0.90:
                ox = -abs(ox)
                ha = "right"
            elif x < -0.90:
                ox = abs(ox)
                ha = "left"
            ax.annotate(
                label, (x, y),
                fontsize=18, ha=ha, va=va,
                xytext=(ox, oy), textcoords="offset points",
                color="#111111", fontweight="semibold",
                annotation_clip=False,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
            )

        ax.set_title(titles[idx], fontsize=23, fontweight="bold", pad=12)

    # --- shared legend (only types that actually appear) ---
    patches = [mpatches.Patch(color=c, label=t.capitalize())
               for t, c in TYPE_COLORS.items() if t in seen_types]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Within-bucket entity co-occurrence networks\n"
        f"(2-core subgraph of top-{top_n} entities by PageRank  ·  edge weight = co-occurrence frequency  ·  node size ∝ PageRank)",
        fontsize=21, y=0.965,
    )
    savefig(fig, "within_bucket_network_fingerprints")


# ─────────────────────────────────────────────────────────────────────────────
# 1b. CROSS-DOMAIN NETWORK FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

def cross_domain_network_fingerprint(top_n: int = 14, exclusive_labels_per_domain: int = 2) -> None:
    """
    Single-panel network showing all five domain hubs with their top-`top_n`
    entities. Shared entities (appearing in ≥2 domains) are drawn as bridge
    nodes in the centre; domain-exclusive entities cluster around their hub.
    Domain hub labels are placed OUTSIDE the hexagon to guarantee readability.
    """
    import math

    ALL_BUCKETS = ["family_matrimonial", "fin_fraud", "land_property",
                   "motor_accidents", "sexual_offences"]

    # ── collect top-N per bucket ─────────────────────────────────────────────
    bucket_top: dict[str, list[dict]] = {}
    for b in ALL_BUCKETS:
        path = WITHIN / f"{b}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        bucket_top[b] = sorted(
            data.get("nodes", []), key=lambda n: n.get("pagerank", 0), reverse=True
        )[:top_n]

    # ── classify entities: shared vs. exclusive ──────────────────────────────
    entity_buckets: dict[str, list[str]] = defaultdict(list)
    entity_info:    dict[str, dict]      = {}
    for b, nodes in bucket_top.items():
        for n in nodes:
            eid = n["entity"]
            entity_buckets[eid].append(b)
            if eid not in entity_info:
                entity_info[eid] = n

    # ── build graph ──────────────────────────────────────────────────────────
    G = nx.Graph()
    for b in bucket_top:
        G.add_node(f"__domain__{b}", node_type="domain", bucket=b,
                   label=BUCKET_DISPLAY[b])
    for eid, buckets_hit in entity_buckets.items():
        n = entity_info[eid]
        G.add_node(eid, node_type="entity", **n, n_domains=len(buckets_hit))
        for b in buckets_hit:
            G.add_edge(f"__domain__{b}", eid, shared=(len(buckets_hit) > 1))

    # ── layout: pentagon for hubs, spring for entities ────────────────────────
    # Pentagon radius = 1.5; entities spring-settle between hubs
    n_domains = len(bucket_top)
    domain_nodes = [f"__domain__{b}" for b in bucket_top]
    fixed_pos: dict = {}
    for i, dn in enumerate(domain_nodes):
        angle = 2 * math.pi * i / n_domains - math.pi / 2
        fixed_pos[dn] = (1.5 * math.cos(angle), 1.5 * math.sin(angle))

    pos = nx.spring_layout(
        G, pos=fixed_pos, fixed=list(fixed_pos.keys()),
        seed=7, k=0.60, iterations=600,
    )

    # ── figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(25, 21))
    ax.set_facecolor("#f7f9fc")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Fixed data-space limits with generous margin so hub labels and leaves never clip
    ax.set_xlim(-3.75, 3.75)
    ax.set_ylim(-3.55, 3.55)
    ax.axis("off")

    # ── edges ────────────────────────────────────────────────────────────────
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if d.get("shared"):
            ax.plot([x0, x1], [y0, y1], color="#e63946",
                    linewidth=2.0, alpha=0.60, zorder=1, clip_on=False)
        else:
            b = G.nodes[v].get("bucket") or G.nodes[u].get("bucket")
            col = BUCKET_COLORS.get(b, "#aaa")
            ax.plot([x0, x1], [y0, y1], color=col,
                    linewidth=1.1, alpha=0.28, zorder=1, clip_on=False)

    # ── domain hub nodes: hexagon marker + EXTERNAL label ────────────────────
    # Label is offset radially outward from the figure centre (0, 0)
    for dn in domain_nodes:
        b   = G.nodes[dn]["bucket"]
        x, y = pos[dn]
        ax.scatter(x, y, s=3200, c=BUCKET_COLORS[b], marker="h",
                   zorder=4, edgecolors="white", linewidths=3.0, alpha=0.95,
                   clip_on=False)

        # radial direction outward from origin → push label beyond the node
        rlen = (x**2 + y**2) ** 0.5 or 1.0
        # offset in data units: 0.38 pushes the label just outside the hexagon
        lx = x + 0.42 * x / rlen
        ly = y + 0.42 * y / rlen
        # horizontal alignment: left for nodes on right half, right for left half
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"
        ax.text(lx, ly, BUCKET_DISPLAY[b].replace(" ", "\n"),
                ha=ha, va=va, fontsize=18, fontweight="bold",
                color=BUCKET_COLORS[b], zorder=6, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=BUCKET_COLORS[b], linewidth=1.2, alpha=0.92),
                clip_on=False)

    # ── entity nodes ──────────────────────────────────────────────────────────
    seen_types: set[str] = set()
    for eid in entity_info:
        if eid not in pos:
            continue
        n     = entity_info[eid]
        ntype = n.get("type", "?")
        seen_types.add(ntype)
        n_dom = G.nodes[eid].get("n_domains", 1)
        color = TYPE_COLORS.get(ntype, "#999")
        size  = 220 + 420 * (n_dom - 1)
        lw    = 2.5 if n_dom > 1 else 0.8
        ax.scatter(*pos[eid], s=size, c=color, zorder=3,
                   edgecolors="#222" if n_dom > 1 else "white",
                   linewidths=lw, alpha=0.93, clip_on=False)

    # ── entity labels: shared entities + top-1 exclusive per domain ───────────
    xs_all = np.array([pos[nd][0] for nd in G.nodes()])
    ys_all = np.array([pos[nd][1] for nd in G.nodes()])
    cx, cy = xs_all.mean(), ys_all.mean()

    # pre-compute top exclusive nodes per domain (for label selection)
    top_exclusive_per_domain: dict[str, set[str]] = {}
    for b in bucket_top:
        candidates = [e for e, bs in entity_buckets.items() if bs == [b] and e in entity_info]
        selected = sorted(
            candidates, key=lambda e: entity_info[e].get("pagerank", 0), reverse=True
        )[:exclusive_labels_per_domain]
        top_exclusive_per_domain[b] = set(selected)

    for eid, n in entity_info.items():
        if eid not in pos:
            continue
        n_dom = G.nodes[eid].get("n_domains", 1)
        if n_dom == 1:
            b = entity_buckets[eid][0]
            if eid not in top_exclusive_per_domain.get(b, set()):
                continue

        label = _wrap_network_label(n["text"], line_width=19, max_chars=46)
        x, y  = pos[eid]
        dx, dy = x - cx, y - cy
        dist   = (dx**2 + dy**2) ** 0.5 or 1.0
        ox = 16 * dx / dist
        oy = 16 * dy / dist
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        weight = "bold" if n_dom > 1 else "normal"
        ax.annotate(
            label, (x, y),
            fontsize=16, ha=ha, va=va, fontweight=weight,
            xytext=(ox, oy), textcoords="offset points",
            color="#111111", annotation_clip=False,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
        )

    # ── legend ────────────────────────────────────────────────────────────────
    type_patches = [mpatches.Patch(color=c, label=t.capitalize())
                    for t, c in TYPE_COLORS.items() if t in seen_types]
    shared_line  = plt.Line2D([0], [0], color="#e63946", linewidth=2.5,
                               label="Cross-domain edge (shared entity)")
    excl_line    = plt.Line2D([0], [0], color="#aaa", linewidth=1.5,
                               label="Within-domain edge")
    ax.legend(handles=type_patches + [shared_line, excl_line],
              loc="lower center", ncol=4, fontsize=16, frameon=False,
              bbox_to_anchor=(0.5, -0.15))

    ax.set_title(
        "Cross-domain entity fingerprint\n"
        f"(top-{top_n} PageRank entities per domain  ·  shared entities in bold  ·  node size ∝ domain count)",
        fontsize=22, fontweight="bold", pad=22,
    )
    savefig(fig, "cross_domain_network_fingerprint")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEGREE CCDF LOG-LOG (power-law diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def degree_ccdf_loglog() -> None:
    print("  Loading graph_full.pkl for CCDF …")
    with open(FULL_PKL, "rb") as f:
        G: nx.Graph = pickle.load(f)

    all_degrees  = np.array([d for _, d in G.degree()])
    case_degrees = np.array([G.degree(n) for n, a in G.nodes(data=True)
                             if a.get("node_type") == "case"])
    entity_degrees = np.array([G.degree(n) for n, a in G.nodes(data=True)
                                if a.get("node_type") not in ("case",)])

    fig, ax = plt.subplots(figsize=(7, 5))
    for degrees, label, color in [
        (all_degrees,    "All nodes",      "#4f86c6"),
        (case_degrees,   "Case nodes",     "#e67e22"),
        (entity_degrees, "Entity nodes",   "#27ae60"),
    ]:
        sorted_d = np.sort(degrees)[::-1]
        ccdf     = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.loglog(sorted_d, ccdf, ".", markersize=2, alpha=0.5, color=color, label=label)

    ax.set_xlabel("Degree (log)")
    ax.set_ylabel("P(D ≥ d) — CCDF (log)")
    ax.set_title("Complementary CDF of node degree\n(log–log scale, power-law tail visible)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "degree_ccdf_loglog")
    del G


# ─────────────────────────────────────────────────────────────────────────────
# 3. PER-BUCKET ENTITY RICHNESS RADAR / TABLE BAR
# ─────────────────────────────────────────────────────────────────────────────

def per_bucket_entity_richness() -> None:
    with open(STATS_J) as f:
        stats = json.load(f)
    bkt_stats = stats.get("per_bucket", {})

    RENAME = {"fin_fraud": "financial_fraud"}
    bkt_stats = {RENAME.get(k, k): v for k, v in bkt_stats.items()}

    ordered = [b for b in BUCKET_DISPLAY if b in bkt_stats and b != "fin_fraud"]
    ENTITY_TYPES = ["statute", "provision", "precedent", "court", "judge", "lawyer"]
    ec = {"statute": "#F1C40F", "provision": "#E67E22", "precedent": "#8338ec",
          "court": "#8E44AD", "judge": "#2980B9", "lawyer": "#7F8C8D"}

    # entities per case
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: unique shared entities per bucket (grouped bar)
    x = np.arange(len(ordered))
    n = len(ENTITY_TYPES)
    width = 0.12
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    ax = axes[0]
    for i, etype in enumerate(ENTITY_TYPES):
        vals = [bkt_stats[b]["entity_type_unique"].get(etype, 0) for b in ordered]
        ax.bar(x + offsets[i], vals, width, label=etype.capitalize(),
               color=ec.get(etype, "#aaa"), edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=22, ha="right")
    ax.set_ylabel("Unique entity nodes")
    ax.set_title("Unique shared-entity nodes per type and domain\n(full graph)")
    ax.legend(fontsize=8)

    # right: mean case degree per bucket
    ax = axes[1]
    means = [bkt_stats[b]["case_degree"]["mean"] for b in ordered]
    medians = [bkt_stats[b]["case_degree"]["median"] for b in ordered]
    colors = [BUCKET_COLORS.get(b, "#888") for b in ordered]
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4, label="Mean")
    ax.plot(x, medians, "D", color="#333", zorder=5, label="Median", markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=22, ha="right")
    ax.set_ylabel("Case node degree")
    ax.set_title("Mean/median case-node degree per domain\n(full graph)")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    savefig(fig, "per_bucket_richness_and_degree")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONNECTIVITY SCORE VIOLIN PLOT (richer than box)
# ─────────────────────────────────────────────────────────────────────────────

def connectivity_violin() -> None:
    print("  Loading graph_sample.pkl for connectivity violin …")
    sample_pkl = ROOT / "outputs" / "graph_sample.pkl"
    with open(sample_pkl, "rb") as f:
        G: nx.Graph = pickle.load(f)

    bucket_scores: dict[str, list] = defaultdict(list)
    for _, d in G.nodes(data=True):
        if d.get("node_type") == "case" and "connectivity_score" in d:
            bucket_scores[d["bucket"]].append(d["connectivity_score"])

    ordered = [b for b in BUCKET_DISPLAY if b in bucket_scores]
    if not ordered:
        print("  (no connectivity scores — skipping)")
        return

    data   = [bucket_scores[b] for b in ordered]
    labels = [BUCKET_DISPLAY[b] for b in ordered]
    colors = [BUCKET_COLORS.get(b, "#888") for b in ordered]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(len(ordered)),
                          showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Connectivity score")
    ax.set_title("Case connectivity score distribution per domain\n(violin = full distribution, white line = median)")
    fig.tight_layout()
    savefig(fig, "connectivity_violin")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP HUBS BY DEGREE — TABLE-STYLE HORIZONTAL BAR
# ─────────────────────────────────────────────────────────────────────────────

def top_hubs_annotated() -> None:
    with open(HUBS_J) as f:
        hubs_data = json.load(f)

    hubs = [h for h in hubs_data.get("top_hubs", []) if h["type"] != "case"][:20]
    if not hubs:
        print("  (no hubs — skipping top_hubs_annotated)")
        return

    type_colors = {
        "statute":   "#e67e22", "provision": "#f39c12", "precedent": "#8e44ad",
        "court":     "#27ae60", "judge":     "#2980b9", "lawyer":    "#7F8C8D",
    }
    labels       = [f"{h['label'][:40]}  [{h['type']}]" for h in hubs]
    degrees      = [h["degree"] for h in hubs]
    bucket_cnts  = [h.get("bucket_count", 0) for h in hubs]
    colors       = [type_colors.get(h["type"], "#888") for h in hubs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # left: degree
    ax = axes[0]
    bars = ax.barh(labels[::-1], degrees[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, degrees[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Degree (connections to cases)")
    ax.set_title("Top hub entities by degree")

    # right: domains bridged
    ax = axes[1]
    ax.barh(labels[::-1], bucket_cnts[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of domains bridged")
    ax.set_title("Domains spanned by each hub")
    ax.set_xlim(0, 6)
    ax.set_xticks(range(6))

    # shared legend
    seen = {}
    patches = []
    for h, c in zip(hubs, colors):
        t = h["type"]
        if t not in seen:
            seen[t] = True
            patches.append(mpatches.Patch(color=c, label=t.capitalize()))
    fig.legend(handles=patches, loc="lower center", ncol=len(seen), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top 20 hub entities — degree and domain bridging", fontsize=13)
    fig.tight_layout()
    savefig(fig, "top_hubs_degree_and_bridging")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHARED vs LOCAL NODE RATIO PER BUCKET
# ─────────────────────────────────────────────────────────────────────────────

def shared_vs_local_ratio() -> None:
    with open(STATS_J) as f:
        stats = json.load(f)

    bkt_stats = stats.get("per_bucket", {})
    RENAME = {"fin_fraud": "financial_fraud"}
    bkt_stats = {RENAME.get(k, k): v for k, v in bkt_stats.items()}
    ordered = [b for b in BUCKET_DISPLAY if b in bkt_stats and b != "fin_fraud"]

    SHARED_TYPES = ["statute", "provision", "precedent", "court", "judge", "lawyer"]
    LOCAL_TYPES  = ["petitioner", "respondent"]

    shared_totals = []
    local_totals  = []
    for b in ordered:
        eu = bkt_stats[b]["entity_type_unique"]
        shared_totals.append(sum(eu.get(t, 0) for t in SHARED_TYPES))
        local_totals.append(sum(eu.get(t, 0) for t in LOCAL_TYPES))

    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, shared_totals, label="Shared entity nodes", color="#4f86c6", edgecolor="white")
    ax.bar(x, local_totals, bottom=shared_totals, label="Local (party) nodes",
           color="#e67e22", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=22, ha="right")
    ax.set_ylabel("Unique node count")
    ax.set_title("Shared-entity vs. local-party node split per domain\n"
                 "(shared nodes carry cross-case edges; local nodes do not)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "shared_vs_local_nodes_per_bucket")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PAGERANK vs DEGREE scatter — full-graph (from graph_stats.json top lists)
# ─────────────────────────────────────────────────────────────────────────────

def pagerank_degree_scatter_from_stats() -> None:
    """Quick scatter using the top-30 PageRank list from graph_stats.json."""
    with open(STATS_J) as f:
        stats = json.load(f)

    entries = stats.get("top_pagerank_global", [])
    if not entries:
        return

    type_colors = {
        "statute": "#F1C40F", "provision": "#E67E22", "precedent": "#8338ec",
        "court": "#8E44AD", "judge": "#2980B9", "lawyer": "#7F8C8D",
        "case": "#4f86c6",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    by_type: dict[str, list] = defaultdict(list)
    for e in entries:
        by_type[e["node_type"]].append(e)

    for ntype, nodes in by_type.items():
        xs = [n["degree"] for n in nodes]
        ys = [n["score"]  for n in nodes]
        ax.scatter(xs, ys, s=80, label=ntype.capitalize(),
                   color=type_colors.get(ntype, "#888"), alpha=0.85,
                   edgecolors="white", linewidths=0.5)
        for n in nodes:
            ax.annotate(n["label"][:20], (n["degree"], n["score"]),
                        fontsize=6.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Degree")
    ax.set_ylabel("PageRank score")
    ax.set_title("PageRank vs. degree — top-30 global nodes\n(labeled, coloured by entity type)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "pagerank_vs_degree_top30_labeled")


# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTCOME SKEW COMPARISON — win/loss rate per bucket (full dataset)
# ─────────────────────────────────────────────────────────────────────────────

def outcome_skew_from_graph_stats() -> None:
    with open(STATS_J) as f:
        stats = json.load(f)
    bkt_stats = stats.get("per_bucket", {})
    RENAME = {"fin_fraud": "financial_fraud"}
    bkt_stats = {RENAME.get(k, k): v for k, v in bkt_stats.items()}
    ordered = [b for b in BUCKET_DISPLAY if b in bkt_stats and b != "fin_fraud"]

    wins, losses, unknowns = [], [], []
    for b in ordered:
        oc = bkt_stats[b].get("outcomes", {})
        w = oc.get("win", 0)
        l = oc.get("loss", 0)
        u = oc.get("unknown", 0)
        wins.append(w); losses.append(l); unknowns.append(u)

    # also compute win-rate
    win_rates = [w / max(w + l, 1) * 100 for w, l in zip(wins, losses)]

    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x, wins,    label="Win",     color="#27ae60")
    ax.bar(x, losses,  bottom=wins,     label="Loss",    color="#e74c3c")
    ax.bar(x, unknowns, bottom=[w+l for w, l in zip(wins, losses)],
           label="Unknown", color="#95a5a6")
    ax.set_xticks(x); ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=22, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title("Outcome distribution per domain\n(graph subset — matched cases)")
    ax.legend()

    ax = axes[1]
    colors = [BUCKET_COLORS.get(b, "#888") for b in ordered]
    bars = ax.bar(x, win_rates, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, v in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([BUCKET_DISPLAY[b] for b in ordered], rotation=22, ha="right")
    ax.set_ylabel("Win rate %")
    ax.set_ylim(0, 100)
    ax.set_title("Win rate among labelled cases per domain\n(graph subset)")
    fig.tight_layout()
    savefig(fig, "outcome_skew_per_bucket")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=== Thesis extras ===\n")

    print("[1/8] within_bucket_network_fingerprints …")
    within_bucket_network_fingerprints()

    print("[1b/8] cross_domain_network_fingerprint …")
    cross_domain_network_fingerprint()

    print("[2/8] degree_ccdf_loglog …")
    degree_ccdf_loglog()

    print("[3/8] per_bucket_entity_richness …")
    per_bucket_entity_richness()

    print("[4/8] connectivity_violin …")
    connectivity_violin()

    print("[5/8] top_hubs_annotated …")
    top_hubs_annotated()

    print("[6/8] shared_vs_local_ratio …")
    shared_vs_local_ratio()

    print("[7/8] pagerank_degree_scatter_from_stats …")
    pagerank_degree_scatter_from_stats()

    print("[8/8] outcome_skew_from_graph_stats …")
    outcome_skew_from_graph_stats()

    print(f"\nDone — all extras saved to {OUT}/")
