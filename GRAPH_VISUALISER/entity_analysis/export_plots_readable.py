"""
Export readable entity analysis plots for thesis use.
Reduced top-N, larger nodes/fonts, more spacing — easier to read in print.

Run: python export_plots_readable.py
Outputs go to outputs/figures_readable/
"""

import json
import os
import glob
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

OUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs")
WITHIN_DIR = os.path.join(OUT_DIR, "within_bucket")
CROSS_DIR  = os.path.join(OUT_DIR, "cross_bucket")
FIG_DIR    = os.path.join(OUT_DIR, "figures_readable")
os.makedirs(FIG_DIR, exist_ok=True)

BUCKET_LABELS = {
    "family_matrimonial": "Family & Matrimonial",
    "land_property":      "Land & Property",
    "motor_accidents":    "Motor Accidents",
    "sexual_offences":    "Sexual Offences",
    "fin_fraud":          "Financial Fraud",
}

TYPE_COLOURS = {
    "STATUTE":   "#e63946",
    "PROVISION": "#f4a261",
    "JUDGE":     "#2a9d8f",
    "COURT":     "#457b9d",
    "LAWYER":    "#a8dadc",
    "PRECEDENT": "#8338ec",
    "GPE":       "#fb8500",
}

BUCKET_COLOURS = {
    "family_matrimonial": "#9B59B6",
    "land_property":      "#1ABC9C",
    "motor_accidents":    "#F39C12",
    "sexual_offences":    "#3498DB",
    "fin_fraud":          "#E74C3C",
    "unknown":            "#adb5bd",
}

# ── readability settings ───────────────────────────────────────────────────────
TOPN_WITHIN    = 25   # was 80  — fewer nodes, easier to label
TOPN_CROSS     = 40   # was 100
BAR_ENTRIES    = 15   # was 30  — shorter bars, bigger text per entry
SPRING_K       = 2.5  # was 1.5 — more spacing between nodes
NODE_SIZE_MIN  = 12   # was 8
NODE_SIZE_MAX  = 55   # was 40
NODE_FONT_SIZE = 11   # was 8/9
TITLE_FONT     = 20
AXIS_FONT      = 14

WIDTH  = 1600
HEIGHT = 1000
SCALE  = 2


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_network_figure(nodes, edges, colour_key="type", top_n=TOPN_WITHIN, title=""):
    if not nodes:
        return go.Figure().update_layout(title="No data")

    top_nodes = sorted(nodes, key=lambda n: n.get("pagerank", 0), reverse=True)[:top_n]
    node_set  = {n["entity"] for n in top_nodes}
    sub_edges = [e for e in edges if e["source"] in node_set and e["target"] in node_set]

    G = nx.Graph()
    for n in top_nodes:
        G.add_node(n["entity"])
    for e in sub_edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    try:
        pos = nx.spring_layout(G, weight="weight", seed=42, k=SPRING_K)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # scale edge width by relative weight for visual clarity
    weights = [e["weight"] for e in sub_edges if e["source"] in pos and e["target"] in pos]
    max_w   = max(weights) if weights else 1

    # draw edges grouped by weight tier (thin / medium / thick)
    def edge_tier(w):
        r = w / max_w
        if r > 0.6:  return (1.8, "#888888")
        if r > 0.2:  return (1.0, "#666666")
        return (0.4, "#444444")

    edge_traces = []
    tiers = {}
    for e in sub_edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        width, colour = edge_tier(e["weight"])
        tiers.setdefault((width, colour), []).append(e)

    for (w, col), elist in tiers.items():
        ex, ey = [], []
        for e in elist:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            ex += [x0, x1, None]; ey += [y0, y1, None]
        edge_traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=w, color=col),
            hoverinfo="none", showlegend=False,
        ))

    # node traces per colour group
    node_traces = []
    groups = {}
    for n in top_nodes:
        key   = n.get("type" if colour_key == "type" else "buckets", ["unknown"])
        group = key if isinstance(key, str) else (key[0] if key else "unknown")
        groups.setdefault(group, []).append(n)

    for group, gnodes in groups.items():
        colour = (TYPE_COLOURS if colour_key == "type" else BUCKET_COLOURS).get(group, "#999")
        xs, ys, texts, hovers, sizes = [], [], [], [], []
        for n in gnodes:
            if n["entity"] not in pos:
                continue
            x, y = pos[n["entity"]]
            xs.append(x); ys.append(y)
            # truncate long labels for readability
            label = n["text"]
            label = label[:22] + "…" if len(label) > 22 else label
            texts.append(label)
            pr  = n.get("pagerank", 0)
            win = n.get("outcome_win_rate")
            win_str = f"{win:.0%}" if win is not None else "n/a"
            hovers.append(
                f"<b>{n['text']}</b><br>Type: {n['type']}<br>"
                f"Frequency: {n.get('frequency', 0)}<br>"
                f"PageRank: {pr:.4f}<br>Win rate: {win_str}<br>"
                f"Buckets: {', '.join(n.get('buckets', []))}"
            )
            sizes.append(max(NODE_SIZE_MIN, min(NODE_SIZE_MAX, pr * 4000)))

        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=group,
            marker=dict(
                size=sizes, color=colour,
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.92,
            ),
            text=texts, textposition="top center",
            textfont=dict(size=NODE_FONT_SIZE, color="#f0f0f0"),
            hovertext=hovers, hoverinfo="text",
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_FONT), x=0.5),
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#141428",
        font=dict(color="#e0e0e0", size=AXIS_FONT),
        legend=dict(
            bgcolor="#1c1c3a", bordercolor="#555",
            font=dict(size=13), borderwidth=1,
            x=1.01, y=0.98,
        ),
        margin=dict(l=20, r=160, t=70, b=20),
        width=WIDTH, height=HEIGHT,
    )
    return fig


def make_bar_figure(entries, metric, title):
    if not entries:
        return go.Figure().update_layout(title="No data")
    df = pd.DataFrame(entries[:BAR_ENTRIES])
    # shorten entity text for bar labels
    df["text"] = df["text"].apply(lambda t: t[:35] + "…" if len(t) > 35 else t)
    fig = px.bar(
        df, x="score", y="text", orientation="h",
        color="type", color_discrete_map=TYPE_COLOURS,
        title=title,
        labels={"score": metric, "text": "Entity"},
    )
    fig.update_layout(
        plot_bgcolor="#0d0d0d", paper_bgcolor="#141428",
        font=dict(color="#e0e0e0", size=AXIS_FONT),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        xaxis=dict(tickfont=dict(size=12)),
        title=dict(font=dict(size=TITLE_FONT), x=0.5),
        legend=dict(font=dict(size=13), bgcolor="#1c1c3a", bordercolor="#555"),
        width=WIDTH, height=HEIGHT,
        margin=dict(l=20, r=20, t=70, b=40),
        bargap=0.25,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def save(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.write_image(path, width=WIDTH, height=HEIGHT, scale=SCALE)
    print(f"  saved → {path}")


def export_within_bucket():
    print("\n── Within-bucket figures (readable) ──")
    files = glob.glob(os.path.join(WITHIN_DIR, "*.json"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]

    for path in files:
        bucket = os.path.basename(path).replace(".json", "")
        label  = BUCKET_LABELS.get(bucket, bucket)
        data   = load_json(path)
        if not data:
            continue
        print(f"\n  {label}")

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        fig = make_network_figure(nodes, edges, colour_key="type", top_n=TOPN_WITHIN,
                                  title=f"Entity Co-occurrence Network — {label}  (top {TOPN_WITHIN})")
        save(fig, f"within_{bucket}_network")

        for metric_key, metric_label in [
            ("top_by_pagerank",    "PageRank"),
            ("top_by_strength",    "Co-occurrence Strength"),
            ("top_by_betweenness", "Betweenness Centrality"),
        ]:
            entries = data.get(metric_key, [])
            if not entries:
                continue
            fig = make_bar_figure(entries, metric_label,
                                  f"Top {BAR_ENTRIES} Entities by {metric_label} — {label}")
            save(fig, f"within_{bucket}_{metric_key}")


def export_cross_bucket():
    print("\n── Cross-bucket figures (readable) ──")
    data = load_json(os.path.join(CROSS_DIR, "cross_bucket_analysis.json"))
    if not data:
        print("  No cross-bucket data — skipping")
        return

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    fig = make_network_figure(nodes, edges, colour_key="type", top_n=TOPN_CROSS,
                              title=f"Cross-Domain Entity Network (by Type, top {TOPN_CROSS})")
    save(fig, "cross_bucket_network_by_type")

    fig = make_network_figure(nodes, edges, colour_key="bucket", top_n=TOPN_CROSS,
                              title=f"Cross-Domain Entity Network (by Domain, top {TOPN_CROSS})")
    save(fig, "cross_bucket_network_by_bucket")

    for metric_key, metric_label in [
        ("top_by_pagerank",    "PageRank"),
        ("top_by_strength",    "Co-occurrence Strength"),
        ("top_by_betweenness", "Betweenness Centrality"),
        ("top_by_bridge",      "Bridge Score (# Domains)"),
    ]:
        entries = data.get(metric_key, [])
        if not entries:
            continue
        fig = make_bar_figure(entries, metric_label,
                              f"Top {BAR_ENTRIES} Cross-Domain Entities by {metric_label}")
        save(fig, f"cross_bucket_{metric_key}")

    by_type = load_json(os.path.join(CROSS_DIR, "by_entity_type.json"))
    if by_type:
        print("\n  Per-type breakdown figures:")
        for etype, ents in by_type.items():
            entries = [{"text": e["text"], "type": e["type"], "score": e["pagerank"]}
                       for e in ents[:BAR_ENTRIES]]
            fig = make_bar_figure(entries, "PageRank",
                                  f"Top {BAR_ENTRIES} {etype} Entities — All Domains")
            save(fig, f"cross_bucket_type_{etype.lower()}_pagerank")


if __name__ == "__main__":
    try:
        import kaleido  # noqa
    except ImportError:
        print("ERROR: kaleido not installed.  pip install 'kaleido==0.2.1'")
        raise SystemExit(1)

    export_within_bucket()
    export_cross_bucket()
    print(f"\nAll readable figures saved to: {FIG_DIR}")
