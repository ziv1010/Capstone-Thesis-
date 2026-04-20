"""
Export entity analysis plots to PNG files for thesis use.
Run: python export_plots.py
Outputs go to outputs/figures/
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
FIG_DIR    = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BUCKET_LABELS = {
    "family_matrimonial_timed_mistral": "Family & Matrimonial",
    "land_property_timed_mistral":      "Land & Property",
    "motor_accidents_timed_mistral":    "Motor Accidents",
    "sexual_offences_timed_mistral":    "Sexual Offences",
    "fin_fraud_timed_mistral":          "Financial Fraud",
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
    "family_matrimonial_timed_mistral": "#e63946",
    "land_property_timed_mistral":      "#2a9d8f",
    "motor_accidents_timed_mistral":    "#e9c46a",
    "sexual_offences_timed_mistral":    "#a8dadc",
    "fin_fraud_timed_mistral":          "#457b9d",
    "unknown":                          "#adb5bd",
}

WIDTH  = 1400
HEIGHT = 900
SCALE  = 2   # 2× = effectively 2800×1800 — good for print


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_network_figure(nodes, edges, colour_key="type", top_n=80, title=""):
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
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    edge_x, edge_y = [], []
    for e in sub_edges:
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#cccccc"),
        hoverinfo="none", showlegend=False,
    )

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
            texts.append(n["text"][:30])
            pr  = n.get("pagerank", 0)
            win = n.get("outcome_win_rate")
            win_str = f"{win:.0%}" if win is not None else "n/a"
            hovers.append(
                f"<b>{n['text']}</b><br>Type: {n['type']}<br>"
                f"Frequency: {n.get('frequency',0)}<br>"
                f"PageRank: {pr:.4f}<br>Win rate: {win_str}"
            )
            sizes.append(max(8, min(40, pr * 3000)))

        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=group,
            marker=dict(size=sizes, color=colour, line=dict(width=1, color="#fff")),
            text=texts, textposition="top center",
            textfont=dict(size=9),
            hovertext=hovers, hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(bgcolor="#16213e", bordercolor="#444", font=dict(size=11)),
        margin=dict(l=20, r=20, t=60, b=20),
        width=WIDTH, height=HEIGHT,
    )
    return fig


def make_bar_figure(entries, metric, title):
    if not entries:
        return go.Figure().update_layout(title="No data")
    df = pd.DataFrame(entries[:30])
    fig = px.bar(
        df, x="score", y="text", orientation="h",
        color="type", color_discrete_map=TYPE_COLOURS,
        title=title,
        labels={"score": metric, "text": "Entity"},
    )
    fig.update_layout(
        plot_bgcolor="#0f0f0f", paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0", size=12),
        yaxis=dict(autorange="reversed"),
        title=dict(font=dict(size=16)),
        width=WIDTH, height=HEIGHT,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def save(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.write_image(path, width=WIDTH, height=HEIGHT, scale=SCALE)
    print(f"  saved → {path}")


def export_within_bucket():
    print("\n── Within-bucket figures ──")
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

        # network
        fig = make_network_figure(nodes, edges, colour_key="type", top_n=80,
                                  title=f"Entity Co-occurrence Network — {label}")
        save(fig, f"within_{bucket}_network")

        # bar charts for each metric
        for metric_key, metric_label in [
            ("top_by_pagerank",    "PageRank"),
            ("top_by_strength",    "Co-occurrence Strength"),
            ("top_by_betweenness", "Betweenness Centrality"),
        ]:
            entries = data.get(metric_key, [])
            if not entries:
                continue
            fig = make_bar_figure(entries, metric_label,
                                  f"Top Entities by {metric_label} — {label}")
            save(fig, f"within_{bucket}_{metric_key}")


def export_cross_bucket():
    print("\n── Cross-bucket figures ──")
    data = load_json(os.path.join(CROSS_DIR, "cross_bucket_analysis.json"))
    if not data:
        print("  No cross-bucket data found — skipping")
        return

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # network coloured by entity type
    fig = make_network_figure(nodes, edges, colour_key="type", top_n=100,
                              title="Cross-Bucket Entity Co-occurrence Network (by Type)")
    save(fig, "cross_bucket_network_by_type")

    # network coloured by source bucket
    fig = make_network_figure(nodes, edges, colour_key="bucket", top_n=100,
                              title="Cross-Bucket Entity Co-occurrence Network (by Domain)")
    save(fig, "cross_bucket_network_by_bucket")

    # bar charts
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
                              f"Top Cross-Bucket Entities by {metric_label}")
        save(fig, f"cross_bucket_{metric_key}")

    # per-entity-type breakdown bar charts
    by_type_path = os.path.join(CROSS_DIR, "by_entity_type.json")
    by_type = load_json(by_type_path)
    if by_type:
        print("\n  Per-type breakdown figures:")
        for etype, ents in by_type.items():
            entries = [{"text": e["text"], "type": e["type"], "score": e["pagerank"]}
                       for e in ents[:20]]
            fig = make_bar_figure(entries, "PageRank",
                                  f"Top {etype} Entities — All Domains (by PageRank)")
            save(fig, f"cross_bucket_type_{etype.lower()}_pagerank")


if __name__ == "__main__":
    try:
        import kaleido  # noqa
    except ImportError:
        print("ERROR: kaleido not installed. Run:  pip install kaleido")
        raise SystemExit(1)

    export_within_bucket()
    export_cross_bucket()
    print(f"\nAll figures saved to: {FIG_DIR}")
