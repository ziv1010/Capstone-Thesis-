"""
Interactive visualizer for entity co-occurrence network analysis.
Two tabs: Within-Bucket and Cross-Bucket.
"""

import json
import os
import glob
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx

# ── paths ─────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
WITHIN_DIR = os.path.join(OUT_DIR, "within_bucket")
CROSS_DIR  = os.path.join(OUT_DIR, "cross_bucket")

BUCKET_COLOURS = {
    "family_matrimonial_timed_mistral": "#e63946",
    "land_property_timed_mistral":      "#2a9d8f",
    "motor_accidents_timed_mistral":    "#e9c46a",
    "sexual_offences_timed_mistral":    "#a8dadc",
    "fin_fraud_timed_mistral":          "#457b9d",
    "unknown":                          "#adb5bd",
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

# ── data loading ──────────────────────────────────────────────────────────────

def load_within_summary():
    path = os.path.join(WITHIN_DIR, "_summary.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_within_bucket(bucket: str) -> dict:
    path = os.path.join(WITHIN_DIR, f"{bucket}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_cross_bucket() -> dict:
    path = os.path.join(CROSS_DIR, "cross_bucket_analysis.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def available_buckets():
    files = glob.glob(os.path.join(WITHIN_DIR, "*.json"))
    return [
        os.path.basename(f).replace(".json", "")
        for f in files
        if not os.path.basename(f).startswith("_")
    ]


# ── graph layout helper ────────────────────────────────────────────────────────

def make_network_figure(nodes: list, edges: list, colour_key: str = "type",
                        top_n: int = 80, title: str = "") -> go.Figure:
    """Build a Plotly network figure from node/edge lists."""
    if not nodes:
        return go.Figure().update_layout(title="No data available — run analyse.py first")

    # take top_n by pagerank
    top_nodes = sorted(nodes, key=lambda n: n.get("pagerank", 0), reverse=True)[:top_n]
    node_set  = {n["entity"] for n in top_nodes}
    sub_edges = [e for e in edges if e["source"] in node_set and e["target"] in node_set]

    # build networkx for layout
    G = nx.Graph()
    for n in top_nodes:
        G.add_node(n["entity"])
    for e in sub_edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    try:
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # edge traces
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

    # node traces per colour group
    node_traces = []
    groups = {}
    for n in top_nodes:
        key = n.get("type" if colour_key == "type" else "buckets", ["unknown"])
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
            pr   = n.get("pagerank", 0)
            freq = n.get("frequency", 0)
            win  = n.get("outcome_win_rate")
            win_str = f"{win:.0%}" if win is not None else "n/a"
            hovers.append(
                f"<b>{n['text']}</b><br>"
                f"Type: {n['type']}<br>"
                f"Frequency: {freq}<br>"
                f"PageRank: {pr:.4f}<br>"
                f"Win rate: {win_str}<br>"
                f"Buckets: {', '.join(n.get('buckets', []))}"
            )
            sizes.append(max(8, min(40, pr * 3000)))

        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=group,
            marker=dict(size=sizes, color=colour, line=dict(width=1, color="#fff")),
            text=texts, textposition="top center",
            textfont=dict(size=8),
            hovertext=hovers, hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="#16213e", bordercolor="#444"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=650,
    )
    return fig


def make_bar_figure(entries: list, metric: str, title: str, colour_key="type") -> go.Figure:
    if not entries:
        return go.Figure().update_layout(title="No data")
    df = pd.DataFrame(entries[:30])
    df["colour"] = df["type"].map(TYPE_COLOURS).fillna("#999")
    fig = px.bar(
        df, x="score", y="text", orientation="h",
        color="type", color_discrete_map=TYPE_COLOURS,
        hover_data=["type"],
        title=title,
        labels={"score": metric, "text": "Entity"},
    )
    fig.update_layout(
        plot_bgcolor="#0f0f0f", paper_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"), yaxis=dict(autorange="reversed"),
        height=600, margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


def nodes_to_df(nodes: list) -> pd.DataFrame:
    if not nodes:
        return pd.DataFrame()
    cols = ["text", "type", "frequency", "case_count", "pagerank",
            "strength", "degree_centrality", "betweenness",
            "outcome_win_rate", "outcome_mean_score", "bucket_count", "buckets"]
    rows = []
    for n in nodes:
        row = {c: n.get(c, "") for c in cols}
        if isinstance(row["buckets"], list):
            row["buckets"] = ", ".join(row["buckets"])
        for k in ["pagerank", "degree_centrality", "betweenness"]:
            if isinstance(row[k], float):
                row[k] = round(row[k], 5)
        if isinstance(row["outcome_win_rate"], float):
            row["outcome_win_rate"] = round(row["outcome_win_rate"], 3)
        if isinstance(row["outcome_mean_score"], float):
            row["outcome_mean_score"] = round(row["outcome_mean_score"], 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# ── app layout ────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Entity Network Analysis")

DARK = {"backgroundColor": "#1a1a2e", "color": "#e0e0e0"}
CARD = {"backgroundColor": "#16213e", "borderRadius": "8px", "padding": "16px", "marginBottom": "16px"}

buckets_available = available_buckets()

app.layout = html.Div(style={**DARK, "fontFamily": "monospace", "minHeight": "100vh", "padding": "20px"}, children=[

    html.H1("Entity Co-occurrence Network Analysis", style={"color": "#a8dadc", "marginBottom": "4px"}),
    html.P("Within-bucket and cross-bucket legal entity connectivity analysis", style={"color": "#888", "marginBottom": "20px"}),

    dcc.Tabs(id="main-tabs", value="within", style={"marginBottom": "20px"}, children=[

        # ── WITHIN-BUCKET TAB ─────────────────────────────────────────────
        dcc.Tab(label="Within-Bucket", value="within", style={"backgroundColor": "#16213e", "color": "#aaa"},
                selected_style={"backgroundColor": "#0f3460", "color": "#a8dadc"}, children=[

            html.Div(style=CARD, children=[
                html.Label("Select Bucket:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="bucket-selector",
                    options=[{"label": b.replace("_timed_mistral", "").replace("_", " ").title(), "value": b}
                             for b in buckets_available],
                    value=buckets_available[0] if buckets_available else None,
                    style={"backgroundColor": "#0f3460", "color": "#000"},
                ),
                html.Div(style={"display": "flex", "gap": "20px", "marginTop": "12px"}, children=[
                    html.Label("Rank by:"),
                    dcc.RadioItems(
                        id="within-metric",
                        options=[
                            {"label": "PageRank",    "value": "top_by_pagerank"},
                            {"label": "Strength",    "value": "top_by_strength"},
                            {"label": "Degree",      "value": "top_by_degree"},
                            {"label": "Betweenness", "value": "top_by_betweenness"},
                        ],
                        value="top_by_pagerank",
                        inline=True,
                        style={"color": "#e0e0e0"},
                    ),
                    html.Label("Network top-N:"),
                    dcc.Slider(id="within-topn", min=20, max=150, step=10, value=60,
                               marks={20: "20", 60: "60", 100: "100", 150: "150"},
                               tooltip={"always_visible": False}),
                ]),
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                html.Div(style=CARD, children=[dcc.Graph(id="within-network")]),
                html.Div(style=CARD, children=[dcc.Graph(id="within-bar")]),
            ]),

            html.Div(style=CARD, children=[
                html.H3("Full Entity Table", style={"color": "#a8dadc"}),
                dash_table.DataTable(
                    id="within-table",
                    sort_action="native",
                    filter_action="native",
                    page_size=25,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#0f3460", "color": "#a8dadc", "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "fontSize": "12px"},
                    style_filter={"backgroundColor": "#0f3460", "color": "#000"},
                ),
            ]),
        ]),

        # ── CROSS-BUCKET TAB ──────────────────────────────────────────────
        dcc.Tab(label="Cross-Bucket", value="cross", style={"backgroundColor": "#16213e", "color": "#aaa"},
                selected_style={"backgroundColor": "#0f3460", "color": "#a8dadc"}, children=[

            html.Div(style=CARD, children=[
                html.Div(style={"display": "flex", "gap": "40px", "alignItems": "center"}, children=[
                    html.Div(children=[
                        html.Label("Rank by:"),
                        dcc.RadioItems(
                            id="cross-metric",
                            options=[
                                {"label": "PageRank",    "value": "top_by_pagerank"},
                                {"label": "Strength",    "value": "top_by_strength"},
                                {"label": "Degree",      "value": "top_by_degree"},
                                {"label": "Betweenness", "value": "top_by_betweenness"},
                                {"label": "Bridge Score (# buckets)", "value": "top_by_bridge"},
                            ],
                            value="top_by_pagerank",
                            inline=True,
                            style={"color": "#e0e0e0"},
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Network top-N:"),
                        dcc.Slider(id="cross-topn", min=20, max=200, step=10, value=80,
                                   marks={20: "20", 80: "80", 150: "150", 200: "200"},
                                   tooltip={"always_visible": False}),
                    ]),
                    html.Div(children=[
                        html.Label("Colour nodes by:"),
                        dcc.RadioItems(
                            id="cross-colour",
                            options=[
                                {"label": "Entity type", "value": "type"},
                                {"label": "First bucket", "value": "bucket"},
                            ],
                            value="type", inline=True,
                            style={"color": "#e0e0e0"},
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Filter by entity type:"),
                        dcc.Dropdown(
                            id="cross-type-filter",
                            options=[{"label": t, "value": t}
                                     for t in ["STATUTE", "PROVISION", "JUDGE", "COURT", "LAWYER", "PRECEDENT", "GPE"]],
                            multi=True,
                            placeholder="All types",
                            style={"backgroundColor": "#0f3460", "color": "#000", "minWidth": "250px"},
                        ),
                    ]),
                ]),
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                html.Div(style=CARD, children=[dcc.Graph(id="cross-network")]),
                html.Div(style=CARD, children=[dcc.Graph(id="cross-bar")]),
            ]),

            # bridge analysis panel
            html.Div(style=CARD, children=[
                html.H3("Bridge Entities (appear in most buckets)", style={"color": "#a8dadc"}),
                dcc.Graph(id="cross-bridge"),
            ]),

            html.Div(style=CARD, children=[
                html.H3("Full Cross-Bucket Entity Table", style={"color": "#a8dadc"}),
                dash_table.DataTable(
                    id="cross-table",
                    sort_action="native",
                    filter_action="native",
                    page_size=25,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#0f3460", "color": "#a8dadc", "fontWeight": "bold"},
                    style_cell={"backgroundColor": "#16213e", "color": "#e0e0e0", "fontSize": "12px"},
                    style_filter={"backgroundColor": "#0f3460", "color": "#000"},
                ),
            ]),
        ]),
    ]),
])


# ── callbacks: within-bucket ───────────────────────────────────────────────────

@app.callback(
    Output("within-network", "figure"),
    Output("within-bar",     "figure"),
    Output("within-table",   "data"),
    Output("within-table",   "columns"),
    Input("bucket-selector", "value"),
    Input("within-metric",   "value"),
    Input("within-topn",     "value"),
)
def update_within(bucket, metric, topn):
    if not bucket:
        empty = go.Figure()
        return empty, empty, [], []

    data = load_within_bucket(bucket)
    if not data:
        msg = go.Figure().update_layout(title="Run analyse.py first to generate data")
        return msg, msg, [], []

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    net_fig = make_network_figure(
        nodes, edges, colour_key="type", top_n=topn,
        title=f"Entity Network — {bucket.replace('_timed_mistral','').replace('_',' ').title()}",
    )

    top_list = data.get(metric, [])
    label_map = {
        "top_by_pagerank":    "PageRank",
        "top_by_strength":    "Co-occurrence Strength",
        "top_by_degree":      "Degree Centrality",
        "top_by_betweenness": "Betweenness Centrality",
    }
    bar_fig = make_bar_figure(top_list, label_map.get(metric, metric),
                              f"Top Entities by {label_map.get(metric,'')}")

    df = nodes_to_df(nodes)
    if df.empty:
        return net_fig, bar_fig, [], []
    cols = [{"name": c, "id": c} for c in df.columns]
    return net_fig, bar_fig, df.to_dict("records"), cols


# ── callbacks: cross-bucket ────────────────────────────────────────────────────

@app.callback(
    Output("cross-network", "figure"),
    Output("cross-bar",     "figure"),
    Output("cross-bridge",  "figure"),
    Output("cross-table",   "data"),
    Output("cross-table",   "columns"),
    Input("cross-metric",      "value"),
    Input("cross-topn",        "value"),
    Input("cross-colour",      "value"),
    Input("cross-type-filter", "value"),
)
def update_cross(metric, topn, colour_key, type_filter):
    data = load_cross_bucket()
    if not data:
        msg = go.Figure().update_layout(title="Run analyse.py first to generate data")
        return msg, msg, msg, [], []

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # apply type filter
    if type_filter:
        filtered_nodes = [n for n in nodes if n["type"] in type_filter]
        filtered_set   = {n["entity"] for n in filtered_nodes}
        filtered_edges = [e for e in edges if e["source"] in filtered_set and e["target"] in filtered_set]
    else:
        filtered_nodes = nodes
        filtered_edges = edges

    net_fig = make_network_figure(
        filtered_nodes, filtered_edges, colour_key=colour_key, top_n=topn,
        title="Cross-Bucket Entity Network",
    )

    top_list = data.get(metric, [])
    if type_filter:
        top_list = [e for e in top_list if e.get("type") in type_filter]
    label_map = {
        "top_by_pagerank":    "PageRank",
        "top_by_strength":    "Co-occurrence Strength",
        "top_by_degree":      "Degree Centrality",
        "top_by_betweenness": "Betweenness Centrality",
        "top_by_bridge":      "Bridge Score (# buckets)",
    }
    bar_fig = make_bar_figure(top_list[:30], label_map.get(metric, metric),
                              f"Top Entities by {label_map.get(metric, '')}")

    # bridge figure: entities in all / most buckets
    bridge_list = data.get("top_by_bridge", [])
    if type_filter:
        bridge_list = [e for e in bridge_list if e.get("type") in type_filter]
    bridge_fig = make_bar_figure(bridge_list[:30], "Bucket Count",
                                 "Bridge Entities — appear in most buckets")

    df = nodes_to_df(filtered_nodes)
    if df.empty:
        return net_fig, bar_fig, bridge_fig, [], []
    cols = [{"name": c, "id": c} for c in df.columns]
    return net_fig, bar_fig, bridge_fig, df.to_dict("records"), cols


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"Starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
