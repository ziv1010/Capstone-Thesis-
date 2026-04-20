#!/usr/bin/env python3
"""
app.py — Legal Case Graph Visualiser (Dash) v2
================================================
3-column layout: sidebar | graph+tabs | details panel

Performance improvements:
  • Static tables/charts (hubs, bridges, top cases, pie, histogram) pre-rendered at
    startup — never recomputed during interaction
  • Main graph callback only rebuilds the graph figure, not all content
  • clickData only (no selectedData) — eliminates double-fire glitch
  • select2d/lasso2d removed from toolbar

UX improvements:
  • Persistent details panel (right column) replaces disruptive modal popup
  • Details panel shows each connected case with the specific entities
    (statute, court, judge, etc.) that link them — grouped by entity type
  • Blue highlight for connected cases, amber for bridge entities
  • Indigo connection-path edges (more visible than white)
  • clickmode="event" (no +select) prevents accidental multi-select

Layout:
  sidebar (2) | graph+tabs (7) | details panel (3)

Run after build_graph.py:
  python app.py [--config config.yaml] [--port 8050]
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import Input, Output, dcc, html, no_update


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── data loading ──────────────────────────────────────────────────────────────

def load_artefacts(out_dir: Path) -> tuple:
    """Returns (G, layout, stats, connections, case_layout)."""
    with open(out_dir / "graph_sample.pkl", "rb") as f:
        G: nx.Graph = pickle.load(f)

    layout_path = out_dir / "layout.pkl"
    if layout_path.exists():
        with open(layout_path, "rb") as f:
            layout: dict = pickle.load(f)
    else:
        print("layout.pkl not found — computing on-the-fly …")
        k   = 2.0 / max(1, np.sqrt(G.number_of_nodes()))
        raw = nx.spring_layout(G, k=k, iterations=50, seed=42)
        layout = {n: list(map(float, xy)) for n, xy in raw.items()}

    with open(out_dir / "stats.json") as f:
        stats: dict = json.load(f)

    connections: dict = {}
    conn_path = out_dir / "case_connections.pkl"
    if conn_path.exists():
        with open(conn_path, "rb") as f:
            connections = pickle.load(f)

    case_layout: dict = {}
    cl_path = out_dir / "case_layout.pkl"
    if cl_path.exists():
        with open(cl_path, "rb") as f:
            case_layout = pickle.load(f)

    return G, layout, stats, connections, case_layout


# ── colour + shape helpers ────────────────────────────────────────────────────

BRIDGE_COLOR    = "#c7d2fe"   # indigo-200
UNASSIGNED_COLOR = "#94A3B8"

NODE_SYMBOLS = {
    "case":       "circle",
    "statute":    "square",
    "provision":  "diamond",
    "precedent":  "triangle-up",
    "court":      "triangle-down",
    "judge":      "cross",
    "lawyer":     "x",
    "petitioner": "triangle-right",
    "respondent": "triangle-left",
}


def node_symbol(node_type: str) -> str:
    return NODE_SYMBOLS.get(node_type, "circle")


def hex_to_rgba(color: str, alpha: float) -> str:
    if not color.startswith("#") or len(color) != 7:
        return color
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def case_is_visible(
    G: nx.Graph,
    node_key: str,
    show_buckets: set[str],
    max_rank: int | None,
) -> bool:
    if node_key not in G:
        return False
    data = G.nodes[node_key]
    if data.get("node_type") != "case":
        return False
    if data.get("bucket", "") not in show_buckets:
        return False
    if max_rank is not None and data.get("connectivity_rank", 1) > max_rank:
        return False
    return True


def case_bucket_counts(
    G: nx.Graph,
    node_key: str,
    show_buckets: set[str],
) -> dict[str, int]:
    data = G.nodes[node_key]
    if data.get("node_type") == "case":
        bucket = data.get("bucket", "")
        return {bucket: 1} if bucket and bucket in show_buckets else {}
    counts: dict[str, int] = defaultdict(int)
    for nb in G.neighbors(node_key):
        nb_data = G.nodes[nb]
        if nb_data.get("node_type") != "case":
            continue
        bucket = nb_data.get("bucket", "")
        if bucket in show_buckets:
            counts[bucket] += 1
    return dict(counts)


def bucket_group_key(G: nx.Graph, node_key: str, show_buckets: set[str]) -> str:
    counts = case_bucket_counts(G, node_key, show_buckets)
    if len(counts) == 1:
        return next(iter(counts))
    if len(counts) > 1:
        return "cross-bucket"
    return "unassigned"


def node_color(G: nx.Graph, node_key: str, bucket_colors: dict, show_buckets: set[str]) -> str:
    grp = bucket_group_key(G, node_key, show_buckets)
    if grp in bucket_colors:
        return bucket_colors[grp]
    if grp == "cross-bucket":
        return BRIDGE_COLOR
    return UNASSIGNED_COLOR


def extract_node_key(click_data: dict | None) -> str | None:
    if not click_data or not click_data.get("points"):
        return None
    for point in click_data["points"]:
        cd = point.get("customdata")
        if cd and len(cd) >= 6:
            return cd[5]
    return None


def edge_key(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def build_focus_context(
    G: nx.Graph,
    selected_node_key: str | None,
    show_buckets: set[str],
    max_rank: int | None,
) -> dict:
    focus: dict = {
        "selected_node":  selected_node_key,
        "highlight_nodes": set(),
        "connected_cases": set(),
        "anchor_cases":    set(),
        "bridge_nodes":    set(),
        "hub_edges":       set(),
        "case_edges":      set(),
        "case_links":      {},
    }
    if not selected_node_key or selected_node_key not in G:
        return focus

    selected_data = G.nodes[selected_node_key]
    selected_type = selected_data.get("node_type")

    if selected_type == "case":
        focus["anchor_cases"].add(selected_node_key)
        case_links: dict[str, set[str]] = defaultdict(set)
        bridge_nodes: set[str] = set()

        for nb in G.neighbors(selected_node_key):
            nb_data = G.nodes[nb]
            if nb_data.get("node_type") == "case":
                continue
            if not nb_data.get("is_shared", False):
                continue
            linked = [
                cn for cn in G.neighbors(nb)
                if cn != selected_node_key and case_is_visible(G, cn, show_buckets, max_rank)
            ]
            if not linked:
                continue
            bridge_nodes.add(nb)
            focus["hub_edges"].add(edge_key(selected_node_key, nb))
            for cn in linked:
                case_links[cn].add(nb)
                focus["hub_edges"].add(edge_key(nb, cn))
                focus["case_edges"].add(edge_key(selected_node_key, cn))

        focus["connected_cases"]  = set(case_links)
        focus["bridge_nodes"]     = bridge_nodes
        focus["highlight_nodes"]  = {selected_node_key} | bridge_nodes | focus["connected_cases"]
        focus["case_links"]       = {cn: sorted(nodes) for cn, nodes in case_links.items()}
        return focus

    # Entity node selected
    direct_cases = [
        cn for cn in G.neighbors(selected_node_key)
        if case_is_visible(G, cn, show_buckets, max_rank)
    ]
    focus["anchor_cases"]    = set(direct_cases)
    focus["connected_cases"] = set(direct_cases)
    focus["bridge_nodes"]    = {selected_node_key}
    focus["highlight_nodes"] = {selected_node_key} | set(direct_cases)
    focus["case_links"]      = {cn: [selected_node_key] for cn in direct_cases}
    for cn in direct_cases:
        focus["hub_edges"].add(edge_key(selected_node_key, cn))
    for i, ca in enumerate(direct_cases):
        for cb in direct_cases[i + 1:]:
            focus["case_edges"].add(edge_key(ca, cb))
    return focus


# ── DETAILS PANEL ────────────────────────────────────────────────────────────

def _chip(label: str, value: str, accent: str | None = None) -> html.Span:
    """Small metric chip: 'label value'."""
    return html.Span(
        [
            html.Span(label + " ", style={"color": "#64748b"}),
            html.Span(value, style={"color": "#e2e8f0", "fontWeight": "600"}),
        ],
        style={
            "backgroundColor": "#1e293b",
            "borderRadius":    "4px",
            "padding":         "2px 7px",
            "fontSize":        "10px",
            "border":          f"1px solid {accent or '#334155'}",
            "marginRight":     "4px",
            "marginBottom":    "4px",
            "display":         "inline-block",
        },
    )


def render_details_panel(
    G: nx.Graph,
    node_key: str | None,
    bucket_colors: dict,
    entity_colors: dict,
    show_buckets: set[str],
    max_rank: int | None,
) -> html.Div:
    if not node_key or node_key not in G:
        return html.Div([
            html.Div(
                "Select a node",
                style={"color": "#475569", "textAlign": "center", "marginTop": "50px", "fontSize": "13px"},
            ),
            html.Div(
                "Click any node in the graph to explore its connections",
                style={"color": "#334155", "textAlign": "center", "fontSize": "11px", "marginTop": "6px"},
            ),
        ])

    d          = G.nodes[node_key]
    node_type  = d.get("node_type", "?")
    label      = d.get("label", node_key)
    degree     = d.get("degree", G.degree(node_key))
    color      = node_color(G, node_key, bucket_colors, show_buckets)
    focus      = build_focus_context(G, node_key, show_buckets, max_rank)
    bkt_counts = case_bucket_counts(G, node_key, show_buckets)
    type_color = entity_colors.get(node_type, "#6366f1")

    # ── header ────────────────────────────────────────────────────────────
    header = html.Div([
        dbc.Badge(
            node_type.upper(),
            style={"backgroundColor": type_color, "fontSize": "10px", "marginBottom": "6px"},
        ),
        html.Div(
            label,
            style={
                "fontWeight":  "700", "fontSize": "13px", "color": "#f1f5f9",
                "lineHeight":  "1.4", "wordBreak": "break-word", "marginBottom": "8px",
            },
        ),
    ])

    # ── metrics row ───────────────────────────────────────────────────────
    if node_type == "case":
        score  = d.get("connectivity_score", 0)
        rank   = d.get("connectivity_rank", 0)
        total  = d.get("bucket_case_count", 0)
        bucket = d.get("bucket", "")
        outcome = d.get("outcome", "")
        metrics = html.Div([
            html.Div([
                html.Span("Bucket ", style={"color": "#64748b", "fontSize": "11px"}),
                dbc.Badge(
                    bucket,
                    style={"backgroundColor": bucket_colors.get(bucket, "#555"), "fontSize": "10px"},
                ),
                *([html.Span(
                    f"  outcome: {outcome}",
                    style={"color": "#94a3b8", "fontSize": "10px", "marginLeft": "6px"},
                )] if outcome and outcome != "unknown" else []),
            ], className="mb-2"),
            html.Div([
                _chip("degree", str(degree)),
                _chip("score", f"{score:.0f}", "#6366f1"),
                _chip("rank", f"#{rank}/{total}", "#22c55e"),
            ]),
        ], className="mb-2")
    else:
        arg_cnt  = d.get("argument_case_count", 0)
        is_cross = len(bkt_counts) > 1
        bucket_pills = [
            dbc.Badge(
                f"{bkt}: {cnt}",
                style={"backgroundColor": bucket_colors.get(bkt, "#555"), "fontSize": "10px"},
                className="me-1",
            )
            for bkt, cnt in sorted(bkt_counts.items(), key=lambda x: -x[1])
        ]
        metrics = html.Div([
            html.Div([
                _chip("degree", str(degree)),
                _chip("arg.cit.", str(arg_cnt), "#f59e0b"),
                *([_chip("bridge", "cross-bucket", "#f59e0b")] if is_cross else []),
            ], className="mb-1"),
            html.Div(
                [html.Span("Used in: ", style={"color": "#64748b", "fontSize": "11px"}), *bucket_pills],
                className="mb-2",
            ) if bucket_pills else html.Div(),
        ])

    # ── connections section ───────────────────────────────────────────────
    n_conn = len(focus["connected_cases"])
    if n_conn == 0:
        conn_section = html.Div(
            "No other visible cases connected under current filters.",
            style={"color": "#475569", "fontSize": "11px", "fontStyle": "italic"},
        )
    else:
        sorted_cases = sorted(
            focus["case_links"].items(),
            key=lambda item: (-len(item[1]), -G.nodes[item[0]].get("connectivity_score", 0)),
        )
        conn_items = []
        for case_node, via_nodes in sorted_cases[:25]:
            cd         = G.nodes[case_node]
            case_bkt   = cd.get("bucket", "")
            case_label = cd.get("label", case_node)
            bkt_color  = bucket_colors.get(case_bkt, "#334155")

            # Group via-nodes by entity type for clarity
            via_by_type: dict[str, list[str]] = defaultdict(list)
            for vn in via_nodes[:12]:
                vd    = G.nodes[vn]
                etype = vd.get("entity_type", vd.get("node_type", "?"))
                via_by_type[etype].append(vd.get("label", vn))

            via_lines = []
            for etype, labels in sorted(via_by_type.items()):
                ec = entity_colors.get(etype.lower(), "#475569")
                via_lines.append(html.Div([
                    html.Span(
                        etype[:5].upper(),
                        style={
                            "backgroundColor": ec,
                            "color": "#fff", "borderRadius": "3px",
                            "padding": "0 4px", "fontSize": "9px",
                            "fontWeight": "bold", "marginRight": "5px",
                            "flexShrink": "0",
                        },
                    ),
                    html.Span(
                        ", ".join(lbl[:32] for lbl in labels[:3])
                        + (f"  +{len(labels)-3}" if len(labels) > 3 else ""),
                        style={"color": "#94a3b8", "fontSize": "10px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "2px"}))

            if len(via_nodes) > 12:
                via_lines.append(html.Div(
                    f"  … +{len(via_nodes)-12} more shared entities",
                    style={"color": "#334155", "fontSize": "10px"},
                ))

            conn_items.append(html.Div([
                html.Div([
                    html.Span("→ ", style={"color": "#6366f1", "fontWeight": "bold", "marginRight": "3px"}),
                    html.Span(case_label[:60], style={"color": "#e2e8f0", "fontSize": "11px", "fontWeight": "600"}),
                ], style={"marginBottom": "3px"}),
                html.Div([
                    dbc.Badge(
                        case_bkt,
                        style={"backgroundColor": bkt_color, "fontSize": "9px"},
                        className="me-1",
                    ),
                    html.Span(
                        f"via {len(via_nodes)} shared entit{'y' if len(via_nodes)==1 else 'ies'}",
                        style={"color": "#64748b", "fontSize": "10px"},
                    ),
                ], className="mb-1"),
                html.Div(via_lines, style={"paddingLeft": "10px"}),
            ], style={
                "borderLeft":    f"3px solid {bkt_color}",
                "paddingLeft":   "8px",
                "marginBottom":  "12px",
            }))

        if len(sorted_cases) > 25:
            conn_items.append(html.Div(
                f"… and {len(sorted_cases)-25} more connected cases",
                style={"color": "#334155", "fontSize": "11px", "fontStyle": "italic"},
            ))

        cap_hint = (
            html.Div(
                f"Graph highlights top 50 of {n_conn} — open Full details ↗ for all.",
                style={"color": "#475569", "fontSize": "10px",
                       "fontStyle": "italic", "marginBottom": "6px"},
            ) if n_conn > 50 else html.Div()
        )

        conn_section = html.Div([
            html.Div(
                f"{n_conn} connected case{'s' if n_conn != 1 else ''}",
                style={
                    "color": "#6366f1", "fontWeight": "600",
                    "fontSize": "12px", "marginBottom": "4px",
                },
            ),
            cap_hint,
            *conn_items,
        ])

    return html.Div([
        header,
        metrics,
        html.Hr(style={"borderColor": "#1e293b", "margin": "8px 0"}),
        conn_section,
    ], style={"fontSize": "12px"})


# ── HUB NETWORK FIGURE ────────────────────────────────────────────────────────

def build_figure(
    G: nx.Graph,
    layout: dict,
    bucket_colors: dict,
    entity_colors: dict,
    show_types: set[str],
    show_buckets: set[str],
    min_degree: int,
    search_text: str,
    max_rank: int | None = None,
    selected_node_key: str | None = None,
) -> go.Figure:
    search_lower = search_text.strip().lower() if search_text else ""

    # ── filter visible nodes ──────────────────────────────────────────────
    visible_nodes: list[str] = []
    for node, d in G.nodes(data=True):
        nt     = d.get("node_type", "unknown")
        bucket = d.get("bucket", "")
        deg    = d.get("degree", G.degree(node))
        if nt not in show_types:
            continue
        if nt == "case":
            if bucket not in show_buckets:
                continue
            if max_rank is not None and d.get("connectivity_rank", 1) > max_rank:
                continue
        if deg < min_degree:
            continue
        if node not in layout:
            continue
        visible_nodes.append(node)

    visible_set  = set(visible_nodes)
    focus        = build_focus_context(G, selected_node_key, show_buckets, max_rank)
    focus_active = bool(focus["highlight_nodes"] & visible_set)

    # ── cap visual highlights to avoid overwhelming the graph ─────────────
    # When a hub entity touches 200+ cases every node lights up — cap at 50
    MAX_VIZ = 50
    if focus_active:
        all_conn_vis = focus["connected_cases"] & visible_set
        if len(all_conn_vis) > MAX_VIZ:
            viz_connected: set[str] = set(sorted(
                all_conn_vis,
                key=lambda n: G.nodes[n].get("connectivity_score", 0),
                reverse=True,
            )[:MAX_VIZ])
        else:
            viz_connected = all_conn_vis
        viz_bridge   = focus["bridge_nodes"] & visible_set
        viz_highlight = ({selected_node_key} if selected_node_key else set()) | viz_connected | viz_bridge
    else:
        viz_connected = set()
        viz_bridge    = set()
        viz_highlight = set()

    # ── edge traces ───────────────────────────────────────────────────────
    ex, ey         = [], []
    ex_arg, ey_arg = [], []
    fx, fy         = [], []
    fx_arg, fy_arg = [], []

    for u, v, ed in G.edges(data=True):
        if u not in visible_set or v not in visible_set:
            continue
        xu, yu = layout[u]
        xv, yv = layout[v]
        is_focus = focus_active and edge_key(u, v) in focus["hub_edges"]
        if ed.get("in_arguments", False) and is_focus:
            fx_arg += [xu, xv, None]; fy_arg += [yu, yv, None]
        elif ed.get("in_arguments", False):
            ex_arg += [xu, xv, None]; ey_arg += [yu, yv, None]
        elif is_focus:
            fx += [xu, xv, None];     fy += [yu, yv, None]
        else:
            ex += [xu, xv, None];     ey += [yu, yv, None]

    traces = [
        go.Scattergl(
            x=ex, y=ey, mode="lines",
            line=dict(
                width=0.4,
                color="rgba(148,163,184,0.07)" if focus_active else "rgba(148,163,184,0.20)",
            ),
            hoverinfo="none", showlegend=False, name="edges",
        ),
        go.Scattergl(
            x=ex_arg, y=ey_arg, mode="lines",
            line=dict(
                width=0.7,
                color="rgba(251,191,36,0.10)" if focus_active else "rgba(251,191,36,0.38)",
            ),
            hoverinfo="none", showlegend=True, name="cited in arguments",
            legendgroup="arg_edges",
        ),
    ]
    if fx:
        traces.append(go.Scattergl(
            x=fx, y=fy, mode="lines",
            line=dict(width=2.5, color="rgba(99,102,241,0.85)"),
            hoverinfo="none", showlegend=True, name="connection path",
            legendgroup="focus_edges",
        ))
    if fx_arg:
        traces.append(go.Scattergl(
            x=fx_arg, y=fy_arg, mode="lines",
            line=dict(width=2.8, color="rgba(251,191,36,0.95)"),
            hoverinfo="none", showlegend=True, name="connection path (arguments)",
            legendgroup="focus_edges",
        ))

    # ── hub labels — drawn last so they appear on top of markers visually.
    # customdata is set so that if this trace "wins" click detection, the node
    # key is still available and the click is processed correctly.
    hub30 = sorted(visible_nodes, key=lambda n: G.nodes[n].get("degree", 0), reverse=True)[:30]
    traces.append(go.Scattergl(
        x=[layout[n][0] for n in hub30],
        y=[layout[n][1] for n in hub30],
        mode="text",
        text=[G.nodes[n].get("label", n)[:30] for n in hub30],
        textposition="top center",
        textfont=dict(size=9, color="rgba(203,213,225,0.62)"),
        customdata=[
            [G.nodes[n].get("label", n), G.nodes[n].get("node_type","?"),
             bucket_group_key(G, n, show_buckets),
             G.nodes[n].get("degree",0), "", n,
             G.nodes[n].get("connectivity_score",0), G.nodes[n].get("connectivity_rank",0)]
            for n in hub30
        ],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<span style='color:#94a3b8'>%{customdata[1]}</span><br>"
            "degree: <b>%{customdata[3]}</b>"
            "<extra></extra>"
        ),
        showlegend=False, name="hub_labels",
    ))

    # ── group nodes by node_type ONLY (one trace per type, ~8 traces total)
    # Previously grouped by node_type × bucket_group → 50+ Scattergl traces,
    # which kills WebGL click detection. Per-point color still shows bucket.
    groups: dict[str, list[str]] = {}
    for node in visible_nodes:
        nt = G.nodes[node].get("node_type", "unknown")
        groups.setdefault(nt, []).append(node)

    all_degs = [G.nodes[n].get("degree", G.degree(n)) for n in visible_nodes] or [1]
    max_deg  = max(all_degs) or 1
    size_map: dict[str, float] = {}

    for nt, nodes in sorted(groups.items()):
        xs, ys, szs, colors, borders, opacities, cdata = [], [], [], [], [], [], []

        for node in nodes:
            d       = G.nodes[node]
            xy      = layout[node]
            deg     = d.get("degree", G.degree(node))
            label   = d.get("label", node)
            outcome = d.get("outcome", "") or ""
            bgrp    = bucket_group_key(G, node, show_buckets)
            color   = node_color(G, node, bucket_colors, show_buckets)
            size    = max(7, min(32, 7 + 25 * (deg / max_deg)))
            base    = size

            xs.append(xy[0]); ys.append(xy[1]); colors.append(color)

            if focus_active:
                if node == selected_node_key:
                    borders.append("#FFFFFF"); opacities.append(1.0); size *= 1.55
                elif node in viz_connected:
                    borders.append("#60a5fa"); opacities.append(1.0); size *= 1.30
                elif node in viz_bridge:
                    borders.append("#fbbf24"); opacities.append(1.0); size *= 1.20
                else:
                    borders.append("rgba(0,0,0,0)"); opacities.append(0.06)
            else:
                is_hit = search_lower and search_lower in label.lower()
                if search_lower:
                    borders.append("#fbbf24" if is_hit else "rgba(0,0,0,0)")
                    opacities.append(1.0 if is_hit else 0.12)
                else:
                    borders.append("rgba(255,255,255,0.20)"); opacities.append(0.88)

            out_line = f"<br>outcome: <b>{outcome}</b>" if outcome and outcome != "unknown" else ""
            cdata.append([
                label, nt, bgrp, deg, out_line, node,
                d.get("connectivity_score", 0),
                d.get("connectivity_rank", 0),
            ])
            szs.append(size); size_map[node] = base

        traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=szs, color=colors, opacity=opacities,
                line=dict(width=1.8, color=borders),
                symbol=node_symbol(nt),
            ),
            customdata=cdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<span style='color:#94a3b8'>%{customdata[1]}</span>"
                "  ·  <b>%{customdata[2]}</b><br>"
                "degree: <b>%{customdata[3]}</b>"
                "%{customdata[4]}"
                "<extra></extra>"
            ),
            name=nt, legendgroup=nt,
        ))

    # ── focus overlay traces (drawn on top of node groups) ───────────────
    if focus_active:
        # Use capped viz_connected so overlay matches what's visually highlighted
        conn_cases = sorted(viz_connected)
        brdg_nodes = sorted(viz_bridge - {selected_node_key})
        sel_nodes  = [n for n in [selected_node_key] if n in visible_set]

        if conn_cases:
            traces.append(go.Scattergl(
                x=[layout[n][0] for n in conn_cases],
                y=[layout[n][1] for n in conn_cases],
                mode="markers",
                marker=dict(
                    size=[size_map.get(n, 10) * 1.4 for n in conn_cases],
                    color=[node_color(G, n, bucket_colors, show_buckets) for n in conn_cases],
                    opacity=0.98,
                    line=dict(width=3.0, color="#60a5fa"),
                    symbol=[node_symbol(G.nodes[n].get("node_type", "case")) for n in conn_cases],
                ),
                customdata=[
                    [G.nodes[n].get("label", n), G.nodes[n].get("node_type","case"),
                     G.nodes[n].get("bucket",""), G.nodes[n].get("degree",0), "", n,
                     G.nodes[n].get("connectivity_score",0), G.nodes[n].get("connectivity_rank",0)]
                    for n in conn_cases
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#60a5fa'>connected case</span>"
                    "  ·  <b>%{customdata[2]}</b><br>"
                    "conn.score: <b>%{customdata[6]:.0f}</b>"
                    "<extra></extra>"
                ),
                showlegend=False, name="connected cases",
            ))

        if brdg_nodes:
            traces.append(go.Scattergl(
                x=[layout[n][0] for n in brdg_nodes],
                y=[layout[n][1] for n in brdg_nodes],
                mode="markers",
                marker=dict(
                    size=[size_map.get(n, 9) * 1.30 for n in brdg_nodes],
                    color=[node_color(G, n, bucket_colors, show_buckets) for n in brdg_nodes],
                    opacity=0.98,
                    line=dict(width=2.8, color="#fbbf24"),
                    symbol=[node_symbol(G.nodes[n].get("node_type", "unknown")) for n in brdg_nodes],
                ),
                customdata=[
                    [G.nodes[n].get("label", n), G.nodes[n].get("node_type","?"),
                     G.nodes[n].get("entity_type","?"), G.nodes[n].get("degree",0), "", n, 0, 0]
                    for n in brdg_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#fbbf24'>bridge entity</span> · %{customdata[2]}<br>"
                    "degree: <b>%{customdata[3]}</b>"
                    "<extra></extra>"
                ),
                showlegend=False, name="bridge entities",
            ))

        if sel_nodes:
            # customdata is REQUIRED here — without it, clicking the selected
            # node returns None and the selection silently breaks
            traces.append(go.Scattergl(
                x=[layout[n][0] for n in sel_nodes],
                y=[layout[n][1] for n in sel_nodes],
                mode="markers+text",
                text=[G.nodes[n].get("label", n)[:40] for n in sel_nodes],
                textposition="top center",
                textfont=dict(size=11, color="#FFFFFF"),
                marker=dict(
                    size=[size_map.get(n, 11) * 1.65 for n in sel_nodes],
                    color=[node_color(G, n, bucket_colors, show_buckets) for n in sel_nodes],
                    opacity=1.0,
                    line=dict(width=3.5, color="#FFFFFF"),
                    symbol=[node_symbol(G.nodes[n].get("node_type", "unknown")) for n in sel_nodes],
                ),
                customdata=[
                    [G.nodes[n].get("label", n), G.nodes[n].get("node_type","?"),
                     bucket_group_key(G, n, show_buckets),
                     G.nodes[n].get("degree",0), "", n,
                     G.nodes[n].get("connectivity_score",0), G.nodes[n].get("connectivity_rank",0)]
                    for n in sel_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#fff'>selected</span>"
                    "  ·  %{customdata[1]}<br>"
                    "degree: <b>%{customdata[3]}</b>"
                    "<extra></extra>"
                ),
                showlegend=False, name="selected node",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_dark_layout(
        uirevision="hub_network",
        title="Hub Network  ·  colors=buckets  ·  shapes=node types  ·  click a node to explore connections",
    ))
    return fig


# ── CASE CONNECTION FIGURE ────────────────────────────────────────────────────

def build_case_network_figure(
    G: nx.Graph,
    case_layout: dict,
    connections: dict,
    bucket_colors: dict,
    show_buckets: set[str],
    max_rank: int | None,
    min_shared: int,
    search_text: str,
    selected_node_key: str | None = None,
) -> go.Figure:
    search_lower = search_text.strip().lower() if search_text else ""

    visible: set[str] = set()
    for node, d in G.nodes(data=True):
        if d.get("node_type") != "case":
            continue
        if d.get("bucket", "") not in show_buckets:
            continue
        if max_rank is not None and d.get("connectivity_rank", 1) > max_rank:
            continue
        if node not in case_layout:
            continue
        visible.add(node)

    focus           = build_focus_context(G, selected_node_key, show_buckets, max_rank)
    focus_case_nodes = (focus["connected_cases"] | focus["anchor_cases"]) & visible
    focus_active    = bool(focus_case_nodes)

    # ── edge tiers ────────────────────────────────────────────────────────
    tiers: dict[int, tuple[list, list]] = {1: ([], []), 2: ([], []), 3: ([], [])}
    focus_xs, focus_ys = [], []
    seen_focus: set[frozenset[str]] = set()
    MAX_EDGES = 4000

    sorted_conns = sorted(
        ((ca, cb, cnt) for (ca, cb), cnt in connections.items()
         if ca in visible and cb in visible and cnt >= min_shared),
        key=lambda x: x[2], reverse=True,
    )[:MAX_EDGES]

    for ca, cb, cnt in sorted_conns:
        x1, y1 = case_layout[ca]
        x2, y2 = case_layout[cb]
        if focus_active and edge_key(ca, cb) in focus["case_edges"]:
            seen_focus.add(edge_key(ca, cb))
            focus_xs.extend([x1, x2, None]); focus_ys.extend([y1, y2, None])
            continue
        tier = 1 if cnt <= 2 else (2 if cnt <= 5 else 3)
        tiers[tier][0].extend([x1, x2, None]); tiers[tier][1].extend([y1, y2, None])

    if focus_active:
        for pair in focus["case_edges"]:
            if pair in seen_focus:
                continue
            ca, cb = tuple(pair)
            if ca not in visible or cb not in visible or ca not in case_layout or cb not in case_layout:
                continue
            x1, y1 = case_layout[ca]; x2, y2 = case_layout[cb]
            focus_xs.extend([x1, x2, None]); focus_ys.extend([y1, y2, None])

    tier_styles = {
        1: ("rgba(100,116,139,0.18)", 0.5, "1-2 shared entities"),
        2: ("rgba(99,102,241,0.35)",  1.2, "3-5 shared entities"),
        3: ("rgba(139,92,246,0.55)",  2.4, "6+ shared entities"),
    }
    traces = []
    for tier, (xs, ys) in tiers.items():
        if not xs:
            continue
        clr, wid, name = tier_styles[tier]
        dim_clr = (
            "rgba(100,116,139,0.05)" if tier == 1 else
            "rgba(99,102,241,0.08)"  if tier == 2 else
            "rgba(139,92,246,0.12)"
        )
        traces.append(go.Scattergl(
            x=xs, y=ys, mode="lines",
            line=dict(width=wid, color=dim_clr if focus_active else clr),
            hoverinfo="none", showlegend=True, name=name,
            legendgroup=f"edge_{tier}",
        ))
    if focus_xs:
        traces.append(go.Scattergl(
            x=focus_xs, y=focus_ys, mode="lines",
            line=dict(width=3.0, color="rgba(99,102,241,0.92)"),
            hoverinfo="none", showlegend=True, name="selected connections",
            legendgroup="focus_edges",
        ))

    # ── node traces grouped by bucket ─────────────────────────────────────
    bucket_nodes: dict[str, list[str]] = defaultdict(list)
    for node in visible:
        bucket_nodes[G.nodes[node].get("bucket", "")].append(node)

    scores   = [G.nodes[n].get("connectivity_score", 0) for n in visible] or [1]
    max_score = max(scores) or 1
    size_map: dict[str, float] = {}

    for bucket, nodes in sorted(bucket_nodes.items()):
        color = bucket_colors.get(bucket, "#BDC3C7")
        xs, ys, szs, borders, opacities, cdata = [], [], [], [], [], []

        for node in nodes:
            d       = G.nodes[node]
            xy      = case_layout[node]
            score   = d.get("connectivity_score", 0)
            rank    = d.get("connectivity_rank", 0)
            label   = d.get("label", node)
            outcome = d.get("outcome", "") or ""
            size    = max(7, min(30, 7 + 23 * (score / max_score)))
            base    = size

            xs.append(xy[0]); ys.append(xy[1])

            if focus_active:
                if node == selected_node_key:
                    borders.append("#FFFFFF"); opacities.append(1.0); size *= 1.55
                elif node in focus_case_nodes:
                    borders.append("#60a5fa"); opacities.append(1.0); size *= 1.28
                else:
                    borders.append("rgba(0,0,0,0)"); opacities.append(0.10)
            else:
                is_hit = search_lower and search_lower in label.lower()
                if search_lower:
                    borders.append("#fbbf24" if is_hit else "rgba(0,0,0,0)")
                    opacities.append(1.0 if is_hit else 0.12)
                else:
                    borders.append(hex_to_rgba(color, 0.70)); opacities.append(0.88)

            out_line = f"<br>outcome: <b>{outcome}</b>" if outcome and outcome != "unknown" else ""
            cdata.append([label, "case", bucket, d.get("degree",0), out_line, node, score, rank])
            szs.append(size); size_map[node] = base

        traces.append(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=szs, color=color, opacity=opacities,
                line=dict(width=1.8, color=borders),
                symbol="circle",
            ),
            customdata=cdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "bucket: <b>%{customdata[2]}</b><br>"
                "conn.score: <b>%{customdata[6]:.0f}</b>  rank: <b>#%{customdata[7]}</b>"
                "%{customdata[4]}"
                "<extra></extra>"
            ),
            name=bucket, legendgroup=f"case_{bucket}",
        ))

    # ── focus overlay ─────────────────────────────────────────────────────
    if focus_active:
        related = sorted(focus_case_nodes - ({selected_node_key} if selected_node_key else set()))
        if related:
            traces.append(go.Scattergl(
                x=[case_layout[n][0] for n in related],
                y=[case_layout[n][1] for n in related],
                mode="markers",
                marker=dict(
                    size=[size_map.get(n,10)*1.35 for n in related],
                    color=[bucket_colors.get(G.nodes[n].get("bucket",""), "#BDC3C7") for n in related],
                    opacity=0.98,
                    line=dict(width=3.0, color="#60a5fa"),
                    symbol="circle",
                ),
                customdata=[
                    [G.nodes[n].get("label",n), "case", G.nodes[n].get("bucket",""),
                     G.nodes[n].get("degree",0), "", n,
                     G.nodes[n].get("connectivity_score",0), G.nodes[n].get("connectivity_rank",0)]
                    for n in related
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#60a5fa'>connected</span>  ·  <b>%{customdata[2]}</b><br>"
                    "conn.score: <b>%{customdata[6]:.0f}</b>"
                    "<extra></extra>"
                ),
                showlegend=False, name="connected cases",
            ))
        if selected_node_key and selected_node_key in visible:
            traces.append(go.Scattergl(
                x=[case_layout[selected_node_key][0]],
                y=[case_layout[selected_node_key][1]],
                mode="markers+text",
                text=[G.nodes[selected_node_key].get("label", selected_node_key)[:40]],
                textposition="top center",
                textfont=dict(size=11, color="#FFFFFF"),
                marker=dict(
                    size=[size_map.get(selected_node_key,11)*1.65],
                    color=[bucket_colors.get(G.nodes[selected_node_key].get("bucket",""), "#BDC3C7")],
                    opacity=1.0,
                    line=dict(width=3.5, color="#FFFFFF"),
                    symbol="circle",
                ),
                hoverinfo="skip", showlegend=False, name="selected",
            ))

    # ── top case labels ───────────────────────────────────────────────────
    top_label = (
        sorted(focus_case_nodes, key=lambda n: G.nodes[n].get("connectivity_score",0), reverse=True)[:15]
        if focus_active else
        sorted(visible, key=lambda n: G.nodes[n].get("connectivity_score",0), reverse=True)[:20]
    )
    traces.append(go.Scattergl(
        x=[case_layout[n][0] for n in top_label],
        y=[case_layout[n][1] for n in top_label],
        mode="text",
        text=[G.nodes[n].get("label", n)[:26] for n in top_label],
        textposition="top center",
        textfont=dict(size=8, color="rgba(203,213,225,0.58)"),
        hoverinfo="skip",  # skip = fully invisible to click detection
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_dark_layout(
        uirevision="case_network",
        title="Case Connection Network  ·  edges=shared legal entities  ·  click a case to explore",
    ))
    return fig


# ── shared layout helper ──────────────────────────────────────────────────────

def _dark_layout(uirevision: str = "stable", title: str = "") -> dict:
    return dict(
        paper_bgcolor = "#0f172a",
        plot_bgcolor  = "#0f172a",
        font          = dict(color="#cbd5e1", family="Inter, system-ui, sans-serif"),
        showlegend    = True,
        legend        = dict(
            bgcolor="rgba(15,23,42,0.92)", bordercolor="#1e293b",
            borderwidth=1, font=dict(size=10),
            x=0.01, y=0.99,
        ),
        hoverlabel    = dict(
            bgcolor="#1e293b", bordercolor="#6366f1",
            font=dict(size=13, color="#f1f5f9"), namelength=-1,
        ),
        margin        = dict(l=0, r=0, t=36 if title else 10, b=0),
        xaxis         = dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis         = dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode     = "closest",
        clickmode     = "event",       # event only — no +select to avoid double-fire
        uirevision    = uirevision,
        title         = dict(text=title, font=dict(size=11, color="#475569"), x=0.5) if title else {},
        # dragmode NOT set — defaults to "zoom" which keeps the pointer cursor on hover
        # and lets click events fire correctly on Scattergl points
    )


# ── chart helpers ─────────────────────────────────────────────────────────────

def bucket_pie(bucket_counts: dict, bucket_colors: dict) -> go.Figure:
    labels = list(bucket_counts.keys())
    values = [bucket_counts[l] for l in labels]
    colors = [bucket_colors.get(l, "#BDC3C7") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, marker_colors=colors,
        textinfo="label+percent", hole=0.42,
        hovertemplate="<b>%{label}</b><br>%{value} cases (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", font=dict(color="#cbd5e1"),
        margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        title=dict(text="Cases per bucket", font=dict(size=12, color="#64748b")),
    )
    return fig


def degree_hist_chart(deg_hist: dict) -> go.Figure:
    edges  = deg_hist.get("bin_edges", [])
    counts = deg_hist.get("counts", [])
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(counts))]
    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color="#6366f1", opacity=0.8,
        hovertemplate="Degree %{x}: <b>%{y}</b> nodes<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=10, t=30, b=60),
        xaxis=dict(title="Degree", tickangle=-45, gridcolor="#1e293b"),
        yaxis=dict(title="# Nodes", gridcolor="#1e293b"),
        title=dict(text="Degree distribution", font=dict(size=12, color="#64748b")),
    )
    return fig


# ── table helpers ─────────────────────────────────────────────────────────────

def _styled_table(header_cells: list, rows: list) -> dbc.Table:
    return dbc.Table(
        [html.Thead(html.Tr(header_cells)), html.Tbody(rows)],
        bordered=False, hover=True, responsive=True, size="sm",
        style={"fontSize": "11px", "color": "#cbd5e1", "backgroundColor": "#0f172a"},
        className="table-dark",
    )


def hub_table(top_hubs: list, bucket_colors: dict) -> dbc.Table:
    header = [html.Th(t, style={"color": "#64748b"}) for t in
              ["#", "Label", "Type", "Deg.", "Arg.Cit.", "Buckets"]]
    rows = []
    for i, h in enumerate(top_hubs[:20], 1):
        pills = [
            dbc.Badge(b[:12], style={"backgroundColor": bucket_colors.get(b, "#555"),
                                     "marginRight": "2px", "fontSize": "9px"})
            for b in h.get("buckets_bridged", [])
        ]
        rows.append(html.Tr([
            html.Td(i, style={"color": "#475569"}),
            html.Td(h["label"][:42], title=h["label"]),
            html.Td(dbc.Badge(h["type"], color="secondary", style={"fontSize": "9px"})),
            html.Td(h["degree"]),
            html.Td(h.get("argument_case_count", 0)),
            html.Td(pills or html.Span("—", style={"color": "#334155"})),
        ]))
    return _styled_table(header, rows)


def bridge_table(bridges: list, bucket_colors: dict) -> dbc.Table:
    if not bridges:
        return html.P("No cross-bucket bridge nodes found.", style={"color": "#64748b"})
    header = [html.Th(t, style={"color": "#64748b"}) for t in
              ["Label", "Type", "Deg.", "Arg.Cit.", "Cases per bucket"]]
    rows = []
    for b in bridges[:25]:
        counts = b.get("bucket_case_counts", {})
        pills = [
            html.Span(f"{bk[:10]}:{n}", style={
                "backgroundColor": bucket_colors.get(bk, "#555"),
                "color": "#fff", "borderRadius": "3px",
                "padding": "1px 4px", "marginRight": "3px", "fontSize": "9px",
            })
            for bk, n in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ]
        rows.append(html.Tr([
            html.Td(b["label"][:40], title=b["label"]),
            html.Td(dbc.Badge(b["entity_type"], color="secondary", style={"fontSize": "9px"})),
            html.Td(b["degree"]),
            html.Td(b.get("argument_case_count", 0)),
            html.Td(pills),
        ]))
    return _styled_table(header, rows)


def top_cases_table(top_cases: list, bucket_colors: dict) -> dbc.Table:
    if not top_cases:
        return html.P("No connectivity data — rebuild the graph.", style={"color": "#64748b"})
    header = [html.Th(t, style={"color": "#64748b"}) for t in
              ["#", "Case", "Bucket", "Conn.Score", "Degree"]]
    rows = []
    for i, c in enumerate(top_cases[:25], 1):
        bkt = c.get("bucket", "")
        rows.append(html.Tr([
            html.Td(i, style={"color": "#475569"}),
            html.Td(c["label"][:48], title=c.get("label", "")),
            html.Td(dbc.Badge(bkt[:14], style={"backgroundColor": bucket_colors.get(bkt, "#555"),
                                               "fontSize": "9px"})),
            html.Td(f"{c.get('connectivity_score', 0):.0f}"),
            html.Td(c.get("degree", 0)),
        ]))
    return _styled_table(header, rows)


def case_connections_table(pairs: list, bucket_colors: dict) -> dbc.Table:
    if not pairs:
        return html.P("No cross-bucket case pairs found.", style={"color": "#64748b"})
    header = [html.Th(t, style={"color": "#64748b"}) for t in
              ["Case A", "Bkt A", "Case B", "Bkt B", "Shared"]]
    rows = []
    for p in pairs[:30]:
        ba, bb = p.get("bucket_a", ""), p.get("bucket_b", "")
        rows.append(html.Tr([
            html.Td(p["case_a"][:36], title=p.get("case_a", "")),
            html.Td(dbc.Badge(ba[:12], style={"backgroundColor": bucket_colors.get(ba, "#555"),
                                              "fontSize": "9px"})),
            html.Td(p["case_b"][:36], title=p.get("case_b", "")),
            html.Td(dbc.Badge(bb[:12], style={"backgroundColor": bucket_colors.get(bb, "#555"),
                                              "fontSize": "9px"})),
            html.Td(dbc.Badge(str(p["shared_entities"]), color="warning", style={"fontSize": "10px"})),
        ]))
    return _styled_table(header, rows)


# ── MODAL BODY ───────────────────────────────────────────────────────────────

def render_modal_body(
    G: nx.Graph,
    node_key: str,
    bucket_colors: dict,
    entity_colors: dict,
    show_buckets: set[str],
    max_rank: int | None,
) -> tuple[html.Div, html.Div]:
    """
    Returns (title_content, body_content) for the full-screen details modal.

    Layout:
      Case node  → BRIDGE ENTITIES section + CONNECTED CASES section
      Entity node → cases organised by bucket, with role descriptions
    """
    d         = G.nodes[node_key]
    node_type = d.get("node_type", "?")
    label     = d.get("label", node_key)
    degree    = d.get("degree", G.degree(node_key))
    focus     = build_focus_context(G, node_key, show_buckets, max_rank)
    bkt_counts = case_bucket_counts(G, node_key, show_buckets)
    type_color = entity_colors.get(node_type, "#6366f1")

    # ── modal title ───────────────────────────────────────────────────────
    title_div = html.Div([
        dbc.Badge(node_type.upper(),
                  style={"backgroundColor": type_color, "fontSize": "11px", "marginRight": "8px"}),
        html.Span(label, style={"fontSize": "15px", "fontWeight": "700", "color": "#f1f5f9"}),
    ])

    # ── shared entity type badge helper ──────────────────────────────────
    def _etype_badge(etype: str) -> html.Span:
        ec = entity_colors.get(etype.lower(), "#475569")
        return html.Span(
            etype[:5].upper(),
            style={
                "backgroundColor": ec, "color": "#fff",
                "borderRadius": "3px", "padding": "0 5px",
                "fontSize": "9px", "fontWeight": "bold",
                "marginRight": "4px",
            },
        )

    # ── CASE NODE ─────────────────────────────────────────────────────────
    if node_type == "case":
        bucket  = d.get("bucket", "")
        score   = d.get("connectivity_score", 0)
        rank    = d.get("connectivity_rank", 0)
        total   = d.get("bucket_case_count", 0)
        outcome = d.get("outcome", "") or ""

        # Stats row
        stats_row = html.Div([
            _chip("bucket", bucket),
            _chip("degree", str(degree)),
            _chip("conn.score", f"{score:.0f}", "#6366f1"),
            _chip("rank", f"#{rank}/{total}", "#22c55e"),
            *([_chip("outcome", outcome)] if outcome and outcome != "unknown" else []),
        ], className="mb-3 d-flex flex-wrap")

        # ── SECTION 1: Bridge entities ────────────────────────────────────
        bridge_nodes_vis = sorted(
            focus["bridge_nodes"],
            key=lambda n: G.nodes[n].get("degree", 0),
            reverse=True,
        )
        if bridge_nodes_vis:
            bridge_rows = []
            for bn in bridge_nodes_vis:
                bd       = G.nodes[bn]
                betype   = bd.get("entity_type", bd.get("node_type","?"))
                blabel   = bd.get("label", bn)
                bdeg     = bd.get("degree", 0)
                arg_cnt  = bd.get("argument_case_count", 0)
                # how many visible cases does this bridge connect to
                linked   = [
                    cn for cn in G.neighbors(bn)
                    if cn != node_key and case_is_visible(G, cn, show_buckets, max_rank)
                ]
                bridge_rows.append(html.Tr([
                    html.Td(_etype_badge(betype)),
                    html.Td(blabel, style={"color": "#e2e8f0", "fontSize": "12px"}),
                    html.Td(str(bdeg), style={"color": "#64748b", "fontSize": "11px"}),
                    html.Td(str(arg_cnt), style={"color": "#f59e0b", "fontSize": "11px"}),
                    html.Td(str(len(linked)), style={"color": "#60a5fa", "fontSize": "11px"}),
                ]))

            bridge_section = html.Div([
                html.H6([
                    html.Span("▸ ", style={"color": "#fbbf24"}),
                    f"Bridge Entities  ({len(bridge_nodes_vis)})",
                ], style={"color": "#cbd5e1", "fontSize": "13px", "marginBottom": "8px"}),
                html.Div(
                    "These are the shared legal entities (statutes, judges, courts …) "
                    "that connect this case to others.",
                    style={"color": "#475569", "fontSize": "11px", "marginBottom": "8px"},
                ),
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Type", style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Entity",  style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Degree",  style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Arg.Cit.",style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Links to", style={"color": "#64748b", "fontSize": "10px"}),
                        ])),
                        html.Tbody(bridge_rows),
                    ],
                    bordered=False, hover=True, size="sm",
                    className="table-dark",
                    style={"fontSize": "11px", "color": "#cbd5e1", "backgroundColor": "#0f172a"},
                ),
            ], style={
                "backgroundColor": "#0a1628",
                "border": "1px solid #1e293b",
                "borderRadius": "8px",
                "padding": "14px",
                "marginBottom": "18px",
            })
        else:
            bridge_section = html.Div()

        # ── SECTION 2: Connected cases ─────────────────────────────────────
        n_conn     = len(focus["connected_cases"])
        n_capped   = min(n_conn, 100)          # show at most 100 in modal
        sorted_cases = sorted(
            focus["case_links"].items(),
            key=lambda item: (-len(item[1]), -G.nodes[item[0]].get("connectivity_score", 0)),
        )

        case_rows = []
        for case_node, via_nodes in sorted_cases[:100]:
            cd        = G.nodes[case_node]
            case_bkt  = cd.get("bucket", "")
            case_lbl  = cd.get("label", case_node)
            bkt_color = bucket_colors.get(case_bkt, "#334155")

            # group via-nodes by entity type
            via_by_type: dict[str, list[str]] = defaultdict(list)
            for vn in via_nodes:
                vd    = G.nodes[vn]
                etype = vd.get("entity_type", vd.get("node_type", "?"))
                via_by_type[etype].append(vd.get("label", vn))

            via_cell = html.Div([
                html.Div([
                    _etype_badge(etype),
                    html.Span(
                        ", ".join(lbl[:40] for lbl in labels[:4])
                        + (f"  +{len(labels)-4}" if len(labels) > 4 else ""),
                        style={"color": "#94a3b8", "fontSize": "10px"},
                    ),
                ], style={"marginBottom": "2px"})
                for etype, labels in sorted(via_by_type.items())
            ])

            case_rows.append(html.Tr([
                html.Td(
                    html.Span("→ " + case_lbl[:55], title=case_lbl),
                    style={"color": "#e2e8f0", "fontSize": "11px", "fontWeight": "600",
                           "borderLeft": f"3px solid {bkt_color}", "paddingLeft": "8px"},
                ),
                html.Td(
                    dbc.Badge(case_bkt, style={"backgroundColor": bkt_color, "fontSize": "9px"}),
                ),
                html.Td(
                    str(len(via_nodes)),
                    style={"color": "#60a5fa", "fontWeight": "600", "fontSize": "11px"},
                ),
                html.Td(via_cell),
            ]))

        cap_note = (
            html.Div(
                f"Showing top 100 of {n_conn} connected cases (sorted by shared entities).",
                style={"color": "#475569", "fontSize": "11px", "fontStyle": "italic", "marginTop": "6px"},
            ) if n_conn > 100 else html.Div()
        )

        if case_rows:
            connected_section = html.Div([
                html.H6([
                    html.Span("▸ ", style={"color": "#60a5fa"}),
                    f"Connected Cases  ({n_conn})",
                ], style={"color": "#cbd5e1", "fontSize": "13px", "marginBottom": "8px"}),
                html.Div(
                    "Cases that share at least one legal entity with this case "
                    "(within the current bucket/rank filters).",
                    style={"color": "#475569", "fontSize": "11px", "marginBottom": "8px"},
                ),
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Case",          style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Bucket",        style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Shared ents",   style={"color": "#64748b", "fontSize": "10px"}),
                            html.Th("Via (by type)", style={"color": "#64748b", "fontSize": "10px"}),
                        ])),
                        html.Tbody(case_rows),
                    ],
                    bordered=False, hover=True, size="sm",
                    className="table-dark",
                    style={"fontSize": "11px", "color": "#cbd5e1", "backgroundColor": "#0f172a"},
                ),
                cap_note,
            ], style={
                "backgroundColor": "#0a1628",
                "border": "1px solid #1e293b",
                "borderRadius": "8px",
                "padding": "14px",
            })
        else:
            connected_section = html.Div(
                "No connected cases visible under current filters.",
                style={"color": "#475569", "fontStyle": "italic", "fontSize": "12px"},
            )

        body = html.Div([stats_row, bridge_section, connected_section])

    # ── ENTITY NODE ───────────────────────────────────────────────────────
    else:
        arg_cnt  = d.get("argument_case_count", 0)
        is_cross = len(bkt_counts) > 1

        stats_row = html.Div([
            _chip("type", node_type),
            _chip("degree", str(degree)),
            _chip("arg.citations", str(arg_cnt), "#f59e0b"),
            *([_chip("bridge", "cross-bucket", "#f59e0b")] if is_cross else []),
        ], className="mb-3 d-flex flex-wrap")

        direct_cases = sorted(
            focus["connected_cases"],
            key=lambda n: G.nodes[n].get("connectivity_score", 0),
            reverse=True,
        )

        if direct_cases:
            summary = html.Div(
                f"Directly used in {len(direct_cases)} visible cases"
                + (f" across {len(bkt_counts)} buckets" if is_cross else ""),
                style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "14px"},
            )

            # Group by bucket
            by_bucket: dict[str, list[str]] = defaultdict(list)
            for cn in direct_cases:
                bkt = G.nodes[cn].get("bucket", "")
                by_bucket[bkt].append(cn)

            bucket_sections = []
            for bkt, nodes in sorted(by_bucket.items(), key=lambda x: -len(x[1])):
                bkt_color = bucket_colors.get(bkt, "#334155")
                rows = []
                for cn in nodes[:60]:
                    cd  = G.nodes[cn]
                    lbl = cd.get("label", cn)
                    sc  = cd.get("connectivity_score", 0)
                    rk  = cd.get("connectivity_rank", 0)
                    ea  = G[cn][node_key].get("in_arguments", False) if G.has_edge(cn, node_key) else False
                    rows.append(html.Tr([
                        html.Td(
                            lbl[:70], title=lbl,
                            style={"color": "#e2e8f0", "fontSize": "11px",
                                   "borderLeft": f"3px solid {bkt_color}", "paddingLeft": "8px"},
                        ),
                        html.Td(f"{sc:.0f}", style={"color": "#6366f1", "fontSize": "11px"}),
                        html.Td(f"#{rk}",    style={"color": "#22c55e", "fontSize": "11px"}),
                        html.Td(
                            dbc.Badge("in arguments", color="warning", style={"fontSize": "9px"})
                            if ea else "",
                        ),
                    ]))
                if len(nodes) > 60:
                    rows.append(html.Tr([
                        html.Td(
                            f"… and {len(nodes)-60} more",
                            colSpan=4,
                            style={"color": "#334155", "fontStyle": "italic", "fontSize": "11px"},
                        ),
                    ]))

                bucket_sections.append(html.Div([
                    html.Div([
                        html.Span("█ ", style={"color": bkt_color}),
                        html.Span(bkt, style={"fontWeight": "600", "color": "#cbd5e1"}),
                        html.Span(f"  ({len(nodes)} cases)",
                                  style={"color": "#64748b", "fontSize": "11px"}),
                    ], style={"marginBottom": "6px"}),
                    dbc.Table(
                        [
                            html.Thead(html.Tr([
                                html.Th("Case",       style={"color": "#64748b", "fontSize": "10px"}),
                                html.Th("Conn.Score", style={"color": "#64748b", "fontSize": "10px"}),
                                html.Th("Rank",       style={"color": "#64748b", "fontSize": "10px"}),
                                html.Th("",           style={"color": "#64748b", "fontSize": "10px"}),
                            ])),
                            html.Tbody(rows),
                        ],
                        bordered=False, hover=True, size="sm",
                        className="table-dark",
                        style={"fontSize": "11px", "color": "#cbd5e1",
                               "backgroundColor": "#0f172a", "marginBottom": "0"},
                    ),
                ], style={
                    "backgroundColor": "#0a1628",
                    "border": "1px solid #1e293b",
                    "borderRadius": "8px",
                    "padding": "14px",
                    "marginBottom": "14px",
                }))

            body = html.Div([stats_row, summary, *bucket_sections])
        else:
            body = html.Div([
                stats_row,
                html.Div("No visible cases under current filters.",
                         style={"color": "#475569", "fontStyle": "italic"}),
            ])

    return title_div, body


# ── app layout ────────────────────────────────────────────────────────────────

def create_app(
    G: nx.Graph,
    layout: dict,
    stats: dict,
    connections: dict,
    case_layout: dict,
    config: dict,
) -> dash.Dash:
    bucket_colors = {b["name"]: b["color"] for b in config.get("buckets", [])}
    entity_colors = config.get("entity_colors", {})

    all_node_types = sorted({d.get("node_type", "?") for _, d in G.nodes(data=True)})
    all_buckets    = sorted({
        d.get("bucket", "") for _, d in G.nodes(data=True)
        if d.get("node_type") == "case"
    })
    max_rank_available = max(
        (d.get("connectivity_rank", 1) for _, d in G.nodes(data=True)
         if d.get("node_type") == "case"),
        default=300,
    )
    slider_max   = min(max_rank_available, 300)
    max_deg_val  = stats.get("max_degree", 1)

    # ── pre-build static content (never recomputed) ───────────────────────
    static_pie     = bucket_pie(stats.get("bucket_counts", {}), bucket_colors)
    static_deg     = degree_hist_chart(stats.get("degree_histogram", {"counts": [], "bin_edges": []}))
    static_hubs    = hub_table(stats.get("top_hubs", []), bucket_colors)
    static_bridges = bridge_table(stats.get("bridge_nodes", []), bucket_colors)
    static_cases   = top_cases_table(stats.get("top_cases_by_connectivity", []), bucket_colors)
    static_pairs   = case_connections_table(stats.get("cross_bucket_case_pairs", []), bucket_colors)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Legal Case Graph",
    )

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    HR = html.Hr(style={"borderColor": "#1e293b", "margin": "10px 0"})

    sidebar = dbc.Col([
        html.Div("FILTERS", style={
            "fontSize": "10px", "color": "#475569", "letterSpacing": "1.5px", "marginBottom": "10px",
        }),

        html.Label("View", style={"fontSize": "11px", "color": "#64748b"}),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label": "  Hub Network",      "value": "hub"},
                {"label": "  Case Connections", "value": "case_net"},
            ],
            value="hub",
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block", "color": "#94a3b8", "fontSize": "12px", "marginBottom": "3px"},
        ),
        HR,

        html.Label("Cases per bucket", style={"fontSize": "11px", "color": "#64748b"}),
        html.Div("top N by connectivity score", style={"fontSize": "9px", "color": "#334155", "marginBottom": "4px"}),
        dcc.Slider(
            id="cases-per-bucket",
            min=5, max=slider_max, step=None, value=min(75, slider_max),
            marks={
                5: "5", 25: "25", 50: "50", 75: "75",
                100: "100", 150: "150", 200: "200",
                **({300: "300"} if slider_max >= 300 else {slider_max: str(slider_max)}),
            },
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        HR,

        html.Label("Buckets", style={"fontSize": "11px", "color": "#64748b"}),
        dcc.Checklist(
            id="filter-buckets",
            options=[{
                "label": html.Span(
                    f"  {bkt}",
                    style={"color": bucket_colors.get(bkt, "#94a3b8"), "fontWeight": "600"},
                ),
                "value": bkt,
            } for bkt in all_buckets],
            value=all_buckets,
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "block", "fontSize": "12px", "marginBottom": "3px"},
        ),
        HR,

        html.Div(id="hub-controls", children=[
            html.Label("Node types", style={"fontSize": "11px", "color": "#64748b"}),
            dcc.Checklist(
                id="filter-node-types",
                options=[{"label": f"  {nt}", "value": nt} for nt in all_node_types],
                value=all_node_types,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "color": "#94a3b8", "fontSize": "12px", "marginBottom": "2px"},
            ),
            HR,
            html.Label("Min. degree", style={"fontSize": "11px", "color": "#64748b"}),
            dcc.Slider(
                id="filter-min-degree",
                min=0, max=max(10, min(50, max_deg_val // 10)),
                step=1, value=0, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            HR,
        ]),

        html.Div(id="case-net-controls", children=[
            html.Label("Min. shared entities", style={"fontSize": "11px", "color": "#64748b"}),
            dcc.Slider(
                id="min-shared-entities",
                min=1, max=10, step=1, value=2, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            HR,
        ], style={"display": "none"}),

        html.Label("Search", style={"fontSize": "11px", "color": "#64748b"}),
        dbc.Input(
            id="search-box", type="text", placeholder="e.g. IPC Section 420",
            debounce=True, size="sm",
            style={
                "backgroundColor": "#0f172a", "color": "#e2e8f0",
                "border": "1px solid #334155", "fontSize": "12px",
            },
        ),
        dbc.Button(
            "Clear selection",
            id="clear-focus-btn",
            color="secondary", size="sm",
            className="mt-2 w-100", outline=True,
            style={"fontSize": "11px"},
        ),
        HR,

        html.Div([
            dbc.Badge(f"{stats['total_nodes']:,} nodes",  color="primary",   className="me-1 mb-1", style={"fontSize": "9px"}),
            dbc.Badge(f"{stats['total_edges']:,} edges",  color="secondary", className="me-1 mb-1", style={"fontSize": "9px"}),
            dbc.Badge(f"avg deg {stats['avg_degree']}",   color="info",      className="me-1 mb-1", style={"fontSize": "9px"}),
            dbc.Badge(f"{stats.get('bridge_node_count',0)} bridges", color="warning", className="me-1 mb-1", style={"fontSize": "9px"}),
            dbc.Badge(f"{stats.get('case_connection_count',0):,} case pairs", color="success", className="me-1 mb-1", style={"fontSize": "9px"}),
        ]),
    ],
    width=2,
    style={
        "backgroundColor": "#060d1a",
        "padding": "12px 10px",
        "height": "100vh",
        "overflowY": "auto",
        "position": "sticky",
        "top": 0,
        "borderRight": "1px solid #1e293b",
    })

    # ── GRAPH + TABS PANEL ────────────────────────────────────────────────
    graph_panel = dbc.Col([
        dcc.Graph(
            id="main-graph",
            style={"height": "65vh"},
            config={
                "scrollZoom":    True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        ),
        dbc.Tabs([
            dbc.Tab(
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(figure=static_pie, style={"height": "230px"},
                                  config={"displayModeBar": False}),
                        width=5,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=static_deg, style={"height": "230px"},
                                  config={"displayModeBar": False}),
                        width=7,
                    ),
                ], className="mt-2"),
                label="Overview",
            ),
            dbc.Tab(
                html.Div(static_hubs,    className="mt-2", style={"maxHeight": "230px", "overflowY": "auto"}),
                label="Top Hubs",
            ),
            dbc.Tab(
                html.Div(static_bridges, className="mt-2", style={"maxHeight": "230px", "overflowY": "auto"}),
                label="Bridges",
            ),
            dbc.Tab(
                html.Div(static_cases,   className="mt-2", style={"maxHeight": "230px", "overflowY": "auto"}),
                label="Top Cases",
            ),
            dbc.Tab(
                html.Div(static_pairs,   className="mt-2", style={"maxHeight": "230px", "overflowY": "auto"}),
                label="Case Connections",
            ),
        ], className="mt-1"),
    ],
    width=7,
    style={"backgroundColor": "#0f172a", "padding": "6px 10px"})

    # ── DETAILS PANEL ────────────────────────────────────────────────────
    details_panel = dbc.Col([
        html.Div([
            html.Span("NODE DETAILS", style={
                "fontSize": "10px", "color": "#475569", "letterSpacing": "1.5px",
            }),
            dbc.Button(
                "Full details ↗",
                id="open-detail-modal",
                color="primary", size="sm", outline=True,
                style={"fontSize": "10px", "padding": "1px 8px", "float": "right"},
                disabled=True,   # enabled by callback when a node is selected
            ),
        ], style={"marginBottom": "10px", "overflow": "hidden"}),
        html.Div(
            id="details-panel",
            children=html.Div([
                html.Div("Select a node", style={
                    "color": "#475569", "textAlign": "center",
                    "marginTop": "50px", "fontSize": "13px",
                }),
                html.Div(
                    "Click any node in the graph to explore its connections",
                    style={"color": "#334155", "textAlign": "center",
                           "fontSize": "11px", "marginTop": "6px"},
                ),
            ]),
            style={
                "overflowY": "auto",
                "height":    "calc(100vh - 84px)",
                "paddingRight": "4px",
            },
        ),
    ],
    width=3,
    style={
        "backgroundColor": "#060d1a",
        "padding": "12px 10px",
        "height": "100vh",
        "overflowY": "hidden",
        "position": "sticky",
        "top": 0,
        "borderLeft": "1px solid #1e293b",
    })

    # ── FULL LAYOUT ───────────────────────────────────────────────────────
    app.layout = dbc.Container([
        dcc.Store(id="selected-node-key"),

        # ── Details modal ─────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(
                html.Div(id="detail-modal-title"),
                style={"backgroundColor": "#0a1628", "borderBottom": "1px solid #1e293b"},
                close_button=True,
            ),
            dbc.ModalBody(
                html.Div(id="detail-modal-body"),
                style={
                    "backgroundColor": "#0f172a", "color": "#cbd5e1",
                    "maxHeight": "78vh", "overflowY": "auto",
                    "padding": "18px 20px",
                },
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="close-detail-modal",
                    color="secondary", size="sm", outline=True,
                    style={"fontSize": "11px"},
                ),
                style={"backgroundColor": "#0a1628", "borderTop": "1px solid #1e293b"},
            ),
        ],
        id="detail-modal",
        is_open=False,
        size="xl",
        scrollable=True,
        backdrop=True,
        style={"color": "#cbd5e1"},
        ),
        dbc.Row(dbc.Col(
            html.Div([
                html.Span(
                    "Legal Case Graph Visualiser",
                    style={"fontWeight": "700", "fontSize": "14px", "color": "#e2e8f0"},
                ),
                html.Span(
                    "  —  nodes connected by shared statutes, courts, judges & precedents"
                    "  ·  click a node to see how it links to other cases",
                    style={"fontSize": "11px", "color": "#475569"},
                ),
            ]),
            width=12,
            style={
                "backgroundColor": "#020617",
                "padding": "8px 16px",
                "borderBottom": "1px solid #1e293b",
            },
        )),
        dbc.Row([sidebar, graph_panel, details_panel], className="g-0"),
    ], fluid=True, style={"backgroundColor": "#0f172a", "minHeight": "100vh"})

    # ── CALLBACKS ─────────────────────────────────────────────────────────

    @app.callback(
        Output("hub-controls",      "style"),
        Output("case-net-controls", "style"),
        Input("view-mode", "value"),
    )
    def toggle_controls(view_mode: str):
        if view_mode == "case_net":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("selected-node-key", "data"),
        Input("main-graph",      "clickData"),
        Input("clear-focus-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_selected_node(click_data, _clear):
        trigger = dash.ctx.triggered_id
        if trigger == "clear-focus-btn":
            return None
        node_key = extract_node_key(click_data)
        return node_key if node_key else no_update

    @app.callback(
        Output("main-graph", "figure"),
        Input("view-mode",           "value"),
        Input("filter-node-types",   "value"),
        Input("filter-buckets",      "value"),
        Input("filter-min-degree",   "value"),
        Input("cases-per-bucket",    "value"),
        Input("min-shared-entities", "value"),
        Input("search-box",          "value"),
        Input("selected-node-key",   "data"),
    )
    def update_graph(
        view_mode, show_types, show_buckets,
        min_degree, cases_per_bucket, min_shared, search_text, selected_node_key,
    ):
        show_types   = set(show_types   or [])
        show_buckets = set(show_buckets or [])
        min_degree   = min_degree or 0
        min_shared   = min_shared or 2
        search_text  = search_text or ""
        max_rank     = int(cases_per_bucket) if cases_per_bucket else None

        if view_mode == "case_net" and case_layout:
            return build_case_network_figure(
                G, case_layout, connections,
                bucket_colors, show_buckets, max_rank, min_shared, search_text,
                selected_node_key=selected_node_key,
            )
        return build_figure(
            G, layout, bucket_colors, entity_colors,
            show_types, show_buckets, min_degree, search_text,
            max_rank=max_rank, selected_node_key=selected_node_key,
        )

    @app.callback(
        Output("details-panel",     "children"),
        Output("open-detail-modal", "disabled"),
        Input("selected-node-key",  "data"),
        Input("filter-buckets",     "value"),
        Input("cases-per-bucket",   "value"),
    )
    def update_details(selected_node_key, show_buckets, cases_per_bucket):
        max_rank = int(cases_per_bucket) if cases_per_bucket else None
        panel = render_details_panel(
            G, selected_node_key, bucket_colors, entity_colors,
            set(show_buckets or []), max_rank,
        )
        btn_disabled = not bool(selected_node_key and selected_node_key in G)
        return panel, btn_disabled

    @app.callback(
        Output("detail-modal",       "is_open"),
        Output("detail-modal-title", "children"),
        Output("detail-modal-body",  "children"),
        Input("open-detail-modal",   "n_clicks"),
        Input("close-detail-modal",  "n_clicks"),
        Input("clear-focus-btn",     "n_clicks"),
        Input("selected-node-key",   "data"),
        Input("filter-buckets",      "value"),
        Input("cases-per-bucket",    "value"),
        prevent_initial_call=True,
    )
    def toggle_detail_modal(
        open_clicks, close_clicks, clear_clicks,
        selected_node_key, show_buckets, cases_per_bucket,
    ):
        trigger = dash.ctx.triggered_id

        # Close triggers
        if trigger in {"close-detail-modal", "clear-focus-btn"}:
            return False, no_update, no_update

        # Node changed while modal might be open — refresh content but don't
        # force open (let current is_open state persist via no_update trick)
        if trigger in {"selected-node-key", "filter-buckets", "cases-per-bucket"}:
            # Only refresh if modal is currently open; we can't read is_open
            # without State, so just let it stay — open_clicks will reopen if needed
            return no_update, no_update, no_update

        # Open button clicked
        if trigger == "open-detail-modal":
            if not selected_node_key or selected_node_key not in G:
                return False, no_update, no_update
            max_rank = int(cases_per_bucket) if cases_per_bucket else None
            title, body = render_modal_body(
                G, selected_node_key, bucket_colors, entity_colors,
                set(show_buckets or []), max_rank,
            )
            return True, title, body

        return no_update, no_update, no_update

    return app


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch Legal Case Graph Visualiser")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--port",   type=int, default=None)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    config  = load_config(args.config)
    out_dir = Path(config.get("output_dir", "outputs"))

    if not (out_dir / "graph_sample.pkl").exists():
        raise SystemExit(
            f"ERROR: {out_dir / 'graph_sample.pkl'} not found.\n"
            "Run build_graph.py first:\n"
            "  python build_graph.py --config config.yaml"
        )

    print("Loading graph artefacts …")
    G, layout, stats, connections, case_layout = load_artefacts(out_dir)
    print(f"  Nodes      : {G.number_of_nodes():,}")
    print(f"  Edges      : {G.number_of_edges():,}")
    print(f"  Case pairs : {len(connections):,}")
    print(f"  Case layout: {'yes' if case_layout else 'not found (case network view disabled)'}")

    app = create_app(G, layout, stats, connections, case_layout, config)

    port  = args.port or config.get("app", {}).get("port", 8050)
    host  = config.get("app", {}).get("host", "0.0.0.0")
    debug = config.get("app", {}).get("debug", False)

    print(f"\nDash app running on http://{host}:{port}")
    print("(SSH tunnel: ssh -L 8050:localhost:8050 <server>)")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
