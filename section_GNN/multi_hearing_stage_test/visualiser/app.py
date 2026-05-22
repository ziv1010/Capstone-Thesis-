#!/usr/bin/env python3
"""
Multi-hearing stage test visualiser (Dash).

Reads only from ../outputs/ — no GPU, no model.

Tabs:
  1. Overview            — summary stats, transition counts, full-path Sankey
  2. Transition Explorer — section-sentence deltas + entity-add frequencies
                           per transition (from transition_aggregates.json)
  3. Case Drill-down     — per-case timeline, section deltas, entity adds/removes
                           between consecutive hearings (from per_case_diffs/),
                           plus likely drivers when per_case_factors/ exists

Run:
  micromamba run -n graph_vis python app.py [--port 8050]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, dcc, html, no_update


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_FONT = dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                 size=12, color="#0f172a")
pio.templates["stagevis"] = pio.templates["plotly_white"]
pio.templates["stagevis"].layout.update(
    font=PLOT_FONT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=14, color="#0f172a", family=PLOT_FONT["family"]),
               x=0.02, xanchor="left", pad=dict(l=4, t=4)),
    margin=dict(l=60, r=20, t=46, b=40),
    xaxis=dict(gridcolor="#eef2f7", linecolor="#e5e7eb", zerolinecolor="#cbd5e1"),
    yaxis=dict(gridcolor="#eef2f7", linecolor="#e5e7eb", zerolinecolor="#cbd5e1"),
    colorway=["#4f46e5", "#16a34a", "#dc2626", "#ea580c", "#0284c7", "#a855f7"],
)
pio.templates.default = "stagevis"


EXP_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = EXP_ROOT / "outputs"
ANALYSIS_DIR = OUT_DIR / "analysis"
INFER_DIR = OUT_DIR / "inference"
PER_CASE_DIR = ANALYSIS_DIR / "per_case_diffs"
PER_CASE_FACTORS_DIR = ANALYSIS_DIR / "per_case_factors"
PER_CASE_RAW_FACTORS_DIR = ANALYSIS_DIR / "per_case_raw_outcome_factors"
INPUT_JSON_DIR = EXP_ROOT / "data" / "input_jsons"

ROLE_COLOR = {
    "PREAMBLE": "#6366f1",
    "FAC": "#475569",
    "ARG_PETITIONER": "#16a34a",
    "ARG_RESPONDENT": "#ea580c",
    "ANALYSIS": "#a855f7",
    "RATIO": "#0d9488",
    "RPC": "#db2777",
    "ISSUE": "#dc2626",
    "PRE_RELIED": "#0284c7",
    "PRE_NOT_RELIED": "#9333ea",
    "STA": "#334155",
    "RLC": "#65a30d",
    "NONE": "#94a3b8",
}

PRED_COLOR = {"WIN": "#16a34a", "LOSE": "#dc2626"}
RAW_OUTCOME_COLOR = {"1": "#16a34a", "0": "#ea580c", "-1": "#dc2626"}
RAW_OUTCOME_NAME = {"1": "WIN", "0": "POSTPONED", "-1": "LOSS"}
TRANSITION_COLORS = {
    "LOSE -> WIN": "#4f46e5",
    "LOSE -> LOSE": "#dc2626",
    "WIN -> WIN": "#16a34a",
    "WIN -> LOSE": "#ea580c",
}
# Underlying label_value_map (config.yaml): -1 → LOSE, 0 → LOSE, 1 → WIN.
LABEL_NUMERIC = {"LOSE": "-1/0", "WIN": "1"}
NUMERIC_TO_NAME = {"-1": "LOSE", "0": "LOSE", "1": "WIN"}


def display_label(label: str) -> str:
    """Bucket-form display: 'LOSE' -> 'LOSE (-1/0)', '-1' -> 'LOSE (-1/0)'."""
    if label in NUMERIC_TO_NAME:
        label = NUMERIC_TO_NAME[label]
    suffix = LABEL_NUMERIC.get(label)
    return f"{label} ({suffix})" if suffix else label


def display_with_raw(raw_score) -> str:
    """Model-label display: -1 -> 'LOSE (-1)', 1 -> 'WIN (1)'."""
    s = str(raw_score).strip()
    name = NUMERIC_TO_NAME.get(s, s)
    return f"{name} ({s})" if s else name


def display_outcome_raw(raw_score) -> str:
    """Actual-outcome display: 0 stays visibly distinct from -1."""
    s = str(raw_score).strip()
    name = RAW_OUTCOME_NAME.get(s, s)
    return f"{name} ({s})" if s else name


def raw_outcome_kind(raw_score) -> str:
    s = str(raw_score).strip()
    if s == "1":
        return "win"
    if s == "0":
        return "warn"
    if s == "-1":
        return "lose"
    return "neutral"


def raw_to_bucket(raw_score, fallback: str = "") -> str:
    return NUMERIC_TO_NAME.get(str(raw_score).strip(), fallback)


def shorten_text(text: str, max_chars: int = 78) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def human_case_title(base_id: str, category: str = "", max_chars: int = 78) -> str:
    title = str(base_id)
    prefix = f"{category}__"
    if category and title.startswith(prefix):
        title = title[len(prefix):]
    if title.endswith("_MERGED"):
        title = title[:-7]
    title = title.replace("_", " ")
    return shorten_text(title, max_chars=max_chars)


def compact_path(items: list[str], formatter=str) -> str:
    if not items:
        return "?"
    if len(items) <= 2:
        return " -> ".join(formatter(v) for v in items)
    return f"{formatter(items[0])} -> {formatter(items[-1])} ({len(items)} hearings)"


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw_outcome_score(base_id: str, stage_index: int, fallback: str = "") -> str:
    path = INPUT_JSON_DIR / f"STAGE{stage_index}__{base_id}.json"
    if not path.exists():
        return str(fallback)
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception:
        return str(fallback)
    value = doc.get("case_outcome_score", fallback)
    return str(value)


def load_data() -> dict:
    transitions = pd.read_csv(ANALYSIS_DIR / "stage_transitions.csv")
    predictions = pd.read_csv(INFER_DIR / "predictions.csv")
    with open(ANALYSIS_DIR / "summary.json") as f:
        summary = json.load(f)
    with open(ANALYSIS_DIR / "transition_counts.json") as f:
        transition_counts = json.load(f)
    aggregates_path = ANALYSIS_DIR / "transition_aggregates.json"
    aggregates = {}
    if aggregates_path.exists():
        with open(aggregates_path) as f:
            aggregates = json.load(f)
    raw_label_lookup: dict[tuple[str, int], str] = {}
    for row in predictions.itertuples():
        stage_index = int(row.stage_index)
        raw_label_lookup[(row.base_case_id, stage_index)] = load_raw_outcome_score(
            row.base_case_id,
            stage_index,
            fallback=str(row.raw_label),
        )
    pred_label_lookup: dict[tuple[str, int], str] = {}
    for row in predictions.itertuples():
        pred_label_lookup[(row.base_case_id, int(row.stage_index))] = str(row.pred_label)

    case_option_meta = {}
    case_options = []
    for r in transitions.sort_values(["true_label", "transition", "base_case_id"]).itertuples():
        stage_indices = range(1, int(r.n_stages) + 1)
        raw_path = [raw_label_lookup.get((r.base_case_id, i), "") for i in stage_indices]
        raw_path = [v for v in raw_path if v != ""]
        final_raw = raw_path[-1] if raw_path else ""
        raw_path_text = " -> ".join(display_outcome_raw(v) for v in raw_path) or "raw ?"
        compact_raw_path = compact_path(raw_path, display_outcome_raw)
        pred_parts = [p.strip() for p in str(r.transition).split("->") if p.strip()]
        compact_pred_path = compact_path(pred_parts)
        status = "changed" if bool(r.changed_prediction) else "stable"
        case_title = human_case_title(r.base_case_id, r.category, max_chars=58)
        compact_label = (
            f"{compact_raw_path} | pred {compact_pred_path} | {status} | {case_title}"
        )
        search_label = (
            f"actual {raw_path_text} | pred {r.transition} | {status} | "
            f"{r.category} | {case_title} | {r.base_case_id}"
        )
        option = {
            "label": compact_label,
            "value": r.base_case_id,
            "search": search_label,
            "title": search_label,
        }
        case_options.append(option)
        case_option_meta[r.base_case_id] = {
            "option": option,
            "case_title": case_title,
            "raw_path": raw_path,
            "final_raw": final_raw,
            "changed_prediction": bool(r.changed_prediction),
            "transition": r.transition,
        }
    return {
        "transitions": transitions,
        "predictions": predictions,
        "summary": summary,
        "transition_counts": transition_counts,
        "aggregates": aggregates,
        "case_options": case_options,
        "case_option_meta": case_option_meta,
        "raw_label_lookup": raw_label_lookup,
        "pred_label_lookup": pred_label_lookup,
    }


def load_per_case(base_id: str) -> dict | None:
    path = PER_CASE_DIR / f"{base_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_per_case_factors(base_id: str) -> dict | None:
    path = PER_CASE_FACTORS_DIR / f"{base_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    payload.setdefault("factor_basis", "prediction")
    return payload


def load_per_case_raw_factors(base_id: str) -> dict | None:
    path = PER_CASE_RAW_FACTORS_DIR / f"{base_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_best_factor_report(base_id: str) -> dict | None:
    prediction_report = load_per_case_factors(base_id)
    if prediction_report:
        return prediction_report
    return load_per_case_raw_factors(base_id)


def load_stage_doc(base_id: str, stage_index: int) -> dict | None:
    path = INPUT_JSON_DIR / f"STAGE{stage_index}__{base_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── overview tab figures ──────────────────────────────────────────────────────

def transition_count_bar(counts: dict) -> go.Figure:
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [TRANSITION_COLORS.get(k, "#94a3b8") for k in labels]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=vals, textposition="outside",
                           textfont=dict(color="#334155", size=11)))
    fig.update_layout(
        title="Transition counts",
        height=380, margin=dict(l=160, r=40, t=46, b=30),
        xaxis_title="cases", yaxis=dict(autorange="reversed"),
        bargap=0.32,
    )
    return fig


def stage_sankey(transitions: pd.DataFrame) -> go.Figure:
    """Full-path Sankey: hearing 1 -> hearing 2 (-> hearing 3 (-> hearing 4))."""
    max_stages = int(transitions["n_stages"].max())
    node_labels: list[str] = []
    node_index: dict[tuple[int, str], int] = {}

    def node_id(stage: int, label: str) -> int:
        key = (stage, label)
        if key not in node_index:
            node_index[key] = len(node_labels)
            node_labels.append(f"H{stage}: {display_label(label)}")
        return node_index[key]

    flows: Counter = Counter()
    for _, row in transitions.iterrows():
        labels = []
        for i in range(1, int(row["n_stages"]) + 1):
            v = row.get(f"stage{i}_pred")
            if isinstance(v, str) and v:
                labels.append(v)
        for i in range(len(labels) - 1):
            a = node_id(i + 1, labels[i])
            b = node_id(i + 2, labels[i + 1])
            flows[(a, b)] += 1

    sources = [s for (s, _), _ in flows.items()]
    targets = [t for (_, t), _ in flows.items()]
    values = [v for _, v in flows.items()]

    def class_of(name: str) -> str:
        body = name.split(": ", 1)[-1]
        for k in PRED_COLOR:
            if body.startswith(k):
                return k
        return ""

    def label_color(name: str) -> str:
        return PRED_COLOR.get(class_of(name), "#607d8b")

    link_colors = []
    for (s, t), _ in flows.items():
        sc = class_of(node_labels[s])
        tc = class_of(node_labels[t])
        if sc == "LOSE" and tc == "WIN":
            link_colors.append("rgba(25,118,210,0.45)")
        elif sc == "WIN" and tc == "LOSE":
            link_colors.append("rgba(239,108,0,0.45)")
        elif sc == tc == "WIN":
            link_colors.append("rgba(46,125,50,0.35)")
        else:
            link_colors.append("rgba(198,40,40,0.35)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            color=[label_color(n) for n in node_labels],
            pad=22, thickness=18,
            line=dict(color="rgba(15,23,42,0.08)", width=1),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(title=f"Prediction flow across hearings (max {max_stages} hearings)",
                      height=420, margin=dict(l=10, r=10, t=46, b=10),
                      font=dict(family=PLOT_FONT["family"], size=12, color="#0f172a"))
    return fig


def confidence_scatter(predictions: pd.DataFrame) -> go.Figure:
    """Per-hearing confidence box/strip per pred_label, faceted by stage_index."""
    fig = go.Figure()
    for stage, sub in predictions.groupby("stage_index"):
        fig.add_trace(go.Box(
            x=sub["pred_label"].astype(str).map(display_label),
            y=sub["confidence"],
            name=f"hearing {int(stage)}",
            boxpoints="outliers",
            marker=dict(size=4, opacity=0.6),
        ))
    fig.update_layout(
        title="Confidence distribution by hearing",
        boxmode="group", height=380,
        xaxis_title="pred_label", yaxis_title="confidence",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def raw_outcome_distribution_bar(predictions: pd.DataFrame,
                                 raw_lookup: dict[tuple[str, int], str] | None = None) -> go.Figure:
    """Final-hearing actual raw outcome counts; 0 and -1 remain separate."""
    final_rows = (
        predictions.sort_values(["base_case_id", "stage_index"])
        .groupby("base_case_id", as_index=False)
        .tail(1)
    )
    raw_values = []
    for row in final_rows.itertuples():
        stage_index = int(row.stage_index)
        raw_values.append(
            raw_lookup.get((row.base_case_id, stage_index), str(row.raw_label))
            if raw_lookup is not None else str(row.raw_label)
        )
    counts = pd.Series(raw_values).astype(str).value_counts().to_dict()
    order = ["0", "-1", "1"]
    labels = [display_outcome_raw(v) for v in order if v in counts]
    vals = [counts[v] for v in order if v in counts]
    colors = [RAW_OUTCOME_COLOR.get(v, "#64748b") for v in order if v in counts]

    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker_color=colors,
        text=vals,
        textposition="outside",
        textfont=dict(color="#334155", size=11),
        customdata=[raw_to_bucket(v) for v in order if v in counts],
        hovertemplate="actual %{x}<br>model bucket %{customdata}<br>cases=%{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Actual raw outcome distribution (final hearing)",
        height=300,
        margin=dict(l=55, r=20, t=46, b=60),
        xaxis_title="actual raw outcome",
        yaxis_title="cases",
        bargap=0.35,
    )
    return fig


# ── transition explorer figures ───────────────────────────────────────────────

def section_delta_bar(agg_for_transition: dict) -> go.Figure:
    sec = agg_for_transition.get("section_sentence_delta", {})
    if not sec:
        return go.Figure().update_layout(title="No section-delta data")
    items = sorted(sec.items(), key=lambda kv: -abs(kv[1].get("mean", 0)))
    roles = [k for k, _ in items]
    means = [v.get("mean", 0) for _, v in items]
    fracs = [v.get("frac_cases_with_change", 0) for _, v in items]
    colors = ["#16a34a" if m > 0 else ("#dc2626" if m < 0 else "#94a3b8") for m in means]
    text = [f"μ={m:+.2f}  ({f*100:.0f}% changed)" for m, f in zip(means, fracs)]
    fig = go.Figure(go.Bar(
        x=means, y=roles, orientation="h",
        marker_color=colors, text=text, textposition="outside",
        textfont=dict(color="#334155", size=11),
    ))
    fig.update_layout(
        title="Mean Δ sentences per rhetorical role (hearing_to − hearing_from)",
        height=420, margin=dict(l=120, r=190, t=46, b=30),
        xaxis_title="Δ sentences", yaxis=dict(autorange="reversed"),
        bargap=0.3,
    )
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    return fig


def entity_add_freq_bar(agg_for_transition: dict, top_n: int = 12) -> go.Figure:
    ent = agg_for_transition.get("entity_label_add_freq", {})
    if not ent:
        return go.Figure().update_layout(title="No entity-add data")
    items = sorted(ent.items(), key=lambda kv: -kv[1].get("count", 0))[:top_n]
    labels = [k for k, _ in items]
    counts = [v.get("count", 0) for _, v in items]
    fracs = [v.get("frac", 0) for _, v in items]
    text = [f"{c} ({f*100:.0f}%)" for c, f in zip(counts, fracs)]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h",
                           marker_color="#4f46e5",
                           text=text, textposition="outside",
                           textfont=dict(color="#334155", size=11)))
    fig.update_layout(
        title="Entity labels with newly-added entities (across cases)",
        height=420, margin=dict(l=140, r=140, t=46, b=30),
        xaxis_title="cases with ≥1 added entity of this label",
        yaxis=dict(autorange="reversed"),
        bargap=0.3,
    )
    return fig


def factor_score_bar(factor_report: dict) -> go.Figure:
    factors = factor_report.get("top_decisive_factors", []) or []
    if not factors:
        return go.Figure().update_layout(title="No scored factors for this case",
                                         height=300)

    items = []
    for f in factors:
        label = str(f.get("label", ""))
        entity = str(f.get("entity", ""))
        score = float(f.get("discriminative_score", 0.0) or 0.0)
        short = entity if len(entity) <= 46 else entity[:43] + "..."
        items.append({
            "name": f"{label}: {short}",
            "score": score,
            "entity": entity,
            "label": label,
        })
    items = sorted(items, key=lambda x: abs(x["score"]), reverse=True)[:10]
    items = list(reversed(items))
    colors = ["#16a34a" if x["score"] > 0 else ("#dc2626" if x["score"] < 0 else "#64748b")
              for x in items]

    fig = go.Figure(go.Bar(
        x=[x["score"] for x in items],
        y=[x["name"] for x in items],
        orientation="h",
        marker_color=colors,
        customdata=[[x["label"], x["entity"]] for x in items],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>score=%{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    basis = "actual-outcome" if factor_report.get("factor_basis") == "raw_outcome" else "prediction"
    fig.update_layout(
        title=f"Top added factors by {basis} transition contrast score",
        height=max(300, 34 * len(items) + 100),
        margin=dict(l=235, r=35, t=46, b=35),
        xaxis_title="higher = more specific to this transition than its control",
        yaxis=dict(automargin=True),
        bargap=0.32,
    )
    return fig


def first_last_section_delta_bar(factor_report: dict) -> go.Figure:
    sec = factor_report.get("section_sentence_delta_first_to_last", {}) or {}
    if not sec:
        return go.Figure().update_layout(title="No first-to-last section delta",
                                         height=280)
    items = sorted(sec.items(), key=lambda kv: abs(kv[1]))[-12:]
    roles = [k for k, _ in items]
    vals = [int(v) for _, v in items]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=roles,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+d}" for v in vals],
        textposition="outside",
        textfont=dict(color="#334155", size=11),
    ))
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    fig.update_layout(
        title="First-to-last hearing section delta",
        height=max(280, 32 * len(items) + 100),
        margin=dict(l=130, r=50, t=46, b=35),
        xaxis_title="sentences added / removed",
        bargap=0.32,
    )
    return fig


# ── case drill-down rendering ─────────────────────────────────────────────────

def stage_text_panel(base_id: str, stage_index: int, date: str,
                     pred: str, true_label: str) -> dbc.AccordionItem:
    """Accordion item showing the source sentences of one hearing, grouped by role."""
    doc = load_stage_doc(base_id, stage_index)
    if doc is None:
        body = html.Div(
            f"Source JSON not found at {INPUT_JSON_DIR.relative_to(EXP_ROOT)}/"
            f"STAGE{stage_index}__{base_id}.json",
            style={"color": "#c62828"},
        )
        title = f"Hearing {stage_index} — (file missing)"
        return dbc.AccordionItem(body, title=title)

    sentences = doc.get("sentences", [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in sentences:
        role = s.get("rhetorical_role", "NONE") or "NONE"
        grouped[role].append(s)

    role_order = [
        "PREAMBLE", "FAC", "ISSUE",
        "ARG_PETITIONER", "ARG_RESPONDENT",
        "PRE_RELIED", "PRE_NOT_RELIED", "STA",
        "ANALYSIS", "RATIO", "RPC", "RLC", "NONE",
    ]
    seen_roles = [r for r in role_order if r in grouped] + \
                 [r for r in grouped if r not in role_order]

    blocks: list = []
    for role in seen_roles:
        items = grouped[role]
        color = ROLE_COLOR.get(role, "#94a3b8")
        header = html.Div([
            html.Span(role,
                      style={"display": "inline-block",
                             "padding": "2px 10px",
                             "borderRadius": "999px",
                             "background": color,
                             "color": "white",
                             "fontWeight": 600,
                             "fontSize": "0.74rem",
                             "letterSpacing": "0.02em"}),
            html.Span(f"  {len(items)} sentence{'s' if len(items) != 1 else ''}",
                      className="muted tiny",
                      style={"marginLeft": "8px"}),
        ], style={"marginBottom": "6px", "marginTop": "12px"})
        sentence_items = [
            html.Li(
                [
                    html.Span(f"#{s.get('sentence_id', '?')}  ",
                              style={"color": "#888", "fontSize": "0.75rem"}),
                    html.Span(s.get("text", "").strip()),
                ],
                style={"marginBottom": "4px", "fontSize": "0.85rem",
                       "lineHeight": "1.35"},
            )
            for s in items
        ]
        blocks.append(html.Div([
            header,
            html.Ul(sentence_items, style={"paddingLeft": "20px",
                                            "marginBottom": "0"}),
        ]))

    raw_score = doc.get("case_outcome_score", "")
    raw_label = doc.get("case_outcome_label", "")
    meta = html.Div([
        html.Span(f"file_id: {doc.get('file_id', '')}",
                  style={"color": "#666", "fontSize": "0.8rem", "marginRight": "12px"}),
        html.Span(f"raw outcome_score: {raw_score} ({raw_label})",
                  style={"color": "#666", "fontSize": "0.8rem", "marginRight": "12px"}),
        html.Span(f"sentences: {len(sentences)}",
                  style={"color": "#666", "fontSize": "0.8rem"}),
    ], style={"marginBottom": "8px"})

    actual = display_outcome_raw(raw_score) if str(raw_score).strip() else display_label(true_label)
    title = (f"Hearing {stage_index}  ·  {date}  ·  "
             f"pred {display_label(pred)}  ·  actual {actual}")
    return dbc.AccordionItem([meta] + blocks, title=title,
                             item_id=f"stage-{stage_index}")


def _pill(text: str, kind: str) -> html.Span:
    """kind: 'win' | 'lose' | 'warn' | 'info' | 'neutral'."""
    return html.Span(text, className=f"pill pill-{kind}")


def _label_pill(label: str, raw: str | None = None) -> html.Span:
    """Render a label badge from a name ('WIN'/'LOSE') or raw score, with semantic class."""
    if raw is not None:
        text = display_with_raw(raw)
        bucket = raw_to_bucket(raw, label)
    else:
        text = display_label(label)
        bucket = raw_to_bucket(label, label)
    kind = "win" if bucket == "WIN" else ("lose" if bucket == "LOSE" else "neutral")
    return _pill(text, kind)


def _raw_outcome_pill(raw: str | None, fallback_label: str = "") -> html.Span:
    if raw is None:
        return _label_pill(fallback_label)
    return _pill(display_outcome_raw(raw), raw_outcome_kind(raw))


def stage_card(stage: dict, true_label: str,
               raw_score: str | None = None,
               pred_raw: str | None = None) -> html.Div:
    pred = stage.get("pred_label", "?")
    conf = stage.get("confidence", 0)
    if raw_score is not None and str(raw_score) in NUMERIC_TO_NAME:
        true_bucket = raw_to_bucket(raw_score)
    else:
        true_bucket = true_label
    is_correct = (pred == true_bucket)

    sec = stage.get("section_sentence_counts", {}) or {}
    ents_by_label = stage.get("entities_by_label", {}) or {}
    sec_rows = sorted(sec.items(), key=lambda kv: -kv[1])
    ent_summary = ", ".join(f"{lbl}:{len(v)}" for lbl, v in
                            sorted(ents_by_label.items(), key=lambda kv: -len(kv[1])))
    card_kind = "win" if pred == "WIN" else "lose"

    head = html.Div([
        html.Span(f"Hearing {stage.get('stage_index', '?')}", className="stage-num"),
        html.Span(stage.get("date", "—"), className="stage-date"),
        html.Span(className="stage-spacer"),
        html.Span("pred", className="muted tiny", style={"marginRight": "4px"}),
        _label_pill(pred, pred_raw),
        html.Span(f"{conf:.3f}", className="mono tiny",
                  style={"marginLeft": "6px", "color": "#334155"}),
        html.Span("actual", className="muted tiny",
                  style={"marginLeft": "12px", "marginRight": "4px"}),
        _raw_outcome_pill(raw_score, true_label),
        html.Span("✓" if is_correct else "✗",
                  style={"marginLeft": "8px", "fontWeight": 700,
                         "color": "#16a34a" if is_correct else "#dc2626",
                         "fontSize": "0.95rem"}),
    ], className="stage-head")

    body = html.Div([
        html.Div([html.B("Sections  "),
                  ", ".join(f"{k}:{v}" for k, v in sec_rows) or "—"],
                 style={"marginBottom": "4px"}),
        html.Div([html.B("Entities  "), ent_summary or "—"]),
    ], className="stage-meta")

    return html.Div([head, body], className=f"stage-card {card_kind}")


def diff_card(diff: dict, true_label: str,
              from_raw: str | None = None,
              to_raw: str | None = None,
              from_pred_raw: str | None = None,
              to_pred_raw: str | None = None) -> dbc.Card:
    sec_delta = diff.get("section_sentence_count_delta", {}) or {}
    sec_items = sorted(sec_delta.items(), key=lambda kv: -abs(kv[1]))
    sec_chips = [
        html.Span(f"{k} {v:+d}",
                  className="chip " + ("chip-up" if v > 0 else "chip-down"))
        for k, v in sec_items
    ]

    ent_diff = diff.get("entity_diff", {}) or {}
    ent_blocks = []
    for label in sorted(ent_diff):
        added = ent_diff[label].get("added", []) or []
        removed = ent_diff[label].get("removed", []) or []
        if not added and not removed:
            continue
        children = [html.B(label, style={"marginRight": "10px",
                                           "color": "#0f172a",
                                           "fontSize": "0.82rem"})]
        if added:
            children.append(html.Span("+ ", style={"color": "#16a34a", "fontWeight": 600}))
            children.append(html.Span(", ".join(added[:8]) + (" …" if len(added) > 8 else ""),
                                      style={"color": "#15803d"}))
        if removed:
            children.append(html.Span("   − ", style={"color": "#dc2626",
                                                        "fontWeight": 600,
                                                        "marginLeft": "8px"}))
            children.append(html.Span(", ".join(removed[:8]) + (" …" if len(removed) > 8 else ""),
                                      style={"color": "#b91c1c"}))
        ent_blocks.append(html.Div(children, style={"fontSize": "0.85rem",
                                                      "marginBottom": "6px",
                                                      "lineHeight": "1.5"}))

    from_pred = diff["from_pred"]
    to_pred = diff["to_pred"]
    if from_pred == to_pred:
        direction_note, direction_kind = "unchanged", "neutral"
    elif to_pred == true_label:
        direction_note, direction_kind = "→ toward truth", "win"
    else:
        direction_note, direction_kind = "→ away from truth", "lose"

    pred_text = (f"{display_with_raw(from_pred_raw)} → {display_with_raw(to_pred_raw)}"
                 if from_pred_raw is not None and to_pred_raw is not None
                 else f"{display_label(from_pred)} → {display_label(to_pred)}")
    pred_kind = "win" if to_pred == "WIN" else "lose"

    if from_raw is not None and to_raw is not None:
        if str(from_raw) == str(to_raw):
            true_text = display_outcome_raw(from_raw)
        else:
            true_text = f"{display_outcome_raw(from_raw)} → {display_outcome_raw(to_raw)}"
        true_kind = raw_outcome_kind(to_raw)
    else:
        true_text = display_label(true_label)
        true_kind = "win" if true_label == "WIN" else "lose"

    head = html.Div([
        html.Span(f"Hearing {diff['from_stage']} → {diff['to_stage']}",
                  style={"fontWeight": 700, "fontSize": "0.92rem", "color": "#0f172a"}),
        html.Span(f"{diff.get('from_date', '')} → {diff.get('to_date', '')}",
                  className="mono tiny muted"),
        html.Span("pred", className="muted tiny",
                  style={"marginLeft": "6px", "marginRight": "4px"}),
        _pill(pred_text, pred_kind),
        html.Span("actual", className="muted tiny",
                  style={"marginLeft": "6px", "marginRight": "4px"}),
        _pill(true_text, true_kind),
        _pill(direction_note, direction_kind),
    ], className="diff-head")

    body = html.Div([
        html.Div("Section sentence deltas", className="section-eyebrow",
                 style={"margin": "0 0 6px 0"}),
        html.Div(sec_chips or html.Span("no change", className="muted tiny"),
                 style={"marginBottom": "12px"}),
        html.Div("Entity changes", className="section-eyebrow",
                 style={"margin": "0 0 6px 0"}),
        html.Div(ent_blocks or html.Span("no change", className="muted tiny")),
    ], className="diff-body")

    return html.Div([head, body], className="diff-card")


def role_badge(role: str) -> html.Span:
    return html.Span(role,
                     style={"display": "inline-block",
                            "padding": "2px 10px",
                            "borderRadius": "999px",
                            "background": ROLE_COLOR.get(role, "#64748b"),
                            "color": "white",
                            "fontWeight": 600,
                            "fontSize": "0.72rem",
                            "letterSpacing": "0.02em"})


def entity_chip(entity: dict) -> html.Span:
    label = str(entity.get("label", "") or "").strip()
    text = str(entity.get("text", "") or "").strip()
    if not text:
        return html.Span()
    return html.Span(f"{label}: {text}" if label else text,
                     className="entity-chip")


def factor_report_panel(factor_report: dict | None) -> html.Div:
    if not factor_report:
        return html.Div(
            "No per-case factor report found. Run scripts/05b_aggregate_transitions.py, "
            "scripts/05c_per_case_factors.py, and scripts/05d_raw_outcome_factors.py "
            "to populate this panel.",
            className="empty-note",
        )

    basis = factor_report.get("factor_basis", "prediction")
    if basis == "raw_outcome":
        basis_label = "Actual raw outcome"
        transition_label = factor_report.get("raw_outcome_transition", factor_report.get("transition", ""))
        support_bits = [
            _pill("raw outcome analysis", "warn"),
            _pill(f"actual {transition_label}", "warn"),
        ]
        prediction_transition = factor_report.get("prediction_transition")
        if prediction_transition:
            support_bits.append(_pill(f"pred {prediction_transition}", "neutral"))
    else:
        basis_label = "Model prediction"
        transition_label = factor_report.get("transition", "")
        support_bits = [
            _pill("prediction analysis", "info"),
            _pill(f"pred {transition_label}", "info"),
        ]

    contrast = factor_report.get("contrast_transition")
    if contrast:
        support_bits.append(_pill(f"contrast {contrast}", "neutral"))
    n_t = factor_report.get("n_transition_cases")
    n_c = factor_report.get("n_contrast_cases")
    if n_t or n_c:
        support_bits.append(_pill(f"support {n_t}/{n_c}", "neutral"))

    factors = factor_report.get("top_decisive_factors", []) or []
    factor_rows = []
    for f in factors[:10]:
        score = float(f.get("discriminative_score", 0.0) or 0.0)
        score_kind = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
        factor_rows.append(html.Div([
            html.Span(f.get("label", ""), className="factor-label"),
            html.Span(f.get("entity", ""), className="factor-entity"),
            html.Span(f"{score:+.4f}", className=f"factor-score {score_kind}"),
        ], className="factor-row"))

    anchor_blocks = []
    for s in (factor_report.get("anchor_sentences", []) or [])[:8]:
        matched = [entity_chip(e) for e in s.get("matched_entities", [])[:6]]
        role = s.get("rhetorical_role", "ROLE")
        anchor_blocks.append(html.Div([
            html.Div([
                role_badge(role),
                html.Span(matched, className="sentence-entities"),
            ], className="sentence-head"),
            html.Div(s.get("text", ""), className="sentence-text"),
        ], className="sentence-block"))

    new_sentence_blocks = []
    for s in (factor_report.get("new_decision_role_sentences", []) or [])[:10]:
        ents = [entity_chip(e) for e in s.get("entities", [])[:6]]
        role = s.get("rhetorical_role", "ROLE")
        new_sentence_blocks.append(html.Div([
            html.Div([
                role_badge(role),
                html.Span(ents, className="sentence-entities"),
            ], className="sentence-head"),
            html.Div(s.get("text", ""), className="sentence-text"),
        ], className="sentence-block"))

    return html.Div([
        html.Div([
            html.Div(basis_label, className="section-eyebrow",
                     style={"margin": "0 0 6px 0"}),
            html.Div(support_bits, className="factor-basis-row"),
        ], className="factor-basis-card"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=factor_score_bar(factor_report),
                                       config={"displayModeBar": False}),
                             className="plot-card"), md=7),
            dbc.Col(html.Div(dcc.Graph(figure=first_last_section_delta_bar(factor_report),
                                       config={"displayModeBar": False}),
                             className="plot-card"), md=5),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div("Ranked added factors", className="section-title"),
                html.Div(factor_rows or html.Div("No added factors ranked.",
                                                 className="empty-note"),
                         className="factor-table"),
            ], md=5),
            dbc.Col([
                html.Div("Anchor sentences", className="section-title"),
                html.Div(anchor_blocks or html.Div(
                    "No anchor sentences matched the ranked factors.",
                    className="empty-note")),
            ], md=7),
        ], className="factor-row-wrap"),
        html.Div("New decision-role sentences", className="section-title"),
        html.Div(new_sentence_blocks or html.Div("No new decision-role sentences found.",
                                                 className="empty-note")),
    ], className="factor-panel")


# ── layout ────────────────────────────────────────────────────────────────────

def build_layout(data: dict) -> html.Div:
    summary = data["summary"]
    transition_options = [
        {"label": k, "value": k} for k in data["aggregates"].get("by_transition", {}).keys()
    ] or [{"label": k, "value": k} for k in data["transition_counts"].keys()]

    n_cases = summary["n_cases"]
    final_acc = summary["final_pred_accuracy"]
    changed = summary["cases_with_changed_prediction"]
    stages_dist = summary["n_stages_distribution"]
    final_raw_counts = (
        data["predictions"].sort_values(["base_case_id", "stage_index"])
        .groupby("base_case_id", as_index=False)
        .tail(1)
    )
    final_raw_counts = pd.Series([
        data["raw_label_lookup"].get((row.base_case_id, int(row.stage_index)), str(row.raw_label))
        for row in final_raw_counts.itertuples()
    ]).astype(str).value_counts().to_dict()

    def stat(label, value, detail, kind=""):
        return html.Div([
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(detail, className="stat-detail"),
        ], className=f"stat-card {kind}")

    stat_grid = html.Div([
        stat("Cases analysed", f"{n_cases:,}", f"{len(data['predictions'])} hearing rows total"),
        stat("Final-pred accuracy", f"{final_acc:.1%}",
             f"{int(round(final_acc*n_cases))} / {n_cases} correct at last hearing",
             kind="success"),
        stat("Changed prediction", f"{changed}",
             f"{changed/n_cases:.0%} of cases shifted at least once",
             kind="warning"),
        stat("LOSE bucket split",
             f"0:{final_raw_counts.get('0', 0)}  -1:{final_raw_counts.get('-1', 0)}",
             "actual POSTPONED (0) and LOSS (-1) both map to model LOSE",
             kind="muted"),
        stat("Hearings distribution",
             " · ".join(f"{k}×{v}" for k, v in stages_dist.items()),
             f"max {max(int(k) for k in stages_dist)} hearings",
             kind="info"),
    ], className="stat-grid")

    overview_tab = html.Div([
        stat_grid,
        html.Div("Prediction flow", className="section-eyebrow"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=stage_sankey(data["transitions"]),
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=7),
            dbc.Col(html.Div(dcc.Graph(figure=transition_count_bar(data["transition_counts"]),
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=5),
        ]),
        html.Div("Model confidence", className="section-eyebrow"),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=confidence_scatter(data["predictions"]),
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=7),
            dbc.Col(html.Div(dcc.Graph(figure=raw_outcome_distribution_bar(
                                            data["predictions"],
                                            data["raw_label_lookup"],
                                        ),
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=5),
        ]),
    ])

    transition_tab = html.Div([
        html.Div("Transition", className="section-eyebrow"),
        html.Div([
            html.Div(dcc.Dropdown(
                id="transition-picker",
                options=transition_options,
                value=transition_options[0]["value"] if transition_options else None,
                clearable=False,
            ), style={"flex": "0 0 320px"}),
            html.Div(id="transition-meta",
                     style={"marginLeft": "16px", "color": "#475569",
                            "fontSize": "0.9rem", "alignSelf": "center"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "16px", "gap": "12px"}),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id="transition-section-delta",
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=6),
            dbc.Col(html.Div(dcc.Graph(id="transition-entity-freq",
                                        config={"displayModeBar": False}),
                              className="plot-card"), md=6),
        ]),
    ])

    case_tab = html.Div([
        html.Div("Case", className="section-eyebrow"),
        html.Div([
            html.Div("Actual outcome", className="filter-label"),
            dcc.RadioItems(
                id="raw-outcome-filter",
                options=[
                    {"label": "All", "value": "ALL"},
                    {"label": "POSTPONED (0)", "value": "0"},
                    {"label": "LOSS (-1)", "value": "-1"},
                    {"label": "WIN (1)", "value": "1"},
                ],
                value="ALL",
                inline=True,
                className="filter-pills",
            ),
        ], className="case-filter-row"),
        dcc.Dropdown(
            id="case-picker",
            options=data["case_options"],
            value=data["case_options"][0]["value"] if data["case_options"] else None,
            clearable=False,
            placeholder="Search by raw outcome, transition, category, or case id",
            optionHeight=42,
            style={"marginBottom": "16px"},
        ),
        html.Div(id="case-summary", style={"marginBottom": "18px"}),
        dbc.Row([
            dbc.Col([
                html.Div("Hearing timeline", className="section-title"),
                html.Div(id="case-stages"),
            ], md=5),
            dbc.Col([
                html.Div("What changed between hearings", className="section-title"),
                html.Div(id="case-diffs"),
            ], md=7),
        ]),
        html.Hr(className="divider"),
        html.Div("Likely drivers", className="section-title"),
        html.Div(
            "Added legal entities are ranked by how strongly they distinguish this "
            "transition from the matched stable transition. Sentence panels show "
            "the later-hearing evidence.",
            className="muted", style={"marginBottom": "12px"},
        ),
        html.Div(id="case-factor-report"),
        html.Hr(className="divider"),
        html.Div("Source text by hearing", className="section-title"),
        html.Div("Sentences grouped by rhetorical-role tag from the upstream "
                  "OpenNyAI pipeline. Click any hearing to expand.",
                  className="muted", style={"marginBottom": "12px"}),
        html.Div(id="case-source-text"),
    ])

    hero = html.Div([
        html.Div("Hearing analysis · GNN ensemble", className="subtitle",
                 style={"textTransform": "uppercase", "letterSpacing": "0.12em",
                        "fontSize": "0.72rem", "marginBottom": "8px",
                        "color": "rgba(255,255,255,0.65)"}),
        html.H1("Multi-hearing test"),
        html.Div(
            "How does the model's verdict prediction shift between hearings of "
            "the same case, and what changes in the case driving it?",
            className="subtitle",
        ),
        html.Div([
            html.Span([html.Span(className="chip-dot"),
                       f"{n_cases} cases"], className="chip"),
            html.Span(f"final-pred acc {final_acc:.1%}", className="chip"),
            html.Span("actual: 0 POSTPONED · -1 LOSS · model maps both to LOSE",
                      className="chip"),
            html.Span(f"reads outputs/analysis · outputs/inference",
                      className="chip"),
        ], className="meta-row"),
    ], className="hero")

    return html.Div([
        hero,
        html.Div([
            html.Div(
                dcc.Tabs(
                    className="modern-tabs",
                    parent_className="modern-tabs-wrap",
                    children=[
                        dcc.Tab(label="Overview", children=overview_tab,
                                className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Transition explorer", children=transition_tab,
                                className="tab", selected_className="tab--selected"),
                        dcc.Tab(label="Case drill-down", children=case_tab,
                                className="tab", selected_className="tab--selected"),
                    ],
                ),
            ),
        ], className="app-shell"),
    ])


# ── app + callbacks ───────────────────────────────────────────────────────────

def build_app(data: dict) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Multi-hearing test visualiser",
    )
    app.layout = build_layout(data)

    @app.callback(
        Output("transition-section-delta", "figure"),
        Output("transition-entity-freq", "figure"),
        Output("transition-meta", "children"),
        Input("transition-picker", "value"),
    )
    def _on_transition(value):
        if not value:
            return no_update, no_update, no_update
        agg = data["aggregates"].get("by_transition", {}).get(value, {})
        n = agg.get("n_cases", data["transition_counts"].get(value, 0))
        meta = f"{value}: {n} cases"
        return section_delta_bar(agg), entity_add_freq_bar(agg), meta

    @app.callback(
        Output("case-picker", "options"),
        Output("case-picker", "value"),
        Input("raw-outcome-filter", "value"),
        State("case-picker", "value"),
    )
    def _filter_cases(raw_filter, current_value):
        options = []
        for base_id, meta in data["case_option_meta"].items():
            if raw_filter != "ALL" and meta.get("final_raw") != raw_filter:
                continue
            options.append(meta["option"])
        if not options:
            return [], None
        values = {o["value"] for o in options}
        next_value = current_value if current_value in values else options[0]["value"]
        return options, next_value

    @app.callback(
        Output("case-summary", "children"),
        Output("case-stages", "children"),
        Output("case-diffs", "children"),
        Output("case-factor-report", "children"),
        Output("case-source-text", "children"),
        Input("case-picker", "value"),
    )
    def _on_case(base_id):
        if not base_id:
            return no_update, no_update, no_update, no_update, no_update
        payload = load_per_case(base_id)
        if payload is None:
            return f"No per-case file for {base_id}", [], [], [], []

        true_label = payload.get("true_label", "?")
        transition = payload.get("transition", "")
        cat = payload.get("category", "")
        split = payload.get("outcome_split", "")
        n_stages = payload.get("n_stages", len(payload.get("stages", [])))

        raw_lookup = data["raw_label_lookup"]
        pred_lookup = data["pred_label_lookup"]
        stages_list = payload.get("stages", [])
        stage_raw = {int(s["stage_index"]): raw_lookup.get((base_id, int(s["stage_index"])))
                     for s in stages_list}
        stage_pred_raw = {int(s["stage_index"]): pred_lookup.get((base_id, int(s["stage_index"])))
                          for s in stages_list}

        raw_values = [stage_raw.get(int(s["stage_index"])) for s in stages_list]
        raw_values = [v for v in raw_values if v is not None]
        final_raw = raw_values[-1] if raw_values else None
        true_kind = "win" if true_label == "WIN" else "lose"
        raw_path_children = []
        for idx, raw in enumerate(raw_values):
            if idx:
                raw_path_children.append(html.Span("->", className="mono tiny muted"))
            raw_path_children.append(_raw_outcome_pill(raw, true_label))
        summary_box = html.Div([
            html.Div([
                html.Div(base_id, style={"fontWeight": 600, "fontSize": "0.95rem",
                                           "color": "#0f172a",
                                           "wordBreak": "break-word"}),
                html.Div([
                    _pill(cat, "neutral"),
                    _pill(split, "neutral"),
                    _pill(f"{n_stages} hearings", "info"),
                ], style={"display": "flex", "gap": "6px", "marginTop": "8px",
                          "flexWrap": "wrap"}),
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Span("Actual", className="muted tiny",
                          style={"marginRight": "6px"}),
                _raw_outcome_pill(final_raw, true_label),
                html.Span("Model bucket", className="muted tiny",
                          style={"marginLeft": "16px", "marginRight": "6px"}),
                _pill(display_label(true_label), true_kind),
                html.Span("Actual path", className="muted tiny",
                          style={"marginLeft": "16px", "marginRight": "6px"}),
                html.Span(raw_path_children or "?", className="raw-path"),
                html.Span("Transition", className="muted tiny",
                          style={"marginLeft": "16px", "marginRight": "6px"}),
                html.Span(transition, className="mono",
                          style={"fontSize": "0.85rem", "color": "#0f172a",
                                 "padding": "2px 8px",
                                 "background": "#f1f5f9", "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center",
                       "flexWrap": "wrap", "gap": "4px"}),
        ], style={"background": "white", "border": "1px solid #e5e7eb",
                   "borderRadius": "12px", "padding": "16px 18px",
                   "boxShadow": "0 1px 2px rgba(15,23,42,0.04)"})

        stage_cards = [
            stage_card(s, true_label,
                       raw_score=stage_raw.get(int(s["stage_index"])),
                       pred_raw=stage_pred_raw.get(int(s["stage_index"])))
            for s in stages_list
        ]
        diff_cards = [
            diff_card(d, true_label,
                      from_raw=stage_raw.get(int(d["from_stage"])),
                      to_raw=stage_raw.get(int(d["to_stage"])),
                      from_pred_raw=stage_pred_raw.get(int(d["from_stage"])),
                      to_pred_raw=stage_pred_raw.get(int(d["to_stage"])))
            for d in payload.get("consecutive_diffs", [])
        ]
        if not diff_cards:
            diff_cards = [html.Div("Single-hearing case — no transitions.",
                                   style={"color": "#666"})]

        text_items = [
            stage_text_panel(
                base_id,
                s.get("stage_index"),
                s.get("date", ""),
                s.get("pred_label", "?"),
                true_label,
            )
            for s in payload.get("stages", [])
        ]
        source_panel = dbc.Accordion(text_items, start_collapsed=True,
                                     always_open=True, flush=True)

        factor_panel = factor_report_panel(load_best_factor_report(base_id))

        return summary_box, stage_cards, diff_cards, factor_panel, source_panel

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data()
    app = build_app(data)
    print(f"[stage-vis] {data['summary']['n_cases']} cases loaded")
    print(f"[stage-vis] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
