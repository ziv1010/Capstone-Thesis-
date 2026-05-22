"""Stage-by-stage visualiser for the Fixed_GPU_OpenNyai pipeline.

Lets you pick a category bucket and a case file and inspect what each pipeline
stage produced for that case:

    Stage 1  NER + RR extract           final_outputs/{bucket}_extract/annotations
    Stage 2  OpenNyai summary           final_outputs/{bucket}_summary_opennyai/enriched_jsons
    Stage 3  Mistral outcome label      final_outputs/{bucket}_labelled_mistral/labelled_jsons
    Stage 4  Cross-validated outcome    cross_validated_outputs/{bucket}/augmented_jsons

Run with run_app.sh (uses the existing 'graph_vis' micromamba env).
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html as dhtml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai")
FINAL_OUTPUTS = ROOT / "final_outputs"
CROSSVAL_OUTPUTS = ROOT / "cross_validated_outputs"

BUCKETS = [
    "family_matrimonial",
    "fin_fraud",
    "food_safety",
    "land_property",
    "motor_accidents",
    "sexual_offences",
]

STAGES: List[Tuple[str, str, str]] = [
    ("extract",  "Stage 1",  "NER + RR"),
    ("summary",  "Stage 2",  "OpenNyai Summary"),
    ("labelled", "Stage 3",  "Mistral Label"),
    ("crossval", "Stage 4",  "Cross-Validated"),
]
STAGE_KEYS = [k for k, _, _ in STAGES]

VIEWS = [
    ("compare",  "Overview",   "Summary across stages"),
    ("extract",  "Stage 1",    "NER + Rhetorical Roles"),
    ("summary",  "Stage 2",    "OpenNyai summary"),
    ("labelled", "Stage 3",    "Mistral outcome label"),
    ("crossval", "Stage 4",    "Cross-validated outcome"),
    ("raw",      "Raw JSON",   "Per-stage payloads"),
]

CROSSVAL_QUESTIONS = {
    "q1": "Did the appellant win the case?",
    "q2": "Was the petition / appeal / application allowed?",
    "q3": "Did the court rule in favour of the petitioner / appellant?",
    "q4": "Did the respondent win the case?",
    "q5": "Was the petition / appeal / application dismissed?",
    "q6": "Did the court rule against the appellant?",
    "q7": "Was the case adjourned, remanded, or deferred without a final decision?",
    "q8": "Is there no clear final judgment in this case?",
}

ENTITY_COLORS = {
    "DATE":         "#fff3cd",
    "STATUTE":      "#d1e7dd",
    "PROVISION":    "#cfe2ff",
    "ORG":          "#f8d7da",
    "GPE":          "#ffe5d0",
    "PRECEDENT":    "#e2d5f5",
    "CASE_NUMBER":  "#fdebd0",
    "WITNESS":      "#d4f1f9",
    "OTHER_PERSON": "#fce8b2",
    "PETITIONER":   "#fbc4ab",
    "RESPONDENT":   "#a8dadc",
    "JUDGE":        "#bee3db",
    "LAWYER":       "#cdb4db",
    "COURT":        "#ffd6a5",
}

MAX_CASE_OPTIONS = 500

# ---------------------------------------------------------------------------
# Path resolution & loading
# ---------------------------------------------------------------------------

def stage_dir(bucket: str, stage: str) -> Optional[Path]:
    if stage == "extract":
        return FINAL_OUTPUTS / f"{bucket}_extract" / "annotations"
    if stage == "summary":
        return FINAL_OUTPUTS / f"{bucket}_summary_opennyai" / "enriched_jsons"
    if stage == "labelled":
        return FINAL_OUTPUTS / f"{bucket}_labelled_mistral" / "labelled_jsons"
    if stage == "crossval":
        return CROSSVAL_OUTPUTS / bucket / "augmented_jsons"
    return None


def stage_file(bucket: str, stage: str, case: str) -> Optional[Path]:
    d = stage_dir(bucket, stage)
    if d is None:
        return None
    return d / f"{case}.json"


def slug_case_name(case: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.]+", "_", case)).strip("_")


def case_search_text(bucket: str, case: str) -> str:
    slug = slug_case_name(case)
    return " ".join(
        [
            case,
            slug,
            f"{bucket}__{slug}",
            f"{bucket}__{slug}_MERGED",
        ]
    ).lower()


@lru_cache(maxsize=64)
def list_cases(bucket: str) -> Tuple[str, ...]:
    d = stage_dir(bucket, "extract")
    if not d or not d.is_dir():
        return tuple()
    return tuple(sorted(p.stem for p in d.glob("*.json")))


@lru_cache(maxsize=128)
def load_stage(bucket: str, stage: str, case: str) -> Optional[dict]:
    p = stage_file(bucket, stage, case)
    if p is None or not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Small UI primitives
# ---------------------------------------------------------------------------

def pill(text, kind: str = "muted", size: str = "") -> dhtml.Span:
    cls = f"pill pill-{kind}" + (f" {size}" if size else "")
    return dhtml.Span(text, className=cls)


def kv_grid(rows: List[Tuple[str, object]]) -> dhtml.Div:
    children: List = []
    for k, v in rows:
        children.append(dhtml.Div(k, className="kv-key"))
        children.append(dhtml.Div(str(v) if v not in (None, "") else "—", className="kv-val"))
    return dhtml.Div(children, className="kv-grid")


def card(title: Optional[str], *body, subtitle: Optional[str] = None) -> dhtml.Div:
    children: List = []
    if title:
        children.append(dhtml.H5(title))
    if subtitle:
        children.append(dhtml.Div(subtitle, className="subtitle"))
    children.extend(body)
    return dhtml.Div(children, className="surface-card")


def empty_card(title: str, msg: str = "Not produced for this case.") -> dhtml.Div:
    return card(title, dhtml.Em(msg, style={"color": "var(--muted)"}))


# ---------------------------------------------------------------------------
# Entity / sentence rendering
# ---------------------------------------------------------------------------

def render_entity_spans(text: str, entities: List[dict], sentence_start: int) -> List:
    if not entities:
        return [text]
    sorted_ents = sorted(
        (e for e in entities if "start" in e and "end" in e),
        key=lambda e: e["start"],
    )
    out: List = []
    cursor = sentence_start
    for ent in sorted_ents:
        s, e = ent["start"], ent["end"]
        if s < cursor:
            continue
        if s > cursor:
            out.append(text[cursor - sentence_start : s - sentence_start])
        label = ent.get("label", "?")
        color = ENTITY_COLORS.get(label, "#eee")
        out.append(
            dhtml.Span(
                [
                    text[s - sentence_start : e - sentence_start],
                    dhtml.Span(label, className="ent-label"),
                ],
                className="ent",
                style={"backgroundColor": color},
                title=f"{label}: {ent.get('text','')}",
            )
        )
        cursor = e
    if cursor - sentence_start < len(text):
        out.append(text[cursor - sentence_start :])
    return out


def rr_pill(role: str) -> dhtml.Span:
    safe = role or "NONE"
    return dhtml.Span(safe, className=f"rr-pill rr-{safe}")


# ---------------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------------

def render_stage1(data: Optional[dict]) -> dhtml.Div:
    if not data:
        return empty_card("Stage 1 · NER + RR")

    sentences = data.get("sentences", []) or []
    ner_by_label = data.get("ner_by_label", {}) or {}
    rr_by_role = data.get("rr_by_role", {}) or {}

    def _count(v):
        if isinstance(v, list):
            return len(v)
        if isinstance(v, int):
            return v
        return 0

    ner_items: List = []
    for label, ents in sorted(ner_by_label.items(), key=lambda kv: -_count(kv[1])):
        n = _count(ents)
        if isinstance(ents, list):
            body = dhtml.Ul(
                [
                    dhtml.Li(
                        f"{e.get('text','')} (sent {e.get('sentence_id','?')})"
                    )
                    for e in ents[:200]
                ],
                style={"maxHeight": "220px", "overflowY": "auto", "margin": 0},
            )
        else:
            body = dhtml.Em("Only counts available — see sentence list for entity spans.")
        ner_items.append(dbc.AccordionItem(body, title=f"{label} · {n}"))

    rr_items: List = []
    for role, items in sorted(rr_by_role.items(), key=lambda kv: -_count(kv[1])):
        n = _count(items)
        if isinstance(items, list):
            body = dhtml.Ul(
                [
                    dhtml.Li(
                        f"sent {it.get('sentence_id','?')}: {it.get('text','')[:200]}"
                    )
                    for it in items[:200]
                ],
                style={"maxHeight": "240px", "overflowY": "auto", "margin": 0},
            )
        else:
            body = dhtml.Em("Only counts available — see sentence list for role badges.")
        rr_items.append(dbc.AccordionItem(body, title=f"{role} · {n}"))

    sent_rows: List = []
    for s in sentences:
        role = s.get("rhetorical_role", "NONE") or "NONE"
        sent_rows.append(
            dhtml.Div(
                [
                    dhtml.Div(f"#{s.get('sentence_id','?')}", className="sent-id"),
                    dhtml.Div(rr_pill(role)),
                    dhtml.Div(
                        render_entity_spans(
                            s.get("text", ""), s.get("entities", []) or [], s.get("start", 0)
                        ),
                        className="sent-text",
                    ),
                ],
                className="sent-row",
            )
        )

    summary_pills = dhtml.Div(
        [
            pill(f"file_id · {data.get('file_id','')}", "muted"),
            pill(f"rr_available · {data.get('rr_available','')}", "info" if data.get("rr_available") else "muted"),
            pill(f"sentences · {len(sentences)}", "accent"),
            pill(f"preamble end · {data.get('preamble_end_char_offset','—')}", "muted"),
        ],
        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
    )

    return dhtml.Div(
        [
            card(
                "Stage 1 · NER + Rhetorical-Role Extract",
                summary_pills,
                dhtml.Div(
                    [
                        dhtml.Div(
                            [
                                dhtml.H6("NER tally"),
                                dbc.Accordion(ner_items, start_collapsed=True, always_open=True)
                                if ner_items
                                else dhtml.Em("No entities extracted.", style={"color": "var(--muted)"}),
                            ],
                            style={"flex": 1, "minWidth": "280px"},
                        ),
                        dhtml.Div(
                            [
                                dhtml.H6("Rhetorical-role tally"),
                                dbc.Accordion(rr_items, start_collapsed=True, always_open=True)
                                if rr_items
                                else dhtml.Em("No RR groups.", style={"color": "var(--muted)"}),
                            ],
                            style={"flex": 1, "minWidth": "280px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                ),
                subtitle="What the OpenNyai NER + Rhetorical-Role models extracted.",
            ),
            card(
                f"Sentences ({len(sent_rows)})",
                dhtml.Div(sent_rows, className="sent-list"),
                subtitle="Sentence-level rhetorical role + inline entity highlights.",
            ),
        ]
    )


def render_stage2(data: Optional[dict]) -> dhtml.Div:
    if not data:
        return empty_card("Stage 2 · OpenNyai Summary")

    summary = data.get("opennyai_summary") or {}
    status = data.get("summary_status", "")
    err = data.get("summary_error", "")

    summary_blocks: List = []
    if isinstance(summary, dict):
        for section, body in summary.items():
            summary_blocks.append(
                dhtml.Div(
                    [
                        dhtml.Div(section, className="section-name"),
                        dhtml.Div(str(body), className="section-body"),
                    ],
                    className="summary-section",
                )
            )
    elif summary:
        summary_blocks.append(dhtml.Pre(str(summary), className="pretty-pre"))

    kept_rows: List = []
    for s in data.get("sentences", []) or []:
        if s.get("in_summary"):
            sec = s.get("summary_section", "") or "—"
            kept_rows.append(
                dhtml.Div(
                    [
                        dhtml.Div(f"#{s.get('sentence_id','?')}", className="sent-id"),
                        dhtml.Div(pill(sec, "success")),
                        dhtml.Div(s.get("text", ""), className="sent-text"),
                    ],
                    className="sent-row",
                )
            )

    status_pill_kind = "success" if status == "success" else ("warning" if status else "muted")

    return dhtml.Div(
        [
            card(
                "Stage 2 · OpenNyai Summary",
                dhtml.Div(
                    [
                        pill(f"status · {status or '—'}", status_pill_kind),
                        pill(f"error · {err}" if err else "no errors", "danger" if err else "muted"),
                        pill(f"kept sentences · {len(kept_rows)}", "accent"),
                    ],
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "12px"},
                ),
                dhtml.H6("opennyai_summary"),
                dhtml.Div(summary_blocks)
                if summary_blocks
                else dhtml.Em("No summary produced.", style={"color": "var(--muted)"}),
                subtitle="Per-section abstractive summary derived from the rhetorical-role extracts.",
            ),
            card(
                f"Sentences kept in summary ({len(kept_rows)})",
                dhtml.Div(kept_rows, className="sent-list")
                if kept_rows
                else dhtml.Em("No sentences flagged in_summary.", style={"color": "var(--muted)"}),
            ),
        ]
    )


def render_outcome_hero(label, score, extra_pills: List = None) -> dhtml.Div:
    safe = str(label or "—").replace("/", "_")
    return dhtml.Div(
        [
            dhtml.Span(str(label or "—"), className=f"outcome-label outcome-{safe}"),
            pill(f"score · {score if score is not None else '—'}", "muted", size="pill-lg"),
            *(extra_pills or []),
        ],
        className="outcome-hero",
    )


def render_stage3(data: Optional[dict]) -> dhtml.Div:
    if not data:
        return empty_card("Stage 3 · Mistral Label")

    label = data.get("case_outcome_label")
    score = data.get("case_outcome_score")
    llm = data.get("llm_case_outcome") or {}

    rpc_block = llm.get("rpc_texts") or []
    decision_text = llm.get("decision_text", "")
    short_expl = llm.get("short_explanation", "")
    raw = llm.get("raw_model_response", "")
    confidence = llm.get("confidence", "—")

    return dhtml.Div(
        [
            card(
                "Stage 3 · Mistral Outcome Label",
                render_outcome_hero(
                    label,
                    score,
                    extra_pills=[pill(f"confidence · {confidence}", "info", size="pill-lg")],
                ),
                kv_grid(
                    [
                        ("backend",  llm.get("backend", "—")),
                        ("model_id", llm.get("model_id", "—")),
                        ("provider", llm.get("provider", "—")),
                    ]
                ),
                dhtml.Hr(),
                dhtml.H6("Short explanation"),
                dhtml.Div(short_expl or dhtml.Em("—"), style={"fontSize": "0.95rem", "marginBottom": "10px"}),
                dhtml.H6("decision_text (RPC sentences fed to the LLM)"),
                dhtml.Pre(decision_text or "—", className="pretty-pre"),
                dhtml.H6("rpc_texts"),
                dhtml.Ul([dhtml.Li(t) for t in rpc_block])
                if rpc_block
                else dhtml.Em("—", style={"color": "var(--muted)"}),
                dhtml.Details(
                    [
                        dhtml.Summary("raw_model_response"),
                        dhtml.Pre(raw or "—", className="pretty-pre"),
                    ],
                    style={"marginTop": "10px"},
                ),
                subtitle="LLM-only outcome label using the rhetorical 'RPC' (court decision) sentences.",
            )
        ]
    )


def render_stage4(data: Optional[dict]) -> dhtml.Div:
    if not data:
        return empty_card("Stage 4 · Cross-Validated")

    label = data.get("case_outcome_label")
    score = data.get("case_outcome_score")
    llm = data.get("llm_case_outcome") or {}
    answers = llm.get("crossval_answers") or {}
    win_score = llm.get("win_score")
    loss_score = llm.get("loss_score")
    neutral_score = llm.get("neutral_score")
    crossval_label = llm.get("crossval_label")
    crossval_conf = llm.get("crossval_confidence")
    raw = llm.get("raw_model_response", "")

    q_rows: List = []
    for i in range(1, 9):
        key = f"q{i}"
        ans = answers.get(key)
        ans_pill = pill("YES", "success") if ans == 1 else (
            pill("NO", "muted") if ans == 0 else pill("—", "warning")
        )
        bucket_lbl = "win" if i <= 3 else ("loss" if i <= 6 else "neutral")
        bucket_pill = pill(bucket_lbl, {"win": "success", "loss": "danger", "neutral": "muted"}[bucket_lbl])
        q_rows.append(
            dhtml.Div(
                [
                    dhtml.Div(key, className="q-key"),
                    dhtml.Div(bucket_pill),
                    dhtml.Div(CROSSVAL_QUESTIONS[key]),
                    dhtml.Div(ans_pill, style={"textAlign": "right"}),
                ],
                className="q-row",
            )
        )

    extra_pills = [
        pill(f"win · {win_score}", "success"),
        pill(f"loss · {loss_score}", "danger"),
        pill(f"neutral · {neutral_score}", "muted"),
        pill(f"crossval_label · {crossval_label}", "accent"),
        pill(f"confidence · {crossval_conf}", "info"),
    ]

    return dhtml.Div(
        [
            card(
                "Stage 4 · Cross-Validated Outcome",
                render_outcome_hero(label, score, extra_pills=extra_pills),
                kv_grid(
                    [
                        ("method",   llm.get("method", "—")),
                        ("model_id", llm.get("model_id", "—")),
                    ]
                ),
                dhtml.Hr(),
                dhtml.H6("8-question cross-validation"),
                dhtml.Div(q_rows, className="q-grid"),
                dhtml.Hr(),
                dhtml.H6("decision_text"),
                dhtml.Pre(llm.get("decision_text", "") or "—", className="pretty-pre"),
                dhtml.Details(
                    [
                        dhtml.Summary("raw_model_response"),
                        dhtml.Pre(raw or "—", className="pretty-pre"),
                    ],
                    style={"marginTop": "10px"},
                ),
                subtitle="Eight YES/NO probes, bucketed into win / loss / neutral and tallied.",
            )
        ]
    )


def render_compare(stages: Dict[str, Optional[dict]]) -> dhtml.Div:
    chips: List = []
    for key, label, _ in STAGES:
        present = stages.get(key) is not None
        chips.append(pill(f"{label} · {'present' if present else 'missing'}",
                          "success" if present else "muted"))
    rows: List[Tuple[str, object]] = []

    s1 = stages.get("extract")
    if s1:
        rows.append(("Stage 1 · sentences",  len(s1.get("sentences", []) or [])))
        rows.append(("Stage 1 · NER labels", len(s1.get("ner_by_label", {}) or {})))
        rows.append(("Stage 1 · rr_available", s1.get("rr_available")))

    s2 = stages.get("summary")
    if s2:
        sm = s2.get("opennyai_summary") or {}
        sec = ", ".join(sm.keys()) if isinstance(sm, dict) else "—"
        rows.append(("Stage 2 · status", s2.get("summary_status", "—")))
        rows.append(("Stage 2 · sections", sec or "—"))

    s3 = stages.get("labelled")
    if s3:
        llm = s3.get("llm_case_outcome") or {}
        rows.append(("Stage 3 · label",      s3.get("case_outcome_label")))
        rows.append(("Stage 3 · score",      s3.get("case_outcome_score")))
        rows.append(("Stage 3 · confidence", llm.get("confidence", "—")))

    s4 = stages.get("crossval")
    if s4:
        llm = s4.get("llm_case_outcome") or {}
        rows.append(("Stage 4 · label",      s4.get("case_outcome_label")))
        rows.append(("Stage 4 · score",      s4.get("case_outcome_score")))
        rows.append(("Stage 4 · confidence", llm.get("crossval_confidence", "—")))

    agree_pill = None
    if s3 and s4:
        agree = s3.get("case_outcome_label") == s4.get("case_outcome_label")
        agree_pill = pill(
            f"Stage 3 vs 4 · {'AGREE' if agree else 'DISAGREE'}",
            "success" if agree else "danger",
            size="pill-lg",
        )

    return dhtml.Div(
        [
            card(
                "Cross-stage overview",
                dhtml.Div(
                    chips + ([agree_pill] if agree_pill else []),
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "14px"},
                ),
                kv_grid(rows),
                subtitle="Quick read on what was produced for this case and how stages 3 / 4 compare.",
            ),
        ]
    )


def render_raw(stages: Dict[str, Optional[dict]]) -> dhtml.Div:
    items: List = []
    for key, group, label in STAGES:
        data = stages.get(key)
        body = (
            dhtml.Pre(json.dumps(data, indent=2, ensure_ascii=False), className="pretty-pre")
            if data
            else dhtml.Em("Not produced for this case.", style={"color": "var(--muted)"})
        )
        items.append(dbc.AccordionItem(body, title=f"{group} · {label}"))
    return card("Raw JSON per stage",
                dbc.Accordion(items, start_collapsed=True, always_open=True))


# ---------------------------------------------------------------------------
# App + layout
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Pipeline Stage Visualiser",
    suppress_callback_exceptions=True,
)
server = app.server


def topbar() -> dhtml.Div:
    return dhtml.Div(
        dbc.Row(
            [
                dbc.Col(
                    dhtml.Div(
                        [
                            dhtml.Div("FG", className="brand-mark"),
                            dhtml.Div(
                                [
                                    dhtml.Div("Pipeline Stage Visualiser", className="brand-title"),
                                    dhtml.Div("Fixed_GPU_OpenNyai · per-case stage inspector",
                                              className="brand-sub"),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    width=4,
                ),
                dbc.Col(
                    [
                        dhtml.Label("Bucket"),
                        dcc.Dropdown(
                            id="bucket-dd",
                            options=[{"label": b, "value": b} for b in BUCKETS],
                            value=BUCKETS[0],
                            clearable=False,
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dhtml.Label("Filter"),
                        dcc.Input(
                            id="case-filter",
                            type="text",
                            placeholder="type to filter…",
                            debounce=True,
                            style={"width": "100%"},
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dhtml.Label(id="case-count", children="Case"),
                        dcc.Dropdown(
                            id="case-dd",
                            options=[],
                            value=None,
                            placeholder="select a case…",
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
            ],
            align="center",
        ),
        className="topbar",
    )


def sidebar_nav(active_view: str, presence: Dict[str, bool]) -> dhtml.Div:
    items: List = []
    items.append(dhtml.Div("Views", className="sidebar-section-title"))
    for vid, label, sub in VIEWS:
        is_active = vid == active_view
        if vid in STAGE_KEYS:
            present = presence.get(vid, False)
            dot_cls = "nav-dot " + ("present" if present else "missing")
            tag = "ok" if present else "—"
        elif vid == "compare":
            dot_cls = "nav-dot present"
            tag = ""
        else:
            dot_cls = "nav-dot"
            tag = ""
        items.append(
            dhtml.Div(
                [
                    dhtml.Span(className=dot_cls),
                    dhtml.Div(
                        [
                            dhtml.Div(label, className="nav-label"),
                            dhtml.Div(sub, style={"fontSize": "0.72rem", "color": "var(--muted)"}),
                        ],
                        style={"flex": 1, "minWidth": 0},
                    ),
                    dhtml.Span(tag, className="nav-tag"),
                ],
                id={"role": "nav", "view": vid},
                className=f"nav-item{' active' if is_active else ''}",
                n_clicks=0,
            )
        )
    items.append(dhtml.Hr())
    items.append(dhtml.Div("Legend", className="sidebar-section-title"))
    items.append(
        dhtml.Div(
            [
                dhtml.Div([dhtml.Span(className="nav-dot present"), dhtml.Span("stage produced")],
                          style={"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "4px"}),
                dhtml.Div([dhtml.Span(className="nav-dot missing"), dhtml.Span("stage missing")],
                          style={"display": "flex", "gap": "8px", "alignItems": "center"}),
            ],
            style={"fontSize": "0.78rem", "color": "var(--muted)", "padding": "0 6px"},
        )
    )
    return dhtml.Div(items, className="sidebar")


app.layout = dhtml.Div(
    [
        dcc.Store(id="active-view", data="compare"),
        topbar(),
        dhtml.Div(
            [
                dhtml.Div(id="sidebar-host"),
                dhtml.Div(id="tab-body", className="main-pane"),
            ],
            className="layout-grid",
        ),
    ],
    className="app-shell",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("case-dd", "options"),
    Output("case-dd", "value"),
    Output("case-count", "children"),
    Input("bucket-dd", "value"),
    Input("case-filter", "value"),
    State("case-dd", "value"),
)
def _update_case_options(bucket, filt, current):
    cases = list_cases(bucket)
    query = (filt or "").lower()
    matches = [c for c in cases if query in case_search_text(bucket, c)] if query else list(cases)
    total = len(matches)
    matches = matches[:MAX_CASE_OPTIONS]
    options = [{"label": c, "value": c, "search": case_search_text(bucket, c)} for c in matches]
    new_val = current if current and current in {o["value"] for o in options} else (matches[0] if matches else None)
    label = f"Case ({total} matches"
    if total > MAX_CASE_OPTIONS:
        label += f", first {MAX_CASE_OPTIONS} shown"
    label += ")"
    return options, new_val, label


@app.callback(
    Output("active-view", "data"),
    Input({"role": "nav", "view": dash.ALL}, "n_clicks"),
    State("active-view", "data"),
    prevent_initial_call=True,
)
def _set_active_view(_clicks, current):
    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return current
    return triggered.get("view", current)


@app.callback(
    Output("sidebar-host", "children"),
    Output("tab-body", "children"),
    Input("active-view", "data"),
    Input("bucket-dd", "value"),
    Input("case-dd", "value"),
)
def _render(active_view, bucket, case):
    if not case:
        sidebar = sidebar_nav(active_view, {k: False for k in STAGE_KEYS})
        return sidebar, dhtml.Div(
            dhtml.Em("Pick a case to view its stage outputs.",
                     style={"color": "var(--muted)"}),
            className="surface-card",
        )

    data = {key: load_stage(bucket, key, case) for key in STAGE_KEYS}
    presence = {k: v is not None for k, v in data.items()}
    sidebar = sidebar_nav(active_view, presence)

    if active_view == "compare":
        body = render_compare(data)
    elif active_view == "extract":
        body = render_stage1(data["extract"])
    elif active_view == "summary":
        body = render_stage2(data["summary"])
    elif active_view == "labelled":
        body = render_stage3(data["labelled"])
    elif active_view == "crossval":
        body = render_stage4(data["crossval"])
    elif active_view == "raw":
        body = render_raw(data)
    else:
        body = dhtml.Em("Unknown view.")
    return sidebar, body


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8053)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
