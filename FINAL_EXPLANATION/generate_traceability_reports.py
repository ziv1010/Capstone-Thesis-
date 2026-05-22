#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import networkx as nx
import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape
from pyvis.network import Network


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
DEFAULT_EXPLANATION_DIR = APP_ROOT / "outputs/target_fold_00_8gpu"
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs/traceability_reports_sample"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/"
    / "combined_dataset_without_food_safety"
)
LEGAL_AUTHORITY_TYPES = {"precedent", "provision", "statute"}
SECTION_TYPES = {
    "preamble",
    "facts",
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
}
LABEL_NAMES = {"1": "WIN", "-1": "LOSS"}
NODE_COLORS = {
    "target_case": "#111827",
    "train_case": "#374151",
    "precedent": "#2563eb",
    "provision": "#059669",
    "statute": "#0f766e",
    "arguments": "#7c3aed",
    "petitioner_arguments": "#8b5cf6",
    "respondent_arguments": "#6d28d9",
    "other_lawyer_arguments": "#5b21b6",
    "facts": "#d97706",
    "preamble": "#b45309",
    "judge": "#be123c",
    "court": "#dc2626",
    "petitioner": "#c2410c",
    "respondent": "#ea580c",
    "lawyer": "#9333ea",
    "petitioner_lawyer": "#a855f7",
    "defence_lawyer": "#7e22ce",
    "relation_type": "#64748b",
    "intermediate": "#94a3b8",
}


REPORT_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Traceability Report - Case {{ report.summary.case_index }}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --soft: #f5f7fa;
      --case: #111827;
      --accent: #2563eb;
      --good: #057a55;
      --warn: #9a5b13;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.45;
    }
    main { max-width: 1280px; margin: 0 auto; padding: 28px 32px 40px; }
    h1, h2, h3 { margin: 0; line-height: 1.2; letter-spacing: 0; }
    h1 { font-size: 28px; max-width: 980px; }
    h2 { font-size: 18px; margin-top: 28px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
    h3 { font-size: 15px; margin: 18px 0 8px; }
    p { margin: 10px 0; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: #eef2f7; padding: 1px 4px; border-radius: 4px; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 14px 0 18px;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: #253044;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 16px 0 22px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .stat .label { color: var(--muted); font-size: 12px; }
    .stat .value { font-weight: 700; font-size: 20px; margin-top: 4px; }
    .reading {
      border-left: 4px solid var(--accent);
      background: #f7faff;
      padding: 12px 14px;
      margin: 16px 0;
    }
    .graph-frame {
      width: 100%;
      height: 760px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }
    th {
      text-align: left;
      color: #344054;
      font-size: 12px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    th, td {
      padding: 8px 10px;
      vertical-align: top;
      border-bottom: 1px solid #edf1f6;
    }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .small { color: var(--muted); font-size: 12px; }
    .marker-both { color: var(--good); font-weight: 700; }
    .marker-attention { color: var(--warn); font-weight: 700; }
    .marker-counterfactual { color: var(--bad); font-weight: 700; }
    .path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .scroll { overflow-x: auto; }
    .provenance-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .prov-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .prov-card h3 { overflow-wrap: anywhere; }
    ol { margin: 8px 0 0 20px; padding: 0; }
    li { margin: 6px 0; }
    .footer { margin-top: 32px; color: var(--muted); font-size: 12px; }
    @media (max-width: 900px) {
      main { padding: 20px 16px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .provenance-grid { grid-template-columns: 1fr; }
      .graph-frame { height: 640px; }
    }
  </style>
</head>
<body>
<main>
  <h1>{{ report.summary.case_id }}</h1>
  <div class="meta">
    <span class="pill">Case index {{ report.summary.case_index }}</span>
    <span class="pill">Target {{ report.summary.target_label_display }}</span>
    <span class="pill">Predicted {{ report.summary.predicted_label_display }}</span>
    <span class="pill">Split {{ report.summary.split }}</span>
    <span class="pill">Generated {{ report.generated_at }}</span>
  </div>

  <div class="grid">
    <div class="stat"><div class="label">Prediction confidence</div><div class="value">{{ report.summary.baseline_pred_proba_fmt }}</div></div>
    <div class="stat"><div class="label">Label 1 probability</div><div class="value">{{ report.summary.baseline_positive_proba_fmt }}</div></div>
    <div class="stat"><div class="label">Local RF nodes</div><div class="value">{{ report.summary.rf_node_count }}</div></div>
    <div class="stat"><div class="label">Local RF edges</div><div class="value">{{ report.summary.rf_edge_count }}</div></div>
  </div>

  <h2>Structured Reading</h2>
  <div class="reading">{{ report.natural_language_explanation }}</div>
  <p class="small">
    Counterfactual delta is the original predicted-class probability minus the probability after masking the evidence group.
    Positive delta means the evidence supported the original prediction; negative delta means masking increased the original prediction's confidence.
  </p>

  <h2>Typed Computation Subgraph</h2>
  <iframe class="graph-frame" srcdoc="{{ graph_html }}"></iframe>

  <h2>Evidence Groups</h2>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>Marker</th>
          <th class="num">CF rank</th>
          <th class="num">Att rank</th>
          <th>Evidence</th>
          <th>Type</th>
          <th>Path family</th>
          <th class="num">Delta</th>
          <th class="num">Abs delta</th>
          <th class="num">Attention</th>
          <th class="num">Support</th>
          <th>Connected label distribution</th>
        </tr>
      </thead>
      <tbody>
        {% for row in report.evidence_groups %}
        <tr>
          <td class="{{ row.marker_css }}">{{ row.marker_symbol|safe }} {{ row.marker_label }}</td>
          <td class="num">{{ row.counterfactual_rank_display }}</td>
          <td class="num">{{ row.attention_rank_display }}</td>
          <td>{{ row.evidence_name }}</td>
          <td>{{ row.evidence_type }}</td>
          <td class="path">{{ row.path_family }}</td>
          <td class="num">{{ row.delta_pred_proba_fmt }}</td>
          <td class="num">{{ row.abs_delta_pred_proba_fmt }}</td>
          <td class="num">{{ row.attention_score_fmt }}</td>
          <td class="num">{{ row.support_train_n_display }}</td>
          <td>{{ row.support_distribution_text }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <h2>Provenance Trail</h2>
  {% if report.provenance_available %}
  <div class="provenance-grid">
    {% for item in report.provenance %}
    <section class="prov-card">
      <h3>{{ item.evidence_name }}</h3>
      <p class="small">{{ item.evidence_type }} via <span class="path">{{ item.path_family }}</span></p>
      {% if item.train_cases %}
      <ol>
        {% for case in item.train_cases %}
        <li>
          {% if case.source_href %}<a href="{{ case.source_href }}">{{ case.case_id }}</a>{% else %}{{ case.case_id }}{% endif %}
          <span class="small">({{ case.label_display }}, index {{ case.case_index }})</span>
          <div class="path">{{ case.path }}</div>
        </li>
        {% endfor %}
      </ol>
      {% else %}
      <p class="small">No connected training cases were found within the configured support hops for this evidence group.</p>
      {% endif %}
    </section>
    {% endfor %}
  </div>
  {% else %}
  <p class="small">{{ report.provenance_note }}</p>
  {% endif %}

  <h2>Machine Artifact</h2>
  <p>JSON: <a href="{{ report.json_filename }}">{{ report.json_filename }}</a></p>
  <p>Standalone graph: <a href="{{ report.graph_filename }}">{{ report.graph_filename }}</a></p>

  <div class="footer">
    Generated from FINAL_EXPLANATION CSVs. This artifact is falsifiable: each evidence row is tied to a concrete mask delta, typed path, attention diagnostic, and source-case provenance when graph cache access is enabled.
  </div>
</main>
</body>
</html>
"""


INDEX_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HGT Traceability Reports</title>
  <style>
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #fff; }
    main { max-width: 1200px; margin: 0 auto; padding: 28px 32px 40px; }
    h1 { font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }
    p { color: #5f6b7a; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 18px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #edf1f6; vertical-align: top; }
    th { background: #f8fafc; border-bottom: 1px solid #d8dee8; color: #344054; font-size: 12px; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .path { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  </style>
</head>
<body>
<main>
  <h1>HGT Traceability Reports</h1>
  <p>{{ rows|length }} per-case reasoning artifacts generated from {{ explanation_dir }}.</p>
  <table>
    <thead>
      <tr>
        <th class="num">Case index</th>
        <th>Case</th>
        <th>Target</th>
        <th>Predicted</th>
        <th class="num">Confidence</th>
        <th>Top evidence</th>
        <th>Top path</th>
        <th>Artifacts</th>
      </tr>
    </thead>
    <tbody>
      {% for row in rows %}
      <tr>
        <td class="num">{{ row.case_index }}</td>
        <td>{{ row.case_id }}</td>
        <td>{{ row.target_label_display }}</td>
        <td>{{ row.predicted_label_display }}</td>
        <td class="num">{{ row.confidence }}</td>
        <td>{{ row.top_evidence }}</td>
        <td class="path">{{ row.top_path }}</td>
        <td><a href="{{ row.report_href }}">HTML</a> | <a href="{{ row.json_href }}">JSON</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def as_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) else out


def as_int(value: Any) -> int | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        out = int(float(value))
    except Exception:
        return None
    return out


def fmt_float(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    return "NA" if number is None else f"{100.0 * number:.{digits}f}%"


def label_display(label: Any) -> str:
    text = str(clean_value(label))
    return f"{LABEL_NAMES.get(text, 'label')} ({text})"


def slugify(value: Any, max_len: int = 70) -> str:
    text = str(clean_value(value) or "case")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "case")[:max_len]


def short_text(value: Any, max_len: int = 90) -> str:
    text = " ".join(str(clean_value(value) or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def json_ready(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    for shard in summary.get("shard_summaries", []):
        if key in shard:
            return shard[key]
    return None


def source_href(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().as_uri()
    except Exception:
        return None


def build_source_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            index.setdefault(path.name, path)
    return index


def evidence_key(row: dict[str, Any] | pd.Series) -> tuple[str, str, Any] | None:
    getter = row.get if isinstance(row, dict) else row.get
    evidence_type = str(clean_value(getter("evidence_type")) or "")
    if not evidence_type:
        return None
    global_index = as_int(getter("evidence_global_index"))
    if global_index is not None:
        return ("global", evidence_type, global_index)
    evidence_id = clean_value(getter("evidence_id"))
    if evidence_id:
        return ("id", evidence_type, str(evidence_id))
    evidence_name = clean_value(getter("evidence_name"))
    if evidence_name:
        return ("name", evidence_type, str(evidence_name))
    return None


def build_support_lookup(support_df: pd.DataFrame) -> dict[tuple[str, str, Any], dict[str, Any]]:
    lookup: dict[tuple[str, str, Any], dict[str, Any]] = {}
    if support_df.empty:
        return lookup
    for row in support_df.to_dict("records"):
        key = evidence_key(row)
        if key:
            lookup.setdefault(key, row)
        evidence_type = str(clean_value(row.get("evidence_type")) or "")
        evidence_id = clean_value(row.get("evidence_id"))
        evidence_name = clean_value(row.get("evidence_name"))
        if evidence_type and evidence_id:
            lookup.setdefault(("id", evidence_type, str(evidence_id)), row)
        if evidence_type and evidence_name:
            lookup.setdefault(("name", evidence_type, str(evidence_name)), row)
    return lookup


def merge_support(row: dict[str, Any], support_lookup: dict[tuple[str, str, Any], dict[str, Any]]) -> dict[str, Any]:
    merged = dict(row)
    key = evidence_key(merged)
    support = support_lookup.get(key) if key else None
    if support:
        for support_key, support_value in support.items():
            if support_key not in merged or clean_value(merged.get(support_key)) is None:
                merged[support_key] = support_value
    return merged


def marker_for(row_id: str, cf_ranks: dict[str, int], att_ranks: dict[str, int]) -> tuple[str, str, str, str]:
    in_cf = row_id in cf_ranks
    in_att = row_id in att_ranks
    if in_cf and in_att:
        return ("both_agree", "&#10003;", "both", "marker-both")
    if in_att:
        return ("attention_only", "&#9651;", "attention only", "marker-attention")
    return ("counterfactual_only", "&#10007;", "counterfactual only", "marker-counterfactual")


def support_distribution_text(row: dict[str, Any]) -> str:
    support_n = as_int(row.get("support_train_n"))
    positive_rate = as_float(row.get("support_positive_rate"))
    negative_rate = as_float(row.get("support_negative_rate"))
    positive_n = as_int(row.get("support_positive_n"))
    negative_n = as_int(row.get("support_negative_n"))
    if support_n is None:
        return "No support row"
    if support_n == 0:
        return "0 connected training cases"
    parts = [f"{support_n:,} train cases"]
    if positive_rate is not None:
        parts.append(f"{fmt_pct(positive_rate)} WIN")
    if negative_rate is not None:
        parts.append(f"{fmt_pct(negative_rate)} LOSS")
    if positive_n is not None and negative_n is not None:
        parts.append(f"counts {positive_n:,}/{negative_n:,}")
    return ", ".join(parts)


def read_tables(explanation_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "case_summary": "case_summary.csv",
        "top": "case_top_explanations.csv",
        "long": "case_counterfactual_groups.csv",
        "support": "connected_case_label_distribution.csv",
        "attention": "attention_counterfactual_overlap.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in required.items():
        path = explanation_dir / filename
        if not path.exists():
            if key in {"support", "attention"}:
                tables[key] = pd.DataFrame()
                continue
            raise FileNotFoundError(f"Missing required input CSV: {path}")
        tables[key] = pd.read_csv(path, low_memory=False)
    return tables


def select_case_indices(case_summary: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    if args.case_index:
        return [int(item) for item in args.case_index]
    work = case_summary.copy()
    if args.all:
        return [int(value) for value in work["case_index"].tolist()]

    limit = max(1, int(args.case_limit))
    if args.sample_strategy == "legal_authority":
        authority = work[work["top_evidence_type"].isin(LEGAL_AUTHORITY_TYPES)].copy()
        if not authority.empty:
            work = authority
    if args.sample_strategy in {"legal_authority", "importance"}:
        work["__importance"] = pd.to_numeric(work["max_abs_delta_pred_proba"], errors="coerce")
        work = work.sort_values("__importance", ascending=False, na_position="last")
    return [int(value) for value in work["case_index"].head(limit).tolist()]


@dataclass
class GraphBundle:
    data: Any
    label_names: list[str]
    train_mask: Any
    source_index: dict[str, Path]
    support_hops: int
    provenance_limit: int


class GraphProvenance:
    def __init__(self, bundle: GraphBundle) -> None:
        self.data = bundle.data
        self.label_names = bundle.label_names
        self.train_mask = bundle.train_mask
        self.source_index = bundle.source_index
        self.support_hops = bundle.support_hops
        self.provenance_limit = bundle.provenance_limit
        self.cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.out_edge_types_by_src: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.in_edge_types_by_dst: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.by_src: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.by_dst: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for edge_type in self.data.edge_types:
            self.out_edge_types_by_src[edge_type[0]].append(edge_type)
            self.in_edge_types_by_dst[edge_type[2]].append(edge_type)
            edge_index = self.data[edge_type].edge_index.detach().cpu().numpy()
            src_order = np.argsort(edge_index[0], kind="mergesort")
            dst_order = np.argsort(edge_index[1], kind="mergesort")
            self.by_src[edge_type] = (edge_index[0][src_order], src_order)
            self.by_dst[edge_type] = (edge_index[1][dst_order], dst_order)

    @staticmethod
    def positions_for(sorted_values: np.ndarray, sorted_positions: np.ndarray, value: int) -> np.ndarray:
        left = np.searchsorted(sorted_values, value, side="left")
        right = np.searchsorted(sorted_values, value, side="right")
        if left == right:
            return np.asarray([], dtype=np.int64)
        return sorted_positions[left:right]

    def connected_train_cases(self, node_type: str, node_index: int) -> list[dict[str, Any]]:
        key = (node_type, int(node_index))
        if key in self.cache:
            return self.cache[key]

        start = (node_type, int(node_index))
        queue: deque[tuple[str, int, int]] = deque([(node_type, int(node_index), 0)])
        seen: set[tuple[str, int]] = {start}
        parent: dict[tuple[str, int], tuple[tuple[str, int], str]] = {}
        hits: list[tuple[int, tuple[str, int]]] = []

        while queue:
            current_type, current_idx, depth = queue.popleft()
            current = (current_type, current_idx)
            if current_type == "case" and bool(self.train_mask[current_idx]):
                hits.append((current_idx, current))
            if depth >= self.support_hops:
                continue

            for edge_type in self.out_edge_types_by_src.get(current_type, []):
                edge_index = self.data[edge_type].edge_index
                sorted_values, sorted_positions = self.by_src[edge_type]
                positions = self.positions_for(sorted_values, sorted_positions, current_idx)
                relation = edge_type[1]
                if positions.size == 0:
                    continue
                for dst_idx in edge_index[1, positions.tolist()].tolist():
                    item = (edge_type[2], int(dst_idx))
                    if item in seen:
                        continue
                    seen.add(item)
                    parent[item] = (current, relation)
                    queue.append((item[0], item[1], depth + 1))

            for edge_type in self.in_edge_types_by_dst.get(current_type, []):
                edge_index = self.data[edge_type].edge_index
                sorted_values, sorted_positions = self.by_dst[edge_type]
                positions = self.positions_for(sorted_values, sorted_positions, current_idx)
                relation = f"inverse:{edge_type[1]}"
                if positions.size == 0:
                    continue
                for src_idx in edge_index[0, positions.tolist()].tolist():
                    item = (edge_type[0], int(src_idx))
                    if item in seen:
                        continue
                    seen.add(item)
                    parent[item] = (current, relation)
                    queue.append((item[0], item[1], depth + 1))

        rows: list[dict[str, Any]] = []
        for case_idx, node in hits[: self.provenance_limit]:
            case_id = str(self.data["case"].case_id[case_idx])
            file_name = str(self.data["case"].file_name[case_idx])
            label = self.label_names[int(self.data["case"].y[case_idx].item())]
            source_path = self.source_index.get(file_name)
            rows.append(
                {
                    "case_index": int(case_idx),
                    "case_id": case_id,
                    "file_name": file_name,
                    "label": label,
                    "label_display": label_display(label),
                    "source_path": str(source_path) if source_path else None,
                    "source_href": source_href(source_path),
                    "path": self.describe_path(node, parent),
                }
            )
        self.cache[key] = rows
        return rows

    def describe_path(
        self,
        node: tuple[str, int],
        parent: dict[tuple[str, int], tuple[tuple[str, int], str]],
    ) -> str:
        chunks: list[str] = [node[0]]
        current = node
        while current in parent:
            prev, relation = parent[current]
            chunks.append(f"<-{relation}-")
            chunks.append(prev[0])
            current = prev
        chunks.reverse()
        return " ".join(chunks)


def load_graph_provenance(
    graph_cache: Path | None,
    predictions_csv: Path | None,
    source_index: dict[str, Path],
    support_hops: int,
    provenance_limit: int,
) -> tuple[GraphProvenance | None, str]:
    if graph_cache is None:
        return None, "No graph cache path was supplied."
    if not graph_cache.exists():
        return None, f"Graph cache does not exist: {graph_cache}"
    try:
        import torch
    except Exception as exc:
        return None, f"Could not import torch; provenance disabled: {exc!r}"
    try:
        bundle = torch.load(graph_cache, map_location="cpu", weights_only=False)
        data = bundle["data"]
        metadata = bundle.get("metadata", {})
        label_names = [str(label) for label in metadata.get("label_names", ["-1", "1"])]
        if predictions_csv and predictions_csv.exists():
            predictions = pd.read_csv(predictions_csv, usecols=["split"])
            train_mask = predictions["split"].astype(str).to_numpy() == "train"
        elif hasattr(data["case"], "train_mask"):
            train_mask = data["case"].train_mask.detach().cpu().numpy().astype(bool)
        else:
            return None, "No predictions CSV or case.train_mask was available for training-case provenance."
        graph_bundle = GraphBundle(
            data=data,
            label_names=label_names,
            train_mask=train_mask,
            source_index=source_index,
            support_hops=support_hops,
            provenance_limit=provenance_limit,
        )
        return GraphProvenance(graph_bundle), "Graph provenance enabled."
    except Exception as exc:
        return None, f"Could not load graph cache; provenance disabled: {exc!r}"


def path_tokens(path_family: Any) -> list[str]:
    text = str(clean_value(path_family) or "")
    if not text:
        return []
    return [token for token in text.split("->") if token]


def relation_color(relation: str) -> str:
    palette = [
        "#2563eb",
        "#059669",
        "#d97706",
        "#be123c",
        "#7c3aed",
        "#0891b2",
        "#c2410c",
        "#4b5563",
    ]
    return palette[abs(hash(relation)) % len(palette)]


def node_color(node_type: str, fallback: str = "#94a3b8") -> str:
    return NODE_COLORS.get(node_type, fallback)


def add_node_once(graph: nx.DiGraph, node_id: str, **attrs: Any) -> None:
    if node_id not in graph:
        graph.add_node(node_id, **attrs)


def add_path_to_graph(
    graph: nx.DiGraph,
    case_node_id: str,
    row: dict[str, Any],
    max_delta: float,
    max_attention: float,
) -> str:
    group_id = str(row["group_id"])
    evidence_type = str(row.get("evidence_type") or "evidence")
    evidence_name = str(row.get("evidence_name") or evidence_type)
    support_text = support_distribution_text(row)
    delta = as_float(row.get("abs_delta_pred_proba")) or 0.0
    attention = as_float(row.get("attention_score")) or 0.0
    size = 18.0 + (38.0 * delta / max(max_delta, 1e-9))
    evidence_node = f"evidence::{group_id}"
    evidence_label = f"{short_text(evidence_name, 42)}\n{support_text}\ndelta {fmt_float(delta)}"
    add_node_once(
        graph,
        evidence_node,
        label=evidence_label,
        title=html.escape(
            f"{evidence_type}: {evidence_name}\nPath: {row.get('path_family')}\n"
            f"Delta: {fmt_float(row.get('delta_pred_proba'))}\n{support_text}"
        ),
        node_type=evidence_type,
        size=size,
        color=node_color(evidence_type),
        shape="dot" if evidence_type != "relation_type" else "diamond",
    )

    tokens = path_tokens(row.get("path_family"))
    edge_width = 1.2 + (6.0 * attention / max(max_attention, 1e-9))
    if len(tokens) >= 3 and len(tokens) % 2 == 1:
        chain: list[str] = []
        for index, token in enumerate(tokens):
            if index % 2 == 1:
                continue
            if token == "case":
                node_id = case_node_id
            elif evidence_type != "relation_type" and (
                index == 0 or index == len(tokens) - 1
            ) and token == evidence_type:
                node_id = evidence_node
            else:
                node_id = f"hop::{group_id}::{index}::{token}"
                add_node_once(
                    graph,
                    node_id,
                    label=token,
                    title=f"Intermediate {token} node on path {row.get('path_family')}",
                    node_type=token,
                    size=14,
                    color=node_color(token, NODE_COLORS["intermediate"]),
                    shape="ellipse",
                )
            chain.append(node_id)
        relations = tokens[1::2]
        for src, relation, dst in zip(chain, relations, chain[1:]):
            if src == dst:
                continue
            graph.add_edge(
                src,
                dst,
                label=relation,
                title=f"{relation}; attention={fmt_float(attention)}",
                color=relation_color(relation),
                width=edge_width,
            )
        if evidence_node not in chain:
            graph.add_edge(
                case_node_id,
                evidence_node,
                label="masked group",
                title="Relation-type or abstract evidence group in the case receptive field.",
                color="#64748b",
                width=edge_width,
            )
    else:
        graph.add_edge(
            case_node_id,
            evidence_node,
            label="in receptive field",
            title=f"Path family: {row.get('path_family')}",
            color="#64748b",
            width=edge_width,
        )
    return evidence_node


def graph_to_pyvis_html(graph: nx.DiGraph) -> str:
    net = Network(
        height="730px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#172033",
        cdn_resources="in_line",
    )
    net.set_options(
        json.dumps(
            {
                "interaction": {"hover": True, "navigationButtons": True},
                "physics": {
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -58,
                        "centralGravity": 0.012,
                        "springLength": 150,
                        "springConstant": 0.08,
                    },
                    "stabilization": {"iterations": 180},
                },
                "edges": {
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
                    "font": {"size": 10, "align": "middle"},
                    "smooth": {"type": "dynamic"},
                },
                "nodes": {
                    "font": {"size": 12, "multi": True},
                    "borderWidth": 1,
                },
            }
        )
    )
    for node_id, attrs in graph.nodes(data=True):
        net.add_node(
            node_id,
            label=str(attrs.get("label", node_id)),
            title=str(attrs.get("title", "")),
            color=str(attrs.get("color", "#94a3b8")),
            size=float(attrs.get("size", 16)),
            shape=str(attrs.get("shape", "dot")),
        )
    for src, dst, attrs in graph.edges(data=True):
        net.add_edge(
            src,
            dst,
            label=str(attrs.get("label", "")),
            title=str(attrs.get("title", "")),
            color=str(attrs.get("color", "#64748b")),
            width=float(attrs.get("width", 1.0)),
        )
    return net.generate_html(notebook=False)


def write_dot(graph: nx.DiGraph, path: Path) -> None:
    try:
        from networkx.drawing.nx_pydot import write_dot as nx_write_dot

        nx_write_dot(graph, path)
    except Exception as exc:
        path.with_suffix(".dot.error.txt").write_text(repr(exc), encoding="utf-8")


def build_report_rows(
    case_rows: pd.DataFrame,
    top_k: int,
    support_lookup: dict[tuple[str, str, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    work = case_rows.copy()
    work["abs_delta_pred_proba"] = pd.to_numeric(work["abs_delta_pred_proba"], errors="coerce")
    work["attention_score"] = pd.to_numeric(work["attention_score"], errors="coerce")
    cf = work.sort_values("abs_delta_pred_proba", ascending=False, na_position="last").head(top_k)
    att = (
        work[work["attention_score"].notna()]
        .sort_values("attention_score", ascending=False, na_position="last")
        .head(top_k)
    )
    cf_ranks = {str(row.group_id): rank for rank, row in enumerate(cf.itertuples(index=False), start=1)}
    att_ranks = {str(row.group_id): rank for rank, row in enumerate(att.itertuples(index=False), start=1)}
    ordered_ids: list[str] = []
    row_by_id: dict[str, dict[str, Any]] = {}
    for frame in (cf, att):
        for row in frame.to_dict("records"):
            group_id = str(row["group_id"])
            row_by_id[group_id] = row
            if group_id not in ordered_ids:
                ordered_ids.append(group_id)

    rows: list[dict[str, Any]] = []
    for group_id in ordered_ids:
        row = merge_support(row_by_id[group_id], support_lookup)
        marker_id, marker_symbol, marker_label, marker_css = marker_for(group_id, cf_ranks, att_ranks)
        row["counterfactual_rank"] = cf_ranks.get(group_id)
        row["attention_rank"] = att_ranks.get(group_id)
        row["marker"] = marker_id
        row["marker_symbol"] = marker_symbol
        row["marker_label"] = marker_label
        row["marker_css"] = marker_css
        row["counterfactual_rank_display"] = row["counterfactual_rank"] or "-"
        row["attention_rank_display"] = row["attention_rank"] or "-"
        row["delta_pred_proba_fmt"] = fmt_float(row.get("delta_pred_proba"))
        row["abs_delta_pred_proba_fmt"] = fmt_float(row.get("abs_delta_pred_proba"))
        row["attention_score_fmt"] = fmt_float(row.get("attention_score"))
        support_n = as_int(row.get("support_train_n"))
        row["support_train_n_display"] = "NA" if support_n is None else f"{support_n:,}"
        row["support_distribution_text"] = support_distribution_text(row)
        rows.append(row)
    return rows


def natural_language(summary: dict[str, Any], evidence_rows: list[dict[str, Any]]) -> str:
    if not evidence_rows:
        return "No evidence groups were available for this case."
    top = next((row for row in evidence_rows if row.get("counterfactual_rank") == 1), evidence_rows[0])
    delta = as_float(top.get("delta_pred_proba")) or 0.0
    direction = "supports" if delta >= 0 else "acts against"
    flip_text = ""
    if str(top.get("prediction_flipped")).lower() in {"true", "1", "yes"}:
        flip_text = " Masking this group also flips the predicted label."
    support_text = support_distribution_text(top)
    agree_count = sum(1 for row in evidence_rows if row.get("marker") == "both_agree")
    cf_only = sum(1 for row in evidence_rows if row.get("marker") == "counterfactual_only")
    att_only = sum(1 for row in evidence_rows if row.get("marker") == "attention_only")
    return (
        f"The frozen HGT predicts {summary['predicted_label_display']} with confidence "
        f"{summary['baseline_pred_proba_fmt']}. Its strongest counterfactual evidence is "
        f"the {top.get('evidence_type')} node or group '{short_text(top.get('evidence_name'), 140)}' "
        f"through path family {top.get('path_family')}. Removing that group changes the predicted-class "
        f"probability by {fmt_float(delta)}, so this evidence {direction} the original prediction. "
        f"The connected training neighbourhood reports {support_text}.{flip_text} "
        f"Attention and counterfactual rankings agree on {agree_count} displayed groups; "
        f"{cf_only} are counterfactual-only and {att_only} are attention-only diagnostics."
    )


def build_case_report(
    case_index: int,
    tables: dict[str, pd.DataFrame],
    support_lookup: dict[tuple[str, str, Any], dict[str, Any]],
    provenance: GraphProvenance | None,
    provenance_note: str,
    top_k: int,
) -> tuple[dict[str, Any], nx.DiGraph]:
    case_summary = tables["case_summary"]
    long = tables["long"]
    attention = tables["attention"]
    summary_rows = case_summary[case_summary["case_index"].astype(int) == int(case_index)]
    if summary_rows.empty:
        raise ValueError(f"case_index {case_index} not found in case_summary.csv")
    summary = {key: clean_value(value) for key, value in summary_rows.iloc[0].to_dict().items()}
    summary["target_label_display"] = label_display(summary.get("target_label"))
    summary["predicted_label_display"] = label_display(summary.get("baseline_pred_label"))
    summary["baseline_pred_proba_fmt"] = fmt_float(summary.get("baseline_pred_proba"))
    summary["baseline_positive_proba_fmt"] = fmt_float(summary.get("baseline_positive_proba"))

    case_rows = long[long["case_index"].astype(int) == int(case_index)].copy()
    if case_rows.empty:
        raise ValueError(f"case_index {case_index} not found in case_counterfactual_groups.csv")
    evidence_rows = build_report_rows(case_rows, top_k, support_lookup)

    attention_row = None
    if not attention.empty and "case_index" in attention:
        match = attention[attention["case_index"].astype(int) == int(case_index)]
        if not match.empty:
            attention_row = {key: clean_value(value) for key, value in match.iloc[0].to_dict().items()}

    graph = nx.DiGraph()
    case_node_id = f"case::{case_index}"
    graph.add_node(
        case_node_id,
        label=f"Target case\n{case_index}\nPred {summary['predicted_label_display']}",
        title=html.escape(str(summary.get("case_id"))),
        node_type="target_case",
        size=34,
        color=NODE_COLORS["target_case"],
        shape="star",
    )

    max_delta = max((as_float(row.get("abs_delta_pred_proba")) or 0.0 for row in evidence_rows), default=1.0)
    max_attention = max((as_float(row.get("attention_score")) or 0.0 for row in evidence_rows), default=1.0)
    evidence_node_by_group: dict[str, str] = {}
    for row in evidence_rows:
        evidence_node_by_group[str(row["group_id"])] = add_path_to_graph(
            graph, case_node_id, row, max_delta, max_attention
        )

    provenance_items: list[dict[str, Any]] = []
    for row in evidence_rows:
        if row.get("counterfactual_rank") is None:
            continue
        train_cases: list[dict[str, Any]] = []
        global_index = as_int(row.get("evidence_global_index"))
        evidence_type = str(row.get("evidence_type") or "")
        if provenance and global_index is not None and evidence_type != "relation_type":
            train_cases = provenance.connected_train_cases(evidence_type, global_index)
            evidence_node = evidence_node_by_group.get(str(row["group_id"]))
            if evidence_node:
                for train_case in train_cases[:2]:
                    train_node = f"train::{train_case['case_index']}"
                    add_node_once(
                        graph,
                        train_node,
                        label=f"Train {train_case['case_index']}\n{train_case['label_display']}",
                        title=html.escape(train_case["case_id"]),
                        node_type="train_case",
                        size=16,
                        color=NODE_COLORS["train_case"],
                        shape="box",
                    )
                    graph.add_edge(
                        evidence_node,
                        train_node,
                        label="source case",
                        title=train_case["path"],
                        color="#374151",
                        width=1.0,
                    )
        provenance_items.append(
            {
                "group_id": row.get("group_id"),
                "evidence_type": row.get("evidence_type"),
                "evidence_name": row.get("evidence_name"),
                "path_family": row.get("path_family"),
                "train_cases": train_cases,
            }
        )

    report = {
        "summary": summary,
        "attention": attention_row,
        "evidence_groups": evidence_rows,
        "provenance_available": provenance is not None,
        "provenance_note": provenance_note,
        "provenance": provenance_items,
        "natural_language_explanation": natural_language(summary, evidence_rows),
    }
    return report, graph


def write_case_artifacts(
    report: dict[str, Any],
    graph: nx.DiGraph,
    output_dir: Path,
    env: Environment,
) -> dict[str, Any]:
    case_index = int(report["summary"]["case_index"])
    slug = slugify(report["summary"]["case_id"])
    base = f"case_{case_index}_{slug}"
    report_dir = output_dir / "cases"
    graph_dir = output_dir / "graphs"
    json_dir = output_dir / "json"
    dot_dir = output_dir / "dot"
    for directory in (report_dir, graph_dir, json_dir, dot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    graph_html = graph_to_pyvis_html(graph)
    graph_path = graph_dir / f"{base}_graph.html"
    graph_path.write_text(graph_html, encoding="utf-8")
    dot_path = dot_dir / f"{base}.dot"
    write_dot(graph, dot_path)

    json_path = json_dir / f"{base}.json"
    json_payload = dict(report)
    json_payload["graph"] = {
        "nodes": [
            {"id": node_id, **{key: json_ready(value) for key, value in attrs.items()}}
            for node_id, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": src,
                "target": dst,
                **{key: json_ready(value) for key, value in attrs.items()},
            }
            for src, dst, attrs in graph.edges(data=True)
        ],
    }
    json_path.write_text(json.dumps(json_ready(json_payload), indent=2), encoding="utf-8")

    report["json_filename"] = os.path.relpath(json_path, report_dir)
    report["graph_filename"] = os.path.relpath(graph_path, report_dir)
    report["generated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    report_template = env.from_string(REPORT_TEMPLATE)
    html_path = report_dir / f"{base}.html"
    html_path.write_text(
        report_template.render(report=report, graph_html=graph_html),
        encoding="utf-8",
    )
    return {
        "case_index": case_index,
        "case_id": report["summary"]["case_id"],
        "target_label_display": report["summary"]["target_label_display"],
        "predicted_label_display": report["summary"]["predicted_label_display"],
        "confidence": report["summary"]["baseline_pred_proba_fmt"],
        "top_evidence": report["summary"].get("top_evidence_name"),
        "top_path": report["summary"].get("top_path_family"),
        "report_href": os.path.relpath(html_path, output_dir),
        "json_href": os.path.relpath(json_path, output_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-case traceability reports for FINAL_EXPLANATION outputs."
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-index", type=int, action="append", help="Case index to render. Repeatable.")
    parser.add_argument("--case-limit", type=int, default=1, help="Number of sample cases when --all is not set.")
    parser.add_argument("--all", action="store_true", help="Render every case in case_summary.csv.")
    parser.add_argument(
        "--sample-strategy",
        choices=["legal_authority", "importance", "first"],
        default="legal_authority",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Counterfactual and attention top-k union to display.")
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--support-hops", type=int, default=2)
    parser.add_argument("--provenance-limit", type=int, default=8)
    parser.add_argument("--no-provenance", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing source JSON judgments. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    explanation_dir = args.explanation_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = read_json(explanation_dir / "run_summary.json")
    if args.graph_cache is None:
        inferred_graph = first_summary_value(run_summary, "graph_cache")
        args.graph_cache = Path(inferred_graph) if inferred_graph else None
    if args.predictions_csv is None:
        inferred_predictions = first_summary_value(run_summary, "predictions_csv")
        args.predictions_csv = Path(inferred_predictions) if inferred_predictions else None

    source_roots = list(args.source_root)
    if not source_roots and DEFAULT_SOURCE_ROOT.exists():
        source_roots.append(DEFAULT_SOURCE_ROOT)
    print(f"[load] explanation_dir={explanation_dir}", flush=True)
    tables = read_tables(explanation_dir)
    selected_cases = select_case_indices(tables["case_summary"], args)
    print(f"[select] cases={len(selected_cases)} first={selected_cases[:5]}", flush=True)

    print(f"[load] source_roots={len(source_roots)}", flush=True)
    source_index = build_source_index(source_roots)
    print(f"[load] source_index_files={len(source_index)}", flush=True)

    provenance = None
    provenance_note = "Provenance was disabled."
    if not args.no_provenance:
        provenance, provenance_note = load_graph_provenance(
            args.graph_cache,
            args.predictions_csv,
            source_index,
            args.support_hops,
            args.provenance_limit,
        )
    print(f"[provenance] {provenance_note}", flush=True)

    support_lookup = build_support_lookup(tables["support"])
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    index_rows: list[dict[str, Any]] = []
    for ordinal, case_index in enumerate(selected_cases, start=1):
        report, graph = build_case_report(
            case_index=case_index,
            tables=tables,
            support_lookup=support_lookup,
            provenance=provenance,
            provenance_note=provenance_note,
            top_k=args.top_k,
        )
        row = write_case_artifacts(report, graph, output_dir, env)
        index_rows.append(row)
        if ordinal % 25 == 0 or ordinal == len(selected_cases):
            print(f"[progress] rendered={ordinal}/{len(selected_cases)}", flush=True)

    index_template = env.from_string(INDEX_TEMPLATE)
    (output_dir / "index.html").write_text(
        index_template.render(rows=index_rows, explanation_dir=str(explanation_dir)),
        encoding="utf-8",
    )
    manifest = {
        "explanation_dir": str(explanation_dir),
        "output_dir": str(output_dir),
        "case_count": len(index_rows),
        "top_k": args.top_k,
        "graph_cache": str(args.graph_cache) if args.graph_cache else None,
        "predictions_csv": str(args.predictions_csv) if args.predictions_csv else None,
        "provenance_note": provenance_note,
        "index": str(output_dir / "index.html"),
        "cases": index_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2),
        encoding="utf-8",
    )
    print(f"[done] index={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
