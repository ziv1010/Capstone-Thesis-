"""
Misclassification diagnostic.

For one case, take each PGE-surfaced top node and ask: in the *training set*,
what is the label distribution of cases that connect to this same node via the
same edge type? If the model misclassified the case, the answer typically shows
that the evidence the explainer surfaced is dominated by training neighbours of
the OTHER class — that is the quantitative reason for the wrong prediction.

Usage:
    python scripts/phase6_misclass_diagnostic.py --case-index 32302
    python scripts/phase6_misclass_diagnostic.py --auto-misclassified 5
    python scripts/phase6_misclass_diagnostic.py --all --skip-plots
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyser.loader import load_config, load_graph_cache, validate_case_ids_bucket  # noqa: E402
from analyser.hetero_utils import get_split_indices, load_fold_splits  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
LEGAL_NEIGHBOUR_NODE_TYPES = {"statute", "provision", "precedent"}


def parse_edge_type(edge_str: str) -> tuple[str, str, str]:
    src, rel, dst = edge_str.split("|")
    return src, rel, dst


def trace_cases_for_node(
    data,
    node_type: str,
    node_index: int,
    edge_type_str: str,
) -> set[int]:
    """Return the set of case node indices that reach `node_index` of `node_type`
    via the surfaced edge type. The surfaced edge can point either toward or
    away from the surfaced node. It is usually one hop short of the case
    (e.g. arguments -> cites_statute -> statute or the reverse relation), so
    we hop one more time along the ownership edge (case -> has_X -> X) to get
    back to cases.
    """
    src, rel, dst = parse_edge_type(edge_type_str)
    if dst == node_type:
        surfaced_side = 1
        intermediate_side = 0
        intermediate_type = src
    elif src == node_type:
        surfaced_side = 0
        intermediate_side = 1
        intermediate_type = dst
    else:
        raise ValueError(f"edge {edge_type_str} does not touch node_type {node_type}")

    edge_index = data.edge_index_dict[(src, rel, dst)]
    mask = edge_index[surfaced_side] == node_index
    intermediate_nodes = edge_index[intermediate_side][mask].tolist()
    if not intermediate_nodes:
        return set()

    if intermediate_type == "case":
        return set(intermediate_nodes)

    # Find the case ownership edge (case, has_*, intermediate_type)
    case_edge_key = None
    for et in data.edge_index_dict:
        if et[0] == "case" and et[2] == intermediate_type and et[1].startswith("has_"):
            case_edge_key = et
            break
    if case_edge_key is None:
        return set()

    case_ei = data.edge_index_dict[case_edge_key]
    intermediate_set = set(intermediate_nodes)
    # vectorise
    interm_tensor = torch.tensor(sorted(intermediate_set), dtype=case_ei.dtype)
    membership = torch.isin(case_ei[1], interm_tensor)
    cases = case_ei[0][membership].unique().tolist()
    return set(cases)


def build_training_case_mask(metadata, num_cases: int) -> torch.Tensor:
    """Boolean mask of length num_cases marking training cases."""
    case_nm = metadata["node_mappings"]["case"]
    splits = metadata["split_assignments"]
    mask = torch.zeros(num_cases, dtype=torch.bool)
    for raw_id, split in splits.items():
        if split != "train":
            continue
        idx = case_nm.get(f"case::{raw_id}")
        if idx is not None:
            mask[idx] = True
    return mask


def build_training_case_mask_from_indices(
    train_indices: list[int], num_cases: int
) -> torch.Tensor:
    mask = torch.zeros(num_cases, dtype=torch.bool)
    if train_indices:
        mask[torch.tensor(train_indices, dtype=torch.long)] = True
    return mask


def label_distribution(
    case_indices: set[int], y: torch.Tensor, train_mask: torch.Tensor
) -> tuple[int, int, int]:
    """(n_train_neighbours, n_label0, n_label1)."""
    if not case_indices:
        return (0, 0, 0)
    idx_tensor = torch.tensor(sorted(case_indices), dtype=torch.long)
    keep = train_mask[idx_tensor]
    train_idx = idx_tensor[keep]
    if train_idx.numel() == 0:
        return (0, 0, 0)
    labels = y[train_idx]
    n0 = int((labels == 0).sum().item())
    n1 = int((labels == 1).sum().item())
    return (int(train_idx.numel()), n0, n1)


def connected_training_case_rows(
    case_indices: set[int],
    data,
    train_mask: torch.Tensor,
    label_names: list[str],
    limit: int,
) -> list[dict]:
    if not case_indices:
        return []
    case_ids = list(data["case"].case_id)
    file_names = list(getattr(data["case"], "file_name", []))
    y = data["case"].y
    train_indices = [idx for idx in sorted(case_indices) if bool(train_mask[idx])]
    if limit > 0:
        train_indices = train_indices[:limit]

    out = []
    for idx in train_indices:
        label_idx = int(y[idx].item())
        item = {
            "case_node_index": int(idx),
            "case_id": str(case_ids[idx]),
            "label": label_names[label_idx],
        }
        if file_names:
            item["file_name"] = str(file_names[idx])
        out.append(item)
    return out


def calculate_weighted_evidence(rows: list[dict], label_names: list[str]) -> dict:
    head_neg, head_pos = label_names
    weighted_n0 = 0.0
    weighted_n1 = 0.0
    importance_total = 0.0

    for row in rows:
        if row["n_train_neighbours"] <= 0:
            continue
        imp = float(row["importance"])
        weighted_n0 += imp * float(row[f"pct_{head_neg}"])
        weighted_n1 += imp * float(row[f"pct_{head_pos}"])
        importance_total += imp

    has_traceable_evidence = importance_total > 0
    if has_traceable_evidence:
        ev0 = weighted_n0 / importance_total
        ev1 = weighted_n1 / importance_total
    else:
        ev0 = ev1 = 0.0

    if not has_traceable_evidence:
        evidence_majority = "untraceable"
    elif ev0 > ev1:
        evidence_majority = head_neg
    elif ev1 > ev0:
        evidence_majority = head_pos
    else:
        evidence_majority = "tie"

    total_importance = sum(float(row["importance"]) for row in rows)
    return {
        f"pct_{head_neg}": ev0,
        f"pct_{head_pos}": ev1,
        "majority_class": evidence_majority,
        "strength": abs(ev0 - ev1),
        "has_traceable_evidence": has_traceable_evidence,
        "n_nodes": len(rows),
        "n_traceable_nodes": sum(1 for row in rows if row["n_train_neighbours"] > 0),
        "traceable_importance_share": (
            importance_total / total_importance if total_importance > 0 else 0.0
        ),
    }


def build_diagnostic_rows(
    node_groups: dict,
    data,
    train_mask: torch.Tensor,
    label_names: list[str],
    trace_cache: dict[tuple[str, int, str], set[int]] | None = None,
    source_scope: str = "legal",
    connected_case_limit: int = 50,
) -> dict:
    y = data["case"].y
    rows = []

    for node_type, nodes in (node_groups or {}).items():
        if not isinstance(nodes, list):
            continue
        for n in nodes:
            edge_type = n.get("edge_type")
            if not edge_type:
                continue
            node_index = int(n["node_index"])
            cache_key = (node_type, node_index, edge_type)
            if trace_cache is not None and cache_key in trace_cache:
                cases = trace_cache[cache_key]
            else:
                try:
                    cases = trace_cases_for_node(
                        data,
                        node_type=node_type,
                        node_index=node_index,
                        edge_type_str=edge_type,
                    )
                except (KeyError, ValueError):
                    cases = set()
                if trace_cache is not None:
                    trace_cache[cache_key] = cases
            n_train, n0, n1 = label_distribution(cases, y, train_mask)
            total = n0 + n1
            if total == 0:
                pct0 = pct1 = 0.0
            else:
                pct0 = n0 / total
                pct1 = n1 / total
            imp = float(n["importance"])
            target_direct = bool(n.get("target_direct", False))
            if n_train > 0:
                trace_status = "traceable"
            elif target_direct:
                trace_status = "case_local_untraceable"
            else:
                trace_status = "shared_untraceable"
            row = {
                "node_type": node_type,
                "node_index": node_index,
                "text": n.get("text", ""),
                "edge_type": edge_type,
                "importance": imp,
                "n_connected_cases": len(cases),
                "n_train_neighbours": n_train,
                f"label_{label_names[0]}": n0,
                f"label_{label_names[1]}": n1,
                f"pct_{label_names[0]}": pct0,
                f"pct_{label_names[1]}": pct1,
                "majority_class": (
                    label_names[0] if n0 > n1 else
                    (label_names[1] if n1 > n0 else "tie")
                ),
                "skew_strength": abs(pct0 - pct1),
                "target_direct": target_direct,
                "connection_scope": n.get("connection_scope", ""),
                "trace_status": trace_status,
                "source_scope": source_scope,
            }
            if source_scope == "legal" and node_type in LEGAL_NEIGHBOUR_NODE_TYPES:
                row["connected_train_cases_limit"] = connected_case_limit
                row["connected_train_cases"] = connected_training_case_rows(
                    cases,
                    data=data,
                    train_mask=train_mask,
                    label_names=label_names,
                    limit=connected_case_limit,
                )
            rows.append(row)

    rows.sort(key=lambda r: r["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["importance_rank"] = rank
    return rows


def topk_sweep(
    rows: list[dict],
    label_names: list[str],
    predicted_label: str,
    cutoffs: list[int],
) -> list[dict]:
    ranked = sorted(rows, key=lambda r: r["importance"], reverse=True)
    out = []
    for cutoff in cutoffs:
        subset = []
        for row in ranked[:cutoff]:
            item = dict(row)
            item.pop("connected_train_cases", None)
            subset.append(item)
        weighted = calculate_weighted_evidence(subset, label_names)
        majority = weighted["majority_class"]
        out.append(
            {
                "k": cutoff,
                "n_ranked_nodes_available": len(ranked),
                "n_nodes_used": len(subset),
                "weighted_evidence": weighted,
                "evidence_majority": majority,
                "evidence_strength": weighted["strength"],
                "supports_prediction": majority == predicted_label,
                "per_node": subset,
            }
        )
    return out


def diagnose_case(
    explanation: dict,
    data,
    metadata,
    train_mask: torch.Tensor,
    label_names: list[str],
    trace_cache: dict[tuple[str, int, str], set[int]] | None = None,
    top_k_cutoffs: list[int] | None = None,
    connected_case_limit: int = 50,
) -> dict:
    target = explanation["target_label"]
    predicted = explanation["predicted_label"]
    misclassified = target != predicted and target != "?"
    cutoffs = top_k_cutoffs or [3, 5, 7]

    graph_node_groups = explanation.get("top_graph_nodes") or {}
    legal_node_groups = explanation.get("top_nodes") or {}
    diagnostic_scope = "full_graph" if graph_node_groups else "legal"

    graph_rows = build_diagnostic_rows(
        graph_node_groups,
        data=data,
        train_mask=train_mask,
        label_names=label_names,
        trace_cache=trace_cache,
        source_scope="full_graph",
        connected_case_limit=connected_case_limit,
    )
    legal_rows = build_diagnostic_rows(
        legal_node_groups,
        data=data,
        train_mask=train_mask,
        label_names=label_names,
        trace_cache=trace_cache,
        source_scope="legal",
        connected_case_limit=connected_case_limit,
    )

    rows = graph_rows if graph_node_groups else legal_rows
    traceable_graph_rows = [row for row in graph_rows if row["n_train_neighbours"] > 0]
    traceable_legal_rows = [row for row in legal_rows if row["n_train_neighbours"] > 0]
    weighted_evidence = calculate_weighted_evidence(rows, label_names)
    legal_weighted_evidence = calculate_weighted_evidence(legal_rows, label_names)
    full_graph_weighted_evidence = calculate_weighted_evidence(graph_rows, label_names)

    return {
        "case_id": explanation["case_id"],
        "case_node_index": explanation["case_node_index"],
        "target_label": target,
        "predicted_label": predicted,
        "confidence": explanation["confidence"],
        "misclassified": misclassified,
        "diagnostic_scope": diagnostic_scope,
        "connected_case_limit": connected_case_limit,
        "weighted_evidence": weighted_evidence,
        "per_node": rows,
        "full_graph_weighted_evidence": full_graph_weighted_evidence,
        "full_graph_per_node": graph_rows,
        "legal_weighted_evidence": legal_weighted_evidence,
        "legal_per_node": legal_rows,
        "traceable_full_graph_weighted_evidence": calculate_weighted_evidence(
            traceable_graph_rows, label_names
        ),
        "traceable_full_graph_per_node": traceable_graph_rows,
        "traceable_legal_weighted_evidence": calculate_weighted_evidence(
            traceable_legal_rows, label_names
        ),
        "traceable_legal_per_node": traceable_legal_rows,
        "top_k_cutoffs": cutoffs,
        "topk_diagnostics": {
            "full_graph": topk_sweep(graph_rows, label_names, predicted, cutoffs),
            "legal": topk_sweep(legal_rows, label_names, predicted, cutoffs),
            "traceable_full_graph": topk_sweep(
                traceable_graph_rows, label_names, predicted, cutoffs
            ),
            "traceable_legal": topk_sweep(
                traceable_legal_rows, label_names, predicted, cutoffs
            ),
        },
    }


def render_markdown(diag: dict, label_names: list[str]) -> str:
    head_neg, head_pos = label_names
    lines = []
    status = "MISCLASSIFIED" if diag["misclassified"] else "correct"
    lines.append(f"# Diagnostic: {diag['case_id']}")
    lines.append("")
    lines.append(f"- **Status**: {status}")
    lines.append(f"- **Target label**: `{diag['target_label']}`")
    lines.append(
        f"- **Predicted label**: `{diag['predicted_label']}`  "
        f"(confidence {diag['confidence']:.3f})"
    )
    we = diag["weighted_evidence"]
    scope_label = "full graph" if diag.get("diagnostic_scope") == "full_graph" else "legal"
    lines.append("")
    lines.append(f"## Where the explainer's {scope_label} evidence points (in the training set)")
    lines.append("")
    lines.append(
        f"Aggregating across traceable top PGE nodes, weighted by their importance:\n\n"
        f"- **{head_neg}**: {we[f'pct_{head_neg}']:.1%}\n"
        f"- **{head_pos}**: {we[f'pct_{head_pos}']:.1%}\n"
        f"- Majority class of evidence: **{we['majority_class']}** "
        f"(skew {we['strength']:.1%})\n"
        f"- Traceable nodes: {we.get('n_traceable_nodes', 0)} / {we.get('n_nodes', 0)}"
    )
    lines.append("")
    if we["majority_class"] == "untraceable":
        lines.append(
            "> **Reading**: the surfaced nodes in this scope are important to "
            "the PGExplainer ranking, but they do not connect back to training "
            "cases under the current trace rule. Treat them as case-local or "
            "untraceable prediction factors, not as label-distribution evidence."
        )
    elif diag["misclassified"]:
        if we["majority_class"] == diag["predicted_label"]:
            mc = we["majority_class"]
            mc_pct = we[f"pct_{mc}"]
            lines.append(
                f"> **Reading**: the model predicted `{diag['predicted_label']}` "
                f"because the surfaced {scope_label} nodes it leaned on "
                f"appear in training cases that are dominantly **{mc}** "
                f"({mc_pct:.1%}). The true label "
                f"`{diag['target_label']}` is under-represented among the training "
                f"neighbours of this case's most informative nodes."
            )
        else:
            lines.append(
                "> **Reading**: the dominant base-rate of the surfaced evidence "
                "does NOT match the prediction, so the misclassification likely "
                "comes from signal outside the currently surfaced PGE nodes."
            )
    lines.append("")
    topk = diag.get("topk_diagnostics", {})
    for scope_key, title in [
        ("full_graph", "Full Graph Top-k Sweep"),
        ("legal", "Legal-only Top-k Sweep"),
        ("traceable_full_graph", "Traceable Full-graph Top-k Sweep"),
    ]:
        sweep = topk.get(scope_key) or []
        if not sweep:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| k | Evidence Majority | Skew | Traceable Nodes | Supports Prediction |")
        lines.append("|---:|---|---:|---:|---|")
        for item in sweep:
            item_we = item.get("weighted_evidence", {})
            lines.append(
                f"| {item.get('k')} | {item_we.get('majority_class', 'untraceable')} | "
                f"{float(item_we.get('strength', 0.0)):.1%} | "
                f"{item_we.get('n_traceable_nodes', 0)} / {item_we.get('n_nodes', 0)} | "
                f"{'yes' if item.get('supports_prediction') else 'no'} |"
            )
        lines.append("")
    legal_rows = [
        r for r in diag.get("legal_per_node", [])
        if r.get("node_type") in LEGAL_NEIGHBOUR_NODE_TYPES
    ]
    if legal_rows:
        lines.append("## Legal shared-case neighbours")
        lines.append("")
        lines.append("| Type | Text | Train cases | Labels | Materialised cases |")
        lines.append("|---|---|---:|---|---:|")
        for r in legal_rows:
            text = r["text"][:60].replace("|", "\\|")
            shown = len(r.get("connected_train_cases", []))
            total = int(r.get("n_train_neighbours", 0) or 0)
            lines.append(
                f"| {r['node_type']} | {text} | {total} | "
                f"{head_neg}: {r.get(f'label_{head_neg}', 0)}, "
                f"{head_pos}: {r.get(f'label_{head_pos}', 0)} | "
                f"{shown} / {total} |"
            )
        lines.append("")
    lines.append("## Per-node evidence breakdown")
    lines.append("")
    lines.append(
        "| Type | Text | Importance | Train cases | "
        f"{head_neg} | {head_pos} | Majority | Skew |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    for r in diag["per_node"]:
        text = r["text"][:60].replace("|", "\\|")
        lines.append(
            f"| {r['node_type']} | {text} | {r['importance']:.3f} | "
            f"{r['n_train_neighbours']} | "
            f"{r[f'label_{head_neg}']} ({r[f'pct_{head_neg}']:.0%}) | "
            f"{r[f'label_{head_pos}']} ({r[f'pct_{head_pos}']:.0%}) | "
            f"{r['majority_class']} | {r['skew_strength']:.0%} |"
        )
    lines.append("")
    return "\n".join(lines)


CATEGORY_COLORS = {
    "statute": "#2ca02c",
    "provision": "#9467bd",
    "precedent": "#ff7f0e",
}
WIN_COLOR = "#1f77b4"   # blue, label = 1
LOSE_COLOR = "#d62728"  # red,  label = -1
NEUTRAL_COLOR = "#bdbdbd"
TARGET_COLOR = "#111111"


def render_diagnostic_subgraph(
    diag: dict, label_names: list[str], out_path: Path, max_nodes: int = 12
) -> None:
    """Simple horizontal bar chart. One row per top PGE node, sorted by
    importance. Each row shows what fraction of training cases that connect
    to that node carry label W (blue) vs L (red). Bottom row aggregates
    the importance-weighted evidence and points at the prediction.
    """
    head_neg, head_pos = label_names  # ["-1", "1"]

    items = sorted(diag["per_node"], key=lambda r: r["importance"], reverse=True)
    items = items[:max_nodes]
    if not items:
        return

    target_lbl = diag["target_label"]
    pred_lbl = diag["predicted_label"]
    conf = diag["confidence"]
    misclass = diag["misclassified"]
    verdict = "MISCLASSIFIED" if misclass else "CORRECT"
    we = diag["weighted_evidence"]
    we_neg = we[f"pct_{head_neg}"]
    we_pos = we[f"pct_{head_pos}"]
    we_majority = we["majority_class"]

    n = len(items)
    fig, ax = plt.subplots(figsize=(12, 1.0 + 0.45 * (n + 2)))

    y_top_node = n  # rows go n..1 (top to bottom for top-imp first)
    for i, r in enumerate(items):
        y = n - i
        n_neg = r[f"label_{head_neg}"]
        n_pos = r[f"label_{head_pos}"]
        total = n_neg + n_pos
        if total == 0:
            ax.barh(y, 1.0, color=NEUTRAL_COLOR, height=0.7, alpha=0.5)
            ax.text(0.5, y, "no training neighbours",
                    ha="center", va="center", fontsize=8, color="#444")
        else:
            pct_pos = n_pos / total
            pct_neg = n_neg / total
            ax.barh(y, pct_pos, color=WIN_COLOR, height=0.7)
            ax.barh(y, pct_neg, left=pct_pos, color=LOSE_COLOR, height=0.7)
            if pct_pos >= 0.08:
                ax.text(pct_pos / 2, y, f"W {int(round(pct_pos*100))}%",
                        ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
            if pct_neg >= 0.08:
                ax.text(pct_pos + pct_neg / 2, y,
                        f"L {int(round(pct_neg*100))}%",
                        ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")

        type_tag = r["node_type"][:4].upper()
        text = str(r["text"])[:38]
        ax.text(-0.02, y, f"[{type_tag}] {text}",
                ha="right", va="center", fontsize=9)
        ax.text(1.02, y,
                f"imp {r['importance']:.2f}  ·  {total} train cases",
                ha="left", va="center", fontsize=8, color="#444")

    sep_y = 0.55
    ax.axhline(sep_y, color="#888", linewidth=0.8, linestyle="--")

    y_agg = 0
    if we_pos + we_neg > 0:
        ax.barh(y_agg, we_pos, color=WIN_COLOR, height=0.7)
        ax.barh(y_agg, we_neg, left=we_pos, color=LOSE_COLOR, height=0.7)
        if we_pos >= 0.08:
            ax.text(we_pos / 2, y_agg,
                    f"W {int(round(we_pos*100))}%",
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
        if we_neg >= 0.08:
            ax.text(we_pos + we_neg / 2, y_agg,
                    f"L {int(round(we_neg*100))}%",
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
    ax.text(-0.02, y_agg, "WEIGHTED EVIDENCE",
            ha="right", va="center", fontsize=10, fontweight="bold")
    pred_word = "WINNING" if pred_lbl == head_pos else "LOSING"
    true_word = "WINNING" if target_lbl == head_pos else "LOSING"
    ev_word = (
        "WINNING" if we_majority == head_pos else
        ("LOSING" if we_majority == head_neg else "TIED")
    )

    if misclass:
        if we_majority == pred_lbl:
            sub = (
                f"Evidence skews {ev_word} → model predicted {pred_word} → "
                f"true label {true_word}.  This explains the misclassification."
            )
            sub_color = "#a30000"
        elif we_majority == "tie":
            sub = (
                f"Evidence is tied but model predicted {pred_word} → "
                "PGE evidence does NOT explain this miss."
            )
            sub_color = "#555"
        else:
            sub = (
                f"Evidence skews {ev_word} but model predicted {pred_word} → "
                f"PGE evidence does NOT explain this miss."
            )
            sub_color = "#555"
    else:
        if we_majority == "tie":
            sub = f"Evidence is tied; model predicted {pred_word}. Correct."
        else:
            sub = (
                f"Evidence skews {ev_word} → matches prediction {pred_word}. "
                f"Correct."
            )
        sub_color = "#005c12"

    title = (
        f"Why did the model predict {pred_word}? — case {diag['case_node_index']}\n"
        f"true={target_lbl}  ·  pred={pred_lbl} (conf {conf:.2f})  ·  {verdict}"
    )
    ax.set_title(title, fontsize=12, pad=14, fontweight="bold")
    fig.text(0.5, 0.02, sub, ha="center", va="bottom",
             fontsize=10, color=sub_color, style="italic", wrap=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n + 0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Share of training cases linked to this node, by label",
                  fontsize=9)

    legend = [
        mpatches.Patch(color=WIN_COLOR, label=f"WINNING (label {head_pos})"),
        mpatches.Patch(color=LOSE_COLOR, label=f"LOSING (label {head_neg})"),
        mpatches.Patch(color=NEUTRAL_COLOR, label="no training neighbours"),
    ]
    ax.legend(handles=legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8, frameon=False)

    fig.tight_layout(rect=(0.18, 0.06, 0.96, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_case_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def iter_case_paths(explanations_dir: Path) -> list[Path]:
    return sorted(explanations_dir.glob("case_*.json"), key=parse_case_index)


def find_all_cases(explanations_dir: Path) -> list[int]:
    return [parse_case_index(p) for p in iter_case_paths(explanations_dir)]


def find_misclassified(explanations_dir: Path, n: int) -> list[int]:
    out = []
    for p in iter_case_paths(explanations_dir):
        with open(p) as f:
            d = json.load(f)
        if d["target_label"] != d["predicted_label"] and d["target_label"] != "?":
            out.append(int(d["case_node_index"]))
            if len(out) >= n:
                break
    return out


def find_correctly_classified(explanations_dir: Path, n: int) -> list[int]:
    out = []
    for p in iter_case_paths(explanations_dir):
        with open(p) as f:
            d = json.load(f)
        if d["target_label"] == d["predicted_label"] and d["target_label"] != "?":
            out.append(int(d["case_node_index"]))
            if len(out) >= n:
                break
    return out


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def topk_summary_for_scope(diag: dict, scope: str) -> dict[str, dict]:
    out = {}
    for item in diag.get("topk_diagnostics", {}).get(scope, []):
        weighted = item.get("weighted_evidence", {})
        out[str(item.get("k"))] = {
            "evidence_majority": item.get("evidence_majority"),
            "evidence_strength": item.get("evidence_strength"),
            "supports_prediction": item.get("supports_prediction"),
            "n_nodes_used": item.get("n_nodes_used"),
            "n_traceable_nodes": weighted.get("n_traceable_nodes", 0),
            "traceable_importance_share": weighted.get("traceable_importance_share", 0.0),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--graph-cache", type=str, default=None)
    ap.add_argument("--phase4-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--case-index", type=int, action="append", default=[])
    ap.add_argument("--all", action="store_true",
                    help="Run diagnostics for every Phase-4 case bundle.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap selected targets after all selectors are applied.")
    ap.add_argument("--auto-misclassified", type=int, default=0,
                    help="Pick this many misclassified cases automatically.")
    ap.add_argument("--auto-correct", type=int, default=0,
                    help="Pick this many correctly classified cases automatically.")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Write JSON/Markdown diagnostics without PNG charts.")
    ap.add_argument(
        "--connected-case-limit",
        type=int,
        default=50,
        help=(
            "Number of connected training cases to store per surfaced legal node. "
            "Use 0 to write every connected training case."
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    expected_bucket = cfg.get("bucket")
    extraction_cfg = cfg.get("extraction", {})
    top_k_cutoffs = [
        int(k) for k in extraction_cfg.get("top_k_diagnostic_cutoffs", [3, 5, 7])
    ]
    graph_cache = Path(args.graph_cache or cfg["graph_cache"])
    output_root = Path(cfg.get("output_root", ROOT / "outputs"))
    explanations_dir = Path(
        args.phase4_dir or output_root / "phase4_explanations" / "cases"
    )
    report_dir = Path(args.output_dir or output_root / "phase6_misclass_diagnostic")

    targets = list(args.case_index)
    if args.all:
        targets.extend(find_all_cases(explanations_dir))
    if args.auto_misclassified:
        targets.extend(find_misclassified(explanations_dir, args.auto_misclassified))
    if args.auto_correct:
        targets.extend(find_correctly_classified(explanations_dir, args.auto_correct))
    targets = list(dict.fromkeys(targets))
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        ap.error("Provide --case-index, --all, --auto-misclassified N, or --auto-correct N")

    print(f"Loading graph cache: {graph_cache}")
    data, metadata = load_graph_cache(graph_cache, expected_bucket=expected_bucket)
    label_names = metadata["label_names"]
    fold_predictions_csv = Path(cfg["checkpoint_dir"]) / "predictions.csv"
    if fold_predictions_csv.exists():
        split_indices = load_fold_splits(
            str(fold_predictions_csv),
            data,
            expected_bucket=expected_bucket,
        )
        print(f"Using fold-specific train split from {fold_predictions_csv}")
    else:
        split_indices = get_split_indices(data)
        print("Using graph-cache train split masks")
    train_mask = build_training_case_mask_from_indices(
        split_indices.get("train", []),
        data["case"].num_nodes,
    )
    print(f"Train mask: {int(train_mask.sum())} cases")

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    trace_cache: dict[tuple[str, int, str], set[int]] = {}
    verbose_cases = len(targets) <= 50
    for ordinal, idx in enumerate(targets, start=1):
        path = explanations_dir / f"case_{idx}.json"
        if not path.exists():
            print(f"  skip: {path} missing")
            continue
        with open(path) as f:
            explanation = json.load(f)
        validate_case_ids_bucket(
            [str(explanation.get("case_id", ""))],
            expected_bucket,
            f"Phase 6 explanation bundle {path}",
        )
        graph_case_id = str(data["case"].case_id[int(idx)])
        if str(explanation.get("case_id", "")) != graph_case_id:
            raise ValueError(
                f"Phase 6 stale bundle mismatch for case_{idx}: "
                f"bundle case_id={explanation.get('case_id')!r}, "
                f"graph case_id={graph_case_id!r}"
            )
        diag = diagnose_case(
            explanation,
            data,
            metadata,
            train_mask,
            label_names,
            trace_cache,
            top_k_cutoffs=top_k_cutoffs,
            connected_case_limit=args.connected_case_limit,
        )

        json_out = report_dir / f"case_{idx}.json"
        md_out = report_dir / f"case_{idx}.md"
        png_out = report_dir / f"case_{idx}_subgraph.png"
        json_out.write_text(json.dumps(diag, indent=2, ensure_ascii=False))
        md_out.write_text(render_markdown(diag, label_names))
        if not args.skip_plots:
            render_diagnostic_subgraph(diag, label_names, png_out)

        we = diag["weighted_evidence"]
        agree = we["majority_class"] == diag["predicted_label"]
        if verbose_cases:
            print(
                f"  case {idx}: target={diag['target_label']} "
                f"pred={diag['predicted_label']} "
                f"evidence_majority={we['majority_class']} "
                f"strength={we['strength']:.1%} "
                f"{'(supports prediction)' if agree else '(does NOT support prediction)'}"
                f"  ->  {display_path(md_out)}"
            )
        elif ordinal == 1 or ordinal % 100 == 0 or ordinal == len(targets):
            print(f"  processed {ordinal}/{len(targets)} cases", flush=True)
        summary_rows.append(
            {
                "case_node_index": idx,
                "bucket": expected_bucket,
                "case_id": diag["case_id"],
                "target_label": diag["target_label"],
                "predicted_label": diag["predicted_label"],
                "confidence": diag["confidence"],
                "diagnostic_scope": diag["diagnostic_scope"],
                "evidence_majority": we["majority_class"],
                "evidence_strength": we["strength"],
                "evidence_supports_prediction": agree,
                "n_nodes": we.get("n_nodes", 0),
                "n_traceable_nodes": we.get("n_traceable_nodes", 0),
                "traceable_importance_share": we.get("traceable_importance_share", 0.0),
                "topk_full_graph": topk_summary_for_scope(diag, "full_graph"),
                "topk_legal": topk_summary_for_scope(diag, "legal"),
                "topk_traceable_full_graph": topk_summary_for_scope(
                    diag, "traceable_full_graph"
                ),
            }
        )

    (report_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {len(summary_rows)} reports under {report_dir}")


if __name__ == "__main__":
    main()
