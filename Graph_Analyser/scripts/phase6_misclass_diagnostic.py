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
from analyser.loader import load_graph_cache  # noqa: E402

GRAPH_CACHE = (
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/data/"
    "timed_bucket_runs/cross_bucket_total_dataset/graph_cache/"
    "case_star_global_graph_cross_bucket_party_args.reasoning_focused.pt"
)
EXPLANATIONS_DIR = ROOT / "outputs" / "phase4_explanations" / "cases"
REPORT_DIR = ROOT / "outputs" / "phase6_misclass_diagnostic"


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
    via the surfaced edge type. The surfaced edge is one hop short of the case
    (e.g. arguments -> cites_statute -> statute), so we hop one more time
    along the reverse ownership edge (case -> has_X -> X) to get back to cases.
    """
    src, rel, dst = parse_edge_type(edge_type_str)
    assert dst == node_type, f"edge {edge_type_str} dst != node_type {node_type}"

    edge_index = data.edge_index_dict[(src, rel, dst)]
    mask = edge_index[1] == node_index
    intermediate_nodes = edge_index[0][mask].tolist()
    if not intermediate_nodes:
        return set()

    if src == "case":
        return set(intermediate_nodes)

    # Find the case ownership edge (case, has_*, src)
    case_edge_key = None
    for et in data.edge_index_dict:
        if et[0] == "case" and et[2] == src and et[1].startswith("has_"):
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


def diagnose_case(
    explanation: dict,
    data,
    metadata,
    train_mask: torch.Tensor,
    label_names: list[str],
) -> dict:
    y = data["case"].y
    target = explanation["target_label"]
    predicted = explanation["predicted_label"]
    misclassified = target != predicted and target != "?"

    rows = []
    weighted_n0 = 0.0
    weighted_n1 = 0.0
    importance_total = 0.0

    for node_type, nodes in explanation["top_nodes"].items():
        for n in nodes:
            cases = trace_cases_for_node(
                data,
                node_type=node_type,
                node_index=int(n["node_index"]),
                edge_type_str=n["edge_type"],
            )
            n_train, n0, n1 = label_distribution(cases, y, train_mask)
            total = n0 + n1
            if total == 0:
                pct0 = pct1 = 0.0
            else:
                pct0 = n0 / total
                pct1 = n1 / total
            imp = float(n["importance"])
            weighted_n0 += imp * pct0
            weighted_n1 += imp * pct1
            importance_total += imp
            rows.append(
                {
                    "node_type": node_type,
                    "text": n["text"],
                    "edge_type": n["edge_type"],
                    "importance": imp,
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
                }
            )

    rows.sort(key=lambda r: r["importance"], reverse=True)

    if importance_total > 0:
        ev0 = weighted_n0 / importance_total
        ev1 = weighted_n1 / importance_total
    else:
        ev0 = ev1 = 0.0
    evidence_majority = label_names[0] if ev0 > ev1 else label_names[1]
    evidence_strength = abs(ev0 - ev1)

    return {
        "case_id": explanation["case_id"],
        "case_node_index": explanation["case_node_index"],
        "target_label": target,
        "predicted_label": predicted,
        "confidence": explanation["confidence"],
        "misclassified": misclassified,
        "weighted_evidence": {
            f"pct_{label_names[0]}": ev0,
            f"pct_{label_names[1]}": ev1,
            "majority_class": evidence_majority,
            "strength": evidence_strength,
        },
        "per_node": rows,
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
    lines.append("")
    lines.append("## Where the explainer's evidence points (in the training set)")
    lines.append("")
    lines.append(
        f"Aggregating across top PGE nodes, weighted by their importance:\n\n"
        f"- **{head_neg}**: {we[f'pct_{head_neg}']:.1%}\n"
        f"- **{head_pos}**: {we[f'pct_{head_pos}']:.1%}\n"
        f"- Majority class of evidence: **{we['majority_class']}** "
        f"(skew {we['strength']:.1%})"
    )
    lines.append("")
    if diag["misclassified"]:
        if we["majority_class"] == diag["predicted_label"]:
            mc = we["majority_class"]
            mc_pct = we[f"pct_{mc}"]
            lines.append(
                f"> **Reading**: the model predicted `{diag['predicted_label']}` "
                f"because the cited statutes/provisions/precedents it leaned on "
                f"appear in training cases that are dominantly **{mc}** "
                f"({mc_pct:.1%}). The true label "
                f"`{diag['target_label']}` is under-represented among the training "
                f"neighbours of this case's most informative nodes."
            )
        else:
            lines.append(
                "> **Reading**: the dominant base-rate of the surfaced evidence "
                "does NOT match the prediction, so the misclassification likely "
                "comes from non-PGE-surfaced signal (e.g., text/structural "
                "embedding) rather than the cited authorities."
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
    ev_word = "WINNING" if we_majority == head_pos else "LOSING"

    if misclass:
        if we_majority == pred_lbl:
            sub = (
                f"Evidence skews {ev_word} → model predicted {pred_word} → "
                f"true label {true_word}.  This explains the misclassification."
            )
            sub_color = "#a30000"
        else:
            sub = (
                f"Evidence skews {ev_word} but model predicted {pred_word} → "
                f"PGE evidence does NOT explain this miss."
            )
            sub_color = "#555"
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


def find_misclassified(n: int) -> list[int]:
    out = []
    for p in EXPLANATIONS_DIR.glob("case_*.json"):
        with open(p) as f:
            d = json.load(f)
        if d["target_label"] != d["predicted_label"] and d["target_label"] != "?":
            out.append(int(d["case_node_index"]))
            if len(out) >= n:
                break
    return out


def find_correctly_classified(n: int) -> list[int]:
    out = []
    for p in EXPLANATIONS_DIR.glob("case_*.json"):
        with open(p) as f:
            d = json.load(f)
        if d["target_label"] == d["predicted_label"] and d["target_label"] != "?":
            out.append(int(d["case_node_index"]))
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", type=int, action="append", default=[])
    ap.add_argument("--auto-misclassified", type=int, default=0,
                    help="Pick this many misclassified cases automatically.")
    ap.add_argument("--auto-correct", type=int, default=0,
                    help="Pick this many correctly classified cases automatically.")
    args = ap.parse_args()

    targets = list(args.case_index)
    if args.auto_misclassified:
        targets.extend(find_misclassified(args.auto_misclassified))
    if args.auto_correct:
        targets.extend(find_correctly_classified(args.auto_correct))
    if not targets:
        ap.error("Provide --case-index or --auto-misclassified N")

    print(f"Loading graph cache ...")
    data, metadata = load_graph_cache(GRAPH_CACHE)
    label_names = metadata["label_names"]
    train_mask = build_training_case_mask(metadata, data["case"].num_nodes)
    print(f"Train mask: {int(train_mask.sum())} cases")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for idx in targets:
        path = EXPLANATIONS_DIR / f"case_{idx}.json"
        if not path.exists():
            print(f"  skip: {path} missing")
            continue
        with open(path) as f:
            explanation = json.load(f)
        diag = diagnose_case(explanation, data, metadata, train_mask, label_names)

        json_out = REPORT_DIR / f"case_{idx}.json"
        md_out = REPORT_DIR / f"case_{idx}.md"
        png_out = REPORT_DIR / f"case_{idx}_subgraph.png"
        json_out.write_text(json.dumps(diag, indent=2))
        md_out.write_text(render_markdown(diag, label_names))
        render_diagnostic_subgraph(diag, label_names, png_out)

        we = diag["weighted_evidence"]
        agree = we["majority_class"] == diag["predicted_label"]
        print(
            f"  case {idx}: target={diag['target_label']} "
            f"pred={diag['predicted_label']} "
            f"evidence_majority={we['majority_class']} "
            f"strength={we['strength']:.1%} "
            f"{'(supports prediction)' if agree else '(does NOT support prediction)'}"
            f"  ->  {md_out.relative_to(ROOT)}"
        )
        summary_rows.append(
            {
                "case_node_index": idx,
                "case_id": diag["case_id"],
                "target_label": diag["target_label"],
                "predicted_label": diag["predicted_label"],
                "confidence": diag["confidence"],
                "evidence_majority": we["majority_class"],
                "evidence_strength": we["strength"],
                "evidence_supports_prediction": agree,
            }
        )

    (REPORT_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nWrote {len(summary_rows)} reports under {REPORT_DIR}")


if __name__ == "__main__":
    main()
