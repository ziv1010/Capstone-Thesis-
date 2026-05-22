#!/usr/bin/env python3
"""
Generate static paper figures from FINAL_EXPLANATION analysis outputs.

Figures:
  fig1_faithfulness_curves
  fig2_community_cluster_sankey
  fig3_contrastive_subgraph_51419_15962
"""
from __future__ import annotations

import argparse
import csv
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import scipy.sparse as sp


ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/FINAL_EXPLANATION")
EXP3 = ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
EXP5 = ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why"
OUT_DIR = ROOT / "figures"


COLORS = {
    "counterfactual": "#1f77b4",
    "attention": "#ff7f0e",
    "random": "#7f7f7f",
    "query": "#d62728",
    "opposite": "#2ca02c",
    "shared": "#6a51a3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return p.parse_args()


def savefig(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[figure] wrote {path}")


def mean_by(items: list[dict], key_fields: tuple[str, ...], value_field: str) -> dict[tuple, float]:
    vals: dict[tuple, list[float]] = defaultdict(list)
    for row in items:
        key = tuple(row[k] for k in key_fields)
        try:
            vals[key].append(float(row[value_field]))
        except (TypeError, ValueError):
            pass
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def fig1_faithfulness(out_dir: Path, formats: list[str]) -> None:
    curves_path = EXP3 / "faithfulness_curves.csv"
    auc_path = EXP3 / "faithfulness_auc_summary.csv"
    rows = list(csv.DictReader(open(curves_path, encoding="utf-8")))
    auc_rows = {r["ranker"]: r for r in csv.DictReader(open(auc_path, encoding="utf-8"))}

    suff = mean_by(rows, ("ranker", "k_requested"), "sufficiency_proba")
    comp = mean_by(rows, ("ranker", "k_requested"), "comprehensiveness_drop")

    rankers = ["counterfactual", "attention", "random"]
    labels = {"counterfactual": "Counterfactual", "attention": "Attention", "random": "Random"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)

    for ranker in rankers:
        xs = sorted({int(k[1]) for k in suff if k[0] == ranker})
        y_s = [suff[(ranker, str(x))] for x in xs]
        y_c = [comp[(ranker, str(x))] for x in xs]
        auc = auc_rows[ranker]
        axes[0].plot(
            xs,
            y_s,
            marker="o",
            linewidth=2.2,
            color=COLORS[ranker],
            label=f"{labels[ranker]}  AUC={float(auc['mean_sufficiency_auc']):.3f}",
        )
        axes[1].plot(
            xs,
            y_c,
            marker="o",
            linewidth=2.2,
            color=COLORS[ranker],
            label=f"{labels[ranker]}  AUC={float(auc['mean_comprehensiveness_auc']):.3f}",
        )

    axes[0].set_title("Sufficiency: keep top-k evidence")
    axes[0].set_ylabel("Mean predicted-class probability")
    axes[1].set_title("Comprehensiveness: remove top-k evidence")
    axes[1].set_ylabel("Mean probability drop")
    for ax in axes:
        ax.set_xlabel("Top-k evidence groups")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(pad=1.0, w_pad=2.2)
    savefig(fig, out_dir, "fig1_faithfulness_curves", formats)
    plt.close(fig)


def draw_ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.34):
    verts = [
        (x0, y0a),
        ((x0 + x1) / 2, y0a),
        ((x0 + x1) / 2, y1a),
        (x1, y1a),
        (x1, y1b),
        ((x0 + x1) / 2, y1b),
        ((x0 + x1) / 2, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig2_sankey(out_dir: Path, formats: list[str]) -> None:
    rows = list(csv.DictReader(open(EXP5 / "case_embedding_clusters.csv", encoding="utf-8")))
    align_rows = list(csv.DictReader(open(EXP5 / "structural_embedding_alignment.csv", encoding="utf-8")))
    community_counts = Counter(int(r["community_id"]) for r in rows)
    top_communities = [cid for cid, _ in community_counts.most_common(12)]

    flow = Counter()
    label_counts: dict[int, Counter] = defaultdict(Counter)
    domain_counts: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        c = int(r["community_id"])
        if c not in top_communities:
            continue
        left = f"C{c}"
        cluster = r["embedding_cluster_id"]
        right = "Noise" if cluster == "-1" else f"Cluster {cluster}"
        flow[(left, right)] += 1
        label_counts[c][r["target_label"]] += 1
        domain_counts[c][r["domain_bucket"]] += 1

    left_order = [f"C{c}" for c in top_communities]
    right_order = ["Cluster 0", "Cluster 1", "Noise"]
    total = sum(flow.values())

    left_totals = {l: sum(flow[(l, r)] for r in right_order) for l in left_order}
    right_totals = {r: sum(flow[(l, r)] for l in left_order) for r in right_order}

    def positions(order, totals, gap=0.016, min_h=0.022):
        y = 0.97
        pos = {}
        usable = 0.94 - gap * (len(order) - 1)
        weights = {item: math.sqrt(max(totals[item], 1)) for item in order}
        weight_sum = sum(weights.values())
        for item in order:
            h = max(min_h, usable * weights[item] / weight_sum)
            pos[item] = (y - h, y)
            y -= h + gap
        return pos

    left_pos = positions(left_order, left_totals)
    right_pos = positions(right_order, right_totals, gap=0.075, min_h=0.10)
    left_cursor = {k: v[0] for k, v in left_pos.items()}
    right_cursor = {k: v[0] for k, v in right_pos.items()}
    cluster_colors = {"Cluster 0": "#2563eb", "Cluster 1": "#16a34a", "Noise": "#8b8f97"}

    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    x_left, x_right = 0.24, 0.77
    bar_w = 0.055

    bg = patches.FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.90,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#f8fafc",
        edgecolor="#e2e8f0",
        linewidth=1.0,
        zorder=-5,
    )
    ax.add_patch(bg)

    for l in left_order:
        y0, y1 = left_pos[l]
        cid = int(l[1:])
        dom = domain_counts[cid].most_common(1)[0][0].replace("_", " ")
        lab = label_counts[cid].most_common(1)[0][0]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_left - bar_w, y0),
                bar_w,
                y1 - y0,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor="#1e293b",
                edgecolor="none",
                alpha=0.95,
            )
        )
        label = f"{l}  ({left_totals[l]:,})\n{dom}; label {lab}"
        ax.text(x_left - bar_w - 0.018, (y0 + y1) / 2, label, ha="right", va="center", fontsize=8.2, color="#111827")

    for r in right_order:
        y0, y1 = right_pos[r]
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_right, y0),
                bar_w,
                y1 - y0,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor=cluster_colors[r],
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax.text(x_right + bar_w + 0.018, (y0 + y1) / 2, f"{r}\n{right_totals[r]:,}", ha="left", va="center", fontsize=9.4, color="#111827")

    for l in left_order:
        left_height = left_pos[l][1] - left_pos[l][0]
        total_l = max(left_totals[l], 1)
        for r in right_order:
            count = flow[(l, r)]
            if not count:
                continue
            h = left_height * count / total_l
            right_height = right_pos[r][1] - right_pos[r][0]
            h_right = right_height * count / max(right_totals[r], 1)
            y0a = left_cursor[l]
            y0b = y0a + h
            y1a = right_cursor[r]
            y1b = y1a + h_right
            alpha = 0.18 + 0.34 * (count / total_l)
            draw_ribbon(ax, x_left, y0a, y0b, x_right, y1a, y1b, cluster_colors[r], alpha=min(alpha, 0.55))
            left_cursor[l] = y0b
            right_cursor[r] = y1b

    metric_text = ""
    if align_rows:
        ar = align_rows[0]
        bits = []
        for key, label in [("nmi", "NMI"), ("ari", "ARI"), ("v_measure", "V")]:
            if key in ar and ar[key]:
                bits.append(f"{label}={float(ar[key]):.3f}")
        metric_text = "  |  ".join(bits)
    ax.text(x_left - 0.02, 1.005, "Structural communities", ha="center", va="bottom", fontsize=12, weight="bold", color="#0f172a")
    ax.text(x_right + 0.03, 1.005, "Embedding clusters", ha="center", va="bottom", fontsize=12, weight="bold", color="#0f172a")
    ax.text(0.5, 0.965, metric_text, ha="center", va="center", fontsize=10, color="#475569")
    fig.tight_layout(pad=0.4)
    savefig(fig, out_dir, "fig2_community_cluster_sankey", formats)
    plt.close(fig)


def wrap_label(text: str, width: int = 34) -> str:
    text = text.replace("precedent:", "precedent: ").replace("provision:", "provision: ").replace("judge:", "judge: ")
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def load_feature_metadata() -> dict[int, dict]:
    out = {}
    with open(EXP5 / "case_feature_metadata.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["feature_index"])] = row
    return out


def fig3_contrastive(out_dir: Path, formats: list[str]) -> None:
    neighbourhood_path = EXP5 / "counterfactual_neighborhoods.csv"
    diff_path = EXP5 / "counterfactual_neighborhood_feature_differences.csv"
    row = None
    for r in csv.DictReader(open(neighbourhood_path, encoding="utf-8")):
        if r["case_index"] == "51419" and r["nearest_opposite_case_index"] == "15962":
            row = r
            break
    if row is None:
        raise RuntimeError("Could not find case 51419 -> 15962 in counterfactual_neighborhoods.csv")

    q_features, o_features = [], []
    for r in csv.DictReader(open(diff_path, encoding="utf-8")):
        if r["case_index"] == "51419" and r["nearest_opposite_case_index"] == "15962":
            item = {
                "rank": int(r["rank"]),
                "type": r["feature_type"],
                "name": r["feature_name"],
                "idf": float(r["idf"]) if r["idf"] else 0.0,
                "skew": r.get("skew_direction", ""),
                "log_odds": r.get("log_odds_vs_base", ""),
            }
            if r["side"] == "query_only":
                q_features.append(item)
            elif r["side"] == "opposite_only":
                o_features.append(item)
    q_features = sorted(q_features, key=lambda x: x["rank"])[:5]
    o_features = sorted(o_features, key=lambda x: x["rank"])[:5]

    feature_meta = load_feature_metadata()
    feature_matrix = sp.load_npz(EXP5 / "case_feature_matrix.npz").tocsr()
    q_set = set(feature_matrix[51419].indices)
    o_set = set(feature_matrix[15962].indices)
    shared_features = []
    for feature_index in sorted(q_set & o_set, key=lambda i: float(feature_meta[i]["idf"]), reverse=True):
        row_meta = feature_meta[feature_index]
        shared_features.append({
            "type": row_meta["feature_type"],
            "name": row_meta["feature_name"],
            "idf": float(row_meta["idf"]),
        })
        if len(shared_features) >= 7:
            break

    max_rows = max(len(shared_features), len(q_features), len(o_features), 4)
    width = 1180
    height = max(560, 160 + max_rows * 68)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    def type_style(feature_type: str) -> tuple[str, str]:
        styles = {
            "precedent": ("#eaf7ef", "#2e8b57"),
            "court": ("#fff7e8", "#b36b00"),
            "judge": ("#fff7e8", "#b36b00"),
            "provision": ("#eaf7fb", "#1f7a9a"),
            "statute": ("#edf2ff", "#4f46e5"),
        }
        return styles.get(feature_type, ("#f4f4f5", "#52525b"))

    def type_label(feature_type: str) -> str:
        return {
            "precedent": "PRECEDENT",
            "court": "COURT",
            "judge": "JUDGE",
            "provision": "PROVISION",
            "statute": "STATUTE",
        }.get(feature_type, feature_type.upper())

    def clean_feature_name(name: str, limit: int = 27) -> str:
        name = " ".join(name.replace("manusc", "MANU/SC/").replace("1983crilj1457", "1983 Cri LJ 1457").split())
        if len(name) <= limit:
            return name
        return name[: max(0, limit - 3)].rstrip() + "..."

    qx, ox = 112, 1068
    q_only_x, shared_x, opp_only_x = 245, 486, 727
    card_w, card_h = 205, 52
    center_y = height / 2

    def stack_y(rows, i):
        block_h = max(len(rows), 1) * 62
        return center_y - block_h / 2 + i * 62

    ax.text(q_only_x + card_w / 2, 38, "Query-only evidence", ha="center", va="center", fontsize=10.5, weight="bold", color="#1f2937")
    ax.text(shared_x + card_w / 2, 38, "Shared evidence", ha="center", va="center", fontsize=10.5, weight="bold", color="#1f2937")
    ax.text(opp_only_x + card_w / 2, 38, "Opposite-only evidence", ha="center", va="center", fontsize=10.5, weight="bold", color="#1f2937")

    def draw_case_circle(x, y, edge, title, subtitle):
        circ = patches.Circle((x, y), 52, facecolor="white", edgecolor=edge, linewidth=2.4)
        ax.add_patch(circ)
        ax.text(x, y - 8, title, ha="center", va="center", fontsize=9.0, weight="bold", color="#1f2937")
        ax.text(x, y + 16, subtitle, ha="center", va="center", fontsize=7.5, color="#64748b")

    def draw_card(x, y, feature):
        face, edge = type_style(feature["type"])
        left, bottom = x, y
        shadow = patches.FancyBboxPatch(
            (left + 2.0, bottom + 3.2),
            card_w,
            card_h,
            boxstyle="round,pad=0.008,rounding_size=8",
            facecolor="#142330",
            edgecolor="none",
            alpha=0.055,
            zorder=0.5,
        )
        ax.add_patch(shadow)
        rect = patches.FancyBboxPatch(
            (left, bottom),
            card_w,
            card_h,
            boxstyle="round,pad=0.008,rounding_size=8",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.45 if feature["type"] not in {"court", "judge"} else 2.7,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(left + 12, bottom + 18, type_label(feature["type"]), ha="left", va="center", fontsize=7.5, weight="bold", color=edge, zorder=3)
        ax.text(left + card_w - 10, bottom + 18, f"idf {feature['idf']:.1f}", ha="right", va="center", fontsize=7.5, weight="bold", color="#64748b", zorder=3)
        ax.text(left + 12, bottom + 38, clean_feature_name(feature["name"]), ha="left", va="center", fontsize=9.0, weight="bold", color="#1f2937", zorder=3)

    def connect_path(points, color="#9cc9b1", dashed=False, alpha=0.42, lw=2.2):
        path = MplPath(points, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=(0, (5.0, 5.0)) if dashed else "solid",
            capstyle="round",
            zorder=1,
        )
        ax.add_patch(patch)

    draw_case_circle(qx, center_y, "#1f7a83", "Case 51419", "target 1")
    draw_case_circle(ox, center_y, "#a85f00", "Case 15962", "target -1")

    for i, feature in enumerate(q_features):
        y = stack_y(q_features, i)
        draw_card(q_only_x, y, feature)
        _, edge = type_style(feature["type"])
        connect_path(
            [(qx + 52, center_y), (185, center_y), (205, y + card_h / 2), (q_only_x, y + card_h / 2)],
            color=edge,
            alpha=0.36,
        )

    for i, feature in enumerate(shared_features):
        y = stack_y(shared_features, i)
        draw_card(shared_x, y, feature)
        _, edge = type_style(feature["type"])
        connect_path(
            [(qx + 52, center_y), (260, center_y), (330, y + card_h / 2), (shared_x, y + card_h / 2)],
            color=edge,
            dashed=True,
            alpha=0.30,
            lw=2.0,
        )
        connect_path(
            [(shared_x + card_w, y + card_h / 2), (790, y + card_h / 2), (900, center_y), (ox - 52, center_y)],
            color=edge,
            dashed=True,
            alpha=0.30,
            lw=2.0,
        )

    for i, feature in enumerate(o_features):
        y = stack_y(o_features, i)
        draw_card(opp_only_x, y, feature)
        _, edge = type_style(feature["type"])
        connect_path(
            [(opp_only_x + card_w, y + card_h / 2), (950, y + card_h / 2), (980, center_y), (ox - 52, center_y)],
            color=edge,
            alpha=0.36,
        )

    fig.tight_layout(pad=0.15)
    savefig(fig, out_dir, "fig3_contrastive_subgraph_51419_15962", formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig1_faithfulness(out_dir, args.formats)
    fig2_sankey(out_dir, args.formats)
    fig3_contrastive(out_dir, args.formats)


if __name__ == "__main__":
    main()
