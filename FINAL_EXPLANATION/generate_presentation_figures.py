#!/usr/bin/env python
"""Slide-ready figures for one case: similarity network + two contrast diagrams.

Renders the same three graphs the Exp-6 visualiser panel shows, as static
PNG/PDF/SVG for a deck.  Data comes from ``presentation_graphs`` and drawing
from ``presentation_figure_style``, so the files match the browser exactly.

    # all three figures for one case
    python generate_presentation_figures.py --case-index 51419

    # which cases make legible figures?
    python generate_presentation_figures.py --suggest 20
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402

from presentation_figure_style import (  # noqa: E402
    CARD_W,
    CASE_R,
    INK,
    MUTED,
    OPPOSITE_EDGE,
    QUERY_EDGE,
    WRONG_INK,
    SAME_EDGE,
    card_metrics,
    connect_path,
    draw_card,
    draw_case_circle,
    draw_cf_legend,
    draw_top_factors,
    new_canvas,
    savefig,
    stack_y,
    type_style,
    wrap_title,
)
from presentation_graphs import (  # noqa: E402
    DEFAULT_EXPLANATION_DIR,
    DEFAULT_PATTERN_DIR,
    CaseNeighborIndex,
    CounterfactualFactorIndex,
    contrast_graph,
    ego_graph,
    showcase_ranking,
)


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = APP_ROOT / "figures"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-index", type=int, default=None, help="Query case to draw.")
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--pool", default="test", choices=["test", "train", "val", "all"])
    parser.add_argument("--k-same", type=int, default=3)
    parser.add_argument("--k-opp", type=int, default=3)
    parser.add_argument("--rows", type=int, default=6, help="Evidence boxes per column in the contrast figures.")
    parser.add_argument(
        "--order",
        default="counterfactual",
        choices=["counterfactual", "evidence"],
        help="Which evidence reaches the visible columns.",
    )
    parser.add_argument("--prefix", default="", help="Filename prefix for the written figures.")
    parser.add_argument(
        "--match",
        default="target",
        choices=["target", "pred"],
        help="Pair cases on their true label (default, the published analysis) or on what the "
        "model predicted. Only 'pred' guarantees the two cases actually got different verdicts.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Denser evidence boxes: add the idf reading and the Δ confidence shift to each badge. "
        "Default is the slide layout — type, name, and a '#3 ▲' rank pill.",
    )
    parser.add_argument(
        "--suggest",
        type=int,
        default=None,
        metavar="N",
        help="Print the N cases whose figures read best, then exit.",
    )
    parser.add_argument("--sample", type=int, default=2500, help="Cases scanned by --suggest.")
    return parser


# ---------------------------------------------------------------------------
# Figure 1: ego similarity network
# ---------------------------------------------------------------------------


def figure_ego(graph: dict[str, Any], out_dir: Path, formats: list[str], prefix: str) -> None:
    center = graph["center"]
    same = [row for row in graph["nodes"] if row["side"] == "same"]
    opposite = [row for row in graph["nodes"] if row["side"] == "opposite"]
    edges = {int(row["target"]): row for row in graph["edges"]}

    width = 1180.0
    rows = max(len(same), len(opposite), 1)
    height = max(560.0, 210.0 + rows * 132.0)
    cx, cy = width / 2, height / 2
    rx, ry = 138.0, 56.0
    node_rx, node_ry = 122.0, 50.0
    col_x = {"same": 196.0, "opposite": width - 196.0}
    tone = {"same": SAME_EDGE, "opposite": OPPOSITE_EDGE}
    fill = {"same": "#2f7a5c", "opposite": "#a2662a"}

    fig, ax = new_canvas(width, height)
    max_shared = max([int(row["shared_total"]) for row in graph["edges"]] + [1])

    def node_y(items: list, i: int) -> float:
        block = max(len(items), 1) * 132.0
        return cy - block / 2 + 66.0 + i * 132.0

    field = "model prediction" if graph.get("match") == "pred" else "true label"
    ax.text(col_x["same"], 44, f"Most similar · same {field}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(col_x["opposite"], 44, f"Most similar · opposite {field}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)

    for side, items in (("same", same), ("opposite", opposite)):
        for i, node in enumerate(items):
            x, y = col_x[side], node_y(items, i)
            edge = edges.get(int(node["case_index"]), {})
            total = int(edge.get("shared_total") or 0)
            lw = 1.6 + math.sqrt(total / max_shared) * 6.5
            anchor_x = cx - rx if side == "same" else cx + rx
            node_edge_x = x + node_rx if side == "same" else x - node_rx
            ax.plot([anchor_x, node_edge_x], [cy, y], color=tone[side], linewidth=lw, alpha=0.55,
                    solid_capstyle="round", zorder=1)
            mid_x, mid_y = (anchor_x + node_edge_x) / 2, (cy + y) / 2
            label = edge.get("shared_label") or ""
            if label:
                ax.text(mid_x, mid_y - 10, label, ha="center", va="center", fontsize=10.5, weight="bold",
                        color="#24323f", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88))
            ax.text(mid_x, mid_y + (7 if label else 0), f"cos {float(edge.get('cosine') or 0):.3f}",
                    ha="center", va="center", fontsize=8.5, weight="bold", color=MUTED, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88))

    def draw_ellipse(x: float, y: float, ex: float, ey: float, face: str, edge: str, meta: dict, width_chars: int, lw: float) -> None:
        ax.add_patch(patches.Ellipse((x, y), ex * 2, ey * 2, facecolor=face, edgecolor=edge, linewidth=lw, zorder=4))
        lines = wrap_title(meta.get("title") or f"Case {meta['case_index']}", width=width_chars, max_lines=2)
        badge = meta.get("label_badge") or "?"
        if meta.get("correct") is False:
            badge = f"true {badge} / pred {meta.get('pred_badge') or '?'}"
        lines.append(f"({badge})   #{meta['case_index']}")
        start = y - (len(lines) - 1) * 7.5
        for i, line in enumerate(lines):
            ax.text(x, start + i * 15, line, ha="center", va="center", zorder=5,
                    fontsize=8.6 if i < len(lines) - 1 else 8.0,
                    weight="bold" if i < len(lines) - 1 else "normal", color="white")

    for side, items in (("same", same), ("opposite", opposite)):
        for i, node in enumerate(items):
            draw_ellipse(col_x[side], node_y(items, i), node_rx, node_ry, fill[side], tone[side], node, 26, 1.6)
    draw_ellipse(cx, cy, rx, ry, "#1f4f66", "#12303f", center, 30, 2.4)

    ax.text(
        60,
        height - 26,
        "Edge labels count shared evidence by type — Judge, Precedent, proVision, Court, Statute; "
        "thickness scales with the total.",
        ha="left",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )
    fig.tight_layout(pad=0.15)
    savefig(fig, out_dir, f"{prefix}ego_similarity_{center['case_index']}", formats)
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# Figure 2/3: contrast diagrams
# ---------------------------------------------------------------------------


def figure_contrast(
    graph: dict[str, Any],
    out_dir: Path,
    formats: list[str],
    prefix: str,
    rows: int,
    stem: str | None = None,
    detail: bool = False,
) -> None:
    side = graph["side"]
    query, other = graph["query"], graph["other"]
    query_only = graph["query_only_features"][:rows]
    shared = graph["shared_features"][:rows]
    other_only = graph["other_only_features"][:rows]

    card_h, pitch = card_metrics(detail)
    max_rows = max(len(shared), len(query_only), len(other_only), 4)
    width = 1180.0
    height = max(600.0, 260.0 + max_rows * pitch)
    center_y = height / 2
    qx, ox = 112.0, 1068.0
    q_only_x, shared_x, opp_only_x = 245.0, 486.0, 727.0
    mid = card_h / 2
    other_heading = "Similar-case-only evidence" if side == "same" else "Opposite-only evidence"
    other_edge = SAME_EDGE if side == "same" else OPPOSITE_EDGE

    fig, ax = new_canvas(width, height)
    for x, heading in (
        (q_only_x, "Query-only evidence"),
        (shared_x, "Shared evidence"),
        (opp_only_x, other_heading),
    ):
        ax.text(x + CARD_W / 2, 44, heading, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)

    for i, feature in enumerate(query_only):
        y = stack_y(center_y, query_only, i, pitch)
        _, edge = type_style(feature["feature_type"])
        connect_path(ax, [(qx + CASE_R, center_y), (185, center_y), (205, y + mid), (q_only_x, y + mid)], color=edge, alpha=0.36)
    for i, feature in enumerate(shared):
        y = stack_y(center_y, shared, i, pitch) + mid
        _, edge = type_style(feature["feature_type"])
        connect_path(ax, [(qx + CASE_R, center_y), (260, center_y), (330, y), (shared_x, y)], color=edge, dashed=True, alpha=0.30, lw=2.0)
        connect_path(ax, [(shared_x + CARD_W, y), (790, y), (900, center_y), (ox - CASE_R, center_y)], color=edge, dashed=True, alpha=0.30, lw=2.0)
    for i, feature in enumerate(other_only):
        y = stack_y(center_y, other_only, i, pitch) + mid
        _, edge = type_style(feature["feature_type"])
        connect_path(ax, [(opp_only_x + CARD_W, y), (950, y), (980, center_y), (ox - CASE_R, center_y)], color=edge, alpha=0.36)

    empty_note = {
        q_only_x: "No query-only evidence",
        shared_x: "No shared evidence",
        opp_only_x: f"No {'similar' if side == 'same' else 'opposite'}-only evidence",
    }
    for column_x, items in ((q_only_x, query_only), (shared_x, shared), (opp_only_x, other_only)):
        if not items:
            ax.text(column_x + CARD_W / 2, center_y, empty_note[column_x], ha="center", va="center",
                    fontsize=9.5, style="italic", color="#8a96a3")
            continue
        for i, feature in enumerate(items):
            draw_card(ax, column_x, stack_y(center_y, items, i, pitch), feature, h=card_h, detail=detail)

    for cx_, meta, edge_ in ((qx, query, QUERY_EDGE), (ox, other, other_edge)):
        draw_case_circle(
            ax, cx_, center_y, edge_,
            label_badge=meta.get("label_badge"), pred_badge=meta.get("pred_badge"),
            correct=meta.get("correct"), case_index=meta.get("case_index"),
            title=meta.get("title"), bucket=meta.get("bucket_label"),
        )
    same_call = str(query.get("pred_label")) == str(other.get("pred_label"))
    verdict = ("the model gave BOTH cases the same label — it did not separate them"
               if same_call else "the model decided these two differently")
    ax.text(width / 2, 22, f"embedding cosine {float(graph['cosine_similarity']):.4f}   ·   {verdict}",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color=WRONG_INK if same_call else MUTED)
    other_role = "Similar case" if side == "same" else "Opposite case"
    notes = [
        draw_top_factors(ax, 60, height - 100, query, "Query case"),
        draw_top_factors(ax, 60, height - 78, other, other_role),
    ]
    note = next((n for n in notes if n), None)
    if note:
        ax.text(60, height - 58, note, ha="left", va="center", fontsize=7.4, style="italic", color=MUTED)
    draw_cf_legend(ax, 60, height - 30)

    fig.tight_layout(pad=0.15)
    savefig(fig, out_dir, stem or f"{prefix}contrast_{side}_{query['case_index']}_{other['case_index']}", formats)
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    neighbors = CaseNeighborIndex(args.pattern_dir)
    factors = CounterfactualFactorIndex(args.explanation_dir)

    if args.suggest is not None:
        rows = showcase_ranking(neighbors, factors, limit=args.suggest, pool=args.pool,
                                sample=args.sample, match=args.match)
        if not rows:
            print("[suggest] no candidates found")
            return
        print(f"{'case':>7}  {'score':>6}  {'rich':>4} {'types':>5} {'shared':>6} {'cosine':>6} {'boxes':>5}  case name")
        for row in rows:
            print(
                f"{row['case_index']:>7}  {row['score']:>6.2f}  {row['rich_edges']:>4} {row['shared_types']:>5} "
                f"{row['shared_total']:>6} {row['mean_cosine']:>6.3f} {row['evidence_boxes']:>5}  {row['title']}"
            )
        return

    if args.case_index is None:
        raise SystemExit("--case-index is required (or use --suggest N to pick one).")

    ego = ego_graph(neighbors, factors, args.case_index, k_same=args.k_same,
                    k_opposite=args.k_opp, pool=args.pool, match=args.match)
    if not ego.get("available"):
        raise SystemExit(f"[error] {ego.get('reason')}")
    figure_ego(ego, args.out_dir, args.formats, args.prefix)

    for side in ("same", "opposite"):
        graph = contrast_graph(
            neighbors, factors, args.case_index, side=side, pool=args.pool,
            limit=args.rows, order=args.order, match=args.match,
        )
        if not graph.get("available"):
            print(f"[skip] {side}: {graph.get('reason')}", flush=True)
            continue
        figure_contrast(graph, args.out_dir, args.formats, args.prefix, args.rows, detail=args.detail)


if __name__ == "__main__":
    main()
