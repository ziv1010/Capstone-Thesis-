"""
visualize_graph_structure.py

Render a publication-quality diagram of the reasoning-focused LegalGraph-LJP
heterogeneous graph: two case-stars (case-local subgraphs) plus the shared
cross-case authorities layer (statute / provision / precedent), faithful to
the edges produced by section_GNN/final_graph/updated_graph/case_star_builder.py
under the policy in section_GNN/final_graph/updated_graph/reasoning_graph_policy.py.

Pure matplotlib — no external graph layout binaries needed. All node positions
are hand-tuned for a clean three-column flow:

       Case A (case-local)  |  shared authorities  |  Case B (case-local)

Usage:
    python visualize_graph_structure.py
    python visualize_graph_structure.py --out figure.pdf --dpi 300
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------
# Paper-style palette
# --------------------------------------------------------------------------
PALETTE = {
    "case":   {"face": "#FFE6B0", "edge": "#B07A1A", "text": "#5A3D0A"},
    "text":   {"face": "#DCE9FB", "edge": "#3D6FB8", "text": "#1F3F75"},
    "party":  {"face": "#E8DAF5", "edge": "#7548AC", "text": "#3F1F70"},
    "court":  {"face": "#CFEBD0", "edge": "#3F8C45", "text": "#1B4F1F"},
    "lawyer": {"face": "#FFD3DC", "edge": "#C46680", "text": "#7A2C44"},
    "shared": {"face": "#A6E3A1", "edge": "#287032", "text": "#0E3A14"},
}

EDGE = {
    "structural": dict(color="#222222", linestyle="-",  linewidth=1.3),
    "has_text":   dict(color="#3563A8", linestyle="--", linewidth=1.1),
    "argues_in":  dict(color="#7F7F7F", linestyle=":",  linewidth=1.4),
    "is_party":   dict(color="#7548AC", linestyle="--", linewidth=1.2),
    "cites":      dict(color="#287032", linestyle="--", linewidth=1.6),
    "belongs":    dict(color="#1B5E20", linestyle="-",  linewidth=2.4),
}

NODE_W, NODE_H = 3.7, 1.10


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    key: str
    label: str
    pos: tuple[float, float]
    kind: str
    sub: str | None = None


def draw_node(ax, node: Node) -> None:
    x, y = node.pos
    style = PALETTE[node.kind]

    # Subtle shadow
    shadow = FancyBboxPatch(
        (x - NODE_W / 2 + 0.06, y - NODE_H / 2 - 0.06), NODE_W, NODE_H,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#000000", edgecolor="none", alpha=0.10, zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=style["face"], edgecolor=style["edge"], linewidth=1.4, zorder=2,
    )
    ax.add_patch(box)

    if node.sub:
        ax.text(x, y + 0.16, node.label, ha="center", va="center",
                fontsize=13, fontweight="bold", color=style["text"], zorder=3)
        ax.text(x, y - 0.26, node.sub, ha="center", va="center",
                fontsize=9.5, color=style["text"], style="italic", alpha=0.85, zorder=3)
    else:
        ax.text(x, y, node.label, ha="center", va="center",
                fontsize=14, fontweight="bold", color=style["text"], zorder=3)


def draw_edge(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    style_name: str,
    label: str | None = None,
    rad: float = 0.0,
    label_pos: float = 0.5,
    label_offset: tuple[float, float] = (0.0, 0.0),
    show_label: bool = True,
) -> None:
    s = EDGE[style_name]
    arrow = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=14,
        shrinkA=18, shrinkB=18,
        zorder=0,
        **s,
    )
    ax.add_patch(arrow)
    if label and show_label:
        # Place label along the curve. For curved edges, offset perpendicular
        # to the line so it sits near the apex.
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
        if rad != 0:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            nx, ny = -dy / length, dx / length
            mx += nx * rad * length * 0.5
            my += ny * rad * length * 0.5
        ax.text(
            mx, my, label,
            fontsize=10, color=s["color"],
            ha="center", va="center", zorder=4,
            bbox=dict(facecolor="white", edgecolor=s["color"], linewidth=0.6,
                      pad=1.6, boxstyle="round,pad=0.25", alpha=1.0),
        )


# --------------------------------------------------------------------------
# Per-case layout
# --------------------------------------------------------------------------
def case_layout(cx: float, side: str) -> dict[str, Node]:
    """
    Per-case star layout — three vertical "party stacks" at the bottom let
    party→role_args edges run as clean vertical lines through the lawyer
    in the middle of each stack.

        Top zone (above case):
            outer column = text segments (preamble, facts, arguments base)
            inner column = court + judge       (faces the shared layer)
                                case (centered)
        Bottom zone (below case):
            three vertical stacks: (petitioner, pet_lawyer, pet_args),
                                   (respondent, def_lawyer, resp_args),
                                   (         , lawyer,      other_args)
            Inner stack faces the shared layer.

    `side` ('left' or 'right') mirrors the layout so role stacks align
    naturally toward the shared layer.
    """
    sign = -1.0 if side == "left" else 1.0   # outward direction

    text_col  = cx + sign * 5.4
    inner_col = cx - sign * 5.4
    case_y    = 2.4

    # Three vertical stacks across the bottom, ordered so the petitioner
    # stack is OUTWARD and the "other lawyer" stack is INWARD.
    if side == "left":
        stack_xs = [cx - 3.6, cx, cx + 3.6]   # pet | def | other (left → right)
    else:
        stack_xs = [cx + 3.6, cx, cx - 3.6]   # mirrored

    p: dict[str, Node] = {}

    # ---- Case node ----
    p["case"] = Node("case", "case", (cx, case_y), "case", sub="local")

    # ---- Outer column: base text segments ----
    p["preamble"]  = Node("preamble",  "preamble",  (text_col, 5.0), "text", sub="rhetorical role")
    p["facts"]     = Node("facts",     "facts",     (text_col, 3.2), "text", sub="rhetorical role")
    p["arguments"] = Node("arguments", "arguments", (text_col, 1.4), "text", sub="base")

    # ---- Inner column: court / judge (facing shared layer) ----
    p["court"] = Node("court", "court", (inner_col, 3.4), "court", sub="case-local")
    p["judge"] = Node("judge", "judge", (inner_col, 1.6), "court", sub="case-local")

    # ---- Bottom three stacks ----
    party_y, lawyer_y, args_y = -0.6, -2.6, -4.6

    p["petitioner"]         = Node("petitioner", "petitioner", (stack_xs[0], party_y),  "party",  sub="case-local")
    p["petitioner_lawyer"]  = Node("petitioner_lawyer", "petitioner_lawyer", (stack_xs[0], lawyer_y), "lawyer", sub="case-local")
    p["petitioner_arguments"] = Node("petitioner_arguments", "petitioner_arguments", (stack_xs[0], args_y), "text", sub="role-specific")

    p["respondent"]         = Node("respondent", "respondent", (stack_xs[1], party_y),  "party",  sub="case-local")
    p["defence_lawyer"]     = Node("defence_lawyer", "defence_lawyer", (stack_xs[1], lawyer_y), "lawyer", sub="case-local")
    p["respondent_arguments"] = Node("respondent_arguments", "respondent_arguments", (stack_xs[1], args_y), "text", sub="role-specific")

    p["lawyer"]                 = Node("lawyer", "lawyer (other)", (stack_xs[2], lawyer_y), "lawyer", sub="case-local")
    p["other_lawyer_arguments"] = Node("other_lawyer_arguments", "other_lawyer_arguments", (stack_xs[2], args_y), "text", sub="role-specific")

    return p


# --------------------------------------------------------------------------
# Edges within one case
# --------------------------------------------------------------------------
def draw_case_edges(ax, p: dict[str, Node], show_labels: bool = True,
                    outward_sign: float = -1.0) -> None:
    case = p["case"].pos

    # case --has_*--> text segments  (no labels — pattern is obvious from arrow color)
    for tgt in ("preamble", "facts", "arguments",
                "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments"):
        draw_edge(ax, case, p[tgt].pos, "has_text", show_label=False)

    # case --has_*/heard_in/decided_by_bench--> party / court / judge / lawyer
    # Label only the non-obvious relations.
    for tgt, rel in [
        ("petitioner",        None),
        ("respondent",        None),
        ("court",             "heard_in"),
        ("judge",             "decided_by_bench"),
        ("petitioner_lawyer", None),
        ("defence_lawyer",    None),
        ("lawyer",            None),
    ]:
        draw_edge(ax, case, p[tgt].pos, "structural",
                  label=rel, label_pos=0.55, show_label=show_labels and rel is not None)

    # lawyer --argues_in--> role-specific arg node
    for src, tgt in [
        ("petitioner_lawyer", "petitioner_arguments"),
        ("defence_lawyer",    "respondent_arguments"),
        ("lawyer",            "other_lawyer_arguments"),
    ]:
        draw_edge(ax, p[src].pos, p[tgt].pos, "argues_in",
                  label="argues_in", show_label=show_labels)

    # party --is_party_in_arguments--> role-specific arg node
    # Both curves bow toward the OUTER side of the case (away from shared
    # layer) so they sit in empty space. Label shown only on the petitioner
    # edge (outer stack, with room to its side) — the respondent edge is the
    # same relation and is identified by its color from the legend.
    is_party_rad = 0.7 * outward_sign
    # Use the short form on the edge — full name "is_party_in_arguments" is in
    # the legend, so the abbreviated label fits in the gap without overlap.
    draw_edge(ax, p["petitioner"].pos, p["petitioner_arguments"].pos, "is_party",
              label="is_party", rad=is_party_rad, label_pos=0.5,
              show_label=show_labels)
    draw_edge(ax, p["respondent"].pos, p["respondent_arguments"].pos, "is_party",
              label=None, rad=is_party_rad, label_pos=0.5, show_label=False)


# --------------------------------------------------------------------------
# Shared layer + cites edges
# --------------------------------------------------------------------------
def shared_layer() -> dict[str, Node]:
    return {
        "statute":   Node("statute",   "statute",   (0.0,  3.6), "shared", sub="shared across cases"),
        "provision": Node("provision", "provision", (0.0,  0.0), "shared", sub="shared across cases"),
        "precedent": Node("precedent", "precedent", (0.0, -3.6), "shared", sub="shared across cases"),
    }


def draw_shared_edges(ax, shared, pa, pb, show_labels: bool = True) -> None:
    # provision --belongs_to_statute--> statute
    draw_edge(ax, shared["provision"].pos, shared["statute"].pos, "belongs",
              label="belongs_to_statute", show_label=show_labels)

    # base arguments --cites_*--> shared (one set of edges per case)
    # Label only the Case A edges (the Case B edges are mirror images of the
    # same relation and are identified by color from the legend).
    cite_targets = [("statute", -3.0), ("provision", 0.0), ("precedent", 3.0)]
    for case_pos, side in [(pa, "left"), (pb, "right")]:
        for shared_key, _ in cite_targets:
            rad = 0.10 if side == "left" else -0.10
            # Label sits very close to the shared layer (label_pos=0.88) so it
            # never overlaps the case's court/judge nodes; only label one side.
            label = f"cites_{shared_key}" if side == "left" else None
            draw_edge(ax, case_pos["arguments"].pos, shared[shared_key].pos, "cites",
                      label=label, rad=rad, label_pos=0.88,
                      show_label=show_labels and label is not None)


# --------------------------------------------------------------------------
# Section backgrounds + legend
# --------------------------------------------------------------------------
def draw_section_bg(ax, x_min, x_max, y_min, y_max, title, face, border):
    rect = FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=0.04,rounding_size=0.30",
        facecolor=face, edgecolor=border, linewidth=1.2, alpha=0.55, zorder=-2,
    )
    ax.add_patch(rect)
    ax.text((x_min + x_max) / 2, y_max - 0.40, title,
            ha="center", va="top", fontsize=16, fontweight="bold", color=border, zorder=-1)


def add_legend(ax) -> None:
    node_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE[k]["face"], markeredgecolor=PALETTE[k]["edge"],
               markersize=14, label=lbl)
        for k, lbl in [
            ("case",   "case  (local)"),
            ("text",   "text segment  (local)"),
            ("party",  "party  (local)"),
            ("court",  "court / judge  (local)"),
            ("lawyer", "lawyer  (local)"),
            ("shared", "statute / provision / precedent  (shared)"),
        ]
    ]
    edge_handles = [
        Line2D([0], [0], color=EDGE["structural"]["color"],
               linestyle=EDGE["structural"]["linestyle"], linewidth=2.0,
               label="structural  (case → entity)"),
        Line2D([0], [0], color=EDGE["has_text"]["color"],
               linestyle=EDGE["has_text"]["linestyle"], linewidth=2.0,
               label="has_*  (case → text segment)"),
        Line2D([0], [0], color=EDGE["argues_in"]["color"],
               linestyle=EDGE["argues_in"]["linestyle"], linewidth=2.4,
               label="argues_in  (lawyer → role arg node)"),
        Line2D([0], [0], color=EDGE["is_party"]["color"],
               linestyle=EDGE["is_party"]["linestyle"], linewidth=2.0,
               label="is_party_in_arguments  (party → role arg node)"),
        Line2D([0], [0], color=EDGE["cites"]["color"],
               linestyle=EDGE["cites"]["linestyle"], linewidth=2.4,
               label="cites_*  (arguments → shared authority)"),
        Line2D([0], [0], color=EDGE["belongs"]["color"],
               linestyle=EDGE["belongs"]["linestyle"], linewidth=2.4,
               label="belongs_to_statute  (provision → statute)"),
    ]
    leg1 = ax.legend(handles=node_handles, loc="lower left",
                     bbox_to_anchor=(0.005, 0.005), fontsize=12,
                     frameon=True, title="Node types", title_fontsize=13,
                     borderpad=0.8, labelspacing=0.6)
    leg1.get_frame().set_edgecolor("#888")
    leg1.get_frame().set_linewidth(0.9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=edge_handles, loc="lower right",
                     bbox_to_anchor=(0.995, 0.005), fontsize=12,
                     frameon=True, title="Edge types", title_fontsize=13,
                     borderpad=0.8, labelspacing=0.6)
    leg2.get_frame().set_edgecolor("#888")
    leg2.get_frame().set_linewidth(0.9)


# --------------------------------------------------------------------------
# Main render
# --------------------------------------------------------------------------
def render(out_path: Path, dpi: int, hide_edge_labels: bool) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(26, 12))
    ax.set_xlim(-19.5, 19.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0)

    # Section backgrounds
    draw_section_bg(ax, -19.4, -3.5, -6.3, 6.3,
                    "Case A   (case-local star)", face="#F4F8FF", border="#5C7FBF")
    draw_section_bg(ax,  -3.3,  3.3, -6.3, 6.3,
                    "Cross-case shared authorities", face="#EFFBEF", border="#5BA664")
    draw_section_bg(ax,   3.5, 19.4, -6.3, 6.3,
                    "Case B   (case-local star)", face="#FFF6F8", border="#BF7090")

    pa = case_layout(cx=-11.5, side="left")
    pb = case_layout(cx=11.5,  side="right")
    shared = shared_layer()

    show_labels = not hide_edge_labels

    # Draw edges first, then nodes on top.
    # outward_sign = -1 for Case A (outer = left), +1 for Case B (outer = right)
    draw_case_edges(ax, pa, show_labels=show_labels, outward_sign=-1.0)
    draw_case_edges(ax, pb, show_labels=show_labels, outward_sign=+1.0)
    draw_shared_edges(ax, shared, pa, pb, show_labels=show_labels)

    for node in pa.values():
        draw_node(ax, node)
    for node in pb.values():
        draw_node(ax, node)
    for node in shared.values():
        draw_node(ax, node)

    add_legend(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("legalgraph_visualization.png"),
                        help="Output path. Extension determines format (.png/.pdf/.svg).")
    parser.add_argument("--dpi", type=int, default=240, help="Output DPI.")
    parser.add_argument("--no-edge-labels", action="store_true",
                        help="Hide colored edge labels for an even cleaner look.")
    args = parser.parse_args()
    render(args.out, args.dpi, args.no_edge_labels)


if __name__ == "__main__":
    main()
