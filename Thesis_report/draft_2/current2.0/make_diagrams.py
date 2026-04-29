"""
Generate three improved diagrams for the thesis presentation:
  1. End-to-end pipeline (stage -> output -> stage -> output ...)
  2. Single-case graph structure (all 17 nodes, every schema edge)
  3. Cross-case sharing and leakage control
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Thesis palette (matches beamer theme)
BLUE = "#113A5C"
GOLD = "#C78B2B"
INK = "#21313F"
SOFT = "#EEF3F7"
ACCENT = "#DCE8F2"

# ----------------------------------------------------------------------------
# 1. END-TO-END PIPELINE
# ----------------------------------------------------------------------------

PIPELINE = [
    ("1. Document\nCollection",        "raw judgment\ntext / PDFs\nbucketed corpora"),
    ("2. OpenNyAI\nEnrichment",        "NER + rhetorical\nroles + summaries\n(annotated JSON)"),
    ("3. Mistral Outcome\nLabelling",  "win / loss /\nprocedural label\n(labelled JSON)"),
    ("4. Multi-Hearing\nMerge",        "one dispute =\none final case\n(merged records)"),
    ("5. Leakage-Safe\nCleaning",      "decision text\nremoved\n(cleaned cases)"),
    ("6. Graph Build +\nEmbeddings",   "PyG HeteroData\n+ bge-m3\n(graph caches)"),
]


def round_box(ax, x, y, w, h, fc, ec, text, fontsize=10, fontweight="normal", txt_color=INK):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=txt_color, fontweight=fontweight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=2.0, style="-|>", mutation=18):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=mutation, zorder=1,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Two rows, 3 stages each with serpentine flow.
    # Each cell = stage (top) + arrow + output (below) within row.
    # Layout: row1 left->right (stages 1,2,3), row2 right->left (stages 4,5,6)

    stage_w, stage_h = 3.0, 1.05
    out_w, out_h = 3.0, 1.05
    cx_left = [2.2, 7.5, 12.8]   # row1 column centers
    cx_right = [12.8, 7.5, 2.2]  # row2 column centers (reversed)

    # Y positions
    y_stage_top = 5.05
    y_out_top = 3.55
    y_stage_bot = 2.05
    y_out_bot = 0.55

    # --- Row 1: stages 1, 2, 3 ---
    for i, cx in enumerate(cx_left):
        s_label, o_label = PIPELINE[i]
        round_box(ax, cx, y_stage_top, stage_w, stage_h, ACCENT, BLUE,
                  s_label, fontsize=11, fontweight="bold", txt_color=BLUE)
        round_box(ax, cx, y_out_top, out_w, out_h, "#FFFBEF", GOLD,
                  o_label, fontsize=9, txt_color=INK)

        # stage -> output (vertical, gold)
        arrow(ax, cx, y_stage_top - stage_h / 2,
              cx, y_out_top + out_h / 2, color=GOLD, lw=2.0)

    # output (i) -> stage (i+1) horizontal in row 1
    for i in range(2):
        x_from = cx_left[i] + out_w / 2
        x_to = cx_left[i + 1] - stage_w / 2
        arrow(ax, x_from, y_out_top + 0.35,
              x_to, y_stage_top - stage_h / 4, color=BLUE, lw=2.0)

    # row1 -> row2 transition: output 3 -> stage 4 (vertical between rows)
    arrow(ax, cx_left[2], y_out_top - out_h / 2,
          cx_right[0], y_stage_bot + stage_h / 2, color=BLUE, lw=2.0)

    # --- Row 2: stages 4, 5, 6 ---
    for i, cx in enumerate(cx_right):
        s_label, o_label = PIPELINE[3 + i]
        round_box(ax, cx, y_stage_bot, stage_w, stage_h, ACCENT, BLUE,
                  s_label, fontsize=11, fontweight="bold", txt_color=BLUE)
        round_box(ax, cx, y_out_bot, out_w, out_h, "#FFFBEF", GOLD,
                  o_label, fontsize=9, txt_color=INK)
        arrow(ax, cx, y_stage_bot - stage_h / 2,
              cx, y_out_bot + out_h / 2, color=GOLD, lw=2.0)

    # output -> next stage in row 2 (right to left)
    for i in range(2):
        x_from = cx_right[i] - out_w / 2
        x_to = cx_right[i + 1] + stage_w / 2
        arrow(ax, x_from, y_out_bot + 0.35,
              x_to, y_stage_bot - stage_h / 4, color=BLUE, lw=2.0)

    # Legend
    legend_x = 0.2
    legend_y = 5.85
    leg_stage = mpatches.Patch(facecolor=ACCENT, edgecolor=BLUE, label="Pipeline stage")
    leg_out = mpatches.Patch(facecolor="#FFFBEF", edgecolor=GOLD, label="Stage output (cached artefact)")
    ax.legend(handles=[leg_stage, leg_out], loc="upper left",
              frameon=False, fontsize=10, ncol=2,
              bbox_to_anchor=(0.0, 1.0))

    plt.tight_layout()
    out_path = OUT / "pipeline_flow_v2.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[pipeline] saved -> {out_path}")


# ----------------------------------------------------------------------------
# 2. SINGLE-CASE GRAPH STRUCTURE (all 17 nodes, every schema edge)
# ----------------------------------------------------------------------------

# Node definition: name -> (display, family, x, y)
# Family colors:
TEXT_FILL = "#FFEFD0"
TEXT_EDGE = GOLD
ENT_FILL = "#DCE8F2"
ENT_EDGE = BLUE
CASE_FILL = "#113A5C"
CASE_EDGE = "#0A2236"

# Layout: case center, three rings.
# Ring 1: text nodes (top half)
# Ring 2: identity entities (bottom half)
# Ring 3: shared legal authorities (right side)

def _polar(cx, cy, r, deg):
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def draw_single_case():
    fig, ax = plt.subplots(figsize=(16, 10.0))
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-8.0, 8.0)
    ax.axis("off")
    ax.set_aspect("equal")

    cx, cy = 0.0, 0.0

    # --- positions for 17 nodes ---
    # Three quadrant strategy:
    #   LEFT  -> text nodes (6)
    #   BOTTOM -> identity entities (7)
    #   TOP-RIGHT -> shared legal authorities (3)
    pos = {"case": (cx, cy)}

    # Text nodes laid out along the left, vertically stacked
    pos["arguments"]               = (-4.5,  0.0)
    pos["facts"]                   = (-7.0,  1.6)
    pos["preamble"]                = (-7.0,  3.4)
    pos["petitioner_arguments"]    = (-7.0, -0.2)
    pos["respondent_arguments"]    = (-7.0, -2.0)
    pos["other_lawyer_arguments"]  = (-7.0, -3.8)

    # Identity entity nodes laid out along the bottom in two tidy rows
    pos["petitioner"]        = (-3.0, -5.5)
    pos["respondent"]        = (-1.0, -5.5)
    pos["petitioner_lawyer"] = ( 1.2, -5.5)
    pos["defence_lawyer"]    = ( 3.4, -5.5)
    pos["lawyer"]            = ( 5.6, -5.5)
    pos["judge"]             = (-1.0, -3.6)
    pos["court"]             = ( 1.2, -3.6)

    # Shared legal authorities on the top-right
    pos["statute"]   = ( 4.0,  4.5)
    pos["provision"] = ( 1.6,  4.5)
    pos["precedent"] = ( 6.4,  4.5)

    # Display labels for all nodes
    labels = {
        "case": "Case",
        "preamble": "Preamble",
        "facts": "Facts",
        "arguments": "Arguments",
        "petitioner_arguments": "Petitioner\nArguments",
        "respondent_arguments": "Respondent\nArguments",
        "other_lawyer_arguments": "Other Lawyer\nArguments",
        "petitioner": "Petitioner",
        "respondent": "Respondent",
        "petitioner_lawyer": "Petitioner\nLawyer",
        "defence_lawyer": "Defence\nLawyer",
        "lawyer": "Lawyer",
        "court": "Court",
        "judge": "Judge",
        "statute": "Statute",
        "provision": "Provision",
        "precedent": "Precedent",
    }

    text_set = {"preamble", "facts", "arguments",
                "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments"}
    shared_set = {"statute", "provision", "precedent"}

    # --- edges from schema.py (only enabled-by-default ones) ---
    edges = [
        # case -> text
        ("case", "preamble", "has_preamble"),
        ("case", "facts", "has_facts"),
        ("case", "arguments", "has_arguments"),
        ("case", "petitioner_arguments", "has_petitioner_arguments"),
        ("case", "respondent_arguments", "has_respondent_arguments"),
        ("case", "other_lawyer_arguments", "has_other_lawyer_arguments"),
        # case -> entities
        ("case", "petitioner", "has_petitioner"),
        ("case", "respondent", "has_respondent"),
        ("case", "court", "heard_in"),
        ("case", "judge", "decided_by_bench"),
        ("case", "lawyer", "has_lawyer"),
        ("case", "petitioner_lawyer", "has_petitioner_lawyer"),
        ("case", "defence_lawyer", "has_defence_lawyer"),
        # arguments -> shared authorities
        ("arguments", "statute", "cites_statute"),
        ("arguments", "provision", "cites_provision"),
        ("arguments", "precedent", "cites_precedent"),
        # provision -> statute hierarchy
        ("provision", "statute", "belongs_to_statute"),
        # lawyer -> argument citations
        ("petitioner_lawyer", "arguments", "citation"),
        ("defence_lawyer", "arguments", "citation"),
        ("petitioner_lawyer", "petitioner_arguments", "citation"),
        ("defence_lawyer", "respondent_arguments", "citation"),
        ("lawyer", "other_lawyer_arguments", "citation"),
        # bridging edges
        ("provision", "arguments", "used_in_arguments"),
        ("statute", "arguments", "used_in_arguments"),
        ("petitioner", "arguments", "is_party_in_arguments"),
        ("respondent", "arguments", "is_party_in_arguments"),
        ("petitioner", "petitioner_arguments", "is_party_in_arguments"),
        ("respondent", "respondent_arguments", "is_party_in_arguments"),
        ("judge", "arguments", "presided_arguments"),
    ]

    # Draw edges first (so nodes sit on top).
    # Per (src,dst) pair, alternate slight curvature so parallel edges fan out.
    pair_count: dict[tuple[str, str], int] = {}
    for src, dst, rel in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        if src == "case" or dst == "case":
            color = BLUE; lw = 1.3; alpha = 0.85
        elif dst in shared_set or src in shared_set:
            color = GOLD; lw = 1.3; alpha = 0.92
        else:
            color = "#7A8A99"; lw = 0.95; alpha = 0.65

        key = (src, dst)
        idx = pair_count.get(key, 0)
        pair_count[key] = idx + 1
        rad = 0.10 + 0.08 * idx
        if (src, dst) == ("provision", "statute"):
            rad = 0.0
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", color=color, linewidth=lw,
            mutation_scale=10, alpha=alpha, zorder=1,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=24, shrinkB=24,
        )
        ax.add_patch(a)

    # Draw nodes on top
    def node_box(name, w=1.85, h=0.85):
        x, y = pos[name]
        if name == "case":
            fc, ec, txt = CASE_FILL, CASE_EDGE, "white"
            w, h = 1.95, 1.05
            fw = "bold"; fs = 14
        elif name in text_set:
            fc, ec, txt = TEXT_FILL, TEXT_EDGE, INK
            fw = "bold"; fs = 9.5
        elif name in shared_set:
            fc, ec, txt = "#FFEAA7", GOLD, INK
            fw = "bold"; fs = 10.5
        else:
            fc, ec, txt = ENT_FILL, ENT_EDGE, INK
            fw = "normal"; fs = 9
        round_box(ax, x, y, w, h, fc, ec, labels[name],
                  fontsize=fs, fontweight=fw, txt_color=txt)

    for n in pos:
        node_box(n)

    # Soft family group labels
    ax.text(-7.0, 4.6, "Text nodes", ha="center", va="center",
            fontsize=10, color=GOLD, fontweight="bold")
    ax.text( 4.0, 5.7, "Shared legal authorities", ha="center", va="center",
            fontsize=10, color=GOLD, fontweight="bold")
    ax.text( 1.3, -6.55, "Local identity entities", ha="center", va="center",
            fontsize=10, color=BLUE, fontweight="bold")

    # Family legend
    leg_case = mpatches.Patch(facecolor=CASE_FILL, edgecolor=CASE_EDGE, label="Case anchor (1)")
    leg_text = mpatches.Patch(facecolor=TEXT_FILL, edgecolor=TEXT_EDGE, label="Text nodes (6)")
    leg_ident = mpatches.Patch(facecolor=ENT_FILL, edgecolor=ENT_EDGE, label="Local identity entities (7)")
    leg_share = mpatches.Patch(facecolor="#FFEAA7", edgecolor=GOLD, label="Shared legal authorities (3)")
    ax.legend(handles=[leg_case, leg_text, leg_ident, leg_share],
              loc="upper left", bbox_to_anchor=(0.0, 1.0),
              fontsize=10, frameon=False, ncol=1)

    # Edge legend (top right)
    ax.plot([], [], color=BLUE, lw=1.6, label="case-incident edge")
    ax.plot([], [], color=GOLD, lw=1.6, label="shared-authority edge")
    ax.plot([], [], color="#7A8A99", lw=1.4, label="bridging / citation edge")
    leg2 = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
                     fontsize=10, frameon=False)
    ax.add_artist(leg2)

    plt.tight_layout()
    out_path = OUT / "single_case_graph_v2.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[single-case] saved -> {out_path}")


# ----------------------------------------------------------------------------
# 3. CROSS-CASE SHARING AND LEAKAGE CONTROL
# ----------------------------------------------------------------------------

def draw_cross_case():
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # --- header ---
    ax.text(8.0, 8.85, "Cross-Case Sharing and Leakage Control",
            ha="center", va="center", fontsize=15, fontweight="bold", color=BLUE)
    ax.text(8.0, 8.40,
            "Shared: statutes, provisions, precedents.   Local: parties, lawyers, judges, courts.",
            ha="center", va="center", fontsize=10, color=INK, style="italic")

    # --- shared authority band (top, gold) ---
    band_y = 7.2
    band = FancyBboxPatch(
        (0.6, band_y - 0.7), 14.8, 1.4,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="#FFF6E1", edgecolor=GOLD, linewidth=1.6, zorder=1,
    )
    ax.add_patch(band)
    ax.text(1.0, band_y + 0.50, "SHARED LAYER",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=GOLD)

    shared_nodes = [
        ("IPC",                       3.6, band_y),
        ("CrPC",                      5.7, band_y),
        ("Section 482",               7.8, band_y),
        ("Supreme Court\nprecedent",  10.2, band_y),
        ("Constitution\nof India",    13.0, band_y),
    ]
    for label, x, y in shared_nodes:
        round_box(ax, x, y, 1.85, 0.8, "#FFEAA7", GOLD, label,
                  fontsize=9, fontweight="bold", txt_color=INK)

    # --- three case clusters ---
    case_y = 3.7
    case_x = [3.0, 8.0, 13.0]
    case_titles = ["Case A", "Case B", "Case C"]

    # Local satellites laid out so labels never overlap each other or the cluster header.
    local_specs = [
        ("Petitioner",   -1.7, -0.5),
        ("Respondent",   -1.7,  0.5),
        ("Pet. Lawyer",   1.7, -0.5),
        ("Def. Lawyer",   1.7,  0.5),
        ("Judge",         0.0, -1.55),
        ("Court",         0.0,  1.45),
    ]

    cluster_w, cluster_h = 4.4, 4.0
    for i, (cx, ttl) in enumerate(zip(case_x, case_titles)):
        # Cluster bounding box
        cluster = FancyBboxPatch(
            (cx - cluster_w / 2, case_y - cluster_h / 2),
            cluster_w, cluster_h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=SOFT, edgecolor="#A8B6C2", linewidth=1.2,
            linestyle="--", zorder=0,
        )
        ax.add_patch(cluster)
        # cluster title goes ABOVE the cluster, not inside it
        ax.text(cx, case_y + cluster_h / 2 + 0.22, f"{ttl} — local cluster",
                ha="center", va="bottom", fontsize=9.5, color="#5A6B7A",
                fontweight="bold")

        # Case node (anchor)
        round_box(ax, cx, case_y, 1.6, 0.9, CASE_FILL, CASE_EDGE,
                  ttl, fontsize=11, fontweight="bold", txt_color="white")

        # local satellites
        for label, dx, dy in local_specs:
            sx, sy = cx + dx, case_y + dy
            round_box(ax, sx, sy, 1.4, 0.55, ENT_FILL, ENT_EDGE, label,
                      fontsize=8, txt_color=INK)
            arrow(ax, cx, case_y, sx, sy, color=BLUE, lw=0.9,
                  style="-", mutation=8)

    # --- shared edges from cases up to authorities (gold) ---
    share_edges = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3),
    ]
    for ci, si in share_edges:
        cx_from = case_x[ci]
        y_from = case_y + cluster_h / 2  # leave from top of cluster
        x_to, y_to = shared_nodes[si][1], shared_nodes[si][2]
        arrow(ax, cx_from, y_from, x_to, y_to - 0.42,
              color=GOLD, lw=1.5, mutation=11)

    # Annotation between clusters and shared band — keep clear of cluster headers
    ax.text(8.0, 6.45,
            "Cross-case messages travel ONLY through shared legal authorities.",
            ha="center", va="center", fontsize=10, color=BLUE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="white", pad=2))

    # --- bottom: leakage control gate strip ---
    gate_y = 0.95
    gate_box = FancyBboxPatch(
        (0.6, gate_y - 0.55), 14.8, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=ACCENT, edgecolor=BLUE, linewidth=1.4, zorder=1,
    )
    ax.add_patch(gate_box)
    ax.text(1.0, gate_y + 0.35, "LEAKAGE GATES",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=BLUE)
    ax.text(8.0, gate_y - 0.05,
            "(1) drop decision fields    (2) remove RPC / RLC sentences    "
            "(3) mask outcome phrases    (4) temporal gating of authorities    "
            "(5) keep identity local",
            ha="center", va="center", fontsize=9.5, color=INK)

    # --- legend (compact, top-left margin) ---
    leg_case = mpatches.Patch(facecolor=CASE_FILL, edgecolor=CASE_EDGE,
                              label="Case anchor")
    leg_local = mpatches.Patch(facecolor=ENT_FILL, edgecolor=ENT_EDGE,
                               label="Local identity entity")
    leg_share = mpatches.Patch(facecolor="#FFEAA7", edgecolor=GOLD,
                               label="Shared legal authority")
    ax.legend(handles=[leg_case, leg_local, leg_share],
              loc="upper left", bbox_to_anchor=(0.005, 0.985),
              fontsize=9.5, frameon=False, ncol=1)

    plt.tight_layout()
    out_path = OUT / "cross_case_v2.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[cross-case] saved -> {out_path}")


if __name__ == "__main__":
    draw_pipeline()
    draw_single_case()
    draw_cross_case()
    print("Done.")
