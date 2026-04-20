"""
generate_deep_diagrams.py
=========================
Generates 5 richly annotated PNGs for the deep GNN documentation.

  1. full_graph_schema.png        – all node types, all edge types + labels
  2. node_feature_storage.png     – feature-tensor breakdown per node type
  3. layer_by_layer.png           – tensor state at L0 → L1 → L2 → L3 (3-layer HGT)
  4. message_passing_hop.png      – complete 2/3-hop receptive field from 'case'
  5. train_val_test_loop.png      – full training loop with early-stopping & metrics

Run:
    micromamba run -p .micromamba/gnn_case_star python document/generate_deep_diagrams.py
"""

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

OUTDIR = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document"

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
C_CASE    = "#3B1F8C"   # deep violet
C_TEXT    = "#1565C0"   # dark blue
C_ENTITY  = "#2E7D32"   # dark green
C_CONTEXT = "#E65100"   # deep orange
C_LEGAL   = "#880E4F"   # dark pink / legal citations
C_EDGE    = "#546E7A"   # blue-grey
C_BRIDGE  = "#F57F17"   # amber – bridging edges
C_CITE    = "#880E4F"   # dark pink – citation edges

FG = "white"

def _wrap(txt, w=14):
    return "\n".join(textwrap.wrap(txt, width=w))


# ══════════════════════════════════════════════════════════════════════════════
# 1. FULL GRAPH SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
def draw_full_graph_schema():
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)

    def box(x, y, w, h, col, label, fontsize=9.5, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.12",
                                       facecolor=col, edgecolor="white",
                                       linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", color=FG,
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                zorder=4, wrap=True)

    def arrow(x1, y1, x2, y2, label="", col=C_EDGE, dash=False):
        ls = (0, (4, 3)) if dash else "solid"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                   linestyle=ls, connectionstyle="arc3,rad=0.05"),
                    zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.22, label, ha="center", va="bottom",
                    color=col, fontsize=7.5, fontstyle="italic",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── case (centre) ─────────────────────────────────────────────────
    box(9.5, 7.2, 3.0, 1.6, C_CASE, "case\n(ROOT)", fontsize=12, bold=True)
    case_cx, case_cy = 11.0, 8.0

    # ── text nodes (left column) ──────────────────────────────────────
    text_positions = [("preamble", 2.0, 12.0), ("facts", 2.0, 8.5), ("arguments", 2.0, 5.0)]
    for label, x, y in text_positions:
        box(x, y, 2.6, 1.2, C_TEXT, label)
        arrow(case_cx, case_cy, x+1.3, y+0.6, f"has_{label}", C_TEXT)

    # ── entity nodes (right side) ─────────────────────────────────────
    entity_positions = [
        ("court\n(shared)", 14.0, 12.5),
        ("judge\n(shared)", 14.0, 10.7),
        ("petitioner\n(local)", 14.0, 8.9),
        ("respondent\n(local)", 14.0, 7.1),
        ("petitioner\nlawyer (shared)", 14.0, 5.3),
        ("defence\nlawyer (shared)", 14.0, 3.5),
    ]
    entity_labels = [
        "heard_in", "decided_by\n_bench", "has_petitioner", "has_respondent",
        "has_petitioner\n_lawyer", "has_defence\n_lawyer"
    ]
    for (label, x, y), elabel in zip(entity_positions, entity_labels):
        box(x, y, 2.7, 1.1, C_ENTITY, label, fontsize=8.5)
        arrow(case_cx, case_cy, x, y+0.55, elabel, C_ENTITY)

    # ── legal citation nodes (bottom left) ────────────────────────────
    legal = [("statute\n(shared)", 5.5, 1.5), ("provision\n(shared)", 8.5, 1.5), ("precedent\n(shared)", 11.5, 1.5)]
    for label, x, y in legal:
        box(x, y, 2.3, 1.0, C_LEGAL, label, fontsize=8.5)

    # arguments → statute / provision / precedent
    args_cx, args_cy = 3.3, 5.6
    arrow(args_cx, args_cy, 6.65, 2.5, "cites_statute", C_LEGAL)
    arrow(args_cx, args_cy, 9.65, 2.5, "cites_provision", C_LEGAL, dash=True)
    arrow(args_cx, args_cy, 12.65, 2.5, "cites_precedent", C_LEGAL, dash=True)

    # provision → statute (belongs_to)
    arrow(8.5+1.15, 2.0, 5.5+1.15, 2.0, "belongs_to_statute", C_LEGAL)

    # ── bridging edges ────────────────────────────────────────────────
    bridge_srcs = [
        ("provision", 9.65, 2.5, "used_in_arguments", 3.3, 5.6),
        ("statute",   6.65, 2.5, "used_in_arguments", 3.3, 5.6),
        ("petitioner",14.0+1.35, 8.9+0.55, "is_party_in_arguments", 5.6, 5.6),
        ("respondent",14.0+1.35, 7.1+0.55, "is_party_in_arguments", 5.6, 5.6),
        ("judge",     14.0+1.35, 10.7+0.55,"presided_arguments",    5.6, 5.6),
    ]
    for _, x1, y1, lbl, x2, y2 in bridge_srcs:
        arrow(x1, y1, x2, y2, lbl, C_BRIDGE, dash=True)

    # ── lawyer → arguments citation edges ────────────────────────────
    for _, x, y in [("pl", 14.0+1.35, 5.3+0.55), ("dl", 14.0+1.35, 3.5+0.55)]:
        arrow(x, y, 4.8, 5.6, "citation", C_CITE, dash=True)

    # ── context nodes (top right) ─────────────────────────────────────
    ctx = [("org(shared)", 17.5, 13.0), ("gpe(shared)", 17.5, 11.2),
           ("date(shared)", 17.5, 9.4), ("case_number\n(shared)", 17.5, 7.6)]
    ctx_labels = ["mentions_org", "mentions_gpe", "has_date", "has_case_number"]
    for (lbl, x, y), clbl in zip(ctx, ctx_labels):
        box(x, y, 2.4, 1.0, C_CONTEXT, lbl, fontsize=8)
        arrow(case_cx, case_cy, x, y+0.5, clbl, C_CONTEXT, dash=True)

    # ── legend ────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color=C_CASE,    label="Root (case)"),
        mpatches.Patch(color=C_TEXT,    label="Text nodes (case-local)"),
        mpatches.Patch(color=C_ENTITY,  label="Entity nodes (shared)"),
        mpatches.Patch(color=C_LEGAL,   label="Legal citation nodes (shared)"),
        mpatches.Patch(color=C_CONTEXT, label="Context nodes (shared)"),
        mpatches.Patch(color=C_BRIDGE,  label="Bridging / shortcut edges"),
        mpatches.Patch(color=C_CITE,    label="Lawyer citation edges"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9,
              bbox_to_anchor=(0.0, 1.0))

    ax.set_title("GNN Graph Schema — All Node & Edge Types", fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/1_full_graph_schema.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 1_full_graph_schema.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. NODE FEATURE STORAGE
# ══════════════════════════════════════════════════════════════════════════════
def draw_node_feature_storage():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.axis("off")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)

    # ── title ─────────────────────────────────────────────────────────
    ax.text(10, 10.5, "Node Feature Storage — data[node_type].x   shape: (N_nodes, embedding_dim + scalar_dim)",
            ha="center", va="center", fontsize=12, fontweight="bold")

    HEADER_Y = 10.0
    node_specs = [
        # (label, colour, x, text_key, embedding_note, scalars)
        ("case", C_CASE, 0.3,
         "text = concat(\n  preamble, facts, arguments\n)",
         "all-MiniLM-L6-v2\n→ 384-d",
         "respondent_count, judge_count,\nlawyer_count, statute_count,\nprovision_count, precedent_count,\npreamble_length, facts_length,\narguments_length, case_year,\npetition_type_known, petition_type_hash\n→ 12 floats [0,1]"),

        ("preamble / facts /\narguments", C_TEXT, 5.1,
         "text = raw section text\n(leakage-masked)",
         "all-MiniLM-L6-v2\n→ 384-d",
         "text_length, is_preamble,\nis_facts, is_arguments,\ncited_statute_count,\ncited_provision_count,\ncited_precedent_count,\npet_lawyer_count, def_lawyer_count,\npetitioner_count, respondent_count,\njudge_count (for args only)\n→ 12 floats [0,1]"),

        ("court / judge /\nlawyer / statute /\nprovision / precedent", C_ENTITY, 10.0,
         "text = canonical entity name\n(normalised string)",
         "all-MiniLM-L6-v2\n→ 384-d",
         "mention_count,\nfirst_seen_{preamble,facts,arguments},\nseen_in_arguments,\nseen_in_preamble,\nlocal_case_frequency,\nglobal_case_frequency,\ndegree, is_shared_node\n→ 10 floats [0,1]"),

        ("org / gpe /\ndate / case_number", C_CONTEXT, 14.9,
         "text = canonical entity name",
         "all-MiniLM-L6-v2\n→ 384-d",
         "(same entity scalars\nas above)\n→ 10 floats [0,1]"),
    ]

    for (label, col, x, text_note, embed_note, scalar_note) in node_specs:
        W = 4.6
        # Header bar
        r = mpatches.FancyBboxPatch((x, 8.5), W, 1.1, boxstyle="round,pad=0.1",
                                    facecolor=col, edgecolor="white", linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + W/2, 9.05, label, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")

        # text source
        ax.text(x + W/2, 8.1, "TEXT SOURCE", ha="center", va="center",
                color=col, fontsize=8, fontweight="bold")
        ax.text(x + W/2, 7.58, text_note, ha="center", va="center",
                color="#333", fontsize=7.5, style="italic",
                bbox=dict(facecolor="#f3f3f3", boxstyle="round,pad=0.2", edgecolor=col, lw=0.8))

        # embedding block
        eb = mpatches.FancyBboxPatch((x+0.1, 5.85), W-0.2, 1.3,
                                     boxstyle="round,pad=0.1",
                                     facecolor=col, alpha=0.18, edgecolor=col, lw=1.2)
        ax.add_patch(eb)
        ax.text(x + W/2, 6.85, "Text Embedding", ha="center", va="center",
                color=col, fontsize=8, fontweight="bold")
        ax.text(x + W/2, 6.40, embed_note, ha="center", va="center",
                color="#222", fontsize=8)

        # scalar block
        sb = mpatches.FancyBboxPatch((x+0.1, 2.0), W-0.2, 3.65,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#eceff1", edgecolor=col, lw=1.2)
        ax.add_patch(sb)
        ax.text(x + W/2, 5.45, "Scalar Features", ha="center", va="center",
                color=col, fontsize=8, fontweight="bold")
        ax.text(x + W/2, 3.8, scalar_note, ha="center", va="center",
                color="#222", fontsize=7.2, linespacing=1.4)

        # concat arrow
        ax.annotate("", xy=(x+W/2, 5.85), xytext=(x+W/2, 5.7),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
        ax.annotate("", xy=(x+W/2, 2.65), xytext=(x+W/2, 1.9),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))

        # final node.x
        fx = mpatches.FancyBboxPatch((x+0.2, 0.6), W-0.4, 1.15,
                                     boxstyle="round,pad=0.12",
                                     facecolor="#263238", edgecolor="white", lw=1.5)
        ax.add_patch(fx)
        ax.text(x + W/2, 1.17, "data[node_type].x\n(shape: N × 396)", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")

        ax.text(x+W/2, 0.22, "CONCAT( 384-d embed  ║  12-d scalars )", ha="center", va="center",
                color="#555", fontsize=7.5)

    fig.text(0.5, 0.01,
             "* For the fallback hashing encoder, embedding_dim = 384.  "
             "* Global nodes carry global_case_frequency & degree populated during graph merge.",
             ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_title("", pad=0)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/2_node_feature_storage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 2_node_feature_storage.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAYER-BY-LAYER TENSOR STATE
# ══════════════════════════════════════════════════════════════════════════════
def draw_layer_by_layer():
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis("off")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)

    ax.text(10, 9.65, "HGT 3-Layer Forward Pass — Tensor State at Each Stage",
            ha="center", va="center", fontsize=13, fontweight="bold")

    stages = [
        ("Stage 0\n(Raw Input)", "#37474F", 0.35,
         "data[node_type].x\nshape: (N, 396)\n= 384-d embed + 12 scalars\n\nAll node types have\ndifferent N but same\nfeature width 396"),
        ("Stage 1\n(After Projection)", C_TEXT, 4.0,
         "After: Linear(396→128)\n  + type_embedding(1×128)\n\nhidden[node_type]\nshape: (N, 128)\n\nAll node types now share\nthe SAME hidden_dim=128"),
        ("Stage 2\nAfter HGTConv L1", C_ENTITY, 8.0,
         "HGTConv aggregates\n1-hop neighbour messages.\n\nhidden[node_type]\nshape: (N, 128)\n\n+ Residual + Dropout(0.25)\n+ LayerNorm + ReLU\n\ncase now sees:\npreamble, facts, arguments,\npetitioner, respondent,\ncourt, judge, lawyers..."),
        ("Stage 3\nAfter HGTConv L2", C_LEGAL, 12.0,
         "HGTConv aggregates\n2-hop messages.\n\nhidden[node_type]\nshape: (N, 128)\n\n+ Residual + Dropout(0.25)\n+ LayerNorm + ReLU\n\ncase now sees:\nstatute, provision,\nprecedent (via arguments)\nand bridging edges"),
        ("Stage 4\nAfter HGTConv L3", C_CITE, 15.65,
         "HGTConv aggregates\n3-hop messages.\n\nhidden[node_type]\nshape: (N, 128)\n\n+ Residual + Dropout(0.25)\n+ LayerNorm + ReLU\n\ncase sees statute via:\ncase→args→provision→statute\n(3 hops)"),
    ]

    W, H = 3.8, 7.0
    SY = 1.0

    for (title, col, x, desc) in stages:
        rect = mpatches.FancyBboxPatch((x, SY), W, H,
                                       boxstyle="round,pad=0.15",
                                       facecolor=col, alpha=0.12,
                                       edgecolor=col, linewidth=2.5, zorder=2)
        ax.add_patch(rect)
        hdr = mpatches.FancyBboxPatch((x, SY+H-1.15), W, 1.15,
                                      boxstyle="round,pad=0.1",
                                      facecolor=col, edgecolor="none", zorder=3)
        ax.add_patch(hdr)
        ax.text(x + W/2, SY+H-0.58, title, ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold", zorder=4)
        ax.text(x + W/2, SY + (H-1.2)/2, desc, ha="center", va="center",
                color="#1a1a1a", fontsize=8.0, linespacing=1.5,
                zorder=4, wrap=True)

    # Arrows between stages
    arrow_y = SY + H/2
    for xi in [0.35 + W, 4.0 + W, 8.0 + W, 12.0 + W]:
        ax.annotate("", xy=(xi + 0.2, arrow_y), xytext=(xi - 0.2 + 0.01, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color="#B0BEC5", lw=2.5), zorder=5)

    # MLP Output box
    mlp_x = 19.6
    mlp = mpatches.FancyBboxPatch((mlp_x - 0.5, 3.0), 0.9, 4.0,
                                   boxstyle="round,pad=0.1",
                                   facecolor=C_CASE, edgecolor="white", lw=1.5)
    ax.add_patch(mlp)
    ax.text(mlp_x, 5.0, "MLP\nHead\n\n→ logits\n(N_case\n× 2)", ha="center", va="center",
            color="white", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(mlp_x - 0.51, 5.0), xytext=(15.65 + W + 0.05, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#B0BEC5", lw=2.5), zorder=5)
    ax.text(17.5, 5.35, "case nodes\nonly", ha="center", va="bottom",
            color="#555", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/3_layer_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 3_layer_by_layer.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPLETE RECEPTIVE FIELD / HOP DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def draw_receptive_field():
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.axis("off")
    ax.set_xlim(-1, 21)
    ax.set_ylim(-0.5, 13)

    ax.text(10, 12.6, "Receptive Field of the 'case' Node — 3 HGT Layers",
            ha="center", va="center", fontsize=13, fontweight="bold")

    def node(x, y, label, col, r=0.52, fontsize=8.5):
        circ = plt.Circle((x, y), r, color=col, zorder=3, linewidth=1.5,
                           edgecolor="white")
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold", zorder=4)

    def edge(x1, y1, x2, y2, lbl, col="#546E7A", dash=False):
        ls = (0, (4, 3)) if dash else "solid"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                   linestyle=ls, connectionstyle="arc3,rad=0.05"))
        mx, my = (x1+x2)/2, (y1+y2)/2
        if lbl:
            ax.text(mx, my+0.25, lbl, ha="center", va="bottom",
                    color=col, fontsize=7,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Layer bands
    for band_x, band_col, band_lbl, anchor_y in [
        (0.0, "#ECEFF1", "HOP-2 Neighbours\n(after Layer 2)", 6.5),
        (5.2, "#E3F2FD", "HOP-1 Neighbours\n(after Layer 1)", 6.5),
        (11.0, "#F3E5F5", "ROOT NODE\n(after Layer 0→3)", 6.5),
        (14.8, "#FFF3E0", "HOP-1 Neighbours\n(right side)", 6.5),
        (18.0, "#F1F8E9", "HOP-3 Bridge\n(Statute via Provision)", 6.5),
    ]:
        bw = 5.0 if band_x < 14 else 2.5
        rect = mpatches.FancyBboxPatch((band_x, -0.3), bw, 12.5,
                                       boxstyle="square,pad=0",
                                       facecolor=band_col, edgecolor="#CFD8DC",
                                       linewidth=1.0, zorder=0, alpha=0.5)
        ax.add_patch(rect)
        ax.text(band_x + bw/2, 12.25, band_lbl, ha="center", va="center",
                color="#555", fontsize=8.5, fontweight="bold")

    # Case node (centre)
    cx, cy = 12.5, 6.0
    node(cx, cy, "case\n(ROOT)", C_CASE, r=0.75, fontsize=10)

    # HOP-1 RIGHT: entity nodes
    h1r = [
        ("court", 15.4, 10.5, C_ENTITY),
        ("judge",  15.4,  8.7, C_ENTITY),
        ("petitioner",  15.4,  7.0, C_ENTITY),
        ("respondent",  15.4,  5.3, C_ENTITY),
        ("pet_lawyer",  15.4,  3.6, C_ENTITY),
        ("def_lawyer",  15.4,  1.9, C_ENTITY),
    ]
    h1r_rels = ["heard_in", "decided_by_bench", "has_petitioner", "has_respondent",
                "has_pet_lawyer", "has_def_lawyer"]
    for (lbl, x, y, col), rel in zip(h1r, h1r_rels):
        node(x, y, lbl, col)
        edge(cx, cy, x, y, rel, col)

    # HOP-1 LEFT: text nodes
    h1l = [
        ("preamble", 6.5, 10.3, C_TEXT),
        ("facts",     6.5,  6.0, C_TEXT),
        ("arguments", 6.5,  1.9, C_TEXT),
    ]
    for (lbl, x, y, col) in h1l:
        node(x, y, lbl, col)
        edge(cx, cy, x, y, f"has_{lbl}", col)

    # HOP-2 LEFT: legal nodes (via arguments)
    args_x, args_y = 6.5, 1.9
    h2_nodes = [
        ("statute",   2.0,  4.5, C_LEGAL, "cites_statute"),
        ("provision", 2.0,  2.5, C_LEGAL, "cites_provision"),
        ("precedent", 2.0,  0.5, C_LEGAL, "cites_precedent"),
    ]
    for (lbl, x, y, col, rel) in h2_nodes:
        node(x, y, lbl, col)
        edge(args_x, args_y, x, y, rel, col, dash=True)

    # Bridging: petition/respondent/judge → arguments
    for (lbl, x, y, col) in [("petitioner", 15.4, 7.0, C_ENTITY),
                               ("respondent", 15.4, 5.3, C_ENTITY),
                               ("judge",      15.4, 8.7, C_ENTITY)]:
        bridge_lbl = "is_party_in_args" if "er" in lbl else "presided_args"
        edge(x, y, args_x, args_y, bridge_lbl, C_BRIDGE, dash=True)

    # Lawyer citation → arguments
    for (_, x, y, _) in h1r[-2:]:
        edge(x, y, args_x, args_y, "citation", C_CITE, dash=True)

    # Provision → Statute (belongs_to) at HOP-3
    stat_x, stat_y = 19.5, 2.5
    node(stat_x, stat_y, "statute\n(3-hop)", C_LEGAL, fontsize=7.5)
    edge(2.0, 2.5, stat_x, stat_y, "belongs_to\n_statute", C_LEGAL, dash=False)

    # Legend
    legend = [
        mpatches.Patch(color=C_CASE,   label="Root (case)"),
        mpatches.Patch(color=C_TEXT,   label="Text nodes"),
        mpatches.Patch(color=C_ENTITY, label="Entity nodes"),
        mpatches.Patch(color=C_LEGAL,  label="Legal citations"),
        mpatches.Patch(color=C_BRIDGE, label="Bridging edges (shortcut)"),
        mpatches.Patch(color=C_CITE,   label="Lawyer citation edges"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/4_receptive_field.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 4_receptive_field.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL / TEST LOOP
# ══════════════════════════════════════════════════════════════════════════════
def draw_training_loop():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.axis("off")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)

    ax.text(9, 12.65, "Training Loop, Early Stopping & Evaluation Procedure",
            ha="center", va="center", fontsize=13, fontweight="bold")

    def box(x, y, w, h, col, label, fontsize=9, lw=2.0):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=col, edgecolor="white",
                                    linewidth=lw, zorder=3)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                color="white", fontsize=fontsize, fontweight="bold", zorder=4)

    def arr(x1, y1, x2, y2, lbl="", col="#546E7A"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                   connectionstyle="arc3,rad=0.0"), zorder=5)
        if lbl:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.12, my, lbl, ha="left", va="center",
                    color=col, fontsize=7.5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── 1. Split ──────────────────────────────────────────────────────
    box(0.4, 10.8, 4.5, 1.5, "#37474F",
        "1.  BUILD SPLIT MASKS\n"
        "mode=random  stratify=True  seed=42\n"
        "train=70%   val=15%   test=15%")

    box(0.4, 8.7, 2.0, 1.7, "#1565C0", "train_mask\n(70%)", fontsize=8.5)
    box(2.6, 8.7, 2.0, 1.7, "#2E7D32", "val_mask\n(15%)", fontsize=8.5)
    box(4.8, 8.7, 2.0, 1.7, "#880E4F", "test_mask\n(15%)", fontsize=8.5)

    arr(2.65, 10.8, 1.4, 10.4)
    arr(2.65, 10.8, 3.6, 10.4)
    arr(2.65, 10.8, 5.8, 10.4)

    # ── 2. Class weights ──────────────────────────────────────────────
    box(0.4, 6.6, 6.5, 1.7, "#4A148C",
        "2.  BALANCED CLASS WEIGHTS\n"
        "sklearn compute_class_weight('balanced')\n"
        "computed on train labels only\n"
        "→ weight tensor for cross_entropy loss")
    arr(2.4, 8.7, 2.4, 8.3)

    # ── 3. Epoch loop ─────────────────────────────────────────────────
    box(7.5, 9.0, 10.0, 3.5, "#263238",
        "3.  EPOCH LOOP  (max=60 epochs)\n\n"
        "   model.train()\n"
        "   logits, _ = model(x_dict, edge_index_dict)   # full-graph transductive forward\n"
        "   loss = cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)\n"
        "   loss.backward()  →  AdamW(lr=1e-3, wd=1e-5).step()\n\n"
        "   model.eval()  →  eval on train + val masks (no grad)", fontsize=8.5)
    arr(0.4+6.5, 7.45, 7.5, 10.75)   # class-weights → epoch loop

    # ── 4. Early stopping ─────────────────────────────────────────────
    box(7.5, 6.5, 4.8, 2.1, "#1B5E20",
        "4. EARLY STOPPING\n"
        "  monitor: val macro_F1\n"
        "  patience = 15 epochs\n"
        "  save best_state_dict if improved")
    arr(9.9, 9.0, 9.9, 8.6)

    # ── 5. Best model restore ─────────────────────────────────────────
    box(12.6, 6.5, 4.8, 2.1, "#B71C1C",
        "5. BEST MODEL RESTORE\n"
        "  load_state_dict(best_state)\n"
        "  model.eval()\n"
        "  final forward → logits + hidden")
    arr(12.3, 7.55, 12.6, 7.55)

    # ── 6. Metrics ────────────────────────────────────────────────────
    box(7.5, 4.1, 10.0, 2.0, "#4E342E",
        "6. FINAL METRICS  (on best model, per split)\n"
        "  accuracy  |  macro F1  |  micro F1  |  per-class P/R/F1/support\n"
        "  ROC-AUC (binary: y_proba[:, 1])   |   PR-AUC\n"
        "  confusion matrix saved as PNG")
    arr(9.9, 6.5, 9.9, 6.1)
    arr(14.9, 6.5, 14.9, 6.1)

    # ── 7. Outputs ────────────────────────────────────────────────────
    box(7.5, 1.5, 4.8, 2.2, "#006064",
        "7. SAVED OUTPUTS\n"
        "  model.pt  (best weights)\n"
        "  metrics.json\n"
        "  predictions.csv\n"
        "  confusion_matrix_test.png", fontsize=8.5)

    box(12.6, 1.5, 4.8, 2.2, "#33691E",
        "8. predictions.csv COLUMNS\n"
        "  case_id | file_name | raw_label\n"
        "  split | target_index | target_label\n"
        "  pred_index | pred_label | confidence", fontsize=8.5)

    arr(9.9, 4.1, 9.9, 3.7)
    arr(9.9, 1.7, 12.6, 1.7, "", "#00897B")

    # ── Transductive note ─────────────────────────────────────────────
    ax.text(9.0, 0.4,
            "★  TRANSDUCTIVE SETTING: HGT runs a single forward pass over the FULL graph every epoch.\n"
            "     Masks select which case-node logits contribute to loss (train) and metrics (val/test).",
            ha="center", va="center", fontsize=8.5, color="#37474F", style="italic",
            bbox=dict(facecolor="#ECEFF1", boxstyle="round,pad=0.3", edgecolor="#90A4AE"))

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/5_training_loop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 5_training_loop.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating deep GNN diagrams…")
    draw_full_graph_schema()
    draw_node_feature_storage()
    draw_layer_by_layer()
    draw_receptive_field()
    draw_training_loop()
    print("All done.")
