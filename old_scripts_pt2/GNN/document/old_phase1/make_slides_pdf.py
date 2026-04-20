"""
make_slides_pdf.py
==================
Generates a rich 12-slide PDF for the GNN Architecture talk.
Requires: matplotlib (already in env)

Run:
    micromamba run -p .micromamba/gnn_case_star \
        python document/make_slides_pdf.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────  constants  ────────────────────────────────────
OUTDIR  = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document")
IMGDIR  = OUTDIR        # PNGs live in the same folder
PDF_OUT = OUTDIR / "gnn_architecture_slides.pdf"

SW, SH = 16, 9          # slide width / height in inches
DPI     = 150

# Brand palette
BG      = "#0D1117"     # near-black slide background
ACCENT  = "#58A6FF"     # bright blue
ACCENT2 = "#3FB950"     # green
ACCENT3 = "#F78166"     # coral
ACCENT4 = "#FFA657"     # orange
ACCENT5 = "#BC8CFF"     # purple
FG      = "#E6EDF3"     # near-white text
MID     = "#8B949E"     # muted grey
CARD    = "#161B22"     # card background
BORDER  = "#30363D"     # card border

# ─────────────────────────────  helpers  ──────────────────────────────────────
def new_slide(pdf: PdfPages, title: str = "", subtitle: str = "") -> tuple:
    fig = plt.figure(figsize=(SW, SH), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, SW)
    ax.set_ylim(0, SH)
    ax.axis("off")

    # top rule
    ax.axhline(SH - 0.55, color=ACCENT, linewidth=2.5, xmin=0.02, xmax=0.98, zorder=5)

    if title:
        ax.text(0.35, SH - 0.28, title,
                color=FG, fontsize=20, fontweight="bold", va="center", zorder=6)
    if subtitle:
        ax.text(0.35, SH - 0.52, subtitle,
                color=MID, fontsize=11, va="center", style="italic", zorder=6)

    # Slide index dot (decorative)
    circ = plt.Circle((0.20, SH - 0.28), 0.10, color=ACCENT, zorder=6)
    ax.add_patch(circ)

    return fig, ax


def card(ax, x, y, w, h, title="", body="", title_col=ACCENT, fontsize=8.5,
         body_fontsize=8, icon=""):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=CARD, edgecolor=BORDER, linewidth=1.2, zorder=3)
    ax.add_patch(r)
    top = y + h - 0.02
    if title:
        label = f"{icon}  {title}" if icon else title
        ax.text(x + 0.15, top - 0.20, label, color=title_col,
                fontsize=fontsize, fontweight="bold", va="top", zorder=4)
        top -= 0.38
    if body:
        ax.text(x + 0.15, top - 0.05, body, color=FG,
                fontsize=body_fontsize, va="top", zorder=4,
                linespacing=1.5, wrap=True)


def metric_pill(ax, x, y, label, value, col):
    r = FancyBboxPatch((x, y), 2.6, 1.0, boxstyle="round,pad=0.08",
                       facecolor=col, alpha=0.18, edgecolor=col, linewidth=1.5, zorder=3)
    ax.add_patch(r)
    ax.text(x + 1.3, y + 0.65, value, color=col, fontsize=20,
            fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(x + 1.3, y + 0.20, label, color=MID, fontsize=8.5,
            ha="center", va="center", zorder=4)


def add_image(ax, path, x, y, w, h):
    try:
        img = plt.imread(str(path))
        ax_img = ax.inset_axes([x/SW, y/SH, w/SW, h/SH], zorder=5)
        ax_img.imshow(img, aspect="auto")
        ax_img.axis("off")
        # thin border
        for spine in ax_img.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1.0)
    except Exception as e:
        ax.text(x + w/2, y + h/2, f"[image: {Path(path).name}]",
                color=MID, fontsize=9, ha="center", va="center",
                style="italic", zorder=5)


def save_slide(pdf, fig):
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────  load metrics  ─────────────────────────────────
METRICS_PATH = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
                    "/GNN/outputs/food_law_final/models/food_law_final_full/metrics.json")
with open(METRICS_PATH) as f:
    M = json.load(f)

history   = M["history"]
epochs    = [h["epoch"] for h in history]
tr_loss   = [h["train_loss"] for h in history]
tr_f1     = [h["train_macro_f1"] for h in history]
val_f1    = [h["val_macro_f1"] for h in history]
best_ep   = M["best_epoch"]

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1  –  Title
# ══════════════════════════════════════════════════════════════════════════════
def slide_title(pdf):
    fig = plt.figure(figsize=(SW, SH), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, SW); ax.set_ylim(0, SH); ax.axis("off")

    # Decorative diagonal stripe
    from matplotlib.patches import Polygon as MPoly
    poly = MPoly([(SW*0.6, SH), (SW, SH), (SW, 0), (SW*0.78, 0)],
                 closed=True, color=ACCENT, alpha=0.06, zorder=1)
    ax.add_patch(poly)

    # Big title
    ax.text(SW/2, 5.8, "Pre-Judgment Legal Outcome Prediction",
            color=FG,     fontsize=26, fontweight="bold",
            ha="center",  va="center", zorder=3)
    ax.text(SW/2, 5.0, "Graph Neural Network Architecture & Training",
            color=ACCENT, fontsize=18, fontweight="bold",
            ha="center",  va="center", zorder=3)

    # Divider
    ax.axhline(4.6, color=BORDER, linewidth=1.5, xmin=0.15, xmax=0.85)

    ax.text(SW/2, 4.2, "Dataset: Food Law Cases  ·  2,105 raw cases  ·  1,466 binary labels",
            color=MID, fontsize=12, ha="center", va="center")
    ax.text(SW/2, 3.7, "Model: HGT (3-layer Heterogeneous Graph Transformer)  ·  hidden_dim=128",
            color=MID, fontsize=12, ha="center", va="center")

    # Key result pills
    pills = [
        (SW/2 - 5.5, 2.0, "Test Macro F1",    "0.709", ACCENT3),
        (SW/2 - 2.7, 2.0, "Val Macro F1",      "0.844", ACCENT2),
        (SW/2 + 0.1, 2.0, "Test Accuracy",      "76.8%", ACCENT),
        (SW/2 + 2.9, 2.0, "Test ROC-AUC",      "0.815", ACCENT5),
    ]
    for x, y, lbl, val, col in pills:
        metric_pill(ax, x, y, lbl, val, col)

    ax.text(SW/2, 1.1, "Ziv Baretto  ·  Capstone Thesis  ·  2026",
            color=MID, fontsize=10, ha="center")

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2  –  Pipeline Overview
# ══════════════════════════════════════════════════════════════════════════════
def slide_pipeline(pdf):
    fig, ax = new_slide(pdf, "End-to-End Pipeline",
                        "From raw JSON case files to binary prediction")

    stages = [
        "("-", "Raw JSON\nCase Files", "2,105 cases\nOpenNyAI output", ACCENT),
        "(">", "Leakage\nFilter", "Drop decision text,\nRPC annotations,\noutcome phrases", ACCENT3),
        "("G", "Case Star\nGraph Builder", "1 case node + text nodes\n+ entity nodes per case", ACCENT5),
        "("W", "Global Graph\nMerge", "Share court, judge,\nstatute, provision\nacross all cases", ACCENT4),
        "("#", "Feature\nExtraction", "MiniLM-L6 embed (384-d)\n+ 12 scalar features\n→ 396-d per node", ACCENT2),
        "("~", "HGT 3-Layer\nTraining", "AdamW, balanced weights\nearly stop @ val F1\nbest epoch: 34", ACCENT),
    ]

    N = len(stages)
    xs = np.linspace(0.6, SW - 0.6, N * 2 - 1)
    box_xs = xs[::2]
    W, H = 2.2, 2.6
    Y = 3.2

    for i, (icon, title, body, col) in enumerate(stages):
        bx = box_xs[i] - W/2
        r = FancyBboxPatch((bx, Y), W, H, boxstyle="round,pad=0.12",
                           facecolor=col, alpha=0.15, edgecolor=col,
                           linewidth=1.8, zorder=3)
        ax.add_patch(r)
        ax.text(bx + W/2, Y + H - 0.28, icon,
                ha="center", va="center", fontsize=18, zorder=4)
        ax.text(bx + W/2, Y + H - 0.75, title, color=col,
                ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=4)
        ax.text(bx + W/2, Y + 0.9, body, color=FG,
                ha="center", va="center", fontsize=7.5,
                linespacing=1.4, zorder=4)
        if i < N - 1:
            ax.annotate("", xy=(box_xs[i+1] - W/2 - 0.08, Y + H/2),
                        xytext=(bx + W + 0.08, Y + H/2),
                        arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.8))

    # Annotation below
    ax.text(SW/2, 1.8, "Transductive setting: full graph is always in memory; only supervision masks change.",
            color=MID, fontsize=9.5, ha="center", style="italic")

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3  –  Full Graph Schema (embedded PNG)
# ══════════════════════════════════════════════════════════════════════════════
def slide_graph_schema(pdf):
    fig, ax = new_slide(pdf, "Graph Schema",
                        "All 18 node types and 25+ edge types in the heterogeneous graph")

    add_image(ax, IMGDIR / "1_full_graph_schema.png",
              x=0.3, y=0.9, w=SW - 0.6, h=SH - 1.5)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4  –  Node Types Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
def slide_node_types(pdf):
    fig, ax = new_slide(pdf, "Node Types",
                        "Three categories: Root · Text · Entity/Authority")

    groups = [
        ("ROOT", ACCENT3, 0.4,
         [("case", "1 per case\nCarries all case-level\nmetadata + combined text\n\nTarget classification node")]),
        ("TEXT NODES\n(case-local)", ACCENT, 3.2,
         [("preamble", "Preamble section text\nleakage-masked"),
          ("facts",    "Facts section text\nleakage-masked"),
          ("arguments","Arguments section text\nHub for all legal\ncitation edges")]),
        ("ENTITY / AUTHORITY", ACCENT2, 8.8,
         [("court★",   "Which court heard it\nShared → cross-case signal"),
          ("judge★",   "Presiding bench\nShared → career-level\nlearning"),
          ("pet_lawyer★","Petitioner advocate\nShared → caseload signal"),
          ("def_lawyer★","Defence advocate\nShared → caseload signal"),
          ("statute★", "Cited law (e.g. IPC)\nShared"),
          ("provision★","Specific section\nShared"),
          ("precedent★","Cited case name"),
          ("petitioner","Filing party (local)"),
          ("respondent","Opposing party (local)")]),
    ]

    for gname, gcol, gx, nodes in groups:
        NW = 1.85
        NH = 1.9
        PAD = 0.12
        total_w = len(nodes) * NW + (len(nodes) - 1) * PAD

        # Group label
        ax.text(gx + total_w/2, 8.25, gname, color=gcol,
                fontsize=9.5, fontweight="bold", ha="center", va="center")

        for j, (ntype, desc) in enumerate(nodes):
            bx = gx + j * (NW + PAD)
            r = FancyBboxPatch((bx, 1.8), NW, NH,
                               boxstyle="round,pad=0.1",
                               facecolor=gcol, alpha=0.12,
                               edgecolor=gcol, linewidth=1.2, zorder=3)
            ax.add_patch(r)
            ax.text(bx + NW/2, 1.8 + NH - 0.28, ntype, color=gcol,
                    fontsize=9, fontweight="bold", ha="center", va="center", zorder=4)
            ax.text(bx + NW/2, 1.8 + NH - 0.75, desc, color=FG,
                    fontsize=7, ha="center", va="center",
                    linespacing=1.35, zorder=4)

    ax.text(SW/2, 1.0,
            "★ Shared nodes: same node object appears across every case that mentions that entity. "
            "This creates cross-case information paths.",
            color=MID, fontsize=8.5, ha="center", style="italic")

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5  –  Edge Semantics
# ══════════════════════════════════════════════════════════════════════════════
def slide_edges(pdf):
    fig, ax = new_slide(pdf, "Edge Types & Their Role in HGT",
                        "Each (src_type, relation, dst_type) triple gets its own learned K/Q/V projections")

    edge_groups = [
        ("Primary  (case ↔ everything)", ACCENT, [
            "case → has_preamble/facts/arguments    Case absorbs all text content",
            "case → heard_in → court                Which court adjudicated",
            "case → decided_by_bench → judge         Bench identity & history",
            "case → has_petitioner/respondent        Filing/opposing party",
            "case → has_petitioner_lawyer /           Advocate identity; shared nodes accumulate",
            "          has_defence_lawyer               caseload signal",
        ]),
        ("Citation  (arguments ↔ legal entities)", ACCENT5, [
            "arguments → cites_statute               Law invoked",
            "arguments → cites_provision             Specific section invoked",
            "arguments → cites_precedent             Precedent relied upon",
            "provision → belongs_to_statute          Legal hierarchy",
            "petitioner_lawyer → citation → arguments  Lawyer tied to specific arguments",
            "defence_lawyer   → citation → arguments  Lawyer tied to defence arguments",
        ]),
        ("Bridging  (shortcuts to reduce hop distance)", ACCENT4, [
            "provision → used_in_arguments           3-hop statute→args→case becomes 2-hop",
            "statute   → used_in_arguments           Same shortcut for statute",
            "petitioner → is_party_in_arguments       Party visible directly at arguments",
            "respondent → is_party_in_arguments       Party visible directly at arguments",
            "judge      → presided_arguments          Judge context at arguments node",
        ]),
    ]

    Y = 7.8
    for gname, gcol, rows in edge_groups:
        ax.text(0.45, Y, gname, color=gcol, fontsize=10.5, fontweight="bold", va="top")
        Y -= 0.38
        for row in rows:
            rel, _, desc = row.partition("   ")
            ax.text(0.65, Y, "·  ", color=gcol, fontsize=8.5, va="top")
            ax.text(0.82, Y, rel, color=FG, fontsize=8.5, va="top",
                    fontfamily="monospace")
            ax.text(5.5, Y, desc.strip(), color=MID, fontsize=8.5, va="top")
            Y -= 0.35
        Y -= 0.20

    # ToUndirected note
    card(ax, 0.4, 0.5, SW - 0.8, 0.85,
         title="After graph construction: ToUndirected() is applied",
         body="Every directed edge gets a reverse counterpart added automatically. "
              "Messages flow symmetrically during HGT convolution.",
         title_col=ACCENT2, fontsize=9)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6  –  Node Feature Storage
# ══════════════════════════════════════════════════════════════════════════════
def slide_features(pdf):
    fig, ax = new_slide(pdf,
                        "Node Feature Tensors  —  data[node_type].x",
                        "Shape: (N_nodes, 396)  =  384-d text embedding  +  12-d scalar features")

    add_image(ax, IMGDIR / "2_node_feature_storage.png",
              x=0.3, y=0.9, w=SW - 0.6, h=SH - 1.55)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7  –  Layer-by-Layer
# ══════════════════════════════════════════════════════════════════════════════
def slide_layers(pdf):
    fig, ax = new_slide(pdf,
                        "Layer-by-Layer Tensor State",
                        "3 HGT layers: what the 'case' node absorbs at each hop")

    add_image(ax, IMGDIR / "3_layer_by_layer.png",
              x=0.3, y=0.9, w=SW - 0.6, h=SH - 1.55)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8  –  Receptive Field
# ══════════════════════════════════════════════════════════════════════════════
def slide_receptive_field(pdf):
    fig, ax = new_slide(pdf,
                        "Multi-Hop Receptive Field",
                        "Every path the 'case' node can see after 3 HGT layers")

    add_image(ax, IMGDIR / "4_receptive_field.png",
              x=0.3, y=0.9, w=SW - 0.6, h=SH - 1.55)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9  –  Train / Val / Test Procedure
# ══════════════════════════════════════════════════════════════════════════════
def slide_training_procedure(pdf):
    fig, ax = new_slide(pdf,
                        "Training Procedure",
                        "Transductive HGT · balanced class weights · early stopping on val macro F1")

    add_image(ax, IMGDIR / "5_training_loop.png",
              x=0.3, y=0.9, w=SW - 0.6, h=SH - 1.55)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10  –  Training Curves  (live from metrics.json)
# ══════════════════════════════════════════════════════════════════════════════
def slide_training_curves(pdf):
    fig, ax = new_slide(pdf,
                        "Training Curves  —  Seed 42  (food_law_final_full)",
                        f"Best epoch: {best_ep}    Best val macro F1: {M['best_val_macro_f1']:.4f}")

    # two sub-axes
    ax_loss = fig.add_axes([0.05, 0.13, 0.43, 0.72], facecolor=CARD)
    ax_f1   = fig.add_axes([0.53, 0.13, 0.43, 0.72], facecolor=CARD)

    for a in [ax_loss, ax_f1]:
        a.set_facecolor(CARD)
        a.tick_params(colors=MID)
        for spine in a.spines.values():
            spine.set_edgecolor(BORDER)
        a.xaxis.label.set_color(MID)
        a.yaxis.label.set_color(MID)
        a.title.set_color(FG)
        a.grid(alpha=0.12, color=BORDER)
        a.axvline(best_ep, color=ACCENT3, linewidth=1.5, linestyle="--", alpha=0.9,
                  label=f"Best epoch {best_ep}")

    ax_loss.plot(epochs, tr_loss, color=ACCENT,  linewidth=2, label="Train Loss")
    ax_loss.set_title("Training Loss", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=FG)

    ax_f1.plot(epochs, tr_f1,  color=ACCENT2, linewidth=2, label="Train Macro F1")
    ax_f1.plot(epochs, val_f1, color=ACCENT3, linewidth=2, label="Val Macro F1")
    ax_f1.axhline(M["val"]["macro_f1"], color=ACCENT5, linewidth=1.5,
                  linestyle=":", label=f"Val final {M['val']['macro_f1']:.3f}")
    ax_f1.set_title("Macro F1", fontsize=11, fontweight="bold")
    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Macro F1")
    ax_f1.set_ylim(0.3, 1.05)
    ax_f1.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=FG)

    for a in [ax_loss, ax_f1]:
        a.tick_params(colors=MID, which="both")

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11  –  Results
# ══════════════════════════════════════════════════════════════════════════════
def slide_results(pdf):
    fig, ax = new_slide(pdf,
                        "Results  —  Best Run (Seed 42)",
                        "Binary classification: lose (0) · win (1)   ·   Food Law dataset, 1,466 cases")

    # ── metric pills ──────────────────────────────────────────────────
    splits = [("TRAIN", M["train"], ACCENT2, 0.5),
              ("VAL",   M["val"],   ACCENT,  5.8),
              ("TEST",  M["test"],  ACCENT3, 11.1)]

    pill_metrics = [
        ("Accuracy",  "accuracy",  ACCENT2),
        ("Macro F1",  "macro_f1",  ACCENT),
        ("ROC-AUC",   "roc_auc",   ACCENT5),
        ("PR-AUC",    "pr_auc",    ACCENT4),
    ]

    PW, PH = 2.2, 0.9
    for sname, sdata, scol, sx in splits:
        ax.text(sx + 1.7, 7.85, sname, color=scol, fontsize=12,
                fontweight="bold", ha="center")
        for pi, (mlabel, mkey, mcol) in enumerate(pill_metrics):
            py = 6.4 - pi * 1.1
            val = sdata.get(mkey, None)
            valstr = f"{val:.3f}" if val is not None else "N/A"
            r = FancyBboxPatch((sx, py), PW*1.55, PH,
                               boxstyle="round,pad=0.08",
                               facecolor=mcol, alpha=0.15,
                               edgecolor=mcol, linewidth=1.2, zorder=3)
            ax.add_patch(r)
            ax.text(sx + PW*1.55/2, py + PH*0.62, valstr, color=mcol,
                    ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)
            ax.text(sx + PW*1.55/2, py + PH*0.20, mlabel, color=MID,
                    ha="center", va="center", fontsize=8, zorder=4)

    # ── per-class table ───────────────────────────────────────────────
    card(ax, 0.4, 0.4, SW - 0.8, 1.5,
         title="Per-Class Results (Test Set — Seed 42)",
         title_col=ACCENT4)

    headers = ["Class", "Precision", "Recall", "F1", "Support"]
    col_xs  = [0.8, 2.8, 5.0, 7.2, 9.4]
    row_y   = 1.5
    for hx, ht in zip(col_xs, headers):
        ax.text(hx, row_y, ht, color=ACCENT4, fontsize=9, fontweight="bold")
    for cls in ["lose", "win"]:
        row_y -= 0.38
        d = M["test"]["per_class"][cls]
        vals = [cls, f"{d['precision']:.3f}", f"{d['recall']:.3f}",
                f"{d['f1']:.3f}", str(d['support'])]
        for vx, vv in zip(col_xs, vals):
            ax.text(vx, row_y, vv, color=FG, fontsize=9)

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12  –  Limitations & Next Steps
# ══════════════════════════════════════════════════════════════════════════════
def slide_limitations(pdf):
    fig, ax = new_slide(pdf,
                        "Limitations & Improvement Directions",
                        "What currently constrains performance and how to address it")

    limits = [
        ("⚠", "Class Imbalance",
         "win:lose ≈ 2.2:1 in binary set.\n"
         "Balanced class weights help but 'lose' F1 still lags (0.578 test).",
         ACCENT3),
        ("⚠", "Overfitting",
         "Train F1 → 1.00 while test F1 = 0.709.\n"
         "Gap suggests model memorises training cases.",
         ACCENT3),
        ("⚠", "Transductive Leakage Risk",
         "Even with masks, ALL nodes (incl. test) participate in message passing.\n"
         "Shared authority nodes carry indirect test-case signal.",
         ACCENT4),
        ("⚠", "String-based Entity Normalisation",
         "Merging authority nodes relies on fuzzy string matching.\n"
         "Noise in entity resolution can mis-merge or fragment nodes.",
         ACCENT4),
    ]
    improvements = [
        ("✦", "Mini-batch HGTLoader", "Inductive training; removes transductive leakage.", ACCENT2),
        ("✦", "Chunked Text Embeddings", "Mean-pool 512-token chunks instead of truncating long arguments.", ACCENT2),
        ("✦", "3-Seed Averaging", "Runs 2 & 3 in progress; average final test metrics.", ACCENT),
        ("✦", "Adversarial Regularisation", "Add noise to shared node embeddings at train time.", ACCENT),
        ("✦", "Temporal Split", "Use year-based split to stress-test out-of-distribution generalisation.", ACCENT5),
    ]

    LW, LH = 7.0, 1.75
    for i, (icon, title, body, col) in enumerate(limits):
        r = i % 2
        c = i // 2
        bx = 0.4 + c * (LW + 0.3)
        by = 5.0 - r * (LH + 0.2)
        r2 = FancyBboxPatch((bx, by), LW, LH, boxstyle="round,pad=0.1",
                            facecolor=col, alpha=0.10,
                            edgecolor=col, linewidth=1.2, zorder=3)
        ax.add_patch(r2)
        ax.text(bx + 0.25, by + LH - 0.30, f"{icon}  {title}",
                color=col, fontsize=9.5, fontweight="bold", va="center", zorder=4)
        ax.text(bx + 0.25, by + LH - 0.70, body,
                color=FG, fontsize=8, va="top", linespacing=1.4, zorder=4)

    # Improvements column
    ax.text(SW - 5.0, 8.3, "IMPROVEMENT DIRECTIONS", color=ACCENT2,
            fontsize=10, fontweight="bold")
    IY = 7.7
    for icon, title, body, col in improvements:
        ax.text(SW - 4.8, IY, f"{icon}  ", color=col, fontsize=10, va="top")
        ax.text(SW - 4.3, IY, title, color=col, fontsize=9.5, fontweight="bold", va="top")
        ax.text(SW - 4.3, IY - 0.33, body, color=MID, fontsize=8, va="top")
        IY -= 1.0

    save_slide(pdf, fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    slides = [
        ("Title Slide",           slide_title),
        ("Pipeline Overview",     slide_pipeline),
        ("Graph Schema",          slide_graph_schema),
        ("Node Types",            slide_node_types),
        ("Edge Semantics",        slide_edges),
        ("Node Feature Tensors",  slide_features),
        ("Layer-by-Layer State",  slide_layers),
        ("Receptive Field",       slide_receptive_field),
        ("Training Procedure",    slide_training_procedure),
        ("Training Curves",       slide_training_curves),
        ("Results",               slide_results),
        ("Limitations",           slide_limitations),
    ]

    with PdfPages(str(PDF_OUT)) as pdf:
        for name, fn in slides:
            print(f"  Rendering: {name} …")
            fn(pdf)

        meta = pdf.infodict()
        meta["Title"]   = "GNN Architecture – Pre-Judgment Legal Outcome Prediction"
        meta["Author"]  = "Ziv Baretto"
        meta["Subject"] = "Capstone Thesis 2026"

    print(f"\n✓  PDF saved to {PDF_OUT}")
    print(f"   {PDF_OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
