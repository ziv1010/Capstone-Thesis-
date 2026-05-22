"""Generate the end-to-end pipeline diagram for the paper.

Produces pipeline_diagram.png and pipeline_diagram.pdf in the same folder.
Run with: micromamba run -n thesis_work python make_pipeline_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Colour palette per stage (light pastel fill, dark border).
STAGE_STYLES = {
    1: {"fill": "#E8F0FA", "border": "#3A6CB8", "title": "Stage 1 — Dataset creation"},
    2: {"fill": "#E8F5E8", "border": "#3F8B3F", "title": "Stage 2 — Graph building"},
    3: {"fill": "#FCEFE0", "border": "#C77A2C", "title": "Stage 3 — LJP (training & evaluation)"},
    4: {"fill": "#F8E5E5", "border": "#B53A3A", "title": "Stage 4 — Explainability & hearing-wise analysis"},
}

BLOCK_FILL = "#FFFFFF"
BLOCK_BORDER = "#444444"
ARROW_COLOR = "#444444"

# ---------------------------------------------------------------------------
# Geometry. Coordinates are in arbitrary "units" — figure size and aspect are
# set later. Each stage row is one band; blocks live inside.
BLOCK_W = 3.6
BLOCK_H = 1.25
H_GAP = 0.55  # horizontal gap between blocks within a row
V_GAP = 0.65  # vertical gap between rows inside a stage (only for stage 4)
STAGE_GAP = 1.1  # vertical gap between stage containers
STAGE_PAD = 0.45  # internal padding inside a stage container

# Block content per stage (lane = 0 main, lane = 1 secondary lane in stage 4).
STAGES = [
    {
        "id": 1,
        "rows": [
            [
                "Raw HC judgments\n(5 domain buckets)",
                "OpenNyAI\nNER + RR + summarizer",
                "LLM cross-validated\noutcome labelling\n(8-Q consistency prompt)",
                "Multi-hearing\nconsolidation\n(case-number merge)",
            ],
        ],
    },
    {
        "id": 2,
        "rows": [
            [
                "Leakage-safe\npreprocessing\n(role drop, phrase mask)",
                "Section assembly\n(6 section texts)",
                "Local case-star +\nglobal authority graph",
                "Node init: text emb\n$\\oplus$ scalars (12 / 12 / 10)",
            ],
        ],
    },
    {
        "id": 3,
        "rows": [
            [
                "Dual text encoder\n(BGE-M3 / InLegalBERT)",
                "HGT backbone\n(typed proj., type embed,\nrel-aware multi-head attn.)",
                "5-fold CV\nper-bucket + cross-bucket",
                "Ablation grid\n(8 graph variants\n$\\pm$ LR decay)",
            ],
        ],
    },
    {
        "id": 4,
        "rows": [
            [
                "Frozen inference +\ncase embeddings",
                "Heterogeneous\nPGExplainer training",
                "Importance ranking\n($L$-hop BFS, max edge score)",
                "Evidence diagnostic +\nembedding $k$-NN +\ntop-$k$ sufficiency",
            ],
            [
                "Stage construction\n(initial / accumulated)",
                "Self-contained\ntest graph + padding",
                "Ensembled inference +\ntransitions, what-changed",
            ],
        ],
    },
]


def draw_block(ax, x, y, text, *, w=BLOCK_W, h=BLOCK_H,
               fill=BLOCK_FILL, border=BLOCK_BORDER, fontsize=9):
    """Place a rounded rectangle with centred multi-line text. (x, y) is centre."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.1,
        edgecolor=border,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a", zorder=4)
    return box


def draw_stage_container(ax, x_left, x_right, y_top, y_bottom, stage_id):
    style = STAGE_STYLES[stage_id]
    box = FancyBboxPatch(
        (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=1.6,
        edgecolor=style["border"],
        facecolor=style["fill"],
        zorder=1,
    )
    ax.add_patch(box)
    # Title pinned to top-left, inside a thin white pill so it sits over the border.
    title_x = x_left + 0.32
    title_y = y_top - 0.04
    ax.text(
        title_x, title_y, style["title"],
        ha="left", va="top",
        fontsize=11, fontweight="bold", color=style["border"], zorder=5,
        bbox=dict(facecolor="white", edgecolor=style["border"],
                  boxstyle="round,pad=0.18", linewidth=0.8),
    )


def draw_arrow(ax, p_from, p_to, *, lw=1.4, color=ARROW_COLOR, mut=14):
    arrow = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>",
        mutation_scale=mut,
        linewidth=lw,
        color=color,
        zorder=2,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def main():
    # Figure-out coordinates -------------------------------------------------
    # Determine width based on the widest single row of any stage.
    max_cols = max(len(row) for stage in STAGES for row in stage["rows"])
    row_width = max_cols * BLOCK_W + (max_cols - 1) * H_GAP
    container_inner_left = -row_width / 2
    container_inner_right = +row_width / 2
    container_outer_left = container_inner_left - STAGE_PAD
    container_outer_right = container_inner_right + STAGE_PAD

    # Walk top-to-bottom, computing y centres for each row.
    cursor_y = 0.0  # top of the current stage container
    stage_geoms = []  # collected per-stage info for arrow drawing

    # First pass: lay out stage geometry and rows.
    for stage in STAGES:
        n_rows = len(stage["rows"])
        # title strip occupies a slice at the top of each container.
        title_strip = 0.55
        rows_height = n_rows * BLOCK_H + (n_rows - 1) * V_GAP
        container_h = title_strip + 2 * STAGE_PAD + rows_height
        y_top = cursor_y
        y_bottom = cursor_y - container_h
        # Row centres start below the title strip.
        first_row_centre_y = y_top - STAGE_PAD - title_strip - BLOCK_H / 2
        row_centres = [
            first_row_centre_y - i * (BLOCK_H + V_GAP) for i in range(n_rows)
        ]
        stage_geoms.append({
            "id": stage["id"],
            "y_top": y_top,
            "y_bottom": y_bottom,
            "rows": stage["rows"],
            "row_centres": row_centres,
        })
        cursor_y = y_bottom - STAGE_GAP

    total_height = -stage_geoms[-1]["y_bottom"]
    total_width = container_outer_right - container_outer_left

    # Build figure -----------------------------------------------------------
    aspect = total_width / total_height
    fig_h = 11.0
    fig_w = fig_h * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(container_outer_left - 0.2, container_outer_right + 0.2)
    ax.set_ylim(stage_geoms[-1]["y_bottom"] - 0.3, 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw stage containers and blocks.
    placed_blocks = {}  # (stage_idx, row_idx, col_idx) -> (cx, cy)
    for s_idx, geom in enumerate(stage_geoms):
        draw_stage_container(
            ax,
            container_outer_left, container_outer_right,
            geom["y_top"], geom["y_bottom"],
            geom["id"],
        )
        for r_idx, (row, cy) in enumerate(zip(geom["rows"], geom["row_centres"])):
            n = len(row)
            row_total_w = n * BLOCK_W + (n - 1) * H_GAP
            x_start = -row_total_w / 2 + BLOCK_W / 2
            for c_idx, text in enumerate(row):
                cx = x_start + c_idx * (BLOCK_W + H_GAP)
                draw_block(ax, cx, cy, text)
                placed_blocks[(s_idx, r_idx, c_idx)] = (cx, cy)

    # Intra-row arrows.
    for s_idx, geom in enumerate(stage_geoms):
        for r_idx, row in enumerate(geom["rows"]):
            for c_idx in range(len(row) - 1):
                cx1, cy1 = placed_blocks[(s_idx, r_idx, c_idx)]
                cx2, cy2 = placed_blocks[(s_idx, r_idx, c_idx + 1)]
                draw_arrow(
                    ax,
                    (cx1 + BLOCK_W / 2 + 0.02, cy1),
                    (cx2 - BLOCK_W / 2 - 0.02, cy2),
                )

    # Inter-stage arrows.
    for s_idx in range(len(stage_geoms) - 1):
        y_from = stage_geoms[s_idx]["y_bottom"]
        y_to = stage_geoms[s_idx + 1]["y_top"]
        draw_arrow(ax, (0, y_from), (0, y_to), lw=1.8, mut=18)

    # Tighten layout and save.
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    png_path = OUT_DIR / "pipeline_diagram.png"
    pdf_path = OUT_DIR / "pipeline_diagram.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
