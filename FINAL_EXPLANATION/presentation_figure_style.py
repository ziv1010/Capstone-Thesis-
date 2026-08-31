#!/usr/bin/env python
"""Matplotlib drawing primitives shared by the static presentation figures.

Lifted out of ``generate_paper_figures.fig3_contrastive`` so the paper figure
and ``generate_presentation_figures.py`` draw identical cards, case nodes and
connector curves, and so both stay in step with the browser renderer in
``visualizer_static/app.js``.

Geometry constants match the SVG renderer 1:1 (card width 205, case circle r=52,
and the slide/detail card heights below) so a figure exported for a slide looks
like the panel on screen.
"""
from __future__ import annotations

import textwrap
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


CARD_W = 205.0
CASE_R = 52.0

# Two card layouts.  "slide" is the default: type + name + a "#3 ▲" pill on the
# type row, because idf and a Δ of 5e-05 are unreadable from the back of a room.
# "detail" keeps idf and the full "CF #3 ▲ Δ0.011" badge on its own row, for the
# paper figures and the thesis.
CARD_H_SLIDE = 52.0
ROW_PITCH_SLIDE = 66.0
CARD_H_DETAIL = 66.0
ROW_PITCH_DETAIL = 80.0

CARD_H = CARD_H_SLIDE
ROW_PITCH = ROW_PITCH_SLIDE

BADGE_W_SLIDE = 62.0
BADGE_W_DETAIL = 124.0


def card_metrics(detail: bool) -> tuple[float, float]:
    """``(card height, row pitch)`` for the chosen layout."""
    return (CARD_H_DETAIL, ROW_PITCH_DETAIL) if detail else (CARD_H_SLIDE, ROW_PITCH_SLIDE)


TYPE_STYLES = {
    "precedent": ("#eaf7ef", "#2e8b57"),
    "court": ("#fff7e8", "#b36b00"),
    "judge": ("#fff7e8", "#b36b00"),
    "provision": ("#eaf7fb", "#1f7a9a"),
    "statute": ("#edf2ff", "#4f46e5"),
}
DEFAULT_STYLE = ("#f4f4f5", "#52525b")

IDENTITY_TYPES = {"court", "judge", "petitioner", "respondent", "lawyer"}

CF_STYLES = {
    "supports": ("#e6f6ed", "#1f7a4d", "▲"),
    "opposes": ("#fdeceb", "#b23a3a", "▼"),
}

QUERY_EDGE = "#1f7a83"
SAME_EDGE = "#2f855a"
OPPOSITE_EDGE = "#a85f00"
INK = "#1f2937"
MUTED = "#64748b"
WRONG_INK = "#b23a3a"


def type_style(feature_type: str) -> tuple[str, str]:
    return TYPE_STYLES.get(str(feature_type), DEFAULT_STYLE)


def type_label(feature_type: str) -> str:
    return str(feature_type).upper()


def clean_feature_name(name: str, limit: int = 27) -> str:
    name = " ".join(
        str(name)
        .replace("manusc", "MANU/SC/")
        .replace("1983crilj1457", "1983 Cri LJ 1457")
        .split()
    )
    if len(name) <= limit:
        return name
    return name[: max(0, limit - 3)].rstrip() + "..."


def wrap_title(title: str, width: int = 24, max_lines: int = 3) -> list[str]:
    lines = textwrap.wrap(" ".join(str(title).split()), width=width, break_long_words=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 1)].rstrip() + "…"
    return lines or [str(title)]


def stack_y(center_y: float, rows: Sequence[Any], i: int, pitch: float = ROW_PITCH) -> float:
    block_h = max(len(rows), 1) * pitch
    return center_y - block_h / 2 + i * pitch


def new_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    """An axes in pixel coordinates with y growing downwards, as in SVG."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def savefig(fig: plt.Figure, out_dir, stem: str, formats: Sequence[str]) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        written.append(path)
        print(f"[figure] wrote {path}", flush=True)
    return written


def connect_path(ax, points, color: str = "#9cc9b1", dashed: bool = False, alpha: float = 0.42, lw: float = 2.2) -> None:
    path = MplPath(points, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=(0, (5.0, 5.0)) if dashed else "solid",
            capstyle="round",
            zorder=1,
        )
    )


def draw_case_circle(
    ax,
    x: float,
    y: float,
    edge: str,
    *,
    label_badge: str,
    case_index: Any,
    pred_badge: str | None = None,
    correct: bool | None = None,
    title: str | None = None,
    bucket: str | None = None,
    radius: float = CASE_R,
) -> None:
    """Compact circle carrying the labels, with the real case name captioned below.

    Both the true label and the model's prediction are shown: the nearest
    opposite-*true*-label case usually gets the *same* prediction, so a circle
    showing only ground truth hides whether the model actually separated the two
    cases at all.
    """
    ax.add_patch(patches.Circle((x, y), radius, facecolor="white", edgecolor=edge, linewidth=2.4, zorder=3))
    if pred_badge is None:
        ax.text(x, y - 16, "true", ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)
        ax.text(x, y + 4, str(label_badge), ha="center", va="center", fontsize=17, weight="bold", color=INK, zorder=4)
    else:
        wrong = correct is False
        ax.text(x, y - 22, f"true  {label_badge}", ha="center", va="center",
                fontsize=10.5, weight="bold", color=INK, zorder=4)
        ax.text(x, y - 2, f"pred  {pred_badge}", ha="center", va="center",
                fontsize=10.5, weight="bold", color=WRONG_INK if wrong else INK, zorder=4)
        if wrong:
            ax.text(x, y + 17, "misclassified", ha="center", va="center",
                    fontsize=7.0, weight="bold", style="italic", color=WRONG_INK, zorder=4)
    ax.text(x, y + (33 if pred_badge is not None else 27), f"#{case_index}",
            ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)
    if not title:
        return
    lines = wrap_title(title, width=24, max_lines=3)
    for i, line in enumerate(lines):
        ax.text(x, y + radius + 22 + i * 14, line, ha="center", va="center", fontsize=8.4, weight="bold", color=INK, zorder=4)
    if bucket:
        ax.text(
            x,
            y + radius + 22 + len(lines) * 14 + 2,
            bucket,
            ha="center",
            va="center",
            fontsize=7.3,
            color=MUTED,
            zorder=4,
        )


def format_delta(delta: Any) -> str:
    """Counterfactual effects span several orders of magnitude; ``%.4f`` would
    print most of them as ``0.0000``."""
    try:
        value = abs(float(delta))
    except (TypeError, ValueError):
        return ""
    if value >= 0.001:
        return f"{value:.3f}"
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("e-0", "e-")


def factor_label(row: dict[str, Any], limit: int = 34) -> str:
    """``judge: udurga prasad rao``, but just ``arguments`` for whole-section
    factors whose name repeats their type."""
    ftype = str(row.get("evidence_type") or "").strip()
    name = str(row.get("evidence_name") or "").strip()
    pretty_type = ftype.replace("_", " ")
    if not name or name == ftype:
        return pretty_type
    return f"{pretty_type}: {clean_feature_name(name, limit=limit)}"


def cf_badge_text(feature: dict[str, Any], detail: bool = False) -> str | None:
    rank = feature.get("cf_evidence_rank")
    if rank is None:
        return None
    parts = [f"CF #{int(rank)}" if detail else f"#{int(rank)}"]
    style = CF_STYLES.get(str(feature.get("cf_direction")))
    if style:
        parts.append(style[2])
    if detail:
        delta = format_delta(feature.get("cf_abs_delta"))
        if delta:
            parts.append(f"Δ{delta}")
    return " ".join(parts)


def draw_card(
    ax,
    x: float,
    y: float,
    feature: dict[str, Any],
    *,
    w: float = CARD_W,
    h: float | None = None,
    show_cf: bool = True,
    detail: bool = False,
) -> None:
    """One evidence box, stamped with its counterfactual rank.

    Slide layout (default): type + name, with a compact "#3 ▲" pill on the type
    row.  ``detail=True`` restores the idf reading and the full
    "CF #3 ▲ Δ0.011" badge on its own row.
    """
    if h is None:
        h = CARD_H_DETAIL if detail else CARD_H_SLIDE
    badge_w = BADGE_W_DETAIL if detail else BADGE_W_SLIDE
    ftype = str(feature.get("feature_type") or feature.get("evidence_type") or "evidence")
    face, edge = type_style(ftype)
    scored = show_cf and feature.get("cf_evidence_rank") is not None
    cf_face, cf_edge, _arrow = CF_STYLES.get(str(feature.get("cf_direction")), ("#eef2f6", "#94a3b8", ""))
    alpha = 0.62 if (show_cf and not scored) else 1.0

    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 2.0, y + 3.2), w, h,
            boxstyle="round,pad=0.008,rounding_size=8",
            facecolor="#142330", edgecolor="none", alpha=0.055, zorder=0.5,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=8",
            facecolor=face, edgecolor=edge,
            linewidth=2.7 if ftype in IDENTITY_TYPES else 1.45,
            alpha=alpha, zorder=2,
        )
    )
    if feature.get("cf_top"):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 3, y + 8), 5, h - 16,
                boxstyle="round,pad=0,rounding_size=2.5",
                facecolor=cf_edge, edgecolor="none", zorder=3,
            )
        )

    ax.text(x + 12, y + 18, type_label(ftype), ha="left", va="center",
            fontsize=7.5, weight="bold", color=edge, zorder=3)
    if detail and feature.get("idf") is not None:
        ax.text(x + w - 10, y + 18, f"idf {float(feature['idf']):.1f}", ha="right", va="center",
                fontsize=7.5, weight="bold", color=MUTED, zorder=3)
    ax.text(
        x + 12, y + (35 if detail else 38),
        clean_feature_name(feature.get("feature_name") or feature.get("evidence_name") or "unnamed evidence"),
        ha="left", va="center", fontsize=9.0, weight="bold", color=INK, zorder=3,
    )
    if not show_cf:
        return

    text = cf_badge_text(feature, detail)
    # Detail keeps the badge on its own row; slide tucks it onto the type row.
    badge_y = (y + h - 13) if detail else (y + 18)
    badge_x = (x + 11) if detail else (x + w - 10 - badge_w)
    if text:
        ax.add_patch(
            patches.FancyBboxPatch(
                (badge_x, badge_y - 8.5), badge_w, 17,
                boxstyle="round,pad=0,rounding_size=8.5",
                facecolor=cf_face, edgecolor=cf_edge, linewidth=1.1, zorder=3,
            )
        )
        ax.text(badge_x + badge_w / 2, badge_y, text, ha="center", va="center",
                fontsize=7.2, weight="bold", color=cf_edge, zorder=4)
        if feature.get("cf_flips"):
            flip_x = (badge_x + badge_w + 6) if detail else (badge_x - 48)
            ax.add_patch(
                patches.FancyBboxPatch(
                    (flip_x, badge_y - 8.5), 42, 17,
                    boxstyle="round,pad=0,rounding_size=8.5",
                    facecolor="#b23a3a", edgecolor="#b23a3a", zorder=3,
                )
            )
            ax.text(flip_x + 21, badge_y, "FLIPS", ha="center", va="center",
                    fontsize=7.2, weight="bold", color="white", zorder=4)
    else:
        note = "no CF data" if feature.get("cf_available") is False else "not scored"
        ax.text(badge_x if detail else x + w - 10, badge_y, note,
                ha="left" if detail else "right", va="center",
                fontsize=7.2, style="italic", color="#97a4b1", zorder=3)


def draw_cf_legend(ax, x: float, y: float) -> None:
    ax.text(
        x, y,
        "▲ supports the decision (masking lowers confidence)      "
        "▼ argues against it (masking raises confidence)      "
        "▏ ribbon = top-3 driving factor",
        ha="left", va="center", fontsize=8.0, color=MUTED,
    )


def draw_top_factors(
    ax,
    x: float,
    y: float,
    case_meta: dict[str, Any],
    role: str,
    limit: int = 3,
) -> str | None:
    """One case's strongest counterfactual factors, in ranked order.

    Worth its own line because the strongest factors are often evidence types
    (parties, arguments, lawyers) that get no box in the contrast diagram — the
    diagram alone would imply the top driver is whichever box ranks highest.
    Returns a footnote to print once if any factor has no box, else ``None``.
    """
    label = f"{role} ({case_meta.get('label_badge') or '?'})"
    if case_meta.get("cf_available") is False:
        ax.text(x, y, f"{label}:   no counterfactual data — masking was only run on test cases",
                ha="left", va="center", fontsize=8.2, style="italic", color=MUTED)
        return None
    rows = [
        row for row in (case_meta.get("top_factors") or [])
        if row.get("cf_evidence_rank") is not None
    ][:limit]
    if not rows:
        return None
    parts = [
        f"#{int(row['cf_evidence_rank'])} "
        f"{CF_STYLES.get(str(row.get('cf_direction')), ('', '', ''))[2]} {factor_label(row)}"
        for row in rows
    ]
    ax.text(x, y, f"{label}:", ha="left", va="center", fontsize=8.2, weight="bold", color=INK)
    ax.text(x + 150, y, "      ".join(parts), ha="left", va="center", fontsize=8.2, color=INK)
    if any(not row.get("has_box", True) for row in rows):
        return "(types with no box in this diagram are still ranked)"
    return None
