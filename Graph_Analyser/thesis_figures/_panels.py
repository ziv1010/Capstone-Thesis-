"""Per-panel renderers.

Each `save_*` function writes ONE artefact for ONE case into `out_dir`.
PNGs for the visual panels and Markdown for the case summary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from _shared import Paths, load_phase4, load_predictions

_ALLOWED_COLOR = "#1f77b4"
_DISMISSED_COLOR = "#d62728"
_TARGET_COLOR = "#111111"

_CATEGORY_COLORS = {
    "statute": "#2ca02c",
    "provision": "#9467bd",
    "precedent": "#ff7f0e",
}


def _row_for(paths: Paths, case_idx: int):
    preds = load_predictions(paths).set_index("node_index")
    return preds.loc[case_idx]


# ---------- (a) t-SNE ----------

def _draw_tsne(ax: Axes, paths: Paths, target_idx: int, split: str = "test") -> None:
    from _shared import compute_or_load_tsne

    points, node_indices = compute_or_load_tsne(paths, split=split)
    preds = load_predictions(paths).set_index("node_index")
    pred_labels = preds.loc[node_indices, "pred_label"].to_numpy()

    is_allowed = np.array(
        [str(p).strip() in ("1", "allowed", "Allowed") for p in pred_labels]
    )
    ax.scatter(points[~is_allowed, 0], points[~is_allowed, 1], s=10,
               c=_DISMISSED_COLOR, alpha=0.35, linewidths=0, label="pred: dismissed")
    ax.scatter(points[is_allowed, 0], points[is_allowed, 1], s=10,
               c=_ALLOWED_COLOR, alpha=0.35, linewidths=0, label="pred: allowed")

    if target_idx in set(node_indices.tolist()):
        j = int(np.where(node_indices == target_idx)[0][0])
        ax.scatter(points[j, 0], points[j, 1], s=360, marker="*",
                   c=_TARGET_COLOR, edgecolors="white", linewidths=1.5,
                   zorder=5, label=f"target (node {target_idx})")
    ax.set_title(f"t-SNE of post-GNN case embeddings — {split} split", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def save_tsne_png(paths: Paths, case_idx: int, out_dir: Path,
                  split: str = "test") -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_tsne(ax, paths, case_idx, split=split)
    fp = out_dir / "panel_a_tsne.png"
    fig.tight_layout(); fig.savefig(fp, dpi=180, bbox_inches="tight"); plt.close(fig)
    return fp


# ---------- (b) PGExplainer subgraph ----------

def _draw_subgraph(ax: Axes, phase4: dict, max_per_cat: int = 3) -> None:
    categories: list[tuple[str, list[dict]]] = []
    top = phase4.get("top_nodes", {}) or {}
    for key in ("statute", "provision", "precedent"):
        items = [it for it in (top.get(key) or []) if it][:max_per_cat]
        if items:
            categories.append((key, items))
    total = sum(len(items) for _, items in categories) or 1
    max_imp = max((it.get("importance", 0.0) for _, items in categories for it in items),
                  default=1.0)
    max_imp = max(max_imp, 1e-6)

    ax.scatter([0], [0], s=1200, c=_TARGET_COLOR, zorder=3)
    case_label = str(phase4.get("case_id", ""))[:50]
    pred = str(phase4.get("predicted_label", "?")).strip()
    conf = phase4.get("confidence", 0.0)
    ax.text(0, -0.17, f"TARGET\npred={pred} ({conf:.2f})",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(0, 0.17, case_label, ha="center", va="bottom", fontsize=9, style="italic")

    angle_step = 2 * np.pi / total
    k = 0; handles: dict[str, bool] = {}
    for cat_name, items in categories:
        color = _CATEGORY_COLORS[cat_name]
        for it in items:
            theta = k * angle_step - np.pi / 2
            x, y = np.cos(theta), np.sin(theta)
            imp = it.get("importance", 0.0)
            lw = 0.8 + 5.0 * (imp / max_imp)
            ax.plot([0, x], [0, y], color=color, linewidth=lw, alpha=0.75, zorder=1)
            ax.scatter([x], [y], s=160, c=color, zorder=2, edgecolors="white")
            text = str(it.get("text", ""))[:42]
            ha = "left" if x >= 0 else "right"
            x_text = x + 0.1 * (1 if x >= 0 else -1)
            ax.text(x_text, y, f"{text}\n(imp {imp:.2f})",
                    ha=ha, va="center", fontsize=8, color="#222222")
            handles[cat_name] = True; k += 1

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=_CATEGORY_COLORS[c], label=c) for c in handles]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("PGExplainer top nodes (edge width ∝ importance)", fontsize=12)


def save_subgraph_png(paths: Paths, case_idx: int, out_dir: Path) -> Path:
    phase4 = load_phase4(paths, case_idx)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_subgraph(ax, phase4)
    fp = out_dir / "panel_b_subgraph.png"
    fig.tight_layout(); fig.savefig(fp, dpi=180, bbox_inches="tight"); plt.close(fig)
    return fp


# ---------- case-level summary ----------

def save_case_summary(paths: Paths, case_idx: int, out_dir: Path) -> Path:
    phase4 = load_phase4(paths, case_idx)
    row = _row_for(paths, case_idx)
    target = str(phase4.get("target_label", "?")).strip()
    pred = str(phase4.get("predicted_label", "?")).strip()
    verdict = "CORRECT" if target == pred else "WRONG"
    conf = phase4.get("confidence", 0.0)
    lines = [
        f"# Case {case_idx} — {verdict}",
        "",
        f"- **Case id:** {phase4.get('case_id', '')}",
        f"- **Split:** {row['split']}",
        f"- **True label:** {target}",
        f"- **Predicted label:** {pred}",
        f"- **Confidence:** {conf:.4f}",
        f"- **Class probabilities:** {phase4.get('class_probabilities', {})}",
        "",
        "## Artefacts in this folder",
        "- `panel_a_tsne.png` — t-SNE of post-GNN embeddings with this case highlighted",
        "- `panel_b_subgraph.png` — PGExplainer top nodes around the target",
    ]
    fp = out_dir / "README.md"
    fp.write_text("\n".join(lines) + "\n")
    return fp


def render_case(paths: Paths, case_idx: int, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "tsne": save_tsne_png(paths, case_idx, out_dir),
        "subgraph": save_subgraph_png(paths, case_idx, out_dir),
        "summary": save_case_summary(paths, case_idx, out_dir),
    }
