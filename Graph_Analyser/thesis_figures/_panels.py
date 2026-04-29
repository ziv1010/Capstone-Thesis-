"""Per-panel renderers.

Each `save_*` function writes ONE artefact for ONE case into `out_dir`.
PNGs for the visual panels; Markdown for the text-heavy panels (neighbours,
LLM explanation) so they can be dropped straight into a thesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from _shared import Paths, load_phase4, load_phase5, load_predictions

_ALLOWED_COLOR = "#1f77b4"
_DISMISSED_COLOR = "#d62728"
_TARGET_COLOR = "#111111"

_CATEGORY_COLORS = {
    "statute": "#2ca02c",
    "provision": "#9467bd",
    "precedent": "#ff7f0e",
    "similar_argument": "#17becf",
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
    sims = phase4.get("similar_case_argument_context", [])[:max_per_cat]
    if sims:
        categories.append(("similar_argument", [
            {"text": s.get("owning_case", "")[:60],
             "importance": s.get("importance", 0.0)} for s in sims
        ]))

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


# ---------- (c) FAISS neighbours as Markdown ----------

def save_neighbours_md(paths: Paths, case_idx: int, out_dir: Path,
                       max_k: int = 5) -> Path:
    phase4 = load_phase4(paths, case_idx)
    neigh = (phase4.get("similar_training_cases") or [])[:max_k]
    target_case_id = str(phase4.get("case_id", ""))
    target_label = str(phase4.get("target_label", "?")).strip()
    pred_label = str(phase4.get("predicted_label", "?")).strip()
    conf = phase4.get("confidence", 0.0)

    lines = [
        f"# Top FAISS neighbours — case {case_idx}",
        "",
        f"**Target case:** {target_case_id}  ",
        f"**True / predicted label:** {target_label} / {pred_label}  ",
        f"**Confidence:** {conf:.4f}",
        "",
        "Retrieved from the post-GNN case embedding space (cosine similarity, "
        "FAISS index built over the train split).",
        "",
        "| Rank | Similarity | True | Pred | Match | Case |",
        "| ---: | ---------: | :--: | :--: | :---: | :--- |",
    ]
    if not neigh:
        lines.append("| — | — | — | — | — | _no neighbours recorded_ |")
    for n in neigh:
        rank = n.get("rank", "?")
        cid = str(n.get("case_id", "")).replace("|", "/")
        sim = n.get("similarity", 0.0)
        tgt = str(n.get("target_label", "?")).strip()
        prd = str(n.get("predicted_label", "?")).strip()
        mark = "✓" if tgt == prd else "✗"
        lines.append(f"| {rank} | {sim:.4f} | {tgt} | {prd} | {mark} | {cid} |")
    fp = out_dir / "panel_c_neighbours.md"
    fp.write_text("\n".join(lines) + "\n")
    return fp


# ---------- (d) LLM explanation as Markdown ----------

def save_llm_md(paths: Paths, case_idx: int, out_dir: Path) -> Path:
    phase5 = load_phase5(paths, case_idx)
    fp = out_dir / "panel_d_llm.md"
    if phase5 is None:
        fp.write_text(
            f"# LLM explanation — case {case_idx}\n\n"
            "_Phase 5 explanation not available for this case._\n"
        )
        return fp
    expl = phase5.get("explanation", "")
    case_id = phase5.get("case_id", "")
    target = str(phase5.get("target_label", "?")).strip()
    pred = str(phase5.get("predicted_label", "?")).strip()
    conf = phase5.get("confidence", 0.0)
    fp.write_text(
        f"# LLM explanation — case {case_idx}\n\n"
        f"**Case:** {case_id}  \n"
        f"**True / predicted label:** {target} / {pred}  \n"
        f"**Confidence:** {conf:.4f}\n\n"
        f"---\n\n{expl}\n"
    )
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
        "- `panel_c_neighbours.md` — top FAISS neighbours in post-GNN space",
        "- `panel_d_llm.md` — Mistral-Small 24B explanation (Phase 5)",
    ]
    fp = out_dir / "README.md"
    fp.write_text("\n".join(lines) + "\n")
    return fp


def render_case(paths: Paths, case_idx: int, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "tsne": save_tsne_png(paths, case_idx, out_dir),
        "subgraph": save_subgraph_png(paths, case_idx, out_dir),
        "neighbours": save_neighbours_md(paths, case_idx, out_dir),
        "llm": save_llm_md(paths, case_idx, out_dir),
        "summary": save_case_summary(paths, case_idx, out_dir),
    }
