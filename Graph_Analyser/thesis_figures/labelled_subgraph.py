#!/usr/bin/env python3
"""Render PGExplainer subgraph panels with cyan-ring nodes labelled by the
TRUE outcome (W = winning / L = losing) of the owning case, plus a global
scan over all phase4 bundles to quantify the W/L imbalance in the cyan ring.

Usage
-----
  micromamba run -n graph_explainer python thesis_figures/labelled_subgraph.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Graph_Analyser")
OUTPUTS = REPO_ROOT / "outputs"
CASES_DIR = OUTPUTS / "phase4_explanations" / "cases"
OUT_DIR = OUTPUTS / "thesis_figures" / "labelled_pair"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_COLORS = {
    "statute": "#2ca02c",
    "provision": "#9467bd",
    "precedent": "#ff7f0e",
}
WIN_COLOR = "#1f77b4"   # blue   (winning, label=1)
LOSE_COLOR = "#d62728"  # red    (losing,  label=-1)
UNK_COLOR = "#7f7f7f"   # grey
TARGET_COLOR = "#111111"


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "phase1_2_inference" / "predictions.csv")


def label_lookup(preds: pd.DataFrame, owning: str) -> str | None:
    if not owning:
        return None
    by_id = preds.set_index("case_id")
    by_file = preds.set_index("file_name")
    if owning in by_id.index:
        row = by_id.loc[owning]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return str(row["target_label"])
    if owning in by_file.index:
        row = by_file.loc[owning]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return str(row["target_label"])
    hits = preds[preds["case_id"].str.contains(owning, regex=False, na=False)]
    if len(hits):
        return str(hits.iloc[0]["target_label"])
    return None


def render_labelled_subgraph(case_idx: int, preds: pd.DataFrame, max_per_cat: int = 3) -> Path:
    bundle = json.loads((CASES_DIR / f"case_{case_idx}.json").read_text())
    case_id = str(bundle.get("case_id", ""))[:55]
    target_lbl = str(bundle.get("target_label", "?")).strip()
    pred_lbl = str(bundle.get("predicted_label", "?")).strip()
    conf = float(bundle.get("confidence", 0.0))
    verdict = "CORRECT" if target_lbl == pred_lbl else "WRONG"

    categories: list[tuple[str, list[dict]]] = []
    top = bundle.get("top_nodes", {}) or {}
    for key in ("statute", "provision", "precedent"):
        items = [it for it in (top.get(key) or []) if it][:max_per_cat]
        if items:
            categories.append((key, items))

    sims = bundle.get("similar_case_argument_context", [])[:max_per_cat]
    sim_items: list[dict] = []
    for s in sims:
        owning = s.get("owning_case", "")
        true_lbl = label_lookup(preds, owning)
        if true_lbl == "1":
            tag, color, tag_full = "W", WIN_COLOR, "WINNING"
        elif true_lbl == "-1":
            tag, color, tag_full = "L", LOSE_COLOR, "LOSING"
        else:
            tag, color, tag_full = "?", UNK_COLOR, "UNKNOWN"
        sim_items.append({
            "text": owning[:55],
            "importance": float(s.get("importance", 0.0)),
            "tag": tag,
            "tag_full": tag_full,
            "color": color,
        })
    if sim_items:
        categories.append(("similar_argument", sim_items))

    total = sum(len(items) for _, items in categories) or 1
    max_imp = max(
        (it.get("importance", 0.0) for _, items in categories for it in items),
        default=1.0,
    )
    max_imp = max(max_imp, 1e-6)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter([0], [0], s=1400, c=TARGET_COLOR, zorder=3)
    ax.text(0, -0.18, f"TARGET\ntrue={target_lbl}  pred={pred_lbl} ({conf:.2f})\n{verdict}",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(0, 0.18, case_id, ha="center", va="bottom", fontsize=9, style="italic")

    angle_step = 2 * np.pi / total
    k = 0
    seen_cats: dict[str, bool] = {}
    for cat_name, items in categories:
        for it in items:
            theta = k * angle_step - np.pi / 2
            x, y = np.cos(theta), np.sin(theta)
            imp = float(it.get("importance", 0.0))
            lw = 0.8 + 5.0 * (imp / max_imp)

            if cat_name == "similar_argument":
                color = it["color"]
                tag = it["tag"]
            else:
                color = CATEGORY_COLORS[cat_name]
                tag = None

            ax.plot([0, x], [0, y], color=color, linewidth=lw, alpha=0.8, zorder=1)
            ax.scatter([x], [y], s=200, c=color, zorder=2, edgecolors="white", linewidths=1.2)

            if tag is not None:
                ax.text(x, y, tag, ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold", zorder=4)

            text = str(it.get("text", ""))[:45]
            label = f"{text}\n(imp {imp:.2f})"
            if tag is not None:
                label = f"{text}\n(imp {imp:.2f} · {it['tag_full']})"
            ha = "left" if x >= 0 else "right"
            x_text = x + 0.12 * (1 if x >= 0 else -1)
            ax.text(x_text, y, label, ha=ha, va="center", fontsize=8, color="#222222")
            seen_cats[cat_name] = True
            k += 1

    legend_handles = []
    for c in seen_cats:
        if c == "similar_argument":
            continue
        legend_handles.append(mpatches.Patch(color=CATEGORY_COLORS[c], label=c))
    if "similar_argument" in seen_cats:
        legend_handles.append(mpatches.Patch(color=WIN_COLOR, label="similar_arg · WINNING (1)"))
        legend_handles.append(mpatches.Patch(color=LOSE_COLOR, label="similar_arg · LOSING (−1)"))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"PGExplainer top nodes — case {case_idx} ({verdict}); "
        f"cyan-ring labelled by TRUE outcome of owning case",
        fontsize=11,
    )

    fp = OUT_DIR / f"panel_b_subgraph_case_{case_idx}_labelled.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fp


def cross_test_scan(preds: pd.DataFrame) -> dict:
    """Per-case cyan-ring W/L tally across every phase4 bundle, segmented by
    target true label and prediction correctness."""
    rows: list[dict] = []
    for f in sorted(CASES_DIR.glob("case_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        sims = d.get("similar_case_argument_context", []) or []
        if not sims:
            continue
        win = lose = unk = 0
        for s in sims:
            lbl = label_lookup(preds, s.get("owning_case", ""))
            if lbl == "1": win += 1
            elif lbl == "-1": lose += 1
            else: unk += 1
        rows.append({
            "case_idx": d["case_node_index"],
            "target_true": str(d.get("target_label", "?")).strip(),
            "predicted":   str(d.get("predicted_label", "?")).strip(),
            "n_sims": len(sims),
            "win": win, "lose": lose, "unk": unk,
        })
    df = pd.DataFrame(rows)
    df["correct"] = df["target_true"] == df["predicted"]

    def _agg(sub):
        n = len(sub)
        if n == 0:
            return {"n_cases": 0}
        total_sims = sub["n_sims"].sum()
        return {
            "n_cases":       int(n),
            "total_sims":    int(total_sims),
            "win_share":     float(sub["win"].sum() / total_sims) if total_sims else 0.0,
            "lose_share":    float(sub["lose"].sum() / total_sims) if total_sims else 0.0,
            "unk_share":     float(sub["unk"].sum() / total_sims) if total_sims else 0.0,
            "all_winning_cases": int((sub["win"] == sub["n_sims"]).sum()),
            "all_losing_cases":  int((sub["lose"] == sub["n_sims"]).sum()),
        }

    summary = {
        "all_explained":              _agg(df),
        "target_winning_correct":     _agg(df[(df["target_true"] == "1")  & (df["correct"])]),
        "target_winning_wrong":       _agg(df[(df["target_true"] == "1")  & (~df["correct"])]),
        "target_losing_correct":      _agg(df[(df["target_true"] == "-1") & (df["correct"])]),
        "target_losing_wrong":        _agg(df[(df["target_true"] == "-1") & (~df["correct"])]),
    }
    return {"summary": summary, "df_head": df.head(8).to_dict(orient="records")}


def _load_or_compute_pca() -> tuple[np.ndarray, np.ndarray]:
    """PCA(2) over the FULL post-GNN case embedding set (all 71813 cases).
    Fast, deterministic, lets us project both train and test points so any
    cyan-ring owning case is visible regardless of split. Cached on disk."""
    cache_fp = OUT_DIR / "pca_full_2d.npz"
    if cache_fp.exists():
        z = np.load(cache_fp)
        return z["points"], z["node_indices"]
    from sklearn.decomposition import PCA
    embs = np.load(OUTPUTS / "phase1_2_inference" / "case_embeddings.npy")
    points = PCA(n_components=2, random_state=0).fit_transform(embs)
    node_indices = np.arange(embs.shape[0], dtype=np.int64)
    np.savez(cache_fp, points=points, node_indices=node_indices)
    return points, node_indices


def render_embedding_panel(case_idx: int, preds: pd.DataFrame) -> Path:
    """Embedding-space panel: 2-D PCA of post-GNN case embeddings (full graph,
    train+val+test). Target highlighted, cyan-ring owning cases overlaid and
    tagged with W/L. Backdrop is sub-sampled for visual clarity."""
    points, node_indices = _load_or_compute_pca()
    idx_to_pos = {int(n): i for i, n in enumerate(node_indices.tolist())}

    preds_by_idx = preds.set_index("node_index")
    pred_labels = preds_by_idx.loc[node_indices, "pred_label"].to_numpy()
    is_winning_pred = np.array(
        [str(p).strip() in ("1", "winning", "Winning") for p in pred_labels]
    )

    bundle = json.loads((CASES_DIR / f"case_{case_idx}.json").read_text())
    target_lbl = str(bundle.get("target_label", "?")).strip()
    pred_lbl   = str(bundle.get("predicted_label", "?")).strip()
    conf       = float(bundle.get("confidence", 0.0))
    verdict = "CORRECT" if target_lbl == pred_lbl else "WRONG"

    by_id = preds.set_index("case_id")
    by_file = preds.set_index("file_name")

    def resolve_node_index(owning: str):
        if owning in by_id.index:
            row = by_id.loc[owning]; row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            return int(row["node_index"]), str(row["split"]), str(row["target_label"])
        if owning in by_file.index:
            row = by_file.loc[owning]; row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            return int(row["node_index"]), str(row["split"]), str(row["target_label"])
        hits = preds[preds["case_id"].str.contains(owning, regex=False, na=False)]
        if len(hits):
            row = hits.iloc[0]
            return int(row["node_index"]), str(row["split"]), str(row["target_label"])
        return None, None, None

    sims = bundle.get("similar_case_argument_context", [])
    overlay = []
    for s in sims:
        owning = s.get("owning_case", "")
        nidx, split, true_lbl = resolve_node_index(owning)
        overlay.append({"owning": owning, "node_index": nidx, "split": split,
                        "true_lbl": true_lbl, "imp": float(s.get("importance", 0.0))})

    fig, ax = plt.subplots(figsize=(9, 7.5))
    rng = np.random.default_rng(0)
    n_back = points.shape[0]
    sample = rng.choice(n_back, size=min(15000, n_back), replace=False)
    sub_points = points[sample]
    sub_winning = is_winning_pred[sample]
    ax.scatter(sub_points[~sub_winning, 0], sub_points[~sub_winning, 1], s=6,
               c=LOSE_COLOR, alpha=0.18, linewidths=0, label="pred = LOSING (−1)")
    ax.scatter(sub_points[sub_winning, 0], sub_points[sub_winning, 1], s=6,
               c=WIN_COLOR, alpha=0.18, linewidths=0, label="pred = WINNING (1)")

    if case_idx in idx_to_pos:
        j = idx_to_pos[case_idx]
        ax.scatter(points[j, 0], points[j, 1], s=420, marker="*",
                   c=TARGET_COLOR, edgecolors="white", linewidths=1.6, zorder=6,
                   label=f"TARGET (node {case_idx})")
        target_xy = points[j]
    else:
        target_xy = None

    off_plot: list[dict] = []
    for o in overlay:
        nidx = o["node_index"]
        if nidx is None or nidx not in idx_to_pos:
            off_plot.append(o)
            continue
        j = idx_to_pos[nidx]
        x, y = points[j]
        if o["true_lbl"] == "1":
            color, tag = WIN_COLOR, "W"
        elif o["true_lbl"] == "-1":
            color, tag = LOSE_COLOR, "L"
        else:
            color, tag = UNK_COLOR, "?"
        ax.scatter([x], [y], s=240, marker="D", c=color,
                   edgecolors="black", linewidths=1.4, zorder=5)
        ax.text(x, y, tag, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=7)
        if target_xy is not None:
            ax.plot([target_xy[0], x], [target_xy[1], y],
                    color="#444444", linewidth=0.7, linestyle="--", alpha=0.7, zorder=2)

    ax.set_title(
        f"Post-GNN case embedding space (PCA-2D, full graph; "
        f"backdrop sub-sampled to 15k for clarity)\n"
        f"Case {case_idx} — true={target_lbl} pred={pred_lbl} conf={conf:.2f} → {verdict}",
        fontsize=10,
    )
    ax.set_xticks([]); ax.set_yticks([])
    extra = [
        mpatches.Patch(color=WIN_COLOR, label="cyan-ring owner · WINNING (1)"),
        mpatches.Patch(color=LOSE_COLOR, label="cyan-ring owner · LOSING (−1)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, fontsize=8, loc="lower right",
              framealpha=0.92)

    fp = OUT_DIR / f"panel_a_embedding_case_{case_idx}_labelled.png"
    fig.tight_layout()
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)

    caption = OUT_DIR / f"panel_a_embedding_case_{case_idx}_caption.md"
    lines = [
        f"# Embedding-space panel — case {case_idx}",
        "",
        f"Backdrop: post-GNN case embeddings (n={len(node_indices)}, full graph), "
        f"projected with PCA(2). Backdrop sub-sampled to 15k points for visual "
        f"clarity, coloured by the model's prediction. Target case shown as ★. "
        f"Cyan-ring owning cases (from PGExplainer's similar_case_argument_context) "
        f"overlaid as W/L diamonds, connected to the target by dashed lines.",
        "",
        "## Cyan-ring owning cases for this target",
        "",
        "| Owning case | Split | True label | In t-SNE? | Importance |",
        "| --- | :---: | :---: | :---: | ---: |",
    ]
    for o in overlay:
        in_plot = "✓" if (o["node_index"] is not None and o["node_index"] in idx_to_pos) else "—"
        true_disp = ("WINNING" if o["true_lbl"] == "1"
                     else "LOSING" if o["true_lbl"] == "-1" else "?")
        lines.append(f"| {o['owning'][:60]} | {o['split'] or '?'} | {true_disp} | {in_plot} | {o['imp']:.3f} |")
    if off_plot:
        lines += ["",
                  f"_{len(off_plot)} owning case(s) could not be resolved to a node index; "
                  f"their split / true-label is recorded above._"]
    caption.write_text("\n".join(lines) + "\n")
    return fp


def main() -> None:
    preds = load_predictions()

    print("Rendering labelled subgraphs …")
    for idx in (33259, 31378):
        fp = render_labelled_subgraph(idx, preds)
        print(f"  saved {fp}")

    print("\nRendering embedding-space (t-SNE) panels …")
    for idx in (33259, 31378):
        fp = render_embedding_panel(idx, preds)
        print(f"  saved {fp}")

    print("\nCross-test scan of cyan-ring W/L imbalance …")
    scan = cross_test_scan(preds)

    summary_path = OUT_DIR / "cyan_ring_label_scan.json"
    summary_path.write_text(json.dumps(scan, indent=2))
    print(f"  saved {summary_path}\n")

    print("Cyan-ring outcome share, segmented by target-true label × correctness:\n")
    rows = []
    for k, v in scan["summary"].items():
        if v.get("n_cases", 0) == 0:
            continue
        rows.append({
            "segment": k,
            "n_cases": v["n_cases"],
            "total_sims": v["total_sims"],
            "win%":  f"{100 * v['win_share']:.1f}",
            "lose%": f"{100 * v['lose_share']:.1f}",
            "unk%":  f"{100 * v['unk_share']:.1f}",
            "all_W_ring": v["all_winning_cases"],
            "all_L_ring": v["all_losing_cases"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
