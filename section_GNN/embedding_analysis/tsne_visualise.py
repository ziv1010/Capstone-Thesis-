#!/usr/bin/env python3
"""
t-SNE visualisation of pre-GNN and post-GNN case node embeddings.

Produces four PNG plots saved alongside the input .npz file:
  <run_name>_tsne_post_by_label.png   — post-GNN, coloured by outcome
  <run_name>_tsne_post_by_domain.png  — post-GNN, coloured by domain
  <run_name>_tsne_pre_by_label.png    — pre-GNN (raw features), coloured by outcome
  <run_name>_tsne_pre_by_domain.png   — pre-GNN, coloured by domain

Usage
-----
  python tsne_visualise.py --embeddings <path/to/run_name_embeddings.npz>

Optional flags:
  --perplexity   t-SNE perplexity (default 30)
  --n-iter       t-SNE iterations (default 1000)
  --sample       Max cases to include (default 5000, subsampled if larger)
  --output-dir   Where to write PNGs (defaults to same dir as .npz)
  --split        Restrict to "train"/"test"/"all" (default "all")
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="t-SNE plots of GNN case embeddings.")
    p.add_argument("--embeddings", required=True, help="Path to *_embeddings.npz from extract_embeddings.py")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n-iter",     type=int,   default=1000)
    p.add_argument("--sample",     type=int,   default=5000,  help="Max cases (random subset if larger)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split",      default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

_LABEL_COLORS  = {"-1": "#d62728", "1": "#1f77b4", "0": "#ff7f0e"}   # red=dismissed blue=allowed
_LABEL_NAMES   = {"-1": "Dismissed (−1)", "1": "Allowed (+1)"}
_DOMAIN_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _scatter(
    coords: np.ndarray,
    color_keys: np.ndarray,
    palette: dict[str, str],
    label_map: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    unique_keys = sorted(set(color_keys))
    for key in unique_keys:
        mask = color_keys == key
        color = palette.get(str(key), "#333333")
        label = label_map.get(str(key), str(key))
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=label, s=6, alpha=0.55, linewidths=0,
        )
    ax.legend(markerscale=3, fontsize=10, framealpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[tsne] saved → {out_path.name}")


def _run_tsne(
    embeddings: np.ndarray,
    perplexity: float,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    import sklearn
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    iter_kwarg = "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2) else "n_iter"
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        **{iter_kwarg: n_iter},
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = tsne.fit_transform(X)
    return coords


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    npz_path = Path(args.embeddings)
    out_dir  = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = npz_path.stem.replace("_embeddings", "")

    data = np.load(npz_path, allow_pickle=True)
    case_ids    = data["case_ids"]
    domains     = data["domains"]
    labels      = data["labels"].astype(str)
    splits      = data["splits"]
    pre_gnn     = data["pre_gnn"]
    post_gnn    = data["post_gnn"]
    label_names = data.get("label_names", np.array(["-1", "1"]))

    print(f"[tsne] loaded {len(case_ids)} cases | pre_gnn={pre_gnn.shape} post_gnn={post_gnn.shape}")

    # ── Filter by split ───────────────────────────────────────
    if args.split != "all":
        keep = splits == args.split
        print(f"[tsne] restricting to split='{args.split}': {keep.sum()} / {len(keep)} cases")
        case_ids  = case_ids[keep]
        domains   = domains[keep]
        labels    = labels[keep]
        splits    = splits[keep]
        pre_gnn   = pre_gnn[keep]
        post_gnn  = post_gnn[keep]

    n = len(case_ids)

    # ── Subsample if too large ─────────────────────────────────
    if n > args.sample:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.sample, replace=False)
        idx.sort()
        print(f"[tsne] subsampling {args.sample} / {n} cases")
        domains  = domains[idx]
        labels   = labels[idx]
        pre_gnn  = pre_gnn[idx]
        post_gnn = post_gnn[idx]
        n        = args.sample

    # ── Build domain palette ───────────────────────────────────
    unique_domains = sorted(set(domains))
    domain_palette = {d: _DOMAIN_PALETTE[i % len(_DOMAIN_PALETTE)] for i, d in enumerate(unique_domains)}
    domain_label_map = {d: d.replace("_timed_mistral", "").replace("_", " ") for d in unique_domains}

    suffix = f"_split{args.split}" if args.split != "all" else ""

    # ── Post-GNN t-SNE ────────────────────────────────────────
    print(f"[tsne] running t-SNE on post-GNN ({n} × {post_gnn.shape[1]}) ...")
    post_coords = _run_tsne(post_gnn, args.perplexity, args.n_iter, args.seed)

    _scatter(
        coords     = post_coords,
        color_keys = labels,
        palette    = _LABEL_COLORS,
        label_map  = _LABEL_NAMES,
        title      = f"Post-GNN embeddings — by outcome\n({run_name})",
        out_path   = out_dir / f"{run_name}_tsne_post_by_label{suffix}.png",
    )
    _scatter(
        coords     = post_coords,
        color_keys = domains,
        palette    = domain_palette,
        label_map  = domain_label_map,
        title      = f"Post-GNN embeddings — by domain\n({run_name})",
        out_path   = out_dir / f"{run_name}_tsne_post_by_domain{suffix}.png",
    )

    # ── Pre-GNN t-SNE ─────────────────────────────────────────
    print(f"[tsne] running t-SNE on pre-GNN ({n} × {pre_gnn.shape[1]}) ...")
    pre_coords = _run_tsne(pre_gnn, args.perplexity, args.n_iter, args.seed)

    _scatter(
        coords     = pre_coords,
        color_keys = labels,
        palette    = _LABEL_COLORS,
        label_map  = _LABEL_NAMES,
        title      = f"Pre-GNN (raw features) — by outcome\n({run_name})",
        out_path   = out_dir / f"{run_name}_tsne_pre_by_label{suffix}.png",
    )
    _scatter(
        coords     = pre_coords,
        color_keys = domains,
        palette    = domain_palette,
        label_map  = domain_label_map,
        title      = f"Pre-GNN (raw features) — by domain\n({run_name})",
        out_path   = out_dir / f"{run_name}_tsne_pre_by_domain{suffix}.png",
    )

    print("[tsne] done.")


if __name__ == "__main__":
    main()
