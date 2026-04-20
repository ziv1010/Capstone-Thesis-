#!/usr/bin/env python3
"""
Create sampled t-SNE visualizations from a completed clustering run.

This script is meant for visual inspection, not for re-clustering. It samples
the saved embedding matrix, reduces it first with PCA, then runs t-SNE in 2D
and optionally 3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "figures_tsne"
DEFAULT_VIEWS = {
    "isometric": (28, 38),
    "top": (88, -90),
    "side": (10, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sampled t-SNE plots from saved case embeddings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing cases_with_clusters.csv and case_embeddings.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where t-SNE figures and projection CSV will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8000,
        help="Number of points to sample before running t-SNE.",
    )
    parser.add_argument(
        "--stratify-by",
        choices=["bucket", "cluster", "bucket_cluster"],
        default="bucket_cluster",
        help="Sampling stratum used to preserve structure in the t-SNE sample.",
    )
    parser.add_argument(
        "--pre-reduce-dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=35.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--learning-rate",
        default="auto",
        help="t-SNE learning rate. Use a float or 'auto'.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1200,
        help="t-SNE max iterations.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Projection dimensions to generate. Typical values: 2 3",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=7.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Point opacity.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and t-SNE.",
    )
    return parser.parse_args()


def stratified_sample_indices(
    labels: Sequence[str],
    max_items: int,
    random_state: int,
) -> np.ndarray:
    labels_array = np.asarray(labels)
    n_items = labels_array.shape[0]
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=int)

    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    unique_labels, counts = np.unique(labels_array, return_counts=True)

    allocations: dict[str, int] = {}
    total = 0
    for label, count in zip(unique_labels, counts):
        take = max(1, int(round(max_items * (int(count) / n_items))))
        take = min(take, int(count))
        allocations[str(label)] = take
        total += take

    if total > max_items:
        while total > max_items:
            largest = max(allocations, key=allocations.get)
            if allocations[largest] <= 1:
                break
            allocations[largest] -= 1
            total -= 1
    elif total < max_items:
        remaining = max_items - total
        ordered = sorted(
            ((str(label), int(count)) for label, count in zip(unique_labels, counts)),
            key=lambda item: item[1],
            reverse=True,
        )
        idx = 0
        while remaining > 0 and ordered:
            label, count = ordered[idx % len(ordered)]
            if allocations[label] < count:
                allocations[label] += 1
                remaining -= 1
            idx += 1

    for label in unique_labels:
        label_indices = np.where(labels_array == label)[0]
        take = allocations[str(label)]
        chosen = rng.choice(label_indices, size=take, replace=False)
        selected.extend(int(value) for value in chosen)

    selected = sorted(set(selected))
    if len(selected) < max_items:
        chosen_set = set(selected)
        leftover = [idx for idx in range(n_items) if idx not in chosen_set]
        extra_take = min(max_items - len(selected), len(leftover))
        if extra_take > 0:
            extra = rng.choice(leftover, size=extra_take, replace=False)
            selected.extend(int(value) for value in extra)
            selected = sorted(selected)

    return np.asarray(selected[:max_items], dtype=int)


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cases_path = input_dir / "cases_with_clusters.csv"
    embeddings_path = input_dir / "case_embeddings.npy"
    if not cases_path.exists():
        raise FileNotFoundError(f"Missing file: {cases_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing file: {embeddings_path}")

    cases_df = pd.read_csv(cases_path)
    embeddings = np.load(embeddings_path)
    if len(cases_df) != embeddings.shape[0]:
        raise ValueError("cases_with_clusters.csv and case_embeddings.npy have different lengths.")
    return cases_df, embeddings


def build_sampling_labels(frame: pd.DataFrame, mode: str) -> list[str]:
    if mode == "bucket":
        return frame["bucket"].astype(str).tolist()
    if mode == "cluster":
        return frame["cluster"].astype(str).tolist()
    return (
        frame["bucket"].astype(str)
        + "::"
        + frame["cluster"].astype(str)
    ).tolist()


def plot_2d_scatter(
    frame: pd.DataFrame,
    label_column: str,
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
) -> None:
    labels = frame[label_column].astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "gist_ncar")

    plt.figure(figsize=(12, 9))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            frame.loc[mask, "tsne_x"],
            frame.loc[mask, "tsne_y"],
            s=point_size,
            alpha=alpha,
            label=label,
            color=cmap(idx % cmap.N),
            rasterized=True,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if len(unique_labels) <= 20:
        plt.legend(loc="best", fontsize=8, markerscale=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_3d_scatter(
    frame: pd.DataFrame,
    label_column: str,
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    elev: float,
    azim: float,
) -> None:
    labels = frame[label_column].astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "gist_ncar")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            frame.loc[mask, "tsne_x"],
            frame.loc[mask, "tsne_y"],
            frame.loc[mask, "tsne_z"],
            s=point_size,
            alpha=alpha,
            label=label,
            color=cmap(idx % cmap.N),
            depthshade=False,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.view_init(elev=elev, azim=azim)
    if len(unique_labels) <= 20:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_df, embeddings = load_inputs(args.input_dir)
    sample_indices = stratified_sample_indices(
        labels=build_sampling_labels(cases_df, args.stratify_by),
        max_items=args.sample_size,
        random_state=args.random_state,
    )
    cases_df = cases_df.iloc[sample_indices].reset_index(drop=True)
    embeddings = embeddings[sample_indices]

    pca_dim = min(args.pre_reduce_dim, embeddings.shape[1], embeddings.shape[0] - 1)
    if pca_dim < 2:
        raise ValueError("Need at least 2 sampled points to run t-SNE.")
    reduced = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(embeddings)
    effective_perplexity = max(1.0, min(args.perplexity, float(len(cases_df) - 1)))

    learning_rate = args.learning_rate
    if learning_rate != "auto":
        learning_rate = float(learning_rate)

    report: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "sample_size": int(len(cases_df)),
        "stratify_by": args.stratify_by,
        "pre_reduce_dim": int(pca_dim),
        "perplexity": effective_perplexity,
        "learning_rate": learning_rate,
        "iterations": args.iterations,
        "dimensions": list(args.dimensions),
    }

    if 2 in args.dimensions:
        tsne_2d = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            init="pca",
            random_state=args.random_state,
            max_iter=args.iterations,
        ).fit_transform(reduced)
        frame_2d = cases_df.copy()
        frame_2d["tsne_x"] = tsne_2d[:, 0]
        frame_2d["tsne_y"] = tsne_2d[:, 1]
        frame_2d.to_csv(args.output_dir / "tsne_projection_2d.csv", index=False)
        plot_2d_scatter(
            frame=frame_2d,
            label_column="bucket",
            title="t-SNE 2D Projection by Bucket",
            output_path=args.output_dir / "tsne_2d_by_bucket.png",
            point_size=args.point_size,
            alpha=args.alpha,
        )
        plot_2d_scatter(
            frame=frame_2d,
            label_column="cluster",
            title="t-SNE 2D Projection by Cluster",
            output_path=args.output_dir / "tsne_2d_by_cluster.png",
            point_size=args.point_size,
            alpha=args.alpha,
        )

    if 3 in args.dimensions:
        tsne_3d = TSNE(
            n_components=3,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            init="pca",
            random_state=args.random_state,
            max_iter=args.iterations,
        ).fit_transform(reduced)
        frame_3d = cases_df.copy()
        frame_3d["tsne_x"] = tsne_3d[:, 0]
        frame_3d["tsne_y"] = tsne_3d[:, 1]
        frame_3d["tsne_z"] = tsne_3d[:, 2]
        frame_3d.to_csv(args.output_dir / "tsne_projection_3d.csv", index=False)
        for view_name, (elev, azim) in DEFAULT_VIEWS.items():
            plot_3d_scatter(
                frame=frame_3d,
                label_column="bucket",
                title=f"t-SNE 3D Projection by Bucket ({view_name})",
                output_path=args.output_dir / f"tsne_3d_by_bucket_{view_name}.png",
                point_size=args.point_size,
                alpha=args.alpha,
                elev=elev,
                azim=azim,
            )
            plot_3d_scatter(
                frame=frame_3d,
                label_column="cluster",
                title=f"t-SNE 3D Projection by Cluster ({view_name})",
                output_path=args.output_dir / f"tsne_3d_by_cluster_{view_name}.png",
                point_size=args.point_size,
                alpha=args.alpha,
                elev=elev,
                azim=azim,
            )

    (args.output_dir / "plot_tsne_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote t-SNE plots to: {args.output_dir}")
    print(f"Sampled points: {len(cases_df):,}")


if __name__ == "__main__":
    main()
