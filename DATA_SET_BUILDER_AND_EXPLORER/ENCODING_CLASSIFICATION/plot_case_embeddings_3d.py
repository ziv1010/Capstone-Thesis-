#!/usr/bin/env python3
"""
Create 3D embedding plots from a completed clustering run.

This script reads:
  - cases_with_clusters.csv
  - case_embeddings.npy

It projects the embeddings to 3D with PCA and writes multi-angle plots colored
by original bucket and by discovered cluster.
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "figures_3d"
DEFAULT_VIEWS = {
    "isometric": (28, 38),
    "top": (88, -90),
    "side": (10, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 3D PCA plots from saved case embeddings."
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
        help="Directory where 3D figures and projection CSV will be written.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional stratified sample size for plotting if the full set is too heavy.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Matplotlib scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Point opacity for the scatter plot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for optional sampling and PCA.",
    )
    return parser.parse_args()


def stratified_sample_indices(
    labels: Sequence[str],
    max_items: int | None,
    random_state: int,
) -> np.ndarray:
    labels_array = np.asarray(labels)
    n_items = labels_array.shape[0]
    if max_items is None or max_items <= 0 or n_items <= max_items:
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


def project_embeddings_3d(
    embeddings: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, list[float]]:
    pca = PCA(n_components=3, random_state=random_state)
    coords = pca.fit_transform(embeddings)
    explained = [float(value) for value in pca.explained_variance_ratio_]
    return coords.astype(np.float32), explained


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
            frame.loc[mask, "projection_3d_x"],
            frame.loc[mask, "projection_3d_y"],
            frame.loc[mask, "projection_3d_z"],
            s=point_size,
            alpha=alpha,
            label=label,
            color=cmap(idx % cmap.N),
            depthshade=False,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
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
        labels=cases_df["bucket"].tolist(),
        max_items=args.max_points,
        random_state=args.random_state,
    )
    cases_df = cases_df.iloc[sample_indices].reset_index(drop=True)
    embeddings = embeddings[sample_indices]

    coords, explained = project_embeddings_3d(
        embeddings=embeddings,
        random_state=args.random_state,
    )
    cases_df["projection_3d_x"] = coords[:, 0]
    cases_df["projection_3d_y"] = coords[:, 1]
    cases_df["projection_3d_z"] = coords[:, 2]

    projection_path = args.output_dir / "case_projection_3d.csv"
    cases_df.to_csv(projection_path, index=False)

    for view_name, (elev, azim) in DEFAULT_VIEWS.items():
        plot_3d_scatter(
            frame=cases_df,
            label_column="bucket",
            title=f"3D PCA Projection by Bucket ({view_name})",
            output_path=args.output_dir / f"scatter3d_by_bucket_{view_name}.png",
            point_size=args.point_size,
            alpha=args.alpha,
            elev=elev,
            azim=azim,
        )
        plot_3d_scatter(
            frame=cases_df,
            label_column="cluster",
            title=f"3D PCA Projection by Cluster ({view_name})",
            output_path=args.output_dir / f"scatter3d_by_cluster_{view_name}.png",
            point_size=args.point_size,
            alpha=args.alpha,
            elev=elev,
            azim=azim,
        )

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "points_plotted": int(len(cases_df)),
        "explained_variance_ratio": explained,
        "views": {
            view_name: {"elev": elev, "azim": azim}
            for view_name, (elev, azim) in DEFAULT_VIEWS.items()
        },
    }
    (args.output_dir / "plot_3d_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote 3D plots to: {args.output_dir}")
    print(f"Points plotted: {len(cases_df):,}")
    print(
        "Explained variance: "
        + ", ".join(f"PC{i + 1}={value:.4f}" for i, value in enumerate(explained))
    )


if __name__ == "__main__":
    main()
