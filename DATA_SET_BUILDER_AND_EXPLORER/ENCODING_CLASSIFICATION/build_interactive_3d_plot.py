#!/usr/bin/env python3
"""
Build an interactive 3D HTML plot from a completed clustering run.

The output is an HTML file that uses Plotly.js from CDN. It does not require
the Python `plotly` package in the micromamba environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "figures_interactive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive 3D HTML plot from saved case embeddings."
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
        help="Directory where the HTML and projection CSV will be written.",
    )
    parser.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Projection method used for the interactive 3D coordinates.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=25000,
        help=(
            "Maximum number of points rendered in the browser. "
            "Set to 0 to disable sampling."
        ),
    )
    parser.add_argument(
        "--stratify-by",
        choices=["bucket", "cluster", "bucket_cluster"],
        default="bucket_cluster",
        help="Sampling stratum used when --max-points is active.",
    )
    parser.add_argument(
        "--pre-reduce-dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE when --method tsne is used.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=35.0,
        help="t-SNE perplexity when --method tsne is used.",
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
        help="t-SNE max iterations when --method tsne is used.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=4.0,
        help="Marker size used in the interactive plot.",
    )
    parser.add_argument(
        "--marker-opacity",
        type=float,
        default=0.70,
        help="Marker opacity used in the interactive plot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and projection.",
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


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    pre_reduce_dim: int,
    perplexity: float,
    learning_rate: str,
    iterations: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if method == "pca":
        pca = PCA(n_components=3, random_state=random_state)
        coords = pca.fit_transform(embeddings)
        meta = {
            "method": "pca",
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
        }
        return coords.astype(np.float32), meta

    pca_dim = min(pre_reduce_dim, embeddings.shape[1], embeddings.shape[0] - 1)
    if pca_dim < 2:
        raise ValueError("Need at least 2 points to run t-SNE.")
    reduced = PCA(n_components=pca_dim, random_state=random_state).fit_transform(embeddings)
    effective_perplexity = max(1.0, min(perplexity, float(embeddings.shape[0] - 1)))
    effective_learning_rate: str | float = learning_rate
    if effective_learning_rate != "auto":
        effective_learning_rate = float(effective_learning_rate)

    coords = TSNE(
        n_components=3,
        perplexity=effective_perplexity,
        learning_rate=effective_learning_rate,
        init="pca",
        random_state=random_state,
        max_iter=iterations,
    ).fit_transform(reduced)
    meta = {
        "method": "tsne",
        "pre_reduce_dim": int(pca_dim),
        "perplexity": effective_perplexity,
        "learning_rate": effective_learning_rate,
        "iterations": int(iterations),
    }
    return coords.astype(np.float32), meta


def make_trace(
    frame: pd.DataFrame,
    label_column: str,
    label_value: str,
    color: str,
    marker_size: float,
    marker_opacity: float,
    visible: bool,
) -> dict[str, object]:
    subset = frame[frame[label_column].astype(str) == label_value]
    customdata = subset[
        [
            "filename",
            "bucket",
            "cluster",
            "nearest_other_bucket",
            "nearest_other_bucket_similarity",
        ]
    ].fillna("").to_numpy().tolist()

    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": label_value,
        "visible": visible,
        "x": subset["interactive_x"].round(6).tolist(),
        "y": subset["interactive_y"].round(6).tolist(),
        "z": subset["interactive_z"].round(6).tolist(),
        "customdata": customdata,
        "text": subset["filename"].astype(str).tolist(),
        "hovertemplate": (
            "<b>%{customdata[0]}</b><br>"
            "Bucket: %{customdata[1]}<br>"
            "Cluster: %{customdata[2]}<br>"
            "Nearest Other Bucket: %{customdata[3]}<br>"
            "Nearest Similarity: %{customdata[4]}<extra></extra>"
        ),
        "marker": {
            "size": marker_size,
            "opacity": marker_opacity,
            "color": color,
        },
    }


def build_plot_payload(
    frame: pd.DataFrame,
    marker_size: float,
    marker_opacity: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bucket_labels = sorted(frame["bucket"].astype(str).unique().tolist())
    cluster_labels = sorted(frame["cluster"].astype(str).unique().tolist())
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
    ]

    traces: list[dict[str, object]] = []
    for idx, label in enumerate(bucket_labels):
        traces.append(
            make_trace(
                frame=frame,
                label_column="bucket",
                label_value=label,
                color=palette[idx % len(palette)],
                marker_size=marker_size,
                marker_opacity=marker_opacity,
                visible=True,
            )
        )

    for idx, label in enumerate(cluster_labels):
        traces.append(
            make_trace(
                frame=frame,
                label_column="cluster",
                label_value=label,
                color=palette[idx % len(palette)],
                marker_size=marker_size,
                marker_opacity=marker_opacity,
                visible=False,
            )
        )

    bucket_visible = [True] * len(bucket_labels) + [False] * len(cluster_labels)
    cluster_visible = [False] * len(bucket_labels) + [True] * len(cluster_labels)

    layout = {
        "title": {
            "text": (
                "Interactive 3D Case Embedding Projection"
                "<br><sup>Use the dropdown to color by bucket or discovered cluster</sup>"
            )
        },
        "template": "plotly_white",
        "showlegend": True,
        "uirevision": "keep-camera",
        "scene": {
            "xaxis": {"title": "Dim 1"},
            "yaxis": {"title": "Dim 2"},
            "zaxis": {"title": "Dim 3"},
            "camera": {
                "eye": {"x": 1.45, "y": 1.45, "z": 1.1},
            },
        },
        "height": 900,
        "margin": {"l": 0, "r": 0, "t": 80, "b": 0},
        "legend": {
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
        },
        "updatemenus": [
            {
                "buttons": [
                    {
                        "label": "Color By Bucket",
                        "method": "update",
                        "args": [
                            {"visible": bucket_visible},
                            {"legend": {"orientation": "v", "x": 1.02, "y": 1.0}},
                        ],
                    },
                    {
                        "label": "Color By Cluster",
                        "method": "update",
                        "args": [
                            {"visible": cluster_visible},
                            {"legend": {"orientation": "v", "x": 1.02, "y": 1.0}},
                        ],
                    },
                ],
                "direction": "left",
                "showactive": True,
                "type": "buttons",
                "x": 0.0,
                "y": 1.08,
            }
        ],
    }
    return traces, layout


def render_html(
    html_path: Path,
    traces: list[dict[str, object]],
    layout: dict[str, object],
    report: dict[str, object],
) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive 3D Case Embeddings</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #f5f5f5;
      color: #1a1a1a;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px;
    }}
    .meta {{
      margin: 0 0 12px;
      font-size: 14px;
      line-height: 1.5;
    }}
    #plot {{
      width: 100%;
      height: 900px;
      background: #ffffff;
      border: 1px solid #dddddd;
      border-radius: 10px;
    }}
    code {{
      background: #eeeeee;
      padding: 2px 6px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="meta">
      Projection method: <code>{report["projection"]["method"]}</code> |
      Points plotted: <code>{report["points_plotted"]}</code> |
      Sampled: <code>{report["sampled"]}</code>
    </p>
    <div id="plot"></div>
  </div>
  <script>
    const traces = {json.dumps(traces, separators=(",", ":"))};
    const layout = {json.dumps(layout, separators=(",", ":"))};
    const config = {{
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
      modeBarButtonsToRemove: ["lasso3d", "select2d", "autoScale2d"]
    }};
    Plotly.newPlot("plot", traces, layout, config);
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_df, embeddings = load_inputs(args.input_dir)
    sampling_labels = build_sampling_labels(cases_df, args.stratify_by)
    max_points = args.max_points if args.max_points is not None else 0
    sample_indices = stratified_sample_indices(
        labels=sampling_labels,
        max_items=max_points,
        random_state=args.random_state,
    )
    sampled = len(sample_indices) != len(cases_df)
    cases_df = cases_df.iloc[sample_indices].reset_index(drop=True)
    embeddings = embeddings[sample_indices]

    coords, projection_meta = project_embeddings(
        embeddings=embeddings,
        method=args.method,
        pre_reduce_dim=args.pre_reduce_dim,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        random_state=args.random_state,
    )
    cases_df["interactive_x"] = coords[:, 0]
    cases_df["interactive_y"] = coords[:, 1]
    cases_df["interactive_z"] = coords[:, 2]

    csv_path = args.output_dir / f"interactive_3d_projection_{args.method}.csv"
    html_path = args.output_dir / f"interactive_3d_{args.method}.html"
    report_path = args.output_dir / f"interactive_3d_{args.method}_report.json"
    cases_df.to_csv(csv_path, index=False)

    traces, layout = build_plot_payload(
        frame=cases_df,
        marker_size=args.marker_size,
        marker_opacity=args.marker_opacity,
    )

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "output_html": str(html_path),
        "output_csv": str(csv_path),
        "points_plotted": int(len(cases_df)),
        "sampled": sampled,
        "projection": projection_meta,
        "stratify_by": args.stratify_by,
    }
    render_html(
        html_path=html_path,
        traces=traces,
        layout=layout,
        report=report,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote interactive HTML to: {html_path}")
    print(f"Wrote projection CSV to: {csv_path}")
    print(f"Points plotted: {len(cases_df):,}")
    print(f"Projection method: {args.method}")


if __name__ == "__main__":
    main()
