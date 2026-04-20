from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _prepare_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_bar_series(
    series: pd.Series,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    rotate_xticks: int = 45,
) -> None:
    p = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate_xticks)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)


def plot_histogram(
    values: pd.Series,
    path: str | Path,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> None:
    p = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values.dropna().values, bins=bins, color="#2ca02c", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
