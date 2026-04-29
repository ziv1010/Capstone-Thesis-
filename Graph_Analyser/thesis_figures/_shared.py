"""Shared loaders + t-SNE caching for the thesis figures."""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GRAPH_ANALYSER_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = GRAPH_ANALYSER_ROOT / "outputs"
FIG_OUTPUT_DIR = OUTPUTS_ROOT / "thesis_figures"
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Paths:
    predictions_csv: Path
    case_embeddings: Path
    phase4_cases_dir: Path
    phase4_manifest: Path
    phase5_dir: Path
    fig_dir: Path

    @classmethod
    def default(cls) -> "Paths":
        p12 = OUTPUTS_ROOT / "phase1_2_inference"
        p4 = OUTPUTS_ROOT / "phase4_explanations"
        p5 = OUTPUTS_ROOT / "phase5_llm_reasoning"
        return cls(
            predictions_csv=p12 / "predictions.csv",
            case_embeddings=p12 / "case_embeddings.npy",
            phase4_cases_dir=p4 / "cases",
            phase4_manifest=p4 / "manifest.json",
            phase5_dir=p5,
            fig_dir=FIG_OUTPUT_DIR,
        )


def load_predictions(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.predictions_csv)


def load_case_embeddings(paths: Paths) -> np.ndarray:
    return np.load(paths.case_embeddings)


def load_phase4(paths: Paths, case_idx: int) -> dict:
    fp = paths.phase4_cases_dir / f"case_{case_idx}.json"
    if not fp.exists():
        raise FileNotFoundError(
            f"No Phase 4 explanation for case {case_idx}. "
            f"Expected {fp}. Check manifest.json for explained cases."
        )
    with open(fp) as f:
        return json.load(f)


def load_phase5(paths: Paths, case_idx: int) -> dict | None:
    fp = paths.phase5_dir / f"explanation_{case_idx}.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def explained_case_indices(paths: Paths) -> list[int]:
    with open(paths.phase4_manifest) as f:
        return list(json.load(f)["case_indices"])


def phase5_case_indices(paths: Paths) -> set[int]:
    """Indices that have a Phase-5 LLM explanation on disk."""
    out: set[int] = set()
    if not paths.phase5_dir.exists():
        return out
    for fp in paths.phase5_dir.glob("explanation_*.json"):
        stem = fp.stem.removeprefix("explanation_")
        try:
            out.add(int(stem))
        except ValueError:
            continue
    return out


def compute_or_load_tsne(
    paths: Paths,
    split: str = "test",
    perplexity: int = 30,
    n_iter: int = 1000,
    seed: int = 0,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """t-SNE on the post-GNN case embeddings for a split, cached on disk.

    Returns (points_2d, node_indices) where node_indices[i] is the row in
    predictions.csv (= node index in the graph) for points_2d[i].
    """
    cache_fp = paths.fig_dir / f"tsne_{split}_p{perplexity}.npz"
    if cache_fp.exists() and not force:
        z = np.load(cache_fp)
        return z["points"], z["node_indices"]

    from sklearn.manifold import TSNE

    preds = load_predictions(paths)
    embs = load_case_embeddings(paths)
    if split != "all":
        mask = preds["split"].to_numpy() == split
        node_indices = preds.loc[mask, "node_index"].to_numpy()
        X = embs[node_indices]
    else:
        node_indices = preds["node_index"].to_numpy()
        X = embs

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    # sklearn renamed n_iter -> max_iter in 1.5; support both.
    try:
        tsne = TSNE(max_iter=n_iter, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=n_iter, **tsne_kwargs)
    points = tsne.fit_transform(X)
    np.savez(cache_fp, points=points, node_indices=node_indices)
    return points, node_indices


def wrap(text: str, width: int = 78) -> str:
    paragraphs = str(text).split("\n")
    return "\n".join(
        textwrap.fill(p, width=width) if p.strip() else "" for p in paragraphs
    )


def label_str(raw) -> str:
    """Map a raw outcome label to a short string."""
    s = str(raw).strip()
    if s in ("-1", "0", "dismissed", "Dismissed"):
        return "dismissed"
    if s in ("1", "allowed", "Allowed"):
        return "allowed"
    return s


def predicted_label_int(phase4: dict) -> int:
    """Return 0 if model predicted -1/dismissed, 1 if it predicted 1/allowed."""
    s = str(phase4.get("predicted_label", "")).strip()
    return 1 if s in ("1", "allowed", "Allowed") else 0


def correct(phase4: dict) -> bool:
    return str(phase4.get("predicted_label")).strip() == str(
        phase4.get("target_label")
    ).strip()
