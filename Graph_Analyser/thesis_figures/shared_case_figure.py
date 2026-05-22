#!/usr/bin/env python3
"""Per-case artefacts for the "shared-case" story.

For each selected case, writes ONE folder containing PGExplainer-focused files:
  panel_a_tsne.png        — t-SNE of post-GNN case embeddings, target highlighted
  panel_b_subgraph.png    — PGExplainer top nodes around the target
  README.md               — case metadata + file index

Output folder per case:
  outputs/thesis_figures/case_<idx>/

Usage
-----
  # Auto-pick cases from Phase 4 explanations:
  python shared_case_figure.py --auto 3

  # Or pass specific case node indices (must appear in phase4 manifest):
  python shared_case_figure.py --case-indices 10134 10419 1210
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    Paths,
    explained_case_indices,
    load_predictions,
)
from _panels import render_case  # noqa: E402


def _auto_pick(paths: Paths, n: int) -> list[int]:
    preds = load_predictions(paths).set_index("node_index")
    explained = explained_case_indices(paths)

    def _rank(pool):
        if not pool:
            return [], []
        rows = preds.loc[pool].copy()
        rows["correct"] = rows["pred_label"].astype(str) == rows["target_label"].astype(str)
        return (
            rows[rows["correct"]].sort_values("confidence", ascending=False).index.tolist(),
            rows[~rows["correct"]].sort_values("confidence", ascending=False).index.tolist(),
        )

    corr, wrong = _rank(explained)

    picks: list[int] = []
    target_correct = max(n - 1, 1)
    for idx in corr:
        if idx not in picks:
            picks.append(idx)
        if len(picks) >= target_correct:
            break
    if n >= 2:
        for idx in wrong:
            if idx not in picks:
                picks.append(idx)
                break
    for src in (corr, wrong):
        for idx in src:
            if idx not in picks:
                picks.append(idx)
            if len(picks) >= n:
                break
        if len(picks) >= n:
            break
    return picks[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-indices", type=int, nargs="+", default=None)
    ap.add_argument("--auto", type=int, default=0, help="auto-pick N explained cases")
    args = ap.parse_args()

    paths = Paths.default()
    if args.case_indices:
        case_indices = args.case_indices
    elif args.auto > 0:
        case_indices = _auto_pick(paths, args.auto)
    else:
        case_indices = _auto_pick(paths, 3)

    print(f"[shared_case_figure] rendering {len(case_indices)} cases: {case_indices}",
          flush=True)

    for idx in case_indices:
        out_dir = paths.fig_dir / f"case_{idx}"
        files = render_case(paths, idx, out_dir)
        print(f"[shared_case_figure] case {idx} → {out_dir}")
        for k, fp in files.items():
            print(f"    {k:>10}  {fp.name}")


if __name__ == "__main__":
    main()
