#!/usr/bin/env python3
"""Per-case artefacts for the "shared-case" story.

For each selected case, writes ONE folder containing four files:
  panel_a_tsne.png        — t-SNE of post-GNN case embeddings, target highlighted
  panel_b_subgraph.png    — PGExplainer top nodes around the target
  panel_c_neighbours.md   — top FAISS neighbours (post-GNN cosine)
  panel_d_llm.md          — Phase 5 LLM explanation
  README.md               — case metadata + file index

Output folder per case:
  outputs/thesis_figures/case_<idx>/

Usage
-----
  # Auto-pick cases (prefers those with both Phase 4 and Phase 5 available):
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
    phase5_case_indices,
)
from _panels import render_case  # noqa: E402


def _auto_pick(paths: Paths, n: int) -> list[int]:
    preds = load_predictions(paths).set_index("node_index")
    explained = explained_case_indices(paths)
    have_phase5 = phase5_case_indices(paths)
    primary = [i for i in explained if i in have_phase5]
    fallback = [i for i in explained if i not in have_phase5]

    def _rank(pool):
        if not pool:
            return [], []
        rows = preds.loc[pool].copy()
        rows["correct"] = rows["pred_label"].astype(str) == rows["target_label"].astype(str)
        return (
            rows[rows["correct"]].sort_values("confidence", ascending=False).index.tolist(),
            rows[~rows["correct"]].sort_values("confidence", ascending=False).index.tolist(),
        )

    corr_p, wrong_p = _rank(primary)
    corr_f, wrong_f = _rank(fallback)

    picks: list[int] = []
    target_correct = max(n - 1, 1)
    for src in (corr_p, corr_f):
        for idx in src:
            if idx not in picks:
                picks.append(idx)
            if len(picks) >= target_correct:
                break
        if len(picks) >= target_correct:
            break
    if n >= 2:
        for src in (wrong_p, wrong_f):
            added = False
            for idx in src:
                if idx not in picks:
                    picks.append(idx); added = True; break
            if added:
                break
    for src in (corr_p, wrong_p, corr_f, wrong_f):
        for idx in src:
            if len(picks) >= n:
                break
            if idx not in picks:
                picks.append(idx)
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
