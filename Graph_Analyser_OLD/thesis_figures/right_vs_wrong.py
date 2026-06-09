#!/usr/bin/env python3
"""Right-vs-wrong: renders per-case artefact folders for one correct and one
wrong prediction, plus a top-level comparison README that links to both.

Output layout:
  outputs/thesis_figures/right_vs_wrong/
    README.md                          — side-by-side metadata + file links
    correct_case_<idx>/                — same layout as shared_case_figure
    wrong_case_<idx>/                  — same layout as shared_case_figure

Usage
-----
  python right_vs_wrong.py
  python right_vs_wrong.py --correct-idx 10134 --wrong-idx 10419
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    Paths,
    explained_case_indices,
    load_phase4,
    load_predictions,
)
from _panels import render_case  # noqa: E402


def _auto_pick(paths: Paths) -> tuple[int, int]:
    preds = load_predictions(paths).set_index("node_index")
    explained = explained_case_indices(paths)
    rows = preds.loc[explained].copy()
    rows["correct"] = rows["pred_label"].astype(str) == rows["target_label"].astype(str)
    corr = rows[rows["correct"]].sort_values("confidence", ascending=False)
    wrong = rows[~rows["correct"]].sort_values("confidence", ascending=False)
    if len(corr) and len(wrong):
        return int(corr.index[0]), int(wrong.index[0])
    rows = preds.loc[explained].copy()
    rows["correct"] = rows["pred_label"].astype(str) == rows["target_label"].astype(str)
    corr = rows[rows["correct"]].sort_values("confidence", ascending=False)
    if len(corr) == 0:
        raise RuntimeError("No explained cases to pick from.")
    print("[right_vs_wrong] no wrong explained cases; using lowest-confidence correct.")
    return int(corr.index[0]), int(corr.index[-1])


def _summary_line(paths: Paths, idx: int) -> str:
    phase4 = load_phase4(paths, idx)
    return (
        f"case {idx} — true={str(phase4.get('target_label')).strip()} "
        f"pred={str(phase4.get('predicted_label')).strip()} "
        f"conf={phase4.get('confidence', 0.0):.4f}"
    )


def _write_comparison_readme(
    paths: Paths, root: Path, correct_idx: int, wrong_idx: int,
    correct_dir: Path, wrong_dir: Path,
) -> Path:
    c_phase4 = load_phase4(paths, correct_idx)
    w_phase4 = load_phase4(paths, wrong_idx)
    fp = root / "README.md"
    rel_c = correct_dir.name
    rel_w = wrong_dir.name
    lines = [
        "# Right vs Wrong — per-case artefacts",
        "",
        "Two cases rendered with the same PGExplainer-focused layout. Compare",
        "the explainer subgraphs between rows — the character of the explanation",
        "typically diverges between a confident-correct and a wrong prediction.",
        "",
        "## Correct prediction",
        "",
        f"- **{_summary_line(paths, correct_idx)}**",
        f"- Case id: {c_phase4.get('case_id', '')}",
        f"- Folder: [`{rel_c}/`]({rel_c}/README.md)",
        f"  - [`panel_a_tsne.png`]({rel_c}/panel_a_tsne.png)",
        f"  - [`panel_b_subgraph.png`]({rel_c}/panel_b_subgraph.png)",
        "",
        "## Wrong prediction",
        "",
        f"- **{_summary_line(paths, wrong_idx)}**",
        f"- Case id: {w_phase4.get('case_id', '')}",
        f"- Folder: [`{rel_w}/`]({rel_w}/README.md)",
        f"  - [`panel_a_tsne.png`]({rel_w}/panel_a_tsne.png)",
        f"  - [`panel_b_subgraph.png`]({rel_w}/panel_b_subgraph.png)",
        "",
    ]
    fp.write_text("\n".join(lines))
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct-idx", type=int, default=None)
    ap.add_argument("--wrong-idx", type=int, default=None)
    args = ap.parse_args()

    paths = Paths.default()
    if args.correct_idx is not None and args.wrong_idx is not None:
        correct_idx, wrong_idx = args.correct_idx, args.wrong_idx
    else:
        correct_idx, wrong_idx = _auto_pick(paths)

    print(f"[right_vs_wrong] correct={correct_idx}  wrong={wrong_idx}", flush=True)

    root = paths.fig_dir / "right_vs_wrong"
    root.mkdir(parents=True, exist_ok=True)
    correct_dir = root / f"correct_case_{correct_idx}"
    wrong_dir = root / f"wrong_case_{wrong_idx}"

    for tag, idx, out_dir in (
        ("correct", correct_idx, correct_dir),
        ("wrong", wrong_idx, wrong_dir),
    ):
        files = render_case(paths, idx, out_dir)
        print(f"[right_vs_wrong] {tag} case {idx} → {out_dir}")
        for k, fp in files.items():
            print(f"    {k:>10}  {fp.name}")

    readme = _write_comparison_readme(
        paths, root, correct_idx, wrong_idx, correct_dir, wrong_dir,
    )
    print(f"[right_vs_wrong] comparison README → {readme}")


if __name__ == "__main__":
    main()
