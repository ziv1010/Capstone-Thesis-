#!/usr/bin/env python
"""Build a compact per-case counterfactual factor index.

``case_counterfactual_groups.csv`` is ~390 MB and roughly half of its rows are
``relation_type`` groups, which never correspond to an evidence box in the
contrast diagrams.  This script keeps only the evidence-node groups and the
columns the visualiser / figure scripts actually need, so both can annotate
evidence boxes without loading the full table.

Output: ``case_counterfactual_factor_index.csv`` in the explanation directory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPLANATION_DIR = (
    APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
)

SOURCE_NAME = "case_counterfactual_groups.csv"
OUTPUT_NAME = "case_counterfactual_factor_index.csv"

SOURCE_COLUMNS = [
    "case_index",
    "case_id",
    "split",
    "target_label",
    "baseline_pred_label",
    "baseline_pred_proba",
    "group_rank_abs",
    "group_kind",
    "evidence_type",
    "evidence_global_index",
    "evidence_name",
    "delta_pred_proba",
    "abs_delta_pred_proba",
    "prediction_flipped",
]

OUTPUT_COLUMNS = [
    "case_index",
    "group_rank_abs",
    "cf_evidence_rank",
    "evidence_type",
    "evidence_global_index",
    "evidence_name",
    "delta_pred_proba",
    "abs_delta_pred_proba",
    "prediction_flipped",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Defaults to <explanation-dir>/{OUTPUT_NAME}",
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser


def build_index(source: Path, chunk_size: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        source,
        usecols=SOURCE_COLUMNS,
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["group_kind"].astype(str) != "relation_type"]
        if chunk.empty:
            continue
        frames.append(chunk.drop(columns=["group_kind"]))
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    factors = pd.concat(frames, ignore_index=True)
    factors["case_index"] = pd.to_numeric(factors["case_index"], errors="coerce").astype("Int64")
    factors["group_rank_abs"] = pd.to_numeric(factors["group_rank_abs"], errors="coerce")
    factors["abs_delta_pred_proba"] = pd.to_numeric(factors["abs_delta_pred_proba"], errors="coerce")
    factors = factors.sort_values(
        ["case_index", "group_rank_abs"], na_position="last", kind="mergesort"
    )
    # Dense rank among evidence-node groups only: this is the number shown on the
    # badge, so that "CF #1" means "the strongest evidence factor for this case"
    # rather than "rank 1 including relation-type groups".
    factors["cf_evidence_rank"] = factors.groupby("case_index").cumcount() + 1
    return factors[OUTPUT_COLUMNS]


def main() -> None:
    args = build_parser().parse_args()
    source = args.explanation_dir / SOURCE_NAME
    if not source.exists():
        raise FileNotFoundError(f"counterfactual groups CSV does not exist: {source}")
    output = args.output or (args.explanation_dir / OUTPUT_NAME)

    start = time.time()
    factors = build_index(source, args.chunk_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(output, index=False)
    cases = int(factors["case_index"].nunique()) if not factors.empty else 0
    print(
        f"[done] rows={len(factors):,} cases={cases:,} "
        f"bytes={output.stat().st_size:,} elapsed={time.time() - start:.1f}s out={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
