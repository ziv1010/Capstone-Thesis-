#!/usr/bin/env python
"""Phase 7: nearest-training-case evidence via frozen GNN embeddings.

Takes the final GNN embedding for each test case, finds the closest
training cases by cosine similarity, and reports their labels and
similarity scores. Serves as a fallback when graph evidence is untraceable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyser.loader import load_config, validate_case_ids_bucket  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--phase4-dir", type=str, default=None)
    ap.add_argument("--phase6-dir", type=str, default=None)
    ap.add_argument("--phase12-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--case-index", type=int, action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only-untraceable", action="store_true")
    ap.add_argument("--auto-untraceable", type=int, default=0)
    ap.add_argument("--auto-misclassified", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nearest-k", type=int, default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true")
    return ap.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_case_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def iter_case_paths(explanations_dir: Path) -> list[Path]:
    return sorted(explanations_dir.glob("case_*.json"), key=parse_case_index)


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def merge_rank_summaries(report_dir: Path, expected_bucket: str | None = None) -> None:
    rows = []
    for path in sorted(report_dir.glob("summary_rank*.json")):
        shard_rows = json.loads(path.read_text())
        if expected_bucket:
            for row in shard_rows:
                if row.get("bucket") != expected_bucket:
                    raise ValueError(
                        f"Stale or cross-bucket Phase 7 shard: {path}. "
                        f"row bucket={row.get('bucket')!r}, expected={expected_bucket!r}"
                    )
        rows.extend(shard_rows)
    (report_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    print(f"[phase7 merge] wrote {len(rows)} rows to {report_dir / 'summary.json'}", flush=True)


def find_all_cases(explanations_dir: Path) -> list[int]:
    return [parse_case_index(p) for p in iter_case_paths(explanations_dir)]


def find_misclassified(explanations_dir: Path, n: int) -> list[int]:
    out = []
    for path in iter_case_paths(explanations_dir):
        bundle = read_json(path)
        if bundle.get("target_label") != bundle.get("predicted_label"):
            out.append(int(bundle["case_node_index"]))
            if len(out) >= n:
                break
    return out


def find_untraceable(phase6_dir: Path, limit: int = 0) -> list[int]:
    summary_path = phase6_dir / "summary.json"
    out = []
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
        for row in rows:
            majority = str(row.get("evidence_majority", ""))
            n_traceable = int(row.get("n_traceable_nodes", 0) or 0)
            if majority == "untraceable" or n_traceable == 0:
                out.append(int(row["case_node_index"]))
                if limit and len(out) >= limit:
                    return out
        return out

    for path in iter_case_paths(phase6_dir):
        diag = read_json(path)
        weighted = diag.get("weighted_evidence", {})
        if weighted.get("majority_class") == "untraceable" or int(weighted.get("n_traceable_nodes", 0) or 0) == 0:
            out.append(int(diag["case_node_index"]))
            if limit and len(out) >= limit:
                break
    return out


def select_targets(args: argparse.Namespace, phase4_dir: Path, phase6_dir: Path) -> list[int]:
    targets = list(args.case_index)
    if args.all:
        targets.extend(find_all_cases(phase4_dir))
    if args.only_untraceable:
        base = targets or find_all_cases(phase4_dir)
        untraceable = set(find_untraceable(phase6_dir))
        targets = [idx for idx in base if idx in untraceable]
    if args.auto_untraceable:
        targets.extend(find_untraceable(phase6_dir, args.auto_untraceable))
    if args.auto_misclassified:
        targets.extend(find_misclassified(phase4_dir, args.auto_misclassified))
    targets = list(dict.fromkeys(int(x) for x in targets))
    if args.limit > 0:
        targets = targets[: args.limit]
    return targets


def format_label(label: str) -> str:
    if str(label) == "1":
        return "Win"
    if str(label) == "-1":
        return "Loss"
    return str(label)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_phase12_artifacts(
    phase12_dir: Path,
    expected_bucket: str | None = None,
) -> dict[str, Any]:
    predictions_csv = phase12_dir / "predictions.csv"
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing {predictions_csv}")
    embeddings = np.load(phase12_dir / "case_embeddings.npy").astype(np.float32)
    predictions_df = pd.read_csv(predictions_csv)
    if expected_bucket:
        validate_case_ids_bucket(
            [str(x) for x in predictions_df["case_id"].tolist()],
            expected_bucket,
            f"Phase 1-2 predictions {predictions_csv}",
        )
    train_df = predictions_df[predictions_df["split"] == "train"].copy()
    train_indices = train_df["node_index"].astype(int).to_numpy() if not train_df.empty else np.array([], dtype=np.int64)
    normalized = normalize_embeddings(embeddings)
    return {
        "normalized_embeddings": normalized,
        "train_df": train_df,
        "train_matrix": normalized[train_indices] if len(train_indices) else np.zeros((0, embeddings.shape[1]), dtype=np.float32),
        "probabilities": np.load(phase12_dir / "probabilities.npy"),
        "case_id_by_node_index": {
            int(row["node_index"]): str(row["case_id"])
            for _, row in predictions_df.iterrows()
        },
    }


def nearest_training_neighbours(
    case_idx: int,
    artifacts: dict[str, Any],
    label_names: list[str],
    nearest_k: int,
) -> dict[str, Any]:
    train_df = artifacts["train_df"]
    if train_df.empty:
        return {"nearest_k": nearest_k, "neighbours": [], "target_label_counts": {}}

    target = artifacts["normalized_embeddings"][case_idx]
    sims = artifacts["train_matrix"] @ target
    order = np.argsort(-sims)[:nearest_k]

    neighbours = []
    target_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    for rank, pos in enumerate(order, start=1):
        row = train_df.iloc[int(pos)]
        node_idx = int(row["node_index"])
        target_label = str(row.get("target_label", label_names[int(row["target_index"])]))
        pred_label = str(row.get("pred_label", label_names[int(row["pred_index"])]))
        target_counts[target_label] += 1
        pred_counts[pred_label] += 1
        neighbours.append(
            {
                "rank": rank,
                "node_index": node_idx,
                "case_id": str(row.get("case_id", f"case_{node_idx}")),
                "target_label": target_label,
                "pred_label": pred_label,
                "confidence": float(row.get("confidence", 0.0)),
                "cosine_similarity": float(sims[int(pos)]),
            }
        )

    majority_target = target_counts.most_common(1)[0][0] if target_counts else "?"
    return {
        "nearest_k": nearest_k,
        "neighbours": neighbours,
        "target_label_counts": dict(target_counts),
        "pred_label_counts": dict(pred_counts),
        "majority_target_label": majority_target,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Phase 7: {report['case_id']}",
        "",
        f"- **Prediction**: `{format_label(report['predicted_label'])}`",
        f"- **Target**: `{format_label(report['target_label'])}`",
        f"- **Confidence**: {float(report.get('confidence', 0.0)):.2%}",
        "",
    ]

    nn = report.get("embedding_neighbours", {})
    lines.extend(
        [
            "## Embedding Nearest Training Cases",
            "",
            f"- **Nearest k**: {nn.get('nearest_k')}",
            f"- **Target-label counts**: `{nn.get('target_label_counts', {})}`",
            f"- **Majority target label**: `{format_label(nn.get('majority_target_label', '?'))}`",
            "",
            "| Rank | Similarity | Target | Pred | Case |",
            "|---:|---:|---|---|---|",
        ]
    )
    for row in nn.get("neighbours", [])[:20]:
        case_id = str(row.get("case_id", ""))[:80].replace("|", "\\|")
        lines.append(
            f"| {row.get('rank')} | {float(row.get('cosine_similarity', 0.0)):.4f} | "
            f"{format_label(row.get('target_label', '?'))} | {format_label(row.get('pred_label', '?'))} | {case_id} |"
        )
    lines.append("")
    return "\n".join(lines)


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    nn = report.get("embedding_neighbours", {})
    predicted = report["predicted_label"]
    return {
        "case_node_index": report["case_node_index"],
        "bucket": report.get("bucket"),
        "case_id": report["case_id"],
        "target_label": report["target_label"],
        "predicted_label": predicted,
        "confidence": report["confidence"],
        "embedding_majority_target_label": nn.get("majority_target_label"),
        "embedding_majority_supports_prediction": nn.get("majority_target_label") == predicted,
        "embedding_target_label_counts": nn.get("target_label_counts", {}),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    phase7_cfg = cfg.get("phase7", {})
    expected_bucket = cfg.get("bucket")
    label_names = ["-1", "1"]

    output_root = Path(cfg.get("output_root", ROOT / "outputs"))
    phase4_dir = Path(args.phase4_dir or output_root / "phase4_explanations" / "cases")
    phase6_dir = Path(args.phase6_dir or output_root / "phase6_misclass_diagnostic")
    phase12_dir = Path(args.phase12_dir or output_root / "phase1_2_inference")
    report_dir = Path(args.output_dir or output_root / "phase7_topk_embedding")

    if args.merge_only:
        merge_rank_summaries(report_dir, expected_bucket=expected_bucket)
        return

    nearest_k = int(args.nearest_k or phase7_cfg.get("nearest_k", 20))

    targets = select_targets(args, phase4_dir, phase6_dir)
    if not targets:
        raise SystemExit(
            "Provide --case-index, --all, --auto-untraceable N, "
            "--auto-misclassified N, or --only-untraceable with available Phase 6 output."
        )
    if args.world_size < 1:
        raise SystemExit("--world-size must be >= 1")
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit("--rank must be in [0, world-size)")
    total_selected = len(targets)
    if args.world_size > 1:
        targets = targets[args.rank :: args.world_size]

    print(
        f"[phase7] selected {len(targets)} cases"
        + (f" for shard {args.rank}/{args.world_size} from {total_selected}" if args.world_size > 1 else ""),
        flush=True,
    )
    print(f"[phase7] loading phase1-2 artifacts: {phase12_dir}", flush=True)
    artifacts = load_phase12_artifacts(phase12_dir, expected_bucket=expected_bucket)

    summary_path = phase12_dir / "summary.json"
    if summary_path.exists():
        phase12_summary = json.loads(summary_path.read_text())
        label_names = phase12_summary.get("label_names", label_names)
        summary_bucket = phase12_summary.get("bucket")
        if expected_bucket and summary_bucket and summary_bucket != expected_bucket:
            raise ValueError(
                f"Phase 1-2 summary bucket {summary_bucket!r} does not match "
                f"config bucket {expected_bucket!r}: {summary_path}"
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for ordinal, case_idx in enumerate(targets, start=1):
        bundle_path = phase4_dir / f"case_{case_idx}.json"
        if not bundle_path.exists():
            print(f"[phase7] skip missing {bundle_path}", flush=True)
            continue
        bundle = read_json(bundle_path)
        validate_case_ids_bucket(
            [str(bundle.get("case_id", ""))],
            expected_bucket,
            f"Phase 7 explanation bundle {bundle_path}",
        )
        predictions_case_id = str(artifacts["case_id_by_node_index"].get(int(case_idx), ""))
        if str(bundle.get("case_id", "")) != predictions_case_id:
            raise ValueError(
                f"Phase 7 stale bundle mismatch for case_{case_idx}: "
                f"bundle case_id={bundle.get('case_id')!r}, "
                f"phase1_2 case_id={predictions_case_id!r}"
            )

        report = {
            "case_node_index": int(case_idx),
            "bucket": expected_bucket,
            "case_id": bundle.get("case_id", f"case_{case_idx}"),
            "target_label": str(bundle.get("target_label", "?")),
            "predicted_label": str(bundle["predicted_label"]),
            "confidence": float(artifacts["probabilities"][case_idx].max()),
            "embedding_neighbours": nearest_training_neighbours(
                case_idx,
                artifacts,
                label_names,
                nearest_k=nearest_k,
            ),
        }

        json_out = report_dir / f"case_{case_idx}.json"
        md_out = report_dir / f"case_{case_idx}.md"
        json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        md_out.write_text(render_markdown(report))
        summary_rows.append(summary_row(report))

        if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(targets):
            print(
                f"[phase7] processed {ordinal}/{len(targets)} "
                f"case={case_idx} -> {display_path(json_out)}",
                flush=True,
            )

    summary_name = "summary.json" if args.world_size == 1 else f"summary_rank{args.rank}.json"
    (report_dir / summary_name).write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False)
    )
    print(f"[phase7] wrote {len(summary_rows)} reports under {report_dir}", flush=True)


if __name__ == "__main__":
    main()
