#!/usr/bin/env python3
"""
Cross-domain test: evaluate the cross_bucket_party_args_lr_decay_kfold model
(trained on all legal domains except food_safety) against the food_safety dataset.

Pipeline:
  1. Write preprocessing + graph-build config (food_safety-specific overrides)
  2. Preprocess food_safety JSONs  → cleaned_cases/   (CPU only)
  3. Build graph                   → graph_cache/      (multi-GPU via CUDA_VISIBLE_DEVICES)
  4. Evaluate 5 fold checkpoints in parallel, one per GPU (round-robin if fewer GPUs)
  5. Aggregate metrics and write summary

Multi-GPU usage:
  --cuda 0,1,2,3   build_graph uses all 4 GPUs for bge-m3 encoding (multi_process=true)
                   fold evals are distributed round-robin across those same GPUs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SECTION_GNN = THIS_DIR.parents[1]

FOOD_SAFETY_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3/food_safety"
)
KFOLD_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/section_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset"
    "/models/cross_bucket_party_args_lr_decay_kfold/kfold"
)
TRAINING_CONFIG = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/section_GNN/runs/cross_bucket_total_dataset/config.yaml"
)

PREPROCESS_SCRIPT = SECTION_GNN / "experiments" / "fixed_open_pipeline" / "preprocess_fixed_open.py"
BUILD_GRAPH_SCRIPT = SECTION_GNN / "final_graph" / "build_graph.py"
EVAL_SCRIPT = SECTION_GNN / "dump2" / "scripts" / "evaluate_saved_model.py"

ARTIFACT_ROOT = THIS_DIR
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def make_env(cuda: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_blocking(script: Path, script_args: list[str], *, env_name: str, cuda: str) -> None:
    cmd = ["micromamba", "run", "-n", env_name, "python", str(script), *script_args]
    print(f"\n[run] CUDA={cuda}  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SECTION_GNN), env=make_env(cuda), check=True)


def run_background(
    script: Path,
    script_args: list[str],
    *,
    env_name: str,
    cuda: str,
    log_path: Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["micromamba", "run", "-n", env_name, "python", str(script), *script_args]
    print(f"[launch] CUDA={cuda}  log={log_path}  {' '.join(cmd)}")
    log_fh = open(log_path, "w")
    return subprocess.Popen(
        cmd,
        cwd=str(SECTION_GNN),
        env=make_env(cuda),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain food_safety eval")
    p.add_argument(
        "--cuda",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated GPU IDs, e.g. '0,1,2,3'. "
             "Build-graph uses all of them; fold evals are distributed round-robin.",
    )
    p.add_argument("--env", default="thesis_work")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def build_food_safety_config() -> dict[str, Any]:
    base = load_yaml(TRAINING_CONFIG)

    processed_dir = ARTIFACT_ROOT / "processed"
    graph_cache_dir = ARTIFACT_ROOT / "graph_cache"
    embeddings_dir = ARTIFACT_ROOT / "embeddings_cache"
    audits_dir = ARTIFACT_ROOT / "audits"
    outputs_dir = ARTIFACT_ROOT / "outputs"

    cfg = dict(base)
    cfg["project"] = dict(base.get("project", {}))
    cfg["project"]["name"] = "cross_domain_food_safety"

    cfg["paths"] = {
        "raw_json_dir": str(FOOD_SAFETY_DIR),
        "processed_dir": str(processed_dir),
        "cleaned_case_dir": str(processed_dir / "cleaned_cases"),
        "normalized_entity_dir": str(processed_dir / "normalized_entities"),
        "embeddings_cache_dir": str(embeddings_dir),
        "graph_cache_dir": str(graph_cache_dir),
        "audits_dir": str(audits_dir),
        "outputs_dir": str(outputs_dir),
    }

    # food_safety filenames don't have __ unlike the cross-bucket training data
    cfg["data"] = dict(base.get("data", {}))
    cfg["data"]["file_glob"] = "*.json"
    cfg["data"]["limit"] = None

    cfg["preprocessing"] = dict(base.get("preprocessing", {}))
    cfg["preprocessing"]["skip_missing_or_unmapped_labels"] = True

    cfg["graph"] = dict(base.get("graph", {}))
    cfg["graph"]["cache_name"] = "case_star_global_graph_food_safety_cross_domain.reasoning_focused.pt"
    cfg["graph"]["debug_sample_size"] = 1

    cfg["features"] = dict(base.get("features", {}))

    return cfg


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["accuracy", "macro_f1", "micro_f1", "roc_auc"]
    agg: dict[str, Any] = {}
    for key in keys:
        values = [r.get(key) for r in fold_results if r.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = mean(values)
            agg[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse GPU list for round-robin assignment to folds
    gpu_ids = [g.strip() for g in args.cuda.split(",") if g.strip()]
    if not gpu_ids:
        sys.exit("[error] --cuda produced an empty GPU list")
    print(f"[info] GPUs available: {gpu_ids}")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Write config ──────────────────────────────────────────────────────
    cfg = build_food_safety_config()
    config_path = ARTIFACT_ROOT / "food_safety_cross_domain_config.yaml"
    dump_yaml(cfg, config_path)
    print(f"[config] Written → {config_path}")

    # ── 2. Preprocess (CPU, no GPU needed) ───────────────────────────────────
    if not args.skip_preprocess:
        preprocess_args = ["--config", str(config_path), "--input-dir", str(FOOD_SAFETY_DIR)]
        if args.limit:
            preprocess_args += ["--limit", str(args.limit)]
        # use first GPU slot in env but it won't actually be used
        run_blocking(PREPROCESS_SCRIPT, preprocess_args, env_name=args.env, cuda=args.cuda)
    else:
        print("[skip] preprocess")

    # ── 3. Build graph (multi-GPU via multi_process=true) ────────────────────
    graph_cache_path = (
        Path(cfg["paths"]["graph_cache_dir"]) / cfg["graph"]["cache_name"]
    )
    if not args.skip_build:
        build_args = ["--config", str(config_path)]
        if args.limit:
            build_args += ["--limit", str(args.limit)]
        # pass all GPUs so sentence-transformers multi-process can use them all
        run_blocking(BUILD_GRAPH_SCRIPT, build_args, env_name=args.env, cuda=args.cuda)
    else:
        print("[skip] build_graph")

    if not graph_cache_path.exists():
        sys.exit(f"[error] Graph cache not found: {graph_cache_path}")

    # ── 4. Evaluate folds in parallel, round-robin across GPUs ──────────────
    log_dir = ARTIFACT_ROOT / "logs"
    fold_specs: list[dict[str, Any]] = []
    for fold_idx in range(N_FOLDS):
        fold_name = f"fold_{fold_idx:02d}"
        checkpoint = KFOLD_DIR / fold_name / "model.pt"
        if not checkpoint.exists():
            print(f"[warn] Checkpoint missing for {fold_name}: {checkpoint}")
            continue
        assigned_gpu = gpu_ids[fold_idx % len(gpu_ids)]
        fold_specs.append({
            "fold_name": fold_name,
            "checkpoint": checkpoint,
            "eval_dir": ARTIFACT_ROOT / "eval_per_fold" / fold_name,
            "gpu": assigned_gpu,
        })

    if not args.skip_eval:
        procs: list[tuple[str, subprocess.Popen]] = []
        for spec in fold_specs:
            eval_args = [
                "--config", str(config_path),
                "--checkpoint", str(spec["checkpoint"]),
                "--graph-cache", str(graph_cache_path),
                "--output-dir", str(spec["eval_dir"]),
                "--title", f"food_safety cross-domain — {spec['fold_name']}",
                "--device", args.device,
            ]
            proc = run_background(
                EVAL_SCRIPT,
                eval_args,
                env_name=args.env,
                cuda=spec["gpu"],
                log_path=log_dir / f"eval_{spec['fold_name']}.log",
            )
            procs.append((spec["fold_name"], proc))

        print(f"\n[wait] Waiting for {len(procs)} fold eval(s) to finish ...")
        failed = []
        for fold_name, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failed.append(fold_name)
                print(f"[error] {fold_name} exited with code {rc}")
            else:
                print(f"[done]  {fold_name}")
        if failed:
            sys.exit(f"[error] These folds failed: {failed}")
    else:
        print("[skip] eval")

    # ── 5. Aggregate ─────────────────────────────────────────────────────────
    fold_metrics: list[dict[str, Any]] = []
    for spec in fold_specs:
        metrics_file = spec["eval_dir"] / "metrics.json"
        if metrics_file.exists():
            m = load_json(metrics_file)
            overall = m.get("overall", {})
            fold_metrics.append({
                "fold": spec["fold_name"],
                "gpu": spec["gpu"],
                "n_cases": m.get("n_cases"),
                "accuracy": overall.get("accuracy"),
                "macro_f1": overall.get("macro_f1"),
                "micro_f1": overall.get("micro_f1"),
                "roc_auc": overall.get("roc_auc"),
            })
        else:
            print(f"[warn] metrics.json not found for {spec['fold_name']}")

    aggregate = aggregate_fold_metrics([
        {k: v for k, v in r.items() if k not in ("fold", "gpu")} for r in fold_metrics
    ])

    summary = {
        "domain": "food_safety",
        "model": "cross_bucket_party_args_lr_decay_kfold",
        "gpus_used": gpu_ids,
        "n_folds_evaluated": len(fold_metrics),
        "food_safety_dir": str(FOOD_SAFETY_DIR),
        "graph_cache": str(graph_cache_path),
        "per_fold": fold_metrics,
        "aggregate": aggregate,
    }
    summary_path = ARTIFACT_ROOT / "cross_domain_summary.json"
    dump_json(summary, summary_path)

    print("\n" + "=" * 60)
    print("CROSS-DOMAIN TEST — food_safety")
    print("=" * 60)
    for row in fold_metrics:
        print(
            f"  {row['fold']} (GPU {row['gpu']})  n={row['n_cases']}  "
            f"acc={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}  "
            f"roc_auc={row.get('roc_auc', float('nan')):.4f}"
        )
    print("-" * 60)
    for key, val in aggregate.items():
        print(f"  {key}: {val:.4f}")
    print(f"\n[done] Summary → {summary_path}")


if __name__ == "__main__":
    main()
