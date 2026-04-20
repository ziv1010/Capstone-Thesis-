#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
UPDATED_GRAPH_ROOT = THIS_DIR.parent
PROJECT_ROOT = UPDATED_GRAPH_ROOT.parent
for path in (UPDATED_GRAPH_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils.io import deep_merge_dict, dump_json, dump_yaml, ensure_dir, load_json, load_yaml


DEFAULT_DATASET_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_cases_remaining_after_8k_each_mistral"
)
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_remaining_after_8k_each_mistral_reasoning"
    / "checkpoint_eval"
)
DEFAULT_HASHING_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_8k_each_mistral_reasoning"
    / "multi_embed_test"
    / "variants"
    / "hashing"
    / "outputs"
    / "models"
    / "reasoning_multi_embed_test__hashing"
)
DEFAULT_BGE_M3_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_8k_each_mistral_reasoning"
    / "multi_embed_test"
    / "variants"
    / "bge_m3"
    / "outputs"
    / "models"
    / "reasoning_multi_embed_test__bge_m3"
)
DEFAULT_E5_LARGE_V2_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_8k_each_mistral_reasoning"
    / "multi_embed_test"
    / "variants"
    / "e5_large_v2"
    / "outputs"
    / "models"
    / "reasoning_multi_embed_test__e5_large_v2"
)
DEFAULT_INCASELAWBERT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_8k_each_mistral_reasoning"
    / "multi_embed_test"
    / "variants"
    / "incaselawbert"
    / "outputs"
    / "models"
    / "reasoning_multi_embed_test__incaselawbert"
)
DEFAULT_INLEGALBERT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "cross_bucket_cases_8k_each_mistral_reasoning"
    / "multi_embed_test"
    / "variants"
    / "inlegalbert"
    / "outputs"
    / "models"
    / "reasoning_multi_embed_test__inlegalbert"
)
DEFAULT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

PREPROCESS_SCRIPT = UPDATED_GRAPH_ROOT / "fixed_open_pipeline" / "preprocess_fixed_open.py"
BUILD_GRAPH_SCRIPT = UPDATED_GRAPH_ROOT / "build_graph.py"
EVAL_SCRIPT = PROJECT_ROOT / "dump2" / "scripts" / "evaluate_saved_model.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the remainder cross-bucket dataset graph for the selected encoder variants and "
            "evaluate saved checkpoints on the full held-out dataset."
        )
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--env-name", "--env", default="thesis_work")
    parser.add_argument("--cuda-visible-devices", default=DEFAULT_CUDA_VISIBLE_DEVICES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=["hashing", "bge_m3"])
    parser.add_argument("--hashing-checkpoint-dir", default=str(DEFAULT_HASHING_CHECKPOINT_DIR))
    parser.add_argument("--bge-m3-checkpoint-dir", default=str(DEFAULT_BGE_M3_CHECKPOINT_DIR))
    parser.add_argument("--e5-large-v2-checkpoint-dir", default=str(DEFAULT_E5_LARGE_V2_CHECKPOINT_DIR))
    parser.add_argument("--incaselawbert-checkpoint-dir", default=str(DEFAULT_INCASELAWBERT_CHECKPOINT_DIR))
    parser.add_argument("--inlegalbert-checkpoint-dir", default=str(DEFAULT_INLEGALBERT_CHECKPOINT_DIR))
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=None,
        help="Optional batch size override for non-hashing text encoders.",
    )
    parser.add_argument(
        "--disable-multi-process",
        action="store_true",
        help="Disable sentence-transformers multi-process encoding for dense encoder variants.",
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def run_python_script(
    script_path: Path,
    script_args: list[str],
    *,
    env_name: str,
    cuda_visible_devices: str,
) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        "micromamba",
        "run",
        "-n",
        env_name,
        "python",
        str(script_path),
        *script_args,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(UPDATED_GRAPH_ROOT), env=env, check=True)


def dataset_slug(dataset_dir: Path) -> str:
    return "".join(character if character.isalnum() else "_" for character in dataset_dir.name).strip("_")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def checkpoint_map(parsed_args: argparse.Namespace) -> dict[str, Path]:
    return {
        "hashing": Path(parsed_args.hashing_checkpoint_dir).resolve(),
        "bge_m3": Path(parsed_args.bge_m3_checkpoint_dir).resolve(),
        "e5_large_v2": Path(parsed_args.e5_large_v2_checkpoint_dir).resolve(),
        "incaselawbert": Path(parsed_args.incaselawbert_checkpoint_dir).resolve(),
        "inlegalbert": Path(parsed_args.inlegalbert_checkpoint_dir).resolve(),
    }


def build_configs(
    *,
    selected_variants: list[str],
    checkpoints: dict[str, Path],
    dataset_dir: Path,
    artifact_root: Path,
    encoder_batch_size: int | None,
    disable_multi_process: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    data_slug = dataset_slug(dataset_dir)
    configs_dir = ensure_dir(artifact_root / "configs")
    shared_root = artifact_root / "shared_preprocess"
    shared_processed = shared_root / "processed"
    shared_cleaned = shared_processed / "cleaned_cases"
    shared_normalized = shared_processed / "normalized_entities"
    shared_audits = shared_root / "audits"
    shared_outputs = shared_root / "outputs"

    if not selected_variants:
        raise ValueError("No variants selected.")

    base_variant = selected_variants[0]
    base_checkpoint_dir = checkpoints[base_variant]
    base_run_config = load_yaml(base_checkpoint_dir / "run_config_snapshot.yaml")
    preprocess_override = {
        "project": {
            "name": f"{base_run_config.get('project', {}).get('name', 'checkpoint_eval')}_{data_slug}_preprocess",
        },
        "paths": {
            "raw_json_dir": str(dataset_dir),
            "processed_dir": str(shared_processed),
            "cleaned_case_dir": str(shared_cleaned),
            "normalized_entity_dir": str(shared_normalized),
            "audits_dir": str(shared_audits),
            "outputs_dir": str(shared_outputs),
        },
    }
    preprocess_cfg = deep_merge_dict(base_run_config, preprocess_override)
    preprocess_config_path = configs_dir / "preprocess_shared.yaml"
    dump_yaml(preprocess_cfg, preprocess_config_path)

    variant_specs: list[dict[str, Any]] = []
    for variant_name in selected_variants:
        checkpoint_dir = checkpoints[variant_name]
        config_template_path = checkpoint_dir / "run_config_snapshot.yaml"
        checkpoint_path = checkpoint_dir / "model.pt"
        training_metrics_path = checkpoint_dir / "metrics.json"
        if not config_template_path.exists():
            raise FileNotFoundError(f"Missing config snapshot for {variant_name}: {config_template_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {variant_name}: {checkpoint_path}")
        if not training_metrics_path.exists():
            raise FileNotFoundError(f"Missing training metrics for {variant_name}: {training_metrics_path}")

        template_cfg = load_yaml(config_template_path)
        variant_root = artifact_root / "variants" / variant_name
        variant_outputs = variant_root / "outputs"
        variant_graph_cache = variant_root / "graph_cache"
        variant_embeddings = variant_root / "embeddings_cache"
        eval_dir = variant_outputs / "checkpoint_eval" / data_slug

        original_cache_name = str(template_cfg.get("graph", {}).get("cache_name", f"{variant_name}.pt"))
        cache_path = Path(original_cache_name)
        cache_stem = cache_path.stem
        cache_suffix = cache_path.suffix or ".pt"
        variant_override = {
            "project": {
                "name": f"{template_cfg.get('project', {}).get('name', 'checkpoint_eval')}_{data_slug}",
            },
            "paths": {
                "raw_json_dir": str(dataset_dir),
                "processed_dir": str(shared_processed),
                "cleaned_case_dir": str(shared_cleaned),
                "normalized_entity_dir": str(shared_normalized),
                "audits_dir": str(shared_audits),
                "embeddings_cache_dir": str(variant_embeddings),
                "graph_cache_dir": str(variant_graph_cache),
                "outputs_dir": str(variant_outputs),
            },
            "graph": {
                "cache_name": f"{cache_stem}.{data_slug}{cache_suffix}",
                "debug_sample_size": 1,
            },
        }
        text_encoder_cfg = dict(template_cfg.get("features", {}).get("text_encoder", {}))
        if str(text_encoder_cfg.get("backend", "")).lower() != "hashing":
            if encoder_batch_size is not None:
                variant_override.setdefault("features", {}).setdefault("text_encoder", {})
                variant_override["features"]["text_encoder"]["batch_size"] = encoder_batch_size
            if disable_multi_process:
                variant_override.setdefault("features", {}).setdefault("text_encoder", {})
                variant_override["features"]["text_encoder"]["multi_process"] = False
        variant_cfg = deep_merge_dict(template_cfg, variant_override)
        config_path = configs_dir / f"{variant_name}.yaml"
        dump_yaml(variant_cfg, config_path)

        variant_specs.append(
            {
                "variant_name": variant_name,
                "config_path": config_path,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_path": checkpoint_path,
                "training_metrics_path": training_metrics_path,
                "graph_cache_path": variant_graph_cache / str(variant_cfg["graph"]["cache_name"]),
                "eval_dir": eval_dir,
            }
        )

    manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "preprocess_script_path": str(PREPROCESS_SCRIPT.resolve()),
        "build_graph_script_path": str(BUILD_GRAPH_SCRIPT.resolve()),
        "evaluate_script_path": str(EVAL_SCRIPT.resolve()),
        "preprocess_config_path": str(preprocess_config_path.resolve()),
        "variants": [
            {
                "variant_name": spec["variant_name"],
                "config_path": str(Path(spec["config_path"]).resolve()),
                "checkpoint_dir": str(Path(spec["checkpoint_dir"]).resolve()),
                "checkpoint_path": str(Path(spec["checkpoint_path"]).resolve()),
                "graph_cache_path": str(Path(spec["graph_cache_path"]).resolve()),
                "eval_dir": str(Path(spec["eval_dir"]).resolve()),
            }
            for spec in variant_specs
        ],
    }
    dump_json(manifest, configs_dir / "run_manifest.json")
    return preprocess_config_path, variant_specs


def collect_results(variant_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for spec in variant_specs:
        training_metrics = load_json(spec["training_metrics_path"])
        eval_metrics = load_json(Path(spec["eval_dir"]) / "metrics.json")
        overall = dict(eval_metrics.get("overall", {}))
        original_test = dict(training_metrics.get("test", {}))

        overall_rows.append(
            {
                "variant": spec["variant_name"],
                "id_test_accuracy": fmt_metric(original_test.get("accuracy")),
                "id_test_macro_f1": fmt_metric(original_test.get("macro_f1")),
                "ood_n_cases": int(eval_metrics.get("n_cases", 0)),
                "ood_accuracy": fmt_metric(overall.get("accuracy")),
                "ood_macro_f1": fmt_metric(overall.get("macro_f1")),
                "ood_micro_f1": fmt_metric(overall.get("micro_f1")),
                "ood_roc_auc": fmt_metric(overall.get("roc_auc")),
                "ood_pr_auc": fmt_metric(overall.get("pr_auc")),
                "checkpoint_dir": str(Path(spec["checkpoint_dir"]).resolve()),
                "eval_dir": str(Path(spec["eval_dir"]).resolve()),
            }
        )

        per_category = dict(eval_metrics.get("per_category", {}))
        for category, metrics in sorted(per_category.items()):
            category_rows.append(
                {
                    "variant": spec["variant_name"],
                    "category": category,
                    "n_samples": int(metrics.get("n_samples", 0)),
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                    "macro_f1": fmt_metric(metrics.get("macro_f1")),
                    "micro_f1": fmt_metric(metrics.get("micro_f1")),
                    "roc_auc": fmt_metric(metrics.get("roc_auc")),
                    "pr_auc": fmt_metric(metrics.get("pr_auc")),
                }
            )

    overall_rows.sort(key=lambda row: float(row["ood_macro_f1"] or 0.0), reverse=True)
    category_rows.sort(key=lambda row: (row["category"], -float(row["macro_f1"] or 0.0), row["variant"]))
    return overall_rows, category_rows


def main() -> None:
    parsed_args = parse_args()
    dataset_dir = Path(parsed_args.dataset_dir).resolve()
    artifact_root = Path(parsed_args.artifact_root).resolve()
    checkpoints = checkpoint_map(parsed_args)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for variant_name in parsed_args.variants:
        if variant_name not in checkpoints:
            raise ValueError(f"Unsupported variant requested: {variant_name}")

    preprocess_config_path, variant_specs = build_configs(
        selected_variants=parsed_args.variants,
        checkpoints=checkpoints,
        dataset_dir=dataset_dir,
        artifact_root=artifact_root,
        encoder_batch_size=parsed_args.encoder_batch_size,
        disable_multi_process=parsed_args.disable_multi_process,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.cuda_visible_devices
    print(f"[info] Dataset dir: {dataset_dir}")
    print(f"[info] Artifact root: {artifact_root}")
    print(f"[info] CUDA_VISIBLE_DEVICES={parsed_args.cuda_visible_devices}")
    print(f"[info] Variants: {parsed_args.variants}")

    if not parsed_args.summarize_only:
        if not parsed_args.skip_preprocess:
            preprocess_args = ["--config", str(preprocess_config_path)]
            if parsed_args.limit is not None:
                preprocess_args.extend(["--limit", str(parsed_args.limit)])
            run_python_script(
                PREPROCESS_SCRIPT,
                preprocess_args,
                env_name=parsed_args.env_name,
                cuda_visible_devices=parsed_args.cuda_visible_devices,
            )

        for spec in variant_specs:
            if not parsed_args.skip_build:
                build_args = ["--config", str(spec["config_path"])]
                if parsed_args.limit is not None:
                    build_args.extend(["--limit", str(parsed_args.limit)])
                run_python_script(
                    BUILD_GRAPH_SCRIPT,
                    build_args,
                    env_name=parsed_args.env_name,
                    cuda_visible_devices=parsed_args.cuda_visible_devices,
                )

            if not parsed_args.skip_eval:
                eval_args = [
                    "--config",
                    str(spec["config_path"]),
                    "--checkpoint",
                    str(spec["checkpoint_path"]),
                    "--graph-cache",
                    str(spec["graph_cache_path"]),
                    "--output-dir",
                    str(spec["eval_dir"]),
                    "--title",
                    f"{spec['variant_name']} on {dataset_dir.name}",
                    "--device",
                    parsed_args.device,
                ]
                run_python_script(
                    EVAL_SCRIPT,
                    eval_args,
                    env_name=parsed_args.env_name,
                    cuda_visible_devices=parsed_args.cuda_visible_devices,
                )

    overall_rows, category_rows = collect_results(variant_specs)
    summary_dir = ensure_dir(artifact_root / "summary")
    write_csv(overall_rows, summary_dir / "overall_comparison.csv")
    write_markdown(overall_rows, summary_dir / "overall_comparison.md")
    write_csv(category_rows, summary_dir / "category_comparison.csv")
    write_markdown(category_rows, summary_dir / "category_comparison.md")
    dump_json(
        {
            "dataset_dir": str(dataset_dir),
            "artifact_root": str(artifact_root),
            "overall_rows": overall_rows,
            "category_rows": category_rows,
        },
        summary_dir / "comparison_summary.json",
    )
    print(f"[done] Summary written under {summary_dir}")


if __name__ == "__main__":
    main()
