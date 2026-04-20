#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
UPDATED_GRAPH_ROOT = THIS_DIR.parent
PROJECT_ROOT = UPDATED_GRAPH_ROOT.parent
DEFAULT_BASE_CONFIG = UPDATED_GRAPH_ROOT / "fixed_open_pipeline" / "fixed_open_reasoning_config.yaml"
DEFAULT_VARIANTS_CONFIG = THIS_DIR / "encoder_variants.yaml"
DEFAULT_MICROMAMBA_PREFIX = (
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star"
)
DEFAULT_CUDA_VISIBLE_DEVICES = "4,5,6,7"
PREPROCESS_SCRIPT = UPDATED_GRAPH_ROOT / "fixed_open_pipeline" / "preprocess_fixed_open.py"
BUILD_GRAPH_SCRIPT = UPDATED_GRAPH_ROOT / "build_graph.py"
TRAIN_SCRIPT = PROJECT_ROOT / "dump2" / "scripts" / "train_gnn.py"
PLUS_MINUS = "\u00B1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML at {path} must parse to a mapping.")
    return data


def dump_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-open reasoning GNN multi-encoder test while keeping the split and graph "
            "configuration fixed, while varying only the training seed across repeated runs."
        )
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--variants-config", default=str(DEFAULT_VARIANTS_CONFIG))
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--micromamba-prefix", default=DEFAULT_MICROMAMBA_PREFIX)
    parser.add_argument("--run-prefix", default="reasoning_multi_embed_test")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES),
        help="Comma-separated GPU ids available to the experiment runner.",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=1,
        help="Number of training repeats per encoder variant. Splits and graph caches stay fixed.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="Stride used to advance the training seed across repeated runs.",
    )
    parser.add_argument(
        "--sequential-variants",
        action="store_true",
        help="Disable parallel per-variant execution and run build/train serially.",
    )
    return parser.parse_args()


def default_artifact_root(base_cfg: dict[str, Any]) -> Path:
    outputs_dir = Path(str(base_cfg.get("paths", {}).get("outputs_dir")))
    return outputs_dir / "multi_embed_test"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
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


def format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def format_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None or std_value is None:
        return ""
    return f"{float(mean_value):.6f} {PLUS_MINUS} {float(std_value):.6f}"


def compute_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def resolve_default_variants(variants_payload: dict[str, Any], available_variants: list[str]) -> list[str]:
    configured_defaults = variants_payload.get("default_variants")
    if configured_defaults is None:
        return available_variants
    if not isinstance(configured_defaults, list):
        raise TypeError("default_variants must be a list when provided.")

    resolved_defaults = [str(item) for item in configured_defaults]
    missing = [name for name in resolved_defaults if name not in available_variants]
    if missing:
        raise KeyError(f"default_variants contains unknown entries: {missing}")
    return resolved_defaults


def run_python_script(
    script_path: Path,
    script_args: list[str],
    *,
    micromamba_prefix: str,
    cuda_visible_devices: str,
) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        "micromamba",
        "run",
        "-p",
        micromamba_prefix,
        "python",
        str(script_path),
        *script_args,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(UPDATED_GRAPH_ROOT), env=env, check=True)


def parse_gpu_ids(cuda_visible_devices: str) -> list[str]:
    gpu_ids = [item.strip() for item in str(cuda_visible_devices).split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids were provided via --cuda-visible-devices.")
    return gpu_ids


def run_variant_pipeline(
    spec: dict[str, Any],
    *,
    micromamba_prefix: str,
    cuda_visible_devices: str,
    limit: int | None,
    skip_build: bool,
    skip_train: bool,
) -> None:
    config_path = Path(spec["config_path"])
    variant_name = str(spec["variant_name"])
    print(f"[variant:{variant_name}] assigned CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    if not skip_build:
        build_args = ["--config", str(config_path)]
        if limit is not None:
            build_args.extend(["--limit", str(limit)])
        run_python_script(
            BUILD_GRAPH_SCRIPT,
            build_args,
            micromamba_prefix=micromamba_prefix,
            cuda_visible_devices=cuda_visible_devices,
        )

    if not skip_train:
        train_args = [
            "--config",
            str(config_path),
            "--run-name",
            str(spec["run_name"]),
        ]
        run_python_script(
            TRAIN_SCRIPT,
            train_args,
            micromamba_prefix=micromamba_prefix,
            cuda_visible_devices=cuda_visible_devices,
        )


def run_variant_pipelines_parallel(
    variant_specs: list[dict[str, Any]],
    *,
    gpu_ids: list[str],
    micromamba_prefix: str,
    limit: int | None,
    skip_build: bool,
    skip_train: bool,
) -> None:
    if not variant_specs:
        return

    work_queue: Queue[dict[str, Any]] = Queue()
    for spec in variant_specs:
        work_queue.put(spec)

    errors: list[tuple[str, str, BaseException]] = []
    errors_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                spec = work_queue.get_nowait()
            except Empty:
                return

            variant_name = str(spec["variant_name"])
            try:
                run_variant_pipeline(
                    spec,
                    micromamba_prefix=micromamba_prefix,
                    cuda_visible_devices=gpu_id,
                    limit=limit,
                    skip_build=skip_build,
                    skip_train=skip_train,
                )
            except BaseException as exc:
                with errors_lock:
                    errors.append((gpu_id, variant_name, exc))
                return
            finally:
                work_queue.task_done()

    active_gpu_ids = gpu_ids[: max(1, min(len(gpu_ids), len(variant_specs)))]
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"variant-gpu-{gpu_id}", daemon=False)
        for gpu_id in active_gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        gpu_id, variant_name, exc = errors[0]
        raise RuntimeError(
            f"Variant pipeline failed for '{variant_name}' on GPU {gpu_id}: {exc}"
        ) from exc


def build_configs(
    *,
    base_cfg: dict[str, Any],
    base_config_path: Path,
    variants_payload: dict[str, Any],
    selected_variants: list[str],
    artifact_root: Path,
    run_prefix: str,
    repeat_runs: int,
    seed_stride: int,
) -> tuple[Path, list[dict[str, Any]]]:
    configs_dir = artifact_root / "configs"
    shared_root = artifact_root / "shared_preprocess"
    shared_processed = shared_root / "processed"
    shared_cleaned = shared_processed / "cleaned_cases"
    shared_normalized = shared_processed / "normalized_entities"
    shared_audits = shared_root / "audits"
    shared_outputs = shared_root / "outputs"

    preprocess_override = {
        "project": {
            "name": f"{base_cfg.get('project', {}).get('name', 'fixed_open')}_multi_embed_test_preprocess",
        },
        "paths": {
            "processed_dir": str(shared_processed),
            "cleaned_case_dir": str(shared_cleaned),
            "normalized_entity_dir": str(shared_normalized),
            "audits_dir": str(shared_audits),
            "outputs_dir": str(shared_outputs),
        },
    }
    preprocess_cfg = deep_merge_dict(base_cfg, preprocess_override)
    preprocess_config_path = configs_dir / "preprocess_shared.yaml"
    dump_yaml(preprocess_cfg, preprocess_config_path)

    base_cache_name = str(base_cfg.get("graph", {}).get("cache_name", "reasoning_focused_graph.pt"))
    cache_stem = Path(base_cache_name).stem
    cache_suffix = Path(base_cache_name).suffix or ".pt"
    variant_specs: list[dict[str, Any]] = []

    variants_cfg = dict(variants_payload.get("variants", {}))
    for variant_name in selected_variants:
        if variant_name not in variants_cfg:
            raise KeyError(f"Unknown encoder variant: {variant_name}")
        variant_payload = dict(variants_cfg[variant_name])
        variant_root = artifact_root / "variants" / variant_name
        variant_outputs = variant_root / "outputs"
        variant_graph_cache = variant_root / "graph_cache"
        variant_embeddings = variant_root / "embeddings_cache"
        run_name = f"{run_prefix}__{variant_name}"
        override = {
            "project": {
                "name": f"{base_cfg.get('project', {}).get('name', 'fixed_open')}_{variant_name}",
            },
            "paths": {
                "processed_dir": str(shared_processed),
                "cleaned_case_dir": str(shared_cleaned),
                "normalized_entity_dir": str(shared_normalized),
                "audits_dir": str(shared_audits),
                "embeddings_cache_dir": str(variant_embeddings),
                "graph_cache_dir": str(variant_graph_cache),
                "outputs_dir": str(variant_outputs),
            },
            "graph": {
                # Keep graph debug artifacts small for quick encoder comparisons.
                "cache_name": f"{cache_stem}.{variant_name}{cache_suffix}",
                "debug_sample_size": 1,
            },
            "features": dict(variant_payload.get("features", {})),
            "training": {
                "repeat_runs": repeat_runs,
                "seed_stride": seed_stride,
            },
        }
        variant_cfg = deep_merge_dict(base_cfg, override)
        config_path = configs_dir / f"{variant_name}.yaml"
        dump_yaml(variant_cfg, config_path)
        variant_specs.append(
            {
                "variant_name": variant_name,
                "description": str(variant_payload.get("description", "")),
                "config_path": config_path,
                "root": variant_root,
                "graph_cache_dir": variant_graph_cache,
                "outputs_dir": variant_outputs,
                "run_name": run_name,
                "graph_cache_name": variant_cfg.get("graph", {}).get("cache_name"),
                "text_encoder": variant_cfg.get("features", {}).get("text_encoder", {}),
            }
        )

    manifest = {
        "base_config_path": str(base_config_path.resolve()),
        "preprocess_script_path": str(PREPROCESS_SCRIPT.resolve()),
        "build_graph_script_path": str(BUILD_GRAPH_SCRIPT.resolve()),
        "train_script_path": str(TRAIN_SCRIPT.resolve()),
        "generated_preprocess_config": str(preprocess_config_path.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selected_variants": selected_variants,
        "project_seed": base_cfg.get("project", {}).get("seed"),
        "split_random_state": base_cfg.get("splits", {}).get("random_state"),
        "training_repeat_runs": repeat_runs,
        "training_seed_stride": seed_stride,
        "shared_cleaned_case_dir": str(shared_cleaned.resolve()),
        "variant_specs": [
            {
                "variant_name": spec["variant_name"],
                "description": spec["description"],
                "config_path": str(Path(spec["config_path"]).resolve()),
                "root": str(Path(spec["root"]).resolve()),
                "run_name": spec["run_name"],
                "text_encoder": spec["text_encoder"],
            }
            for spec in variant_specs
        ],
    }
    dump_json(manifest, artifact_root / "configs" / "run_manifest.json")
    return preprocess_config_path, variant_specs


def summarize_training_runs(run_dir: Path) -> dict[str, Any]:
    multi_run_summary_path = run_dir / "multi_run_summary.json"
    if multi_run_summary_path.exists():
        multi_run_summary = load_json(multi_run_summary_path)
        run_summaries = list(multi_run_summary.get("runs") or [])
        if not run_summaries:
            raise RuntimeError(f"Repeated-run summary exists but contains no runs: {multi_run_summary_path}")

        best_run_label = str(multi_run_summary.get("best_run_label", ""))
        best_run = next(
            (run for run in run_summaries if str(run.get("run_label", "")) == best_run_label),
            None,
        )
        if best_run is None:
            raise RuntimeError(
                f"Could not find best run '{best_run_label}' inside {multi_run_summary_path}"
            )

        val_macro_f1_values = [float((run.get("val") or {}).get("macro_f1", 0.0)) for run in run_summaries]
        test_macro_f1_values = [float((run.get("test") or {}).get("macro_f1", 0.0)) for run in run_summaries]
        val_mean, val_std = compute_mean_std(val_macro_f1_values)
        test_mean, test_std = compute_mean_std(test_macro_f1_values)

        return {
            "repeat_runs": int(multi_run_summary.get("repeat_runs", len(run_summaries))),
            "base_seed": multi_run_summary.get("base_seed"),
            "seed_stride": multi_run_summary.get("seed_stride"),
            "val_macro_f1_mean": val_mean,
            "val_macro_f1_std": val_std,
            "test_macro_f1_mean": test_mean,
            "test_macro_f1_std": test_std,
            "best_run_label": best_run_label,
            "best_seed": best_run.get("seed"),
            "best_epoch": best_run.get("best_epoch"),
            "best_val_macro_f1": float((best_run.get("val") or {}).get("macro_f1", 0.0)),
            "best_test_macro_f1": float((best_run.get("test") or {}).get("macro_f1", 0.0)),
        }

    metrics = load_json(run_dir / "metrics.json")
    val_macro_f1 = float((metrics.get("val") or {}).get("macro_f1", 0.0))
    test_macro_f1 = float((metrics.get("test") or {}).get("macro_f1", 0.0))
    return {
        "repeat_runs": 1,
        "base_seed": None,
        "seed_stride": None,
        "val_macro_f1_mean": val_macro_f1,
        "val_macro_f1_std": 0.0,
        "test_macro_f1_mean": test_macro_f1,
        "test_macro_f1_std": 0.0,
        "best_run_label": "run_01_seed_single",
        "best_seed": None,
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_macro_f1": val_macro_f1,
        "best_test_macro_f1": test_macro_f1,
    }


def collect_variant_results(variant_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {"baseline_variant": None, "variants": {}}

    baseline_split_assignments: Any | None = None
    baseline_node_counts: Any | None = None
    baseline_relation_mappings: Any | None = None
    baseline_case_split_counts: Any | None = None

    for spec in variant_specs:
        graph_cache_dir = Path(spec["graph_cache_dir"])
        outputs_dir = Path(spec["outputs_dir"])
        run_dir = outputs_dir / "models" / str(spec["run_name"])

        graph_metadata = load_json(graph_cache_dir / "graph_metadata.reasoning_focused.json")
        split_assignments = load_json(graph_cache_dir / "split_assignments.reasoning_focused.json")
        relation_mappings = load_json(graph_cache_dir / "relation_mappings.reasoning_focused.json")
        text_encoder = dict(spec.get("text_encoder", {}))
        training_summary = summarize_training_runs(run_dir)

        if baseline_split_assignments is None:
            checks["baseline_variant"] = spec["variant_name"]
            baseline_split_assignments = split_assignments
            baseline_node_counts = graph_metadata.get("node_counts")
            baseline_relation_mappings = relation_mappings
            baseline_case_split_counts = graph_metadata.get("case_split_counts")

        variant_check = {
            "same_split_assignments_as_baseline": split_assignments == baseline_split_assignments,
            "same_node_counts_as_baseline": graph_metadata.get("node_counts") == baseline_node_counts,
            "same_relation_mappings_as_baseline": relation_mappings == baseline_relation_mappings,
            "same_case_split_counts_as_baseline": (
                graph_metadata.get("case_split_counts") == baseline_case_split_counts
            ),
        }
        checks["variants"][spec["variant_name"]] = variant_check

        rows.append(
            {
                "variant": spec["variant_name"],
                "backend": text_encoder.get("backend", ""),
                "model_name": text_encoder.get("model_name", ""),
                "repeat_runs": training_summary.get("repeat_runs", 1),
                "train_seed_base": training_summary.get("base_seed", ""),
                "train_seed_stride": training_summary.get("seed_stride", ""),
                "embedding_dim": graph_metadata.get("embedding_dim", ""),
                "feature_dim": graph_metadata.get("feature_dim", ""),
                "val_macro_f1_mean": format_float(training_summary.get("val_macro_f1_mean")),
                "val_macro_f1_std": format_float(training_summary.get("val_macro_f1_std")),
                "val_macro_f1_mean_std": format_mean_std(
                    training_summary.get("val_macro_f1_mean"),
                    training_summary.get("val_macro_f1_std"),
                ),
                "test_macro_f1_mean": format_float(training_summary.get("test_macro_f1_mean")),
                "test_macro_f1_std": format_float(training_summary.get("test_macro_f1_std")),
                "test_macro_f1_mean_std": format_mean_std(
                    training_summary.get("test_macro_f1_mean"),
                    training_summary.get("test_macro_f1_std"),
                ),
                "best_run_label": training_summary.get("best_run_label", ""),
                "best_seed": training_summary.get("best_seed", ""),
                "best_epoch": training_summary.get("best_epoch", ""),
                "best_val_macro_f1": format_float(training_summary.get("best_val_macro_f1")),
                "best_test_macro_f1": format_float(training_summary.get("best_test_macro_f1")),
            }
        )

    return rows, checks


def validate_checks(checks: dict[str, Any]) -> None:
    for variant_name, variant_check in checks.get("variants", {}).items():
        if not all(bool(value) for value in variant_check.values()):
            raise RuntimeError(
                f"Consistency check failed for variant '{variant_name}': {variant_check}"
            )


def main() -> None:
    parsed_args = parse_args()
    if parsed_args.repeat_runs < 1:
        raise ValueError("--repeat-runs must be at least 1.")
    if parsed_args.seed_stride < 1:
        raise ValueError("--seed-stride must be at least 1.")
    gpu_ids = parse_gpu_ids(parsed_args.cuda_visible_devices)

    base_config_path = Path(parsed_args.base_config).resolve()
    variants_config_path = Path(parsed_args.variants_config).resolve()
    base_cfg = load_yaml(base_config_path)
    variants_payload = load_yaml(variants_config_path)
    available_variants = list((variants_payload.get("variants") or {}).keys())
    default_variants = resolve_default_variants(variants_payload, available_variants)
    selected_variants = parsed_args.variants or default_variants
    artifact_root = (
        Path(parsed_args.artifact_root).resolve()
        if parsed_args.artifact_root
        else default_artifact_root(base_cfg)
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.cuda_visible_devices
    print(f"[info] CUDA_VISIBLE_DEVICES is pinned to {parsed_args.cuda_visible_devices}")
    print(f"[info] Parsed GPU ids: {gpu_ids}")
    print(f"[info] Using preprocess script: {PREPROCESS_SCRIPT}")
    print(f"[info] Using build graph script: {BUILD_GRAPH_SCRIPT}")
    print(f"[info] Using train script: {TRAIN_SCRIPT}")
    print(
        "[info] Each build/train subprocess is restricted to its assigned visible GPU. "
        "The training code still uses torch.device('cuda'), so each subprocess uses the first "
        "and only visible CUDA device."
    )

    preprocess_config_path, variant_specs = build_configs(
        base_cfg=base_cfg,
        base_config_path=base_config_path,
        variants_payload=variants_payload,
        selected_variants=selected_variants,
        artifact_root=artifact_root,
        run_prefix=parsed_args.run_prefix,
        repeat_runs=parsed_args.repeat_runs,
        seed_stride=parsed_args.seed_stride,
    )

    if not parsed_args.summarize_only:
        if not parsed_args.skip_preprocess:
            preprocess_args = ["--config", str(preprocess_config_path)]
            if parsed_args.limit is not None:
                preprocess_args.extend(["--limit", str(parsed_args.limit)])
            run_python_script(
                PREPROCESS_SCRIPT,
                preprocess_args,
                micromamba_prefix=parsed_args.micromamba_prefix,
                cuda_visible_devices=parsed_args.cuda_visible_devices,
            )

        if len(variant_specs) == 1:
            # Give a single selected variant the full visible GPU set so encoder-level multi-GPU can activate.
            run_variant_pipeline(
                variant_specs[0],
                micromamba_prefix=parsed_args.micromamba_prefix,
                cuda_visible_devices=parsed_args.cuda_visible_devices,
                limit=parsed_args.limit,
                skip_build=parsed_args.skip_build,
                skip_train=parsed_args.skip_train,
            )
        elif parsed_args.sequential_variants or len(gpu_ids) == 1:
            for spec in variant_specs:
                run_variant_pipeline(
                    spec,
                    micromamba_prefix=parsed_args.micromamba_prefix,
                    cuda_visible_devices=parsed_args.cuda_visible_devices,
                    limit=parsed_args.limit,
                    skip_build=parsed_args.skip_build,
                    skip_train=parsed_args.skip_train,
                )
        else:
            print(
                "[info] Running variant build/train pipelines in parallel across GPUs. "
                "Console output from different variants may interleave."
            )
            run_variant_pipelines_parallel(
                variant_specs,
                gpu_ids=gpu_ids,
                micromamba_prefix=parsed_args.micromamba_prefix,
                limit=parsed_args.limit,
                skip_build=parsed_args.skip_build,
                skip_train=parsed_args.skip_train,
            )

    rows, checks = collect_variant_results(variant_specs)
    validate_checks(checks)

    summary_dir = artifact_root / "summary"
    write_csv(rows, summary_dir / "encoder_macro_f1_comparison.csv")
    write_markdown(rows, summary_dir / "encoder_macro_f1_comparison.md")
    dump_json(rows, summary_dir / "encoder_macro_f1_comparison.json")
    dump_json(checks, summary_dir / "consistency_checks.json")

    print(f"[done] Summary written to {summary_dir}")
    for row in rows:
        print(
            f"[summary] {row['variant']}: "
            f"val_macro_f1={row['val_macro_f1_mean_std']} "
            f"test_macro_f1={row['test_macro_f1_mean_std']} "
            f"best_run={row['best_run_label']} "
            f"best_seed={row['best_seed']}"
        )


if __name__ == "__main__":
    main()
