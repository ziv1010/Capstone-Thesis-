#!/usr/bin/env python3
"""Orchestrate the full paper-faithful NER and RR replications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    collect_gpu_info,
    collect_machine_info,
    configure_logger,
    export_pip_freeze,
    git_commit,
    now_utc_iso,
    read_json,
    run_subprocess,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+", default=["ner", "rr"], choices=["ner", "rr"])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--rr_device", default="cuda:0")
    return parser.parse_args()


def env_exists(env_name: str) -> bool:
    result = run_subprocess(["micromamba", "env", "list"], capture_output=True)
    return env_name in result.stdout


def ensure_rr_environment(root: Path, logger: object) -> None:
    env_name = "opennyai_rr_paper_py38"
    if env_exists(env_name):
        logger.info("micromamba env %s already exists", env_name)
        return

    run_subprocess(["micromamba", "create", "-y", "-n", env_name, "python=3.8", "pip"], logger=logger)
    run_subprocess(
        [
            "micromamba",
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu111",
            "torch==1.9.0+cu111",
            "torchvision==0.10.0+cu111",
        ],
        logger=logger,
    )
    run_subprocess(
        [
            "micromamba",
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "allennlp==2.7.0",
            "transformers==4.9.2",
            "spacy==3.1.2",
            "datasets==1.11.0",
            "scikit-learn==0.24.2",
            "prettytable==2.2.0",
            "sentencepiece==0.1.96",
            "jsonnet==0.17.0",
            "numpy==1.21.2",
        ],
        logger=logger,
    )


def ner_model_installed(root: Path) -> bool:
    result = run_subprocess(
        [
            "micromamba",
            "run",
            "-n",
            "opennyai_py38",
            "python",
            "-c",
            "import spacy; print(sorted(spacy.util.get_installed_models()))",
        ],
        capture_output=True,
    )
    return "en_legal_ner_trf" in result.stdout and "en_core_web_sm" in result.stdout


def ensure_ner_models(root: Path, logger: object) -> None:
    if ner_model_installed(root):
        logger.info("NER spaCy models are already installed in opennyai_py38")
        return

    manifest = read_json(root / "replication_manifest.json")
    ner_wheel = manifest["models"]["en_legal_ner_trf"]["path"]
    preamble_wheel = manifest["models"]["en_core_web_sm_3_2_0"]["path"]
    run_subprocess(
        [
            "micromamba",
            "run",
            "-n",
            "opennyai_py38",
            "python",
            "-m",
            "pip",
            "install",
            ner_wheel,
            preamble_wheel,
        ],
        logger=logger,
    )


def build_summary(root: Path, components: list[str], rr_device: str) -> dict[str, object]:
    manifest = read_json(root / "replication_manifest.json")
    summary = {
        "run_date": now_utc_iso(),
        "machine_info": collect_machine_info(),
        "gpu_info": collect_gpu_info(),
        "repos": manifest["repos"],
        "datasets": manifest["datasets"],
        "models": manifest["models"],
        "paper_targets": {
            "ner_strict_f1": 0.9108,
            "rr_weighted_f1": 0.79,
        },
        "comparison_to_paper": {},
        "fallbacks_used": [],
        "failures": {},
    }
    if "ner" in components:
        ner_metrics = read_json(root / "outputs" / "ner" / "test_metrics.json")
        summary["ner"] = ner_metrics
        summary["comparison_to_paper"]["ner"] = {
            "target_strict_f1": 0.9108,
            "observed_strict_f1": ner_metrics["f1"],
            "delta": ner_metrics["f1"] - 0.9108,
        }
        summary["failures"]["ner"] = ner_metrics.get("failed_documents", [])
    if "rr" in components:
        rr_metrics = read_json(root / "outputs" / "rr" / "test_metrics.json")
        summary["rr"] = rr_metrics
        summary["comparison_to_paper"]["rr"] = {
            "target_weighted_f1": 0.79,
            "observed_weighted_f1": rr_metrics["weighted_f1"],
            "delta": rr_metrics["weighted_f1"] - 0.79,
            "requested_device": rr_device,
            "runtime_device": rr_metrics["runtime"]["runtime_device"],
        }
        if rr_metrics["runtime"].get("fallback_used"):
            summary["fallbacks_used"].append(rr_metrics["runtime"]["fallback_used"])
        summary["failures"]["rr"] = rr_metrics.get("failed_documents", [])
    return summary


def build_summary_markdown(summary: dict[str, object], root: Path, components: list[str], rr_device: str) -> str:
    lines = [
        "# Final Replication Summary",
        "",
        f"- Run date: `{summary['run_date']}`",
        f"- Root: `{root}`",
        f"- NER env: `opennyai_py38`",
        f"- RR env: `opennyai_rr_paper_py38`",
        "",
        "## Commands",
        "",
        f"- `python scripts/fetch_assets.py --root {root}`",
    ]
    if "ner" in components:
        lines.append(
            f"- `micromamba run -n opennyai_py38 python {root / 'scripts' / 'run_ner_paper_replication.py'} "
            f"--dataset_root {root / 'datasets'} --output_dir {root / 'outputs' / 'ner'} --use_gpu --gpu_id 0`"
        )
    if "rr" in components:
        lines.append(
            f"- `micromamba run -n opennyai_rr_paper_py38 python {root / 'scripts' / 'run_rr_paper_replication.py'} "
            f"--dataset_root {root / 'datasets'} --output_dir {root / 'outputs' / 'rr'} --device {rr_device}`"
        )
    lines.extend(["", "## Results", ""])
    if "ner" in components:
        ner = summary["ner"]
        lines.append(
            f"- NER strict F1: `{ner['f1']:.4f}` vs paper target `0.9108` "
            f"(delta `{summary['comparison_to_paper']['ner']['delta']:.4f}`)"
        )
    if "rr" in components:
        rr = summary["rr"]
        lines.append(
            f"- RR weighted F1: `{rr['weighted_f1']:.4f}` vs paper target `0.7900` "
            f"(delta `{summary['comparison_to_paper']['rr']['delta']:.4f}`)"
        )
        lines.append(f"- RR runtime device: `{rr['runtime']['runtime_device']}`")
        if rr["runtime"].get("fallback_used"):
            lines.append("- RR GPU fallback was used.")
    lines.extend(["", "## Repos", ""])
    lines.append(f"- legal_NER commit: `{git_commit(root / 'external' / 'legal_NER')}`")
    lines.append(
        f"- rhetorical-role-baseline commit: `{git_commit(root / 'external' / 'rhetorical-role-baseline')}`"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    logger = configure_logger("run_all_replications", root / "outputs" / "logs" / "run_all.log")
    logger.info("CLI command: %s", " ".join(sys.argv))

    run_subprocess(["python3", str(root / "scripts" / "fetch_assets.py"), "--root", str(root)], logger=logger)
    ensure_ner_models(root, logger)
    export_pip_freeze("opennyai_py38", root / "envs" / "ner_runtime.txt")
    ensure_rr_environment(root, logger)

    if "ner" in args.components:
        run_subprocess(
            [
                "micromamba",
                "run",
                "-n",
                "opennyai_py38",
                "python",
                str(root / "scripts" / "run_ner_paper_replication.py"),
                "--dataset_root",
                str(root / "datasets"),
                "--output_dir",
                str(root / "outputs" / "ner"),
                "--use_gpu",
                "--gpu_id",
                "0",
            ],
            logger=logger,
        )
    if "rr" in args.components:
        run_subprocess(
            [
                "micromamba",
                "run",
                "-n",
                "opennyai_rr_paper_py38",
                "python",
                str(root / "scripts" / "run_rr_paper_replication.py"),
                "--dataset_root",
                str(root / "datasets"),
                "--output_dir",
                str(root / "outputs" / "rr"),
                "--device",
                args.rr_device,
            ],
            logger=logger,
        )

    summary = build_summary(root, list(args.components), args.rr_device)
    write_json(root / "reports" / "final_replication_summary.json", summary)
    write_text(
        root / "reports" / "final_replication_summary.md",
        build_summary_markdown(summary, root, list(args.components), args.rr_device),
    )

    manifest = read_json(root / "replication_manifest.json")
    manifest["run_summary"] = summary
    write_json(root / "replication_manifest.json", manifest)
    logger.info("Wrote final reports to %s", root / "reports")


if __name__ == "__main__":
    main()
