#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


SECTION_GNN = Path(__file__).resolve().parents[2]
BUCKETS = (
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
    "cross_bucket_total_dataset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configs for the central-authority removal ablation.")
    parser.add_argument(
        "--entity-config-root",
        default=str(SECTION_GNN / "ablations" / "entity_resolved_data" / "configs"),
    )
    parser.add_argument(
        "--data-root",
        default=str(SECTION_GNN / "data" / "ablations" / "remove_central_authorities"),
    )
    parser.add_argument(
        "--outputs-root",
        default=str(SECTION_GNN / "outputs" / "ablations" / "remove_central_authorities"),
    )
    parser.add_argument(
        "--config-root",
        default=str(SECTION_GNN / "ablations" / "remove_central_authorities" / "configs"),
    )
    parser.add_argument("--only", choices=["party", "section", "both"], default="both")
    parser.add_argument(
        "--lr-mode",
        choices=["decay", "none"],
        default="decay",
        help="Training schedule to encode in generated configs. Default: decay.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def short_name(bucket: str) -> str:
    if bucket == "cross_bucket_total_dataset":
        return "cross_bucket"
    return bucket.removesuffix("_timed_mistral")


def retarget_config(
    cfg: dict[str, Any],
    *,
    bucket: str,
    data_root: Path,
    outputs_root: Path,
    variant: str,
    lr_mode: str,
) -> dict[str, Any]:
    short = short_name(bucket)
    out = copy.deepcopy(cfg)
    bucket_data_root = data_root / bucket
    bucket_outputs_root = outputs_root / bucket

    if variant == "party":
        project_name = f"{short}_central_authorities_removed_party_args_preamble"
        cache_name = f"case_star_central_authorities_removed_{short}_party_args_preamble_lr_decay.reasoning_focused.pt"
    elif variant == "section":
        project_name = f"{short}_central_authorities_removed_section_sep"
        cache_name = f"case_star_central_authorities_removed_{short}_section_sep_lr_decay.reasoning_focused.pt"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    project_name = f"{project_name}_{'lr_decay' if lr_mode == 'decay' else 'no_lr'}"
    out["project"]["name"] = project_name
    paths = out.setdefault("paths", {})
    paths["processed_dir"] = str(bucket_data_root / "processed")
    paths["cleaned_case_dir"] = str(bucket_data_root / "processed" / "cleaned_cases")
    paths["normalized_entity_dir"] = str(bucket_data_root / "processed" / "normalized_entities")
    paths["embeddings_cache_dir"] = str(bucket_data_root / "embeddings_cache" / variant)
    paths["graph_cache_dir"] = str(bucket_data_root / "graph_cache" / variant)
    paths["audits_dir"] = str(bucket_data_root / "audits")
    paths["outputs_dir"] = str(bucket_outputs_root)
    out.setdefault("graph", {})["cache_name"] = cache_name
    if lr_mode == "none":
        training = out.setdefault("training", {})
        training.update(
            {
                "epochs": 60,
                "use_early_stopping": True,
                "early_stopping_patience": 15,
            }
        )
        for key in ("lr_scheduler", "lr_scheduler_factor", "lr_scheduler_patience", "lr_min"):
            training.pop(key, None)
    return out


def main() -> None:
    args = parse_args()
    entity_config_root = Path(args.entity_config_root)
    data_root = Path(args.data_root)
    outputs_root = Path(args.outputs_root)
    config_root = Path(args.config_root)
    variants = []
    if args.only in {"party", "both"}:
        variants.append("party")
    if args.only in {"section", "both"}:
        variants.append("section")

    for variant in variants:
        for bucket in BUCKETS:
            source_config = entity_config_root / variant / bucket / "config.yaml"
            if not source_config.is_file():
                raise FileNotFoundError(f"Entity-resolved config missing: {source_config}")
            cfg = retarget_config(
                load_yaml(source_config),
                bucket=bucket,
                data_root=data_root,
                outputs_root=outputs_root,
                variant=variant,
                lr_mode=args.lr_mode,
            )
            target_config = config_root / variant / bucket / "config.yaml"
            save_yaml(target_config, cfg)
            print(f"[config] wrote {target_config}")


if __name__ == "__main__":
    main()
