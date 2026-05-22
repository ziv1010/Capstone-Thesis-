#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


SECTION_GNN = Path(__file__).resolve().parents[2]
RESOLVED_ROOT = (
    SECTION_GNN.parent
    / "DATA_SET_BUILDER_AND_EXPLORER"
    / "Timeline_Maker"
    / "output_merged_v3_resolved"
)

BUCKETS: dict[str, str] = {
    "family_matrimonial_timed_mistral": "family_matrimonial",
    "fin_fraud_timed_mistral": "fin_fraud",
    "land_property_timed_mistral": "land_property",
    "motor_accidents_timed_mistral": "motor_accidents",
    "sexual_offences_timed_mistral": "sexual_offences",
    "cross_bucket_total_dataset": "combined_dataset_without_food_safety",
}

SECTION_NO_NAMES_INCLUDE_SECTIONS = [
    "facts",
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
]
SECTION_NO_NAMES_CASE_TEXT_SECTIONS = [
    "facts",
    "arguments",
]
SECTION_NO_NAMES_INCLUDE_NODE_TYPES = [
    "case",
    "facts",
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
    "statute",
    "provision",
    "precedent",
]
SECTION_NO_NAMES_CASE_SCALARS = [
    "statute_count",
    "provision_count",
    "precedent_count",
    "facts_length",
    "arguments_length",
    "case_year",
    "petition_type_known",
    "petition_type_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configs for the entity-resolved data ablation.")
    parser.add_argument("--resolved-root", default=str(RESOLVED_ROOT))
    parser.add_argument("--data-root", default=str(SECTION_GNN / "data" / "ablations" / "entity_resolved_data"))
    parser.add_argument("--outputs-root", default=str(SECTION_GNN / "outputs" / "ablations" / "entity_resolved_data"))
    parser.add_argument("--config-root", default=str(SECTION_GNN / "ablations" / "entity_resolved_data" / "configs"))
    parser.add_argument("--only", choices=["party", "section", "section_no_names", "both", "all"], default="both")
    parser.add_argument("--lr-mode", choices=["decay", "none"], default="decay")
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


def apply_lr_decay(cfg: dict[str, Any]) -> None:
    training = cfg.setdefault("training", {})
    training.update(
        {
            "epochs": 90,
            "use_early_stopping": True,
            "early_stopping_patience": 20,
            "lr_scheduler": "reduce_on_plateau",
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 8,
            "lr_min": 0.000001,
        }
    )


def apply_no_lr_decay(cfg: dict[str, Any]) -> None:
    training = cfg.setdefault("training", {})
    training.update(
        {
            "epochs": 60,
            "use_early_stopping": True,
            "early_stopping_patience": 15,
        }
    )
    for key in ("lr_scheduler", "lr_scheduler_factor", "lr_scheduler_patience", "lr_min"):
        training.pop(key, None)


def apply_section_sep_no_names(cfg: dict[str, Any]) -> None:
    graph = cfg.setdefault("graph", {})
    graph["respect_explicit_includes"] = True
    graph["include_sections"] = list(SECTION_NO_NAMES_INCLUDE_SECTIONS)
    graph["case_text_sections"] = list(SECTION_NO_NAMES_CASE_TEXT_SECTIONS)
    graph["include_node_types"] = list(SECTION_NO_NAMES_INCLUDE_NODE_TYPES)
    graph["share_party_nodes"] = False
    graph["shareable_node_types"] = ["statute", "provision", "precedent"]

    features = cfg.setdefault("features", {})
    features["case_scalar_names"] = list(SECTION_NO_NAMES_CASE_SCALARS)


def retarget_config(
    cfg: dict[str, Any],
    *,
    bucket: str,
    source_dir: Path,
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
        project_name = f"{short}_entity_resolved_party_args_preamble"
        cache_name = f"case_star_entity_resolved_{short}_party_args_preamble_lr_decay.reasoning_focused.pt"
    elif variant == "section":
        project_name = f"{short}_entity_resolved_section_sep"
        cache_name = f"case_star_entity_resolved_{short}_section_sep_lr_decay.reasoning_focused.pt"
    elif variant == "section_no_names":
        project_name = f"{short}_entity_resolved_section_sep_no_names"
        cache_name = f"case_star_entity_resolved_{short}_section_sep_no_names.reasoning_focused.pt"
        apply_section_sep_no_names(out)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    out["project"]["name"] = f"{project_name}_{'lr_decay' if lr_mode == 'decay' else 'no_lr'}"
    paths = out.setdefault("paths", {})
    paths["raw_json_dir"] = str(source_dir)
    paths["processed_dir"] = str(bucket_data_root / "processed")
    paths["cleaned_case_dir"] = str(bucket_data_root / "processed" / "cleaned_cases")
    paths["normalized_entity_dir"] = str(bucket_data_root / "processed" / "normalized_entities")
    paths["embeddings_cache_dir"] = str(bucket_data_root / "embeddings_cache" / variant)
    paths["graph_cache_dir"] = str(bucket_data_root / "graph_cache" / variant)
    paths["audits_dir"] = str(bucket_data_root / "audits")
    paths["outputs_dir"] = str(bucket_outputs_root)

    out.setdefault("data", {})["file_glob"] = "*.json"
    out.setdefault("graph", {})["cache_name"] = cache_name
    if lr_mode == "decay":
        apply_lr_decay(out)
    else:
        apply_no_lr_decay(out)
    return out


def main() -> None:
    args = parse_args()
    resolved_root = Path(args.resolved_root)
    data_root = Path(args.data_root)
    outputs_root = Path(args.outputs_root)
    config_root = Path(args.config_root)

    variants: list[tuple[str, Path]] = []
    if args.only in {"party", "both", "all"}:
        variants.append(("party", SECTION_GNN / "runs_v2" / "party_args_preamble_lr_decay"))
    if args.only in {"section", "both", "all"}:
        variants.append(("section", SECTION_GNN / "ablations" / "section_sep_enc_lr_decay"))
    if args.only in {"section_no_names", "all"}:
        variants.append(("section_no_names", SECTION_GNN / "ablations" / "section_sep_enc_lr_decay"))

    for bucket, source_name in BUCKETS.items():
        source_dir = resolved_root / source_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Resolved source directory missing for {bucket}: {source_dir}")

        for variant, base_root in variants:
            base_cfg = base_root / bucket / "config.yaml"
            if not base_cfg.is_file():
                raise FileNotFoundError(f"Base config missing for {variant}/{bucket}: {base_cfg}")
            cfg = retarget_config(
                load_yaml(base_cfg),
                bucket=bucket,
                source_dir=source_dir,
                data_root=data_root,
                outputs_root=outputs_root,
                variant=variant,
                lr_mode=args.lr_mode,
            )
            config_path = config_root / variant / bucket / "config.yaml"
            save_yaml(config_path, cfg)
            print(f"[config] wrote {config_path}")


if __name__ == "__main__":
    main()
