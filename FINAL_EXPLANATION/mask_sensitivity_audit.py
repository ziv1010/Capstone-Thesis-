#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import pattern_explainability_common as common
from explain_hgt import load_yaml
from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.training.metrics import compute_metrics


DEFAULT_EXPLANATION_DIR = (
    common.APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
)
DEFAULT_FULL_GRAPH_DIR = (
    common.APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph"
)
DEFAULT_IDENTITY_TYPES = (
    "judge",
    "court",
    "petitioner",
    "respondent",
    "lawyer",
    "petitioner_lawyer",
    "defence_lawyer",
)
IDENTITY_MASKS: dict[str, tuple[str, ...]] = {
    "no_judge": ("judge",),
    "no_parties": ("petitioner", "respondent"),
    "no_lawyers": ("lawyer", "petitioner_lawyer", "defence_lawyer"),
    "no_court": ("court",),
    "no_all_identities": DEFAULT_IDENTITY_TYPES,
}


@dataclass(frozen=True)
class MaskSpec:
    mask_name: str
    mask_family: str
    node_indices_by_type: dict[str, set[int] | None]
    description: str
    top_k_hubs: int | None = None


@dataclass
class PredictionBundle:
    probs: np.ndarray
    pred: np.ndarray
    confidence: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc inference masking audit for the frozen HGT model. "
            "Masks identity node families and top-k hub authorities without retraining."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--full-graph-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--identity-masks",
        default=",".join(IDENTITY_MASKS),
        help=(
            "Comma-separated identity masks to run. Available: "
            + ",".join(IDENTITY_MASKS)
        ),
    )
    parser.add_argument(
        "--identity-types",
        default=",".join(DEFAULT_IDENTITY_TYPES),
        help="Node types included by no_all_identities.",
    )
    parser.add_argument(
        "--hub-ks",
        default="10,25,50",
        help="Comma-separated top-k hub-authority masks. Use empty string to skip.",
    )
    parser.add_argument(
        "--authority-roles-csv",
        type=Path,
        default=None,
        help=(
            "Experiment-9 authority role CSV. Defaults to "
            "--full-graph-dir/authority_role_classification_res_<resolution>.csv."
        ),
    )
    parser.add_argument("--hub-resolution", type=float, default=1.0)
    parser.add_argument("--hub-role", default="hub")
    parser.add_argument(
        "--hub-sort",
        default="n_cases",
        help="Column used to rank hub authorities when --authority-roles-csv is available.",
    )
    parser.add_argument(
        "--domain-rank-by",
        default="accuracy_drop",
        choices=(
            "accuracy_drop",
            "confidence_drop",
            "macro_f1_drop",
            "mean_confidence_drop",
            "mean_original_pred_proba_drop",
            "flip_rate",
        ),
    )
    parser.add_argument("--top-domain-drops", type=int, default=5)
    parser.add_argument("--skip-case-output", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for part in parse_csv_list(raw):
        value = int(part)
        if value <= 0:
            raise ValueError(f"Expected positive top-k value, got {value}.")
        values.append(value)
    return sorted(set(values))


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def load_model(
    data: Any,
    label_names: list[str],
    cfg: dict[str, Any],
    model_path: Path,
    device: torch.device,
) -> HeteroLegalOutcomeGNN:
    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    model = HeteroLegalOutcomeGNN(
        metadata=data.metadata(),
        input_dims=input_dims,
        out_dim=len(label_names),
        cfg=cfg.get("model", {}),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_x_dict(data: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {node_type: data[node_type].x.to(device, non_blocking=True) for node_type in data.node_types}


def build_edge_index_dict(
    data: Any,
    mask: MaskSpec | None,
    device: torch.device,
) -> tuple[dict[tuple[str, str, str], torch.Tensor], dict[str, Any]]:
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    stats = {
        "n_masked_edges": 0,
        "n_original_edges": 0,
        "n_remaining_edges": 0,
        "n_edge_types_changed": 0,
        "edge_type_mask_counts": {},
    }
    node_indices_by_type = mask.node_indices_by_type if mask else {}

    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index.detach().cpu()
        edge_count = int(edge_index.shape[1])
        stats["n_original_edges"] += edge_count
        keep = torch.ones(edge_count, dtype=torch.bool)

        if mask is not None and edge_count:
            src_nodes = node_indices_by_type.get(src_type, "not_masked")
            dst_nodes = node_indices_by_type.get(dst_type, "not_masked")
            if src_nodes is None:
                keep &= False
            elif src_nodes != "not_masked" and src_nodes:
                src_tensor = torch.tensor(sorted(src_nodes), dtype=edge_index.dtype)
                keep &= ~torch.isin(edge_index[0], src_tensor)
            if dst_nodes is None:
                keep &= False
            elif dst_nodes != "not_masked" and dst_nodes:
                dst_tensor = torch.tensor(sorted(dst_nodes), dtype=edge_index.dtype)
                keep &= ~torch.isin(edge_index[1], dst_tensor)

        kept_edges = int(keep.sum().item())
        masked_edges = edge_count - kept_edges
        if masked_edges:
            stats["n_edge_types_changed"] += 1
            stats["edge_type_mask_counts"][common_edge_type_to_str(edge_type)] = masked_edges
        stats["n_masked_edges"] += masked_edges
        stats["n_remaining_edges"] += kept_edges
        if kept_edges:
            current = edge_index[:, keep].contiguous()
        else:
            current = torch.empty((2, 0), dtype=torch.long)
        edge_index_dict[edge_type] = current.to(device, non_blocking=True)
    return edge_index_dict, stats


def common_edge_type_to_str(edge_type: tuple[str, str, str]) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


@torch.no_grad()
def forward_selected_cases(
    model: HeteroLegalOutcomeGNN,
    x_dict: dict[str, torch.Tensor],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    case_indices: np.ndarray,
    device: torch.device,
) -> PredictionBundle:
    logits, _ = model(x_dict, edge_index_dict)
    index = torch.as_tensor(np.asarray(case_indices, dtype=np.int64).copy(), device=device)
    probs = torch.softmax(logits[index], dim=-1).detach().cpu().numpy()
    pred = probs.argmax(axis=1).astype(np.int64, copy=False)
    confidence = probs.max(axis=1)
    return PredictionBundle(probs=probs, pred=pred, confidence=confidence)


def selected_case_indices(
    case_meta: pd.DataFrame,
    split: str,
    case_limit: int | None,
) -> np.ndarray:
    if split == "all":
        values = case_meta["case_index"].to_numpy(dtype=np.int64)
    else:
        values = case_meta.loc[
            case_meta["split"].astype(str) == split, "case_index"
        ].to_numpy(dtype=np.int64)
    if case_limit is not None:
        values = values[:case_limit]
    if values.size == 0:
        raise ValueError(f"No cases selected for split={split!r}.")
    return values


def metric_subset(
    y_true: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    return compute_metrics(
        y_true=y_true,
        y_pred=pred,
        y_proba=probs,
        label_names=label_names,
    )


def metric_row(
    mask: MaskSpec,
    baseline: PredictionBundle,
    masked: PredictionBundle,
    y_true: np.ndarray,
    label_names: list[str],
    n_masked_edges: int,
    n_original_edges: int,
    n_edge_types_changed: int,
    n_masked_authorities: int,
) -> dict[str, Any]:
    base_metrics = metric_subset(y_true, baseline.pred, baseline.probs, label_names)
    mask_metrics = metric_subset(y_true, masked.pred, masked.probs, label_names)
    row_idx = np.arange(len(y_true))
    masked_original_pred_proba = masked.probs[row_idx, baseline.pred]
    original_pred_proba_drop = baseline.confidence - masked_original_pred_proba
    confidence_drop = float(np.mean(original_pred_proba_drop))
    return {
        "mask_name": mask.mask_name,
        "mask_family": mask.mask_family,
        "description": mask.description,
        "masked_node_types": "|".join(sorted(mask.node_indices_by_type)),
        "top_k_hubs": mask.top_k_hubs if mask.top_k_hubs is not None else "",
        "n_masked_authorities": n_masked_authorities,
        "n_cases": int(len(y_true)),
        "n_masked_edges": int(n_masked_edges),
        "n_original_edges": int(n_original_edges),
        "masked_edge_share": n_masked_edges / n_original_edges if n_original_edges else 0.0,
        "n_edge_types_changed": int(n_edge_types_changed),
        "baseline_accuracy": safe_float(base_metrics.get("accuracy")),
        "masked_accuracy": safe_float(mask_metrics.get("accuracy")),
        "accuracy_drop": safe_float(base_metrics.get("accuracy") - mask_metrics.get("accuracy")),
        "baseline_macro_f1": safe_float(base_metrics.get("macro_f1")),
        "masked_macro_f1": safe_float(mask_metrics.get("macro_f1")),
        "macro_f1_drop": safe_float(base_metrics.get("macro_f1") - mask_metrics.get("macro_f1")),
        "baseline_mean_confidence": float(np.mean(baseline.confidence)),
        "masked_mean_confidence": float(np.mean(masked.confidence)),
        "mean_confidence_drop": float(np.mean(baseline.confidence) - np.mean(masked.confidence)),
        "masked_mean_original_pred_proba": float(np.mean(masked_original_pred_proba)),
        "confidence_drop": confidence_drop,
        "mean_original_pred_proba_drop": confidence_drop,
        "median_original_pred_proba_drop": float(np.median(original_pred_proba_drop)),
        "flip_rate": float(np.mean(masked.pred != baseline.pred)),
    }


def domain_rows(
    mask: MaskSpec,
    baseline: PredictionBundle,
    masked: PredictionBundle,
    y_true: np.ndarray,
    label_names: list[str],
    domains: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in sorted(set(domains.tolist())):
        domain_mask = domains == domain
        if not bool(domain_mask.any()):
            continue
        base = PredictionBundle(
            probs=baseline.probs[domain_mask],
            pred=baseline.pred[domain_mask],
            confidence=baseline.confidence[domain_mask],
        )
        masked_domain = PredictionBundle(
            probs=masked.probs[domain_mask],
            pred=masked.pred[domain_mask],
            confidence=masked.confidence[domain_mask],
        )
        metrics = metric_row(
            mask=mask,
            baseline=base,
            masked=masked_domain,
            y_true=y_true[domain_mask],
            label_names=label_names,
            n_masked_edges=0,
            n_original_edges=0,
            n_edge_types_changed=0,
            n_masked_authorities=0,
        )
        rows.append(
            {
                "mask_name": mask.mask_name,
                "mask_family": mask.mask_family,
                "domain_bucket": domain,
                "n_cases": int(domain_mask.sum()),
                "baseline_accuracy": metrics["baseline_accuracy"],
                "masked_accuracy": metrics["masked_accuracy"],
                "accuracy_drop": metrics["accuracy_drop"],
                "baseline_macro_f1": metrics["baseline_macro_f1"],
                "masked_macro_f1": metrics["masked_macro_f1"],
                "macro_f1_drop": metrics["macro_f1_drop"],
                "baseline_mean_confidence": metrics["baseline_mean_confidence"],
                "masked_mean_confidence": metrics["masked_mean_confidence"],
                "mean_confidence_drop": metrics["mean_confidence_drop"],
                "confidence_drop": metrics["confidence_drop"],
                "mean_original_pred_proba_drop": metrics["mean_original_pred_proba_drop"],
                "flip_rate": metrics["flip_rate"],
            }
        )
    return rows


def case_prediction_rows(
    mask: MaskSpec,
    case_meta: pd.DataFrame,
    case_indices: np.ndarray,
    y_true: np.ndarray,
    label_names: list[str],
    baseline: PredictionBundle,
    masked: PredictionBundle,
) -> pd.DataFrame:
    rows = case_meta.loc[case_indices].copy()
    row_idx = np.arange(len(case_indices))
    rows["mask_name"] = mask.mask_name
    rows["mask_family"] = mask.mask_family
    rows["target_index"] = y_true
    rows["baseline_pred_index"] = baseline.pred
    rows["baseline_pred_label"] = [label_names[idx] for idx in baseline.pred]
    rows["baseline_confidence"] = baseline.confidence
    rows["masked_pred_index"] = masked.pred
    rows["masked_pred_label"] = [label_names[idx] for idx in masked.pred]
    rows["masked_confidence"] = masked.confidence
    rows["masked_original_pred_proba"] = masked.probs[row_idx, baseline.pred]
    rows["original_pred_proba_drop"] = (
        baseline.confidence - rows["masked_original_pred_proba"].to_numpy()
    )
    rows["prediction_flipped"] = masked.pred != baseline.pred
    rows["masked_correct"] = masked.pred == y_true
    return rows[
        [
            "case_index",
            "case_id",
            "file_name",
            "split",
            "domain_bucket",
            "target_label",
            "target_index",
            "mask_name",
            "mask_family",
            "baseline_pred_label",
            "baseline_confidence",
            "masked_pred_label",
            "masked_confidence",
            "masked_original_pred_proba",
            "original_pred_proba_drop",
            "prediction_flipped",
            "correct",
            "masked_correct",
        ]
    ]


def build_identity_masks(identity_masks_raw: str, identity_types_raw: str, data: Any) -> list[MaskSpec]:
    selected_names = parse_csv_list(identity_masks_raw)
    identity_types = tuple(parse_csv_list(identity_types_raw))
    masks: list[MaskSpec] = []
    for mask_name in selected_names:
        if mask_name not in IDENTITY_MASKS:
            raise ValueError(
                f"Unknown identity mask {mask_name!r}. Available: {sorted(IDENTITY_MASKS)}"
            )
        node_types = identity_types if mask_name == "no_all_identities" else IDENTITY_MASKS[mask_name]
        present_types = tuple(node_type for node_type in node_types if node_type in data.node_types)
        masks.append(
            MaskSpec(
                mask_name=mask_name,
                mask_family="identity",
                node_indices_by_type={node_type: None for node_type in present_types},
                description="Mask all incident edges for " + ", ".join(present_types),
            )
        )
    return masks


def authority_roles_path(args: argparse.Namespace) -> Path:
    if args.authority_roles_csv is not None:
        return args.authority_roles_csv
    return args.full_graph_dir / f"authority_role_classification_res_{args.hub_resolution:.2f}.csv"


def collect_authority_to_cases(data: Any) -> dict[tuple[str, int], set[int]]:
    auth_to_cases: dict[tuple[str, int], set[int]] = {}
    section_to_cases: dict[str, dict[int, set[int]]] = {}
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        edge_index = data[edge_type].edge_index.cpu().numpy()
        if src_type == "case" and dst_type in common.LEGAL_AUTHORITY_TYPES:
            for c_idx, a_idx in zip(edge_index[0], edge_index[1], strict=False):
                auth_to_cases.setdefault((dst_type, int(a_idx)), set()).add(int(c_idx))
        if src_type == "case" and dst_type in common.SECTION_NODE_TYPES:
            section_map = section_to_cases.setdefault(dst_type, {})
            for c_idx, sec_idx in zip(edge_index[0], edge_index[1], strict=False):
                section_map.setdefault(int(sec_idx), set()).add(int(c_idx))

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if (
            src_type in common.SECTION_NODE_TYPES
            and dst_type in common.LEGAL_AUTHORITY_TYPES
            and relation.startswith("cites_")
        ):
            section_map = section_to_cases.get(src_type, {})
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for sec_idx, a_idx in zip(edge_index[0], edge_index[1], strict=False):
                cases = section_map.get(int(sec_idx), set())
                if cases:
                    auth_to_cases.setdefault((dst_type, int(a_idx)), set()).update(cases)
    return auth_to_cases


def fallback_authority_table(data: Any) -> pd.DataFrame:
    auth_to_cases = collect_authority_to_cases(data)
    node_ids = {
        node_type: [str(value) for value in data[node_type].node_id]
        for node_type in common.LEGAL_AUTHORITY_TYPES
        if node_type in data.node_types and hasattr(data[node_type], "node_id")
    }
    rows: list[dict[str, Any]] = []
    for (node_type, node_idx), cases in auth_to_cases.items():
        ids = node_ids.get(node_type, [])
        rows.append(
            {
                "feature_type": node_type,
                "feature_local_index": int(node_idx),
                "feature_name": (
                    common.display_node_id(node_type, ids[node_idx], max_len=200)
                    if node_idx < len(ids)
                    else f"{node_type}::{node_idx}"
                ),
                "n_cases": int(len(cases)),
                "n_unique_cases": int(len(cases)),
                "role": "degree_fallback",
            }
        )
    return pd.DataFrame(rows)


def load_hub_authorities(args: argparse.Namespace, data: Any) -> tuple[pd.DataFrame, str]:
    roles_path = authority_roles_path(args)
    if roles_path.exists():
        roles = pd.read_csv(roles_path)
        if "role" in roles.columns:
            roles = roles[roles["role"].astype(str) == str(args.hub_role)].copy()
        source = str(roles_path)
    else:
        print(
            f"[warn] authority role CSV not found at {roles_path}; "
            "falling back to top legal authorities by unique citing cases.",
            flush=True,
        )
        roles = fallback_authority_table(data)
        source = "degree_fallback"

    required = {"feature_type", "feature_local_index", "feature_name"}
    missing = required.difference(roles.columns)
    if missing:
        raise ValueError(f"Authority table is missing columns: {sorted(missing)}")
    sort_col = args.hub_sort if args.hub_sort in roles.columns else "n_unique_cases"
    if sort_col not in roles.columns:
        sort_col = "feature_local_index"
    roles = roles.copy()
    roles[sort_col] = pd.to_numeric(roles[sort_col], errors="coerce").fillna(0)
    roles = roles.sort_values(sort_col, ascending=False)
    roles = roles.drop_duplicates(["feature_type", "feature_local_index"], keep="first")
    return roles.reset_index(drop=True), source


def build_hub_masks(args: argparse.Namespace, data: Any) -> tuple[list[MaskSpec], pd.DataFrame, str]:
    hub_ks = parse_int_csv(args.hub_ks)
    if not hub_ks:
        return [], pd.DataFrame(), ""
    hubs, source = load_hub_authorities(args, data)
    if hubs.empty:
        raise ValueError("No hub authorities were available for masking.")

    max_k = max(hub_ks)
    selected = hubs.head(max_k).copy()
    selected["hub_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)

    masks: list[MaskSpec] = []
    for k in hub_ks:
        subset = selected.head(k)
        node_indices: dict[str, set[int] | None] = {}
        for node_type, group in subset.groupby("feature_type"):
            node_indices[str(node_type)] = {
                int(value) for value in group["feature_local_index"].tolist()
            }
        masks.append(
            MaskSpec(
                mask_name=f"remove_top_{k}_hubs",
                mask_family="hub_authority",
                node_indices_by_type=node_indices,
                description=f"Mask incident edges for top {k} hub legal authorities.",
                top_k_hubs=k,
            )
        )
    return masks, selected, source


def masked_authority_count(mask: MaskSpec) -> int:
    total = 0
    for indices in mask.node_indices_by_type.values():
        if indices is not None:
            total += len(indices)
    return total


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = args.output_dir or args.explanation_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = common.infer_artifact_paths(
        explanation_dir=args.explanation_dir,
        graph_cache=args.graph_cache,
        predictions_csv=args.predictions_csv,
        config=args.config,
        model_path=args.model_path,
    )
    print(f"[load] graph={paths['graph_cache']}", flush=True)
    bundle = torch.load(paths["graph_cache"], map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle.get("metadata", {})
    label_names = [str(label) for label in metadata.get("label_names", ["-1", "1"])]
    cfg = load_yaml(paths["config"])
    predictions = common.load_predictions(
        paths["predictions_csv"], expected_cases=int(data["case"].num_nodes)
    )
    case_meta = common.build_case_metadata(data, predictions)
    case_indices = selected_case_indices(case_meta, args.split, args.case_limit)
    y_true = data["case"].y.detach().cpu().numpy()[case_indices].astype(np.int64, copy=False)
    domains = case_meta.loc[case_indices, "domain_bucket"].astype(str).to_numpy()

    identity_masks = build_identity_masks(args.identity_masks, args.identity_types, data)
    hub_masks, hub_table, hub_source = build_hub_masks(args, data)
    masks = identity_masks + hub_masks
    if not masks:
        raise ValueError("No masks requested.")

    device = choose_device(args.device)
    print(f"[load] model={paths['model_path']} device={device}", flush=True)
    model = load_model(data, label_names, cfg, paths["model_path"], device)
    x_dict = build_x_dict(data, device)

    print(f"[baseline] cases={len(case_indices)} split={args.split}", flush=True)
    baseline_edges, baseline_edge_stats = build_edge_index_dict(data, None, device)
    baseline = forward_selected_cases(model, x_dict, baseline_edges, case_indices, device)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    all_domain_rows: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    edge_stats_by_mask: dict[str, Any] = {}

    for mask in masks:
        print(f"[mask] {mask.mask_name}", flush=True)
        masked_edges, edge_stats = build_edge_index_dict(data, mask, device)
        masked = forward_selected_cases(model, x_dict, masked_edges, case_indices, device)
        n_authorities = masked_authority_count(mask)
        summary_rows.append(
            metric_row(
                mask=mask,
                baseline=baseline,
                masked=masked,
                y_true=y_true,
                label_names=label_names,
                n_masked_edges=int(edge_stats["n_masked_edges"]),
                n_original_edges=int(edge_stats["n_original_edges"]),
                n_edge_types_changed=int(edge_stats["n_edge_types_changed"]),
                n_masked_authorities=n_authorities,
            )
        )
        all_domain_rows.extend(
            domain_rows(
                mask=mask,
                baseline=baseline,
                masked=masked,
                y_true=y_true,
                label_names=label_names,
                domains=domains,
            )
        )
        if not args.skip_case_output:
            case_frames.append(
                case_prediction_rows(
                    mask=mask,
                    case_meta=case_meta,
                    case_indices=case_indices,
                    y_true=y_true,
                    label_names=label_names,
                    baseline=baseline,
                    masked=masked,
                )
            )
        edge_stats_by_mask[mask.mask_name] = edge_stats
        del masked_edges, masked
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "mask_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    domain_df = pd.DataFrame(all_domain_rows)
    if not domain_df.empty:
        domain_df = domain_df.sort_values(
            ["mask_name", args.domain_rank_by, "n_cases"],
            ascending=[True, False, False],
        )
    domain_path = output_dir / "mask_sensitivity_by_domain.csv"
    domain_df.to_csv(domain_path, index=False)

    top_domain_rows = []
    if not domain_df.empty:
        for mask_name, group in domain_df.groupby("mask_name", sort=False):
            top_domain_rows.append(group.head(args.top_domain_drops))
    top_domain_df = (
        pd.concat(top_domain_rows, ignore_index=True)
        if top_domain_rows
        else pd.DataFrame()
    )
    top_domain_path = output_dir / "mask_sensitivity_top_domain_drops.csv"
    top_domain_df.to_csv(top_domain_path, index=False)

    case_path = None
    if case_frames:
        case_path = output_dir / "mask_sensitivity_case_predictions.csv"
        pd.concat(case_frames, ignore_index=True).to_csv(case_path, index=False)

    hub_path = None
    if not hub_table.empty:
        hub_path = output_dir / "mask_sensitivity_hub_authorities.csv"
        hub_table.to_csv(hub_path, index=False)

    manifest = {
        "model_path": str(paths["model_path"]),
        "graph_cache": str(paths["graph_cache"]),
        "config": str(paths["config"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "explanation_dir": str(args.explanation_dir),
        "full_graph_dir": str(args.full_graph_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "case_limit": args.case_limit,
        "n_cases": int(len(case_indices)),
        "device": str(device),
        "label_names": label_names,
        "baseline_edge_stats": baseline_edge_stats,
        "edge_stats_by_mask": edge_stats_by_mask,
        "hub_authority_source": hub_source,
        "hub_ks": parse_int_csv(args.hub_ks),
        "identity_masks": parse_csv_list(args.identity_masks),
        "domain_rank_by": args.domain_rank_by,
        "outputs": {
            "summary": str(summary_path),
            "by_domain": str(domain_path),
            "top_domain_drops": str(top_domain_path),
            "case_predictions": str(case_path) if case_path else None,
            "hub_authorities": str(hub_path) if hub_path else None,
        },
        "interpretation": (
            "Accuracy/macro-F1 drops compare the frozen model under post-hoc edge "
            "masking against the unmasked inference baseline. Flip rate is the share "
            "of selected cases whose predicted class changes. mean_original_pred_proba_drop "
            "tracks confidence in the original unmasked prediction after masking."
        ),
        "elapsed_seconds": time.time() - start,
    }
    write_json(output_dir / "mask_sensitivity_manifest.json", manifest)
    print(f"[done] wrote {summary_path}", flush=True)
    print(f"[done] wrote {domain_path}", flush=True)
    print(f"[done] elapsed_min={(time.time() - start) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
