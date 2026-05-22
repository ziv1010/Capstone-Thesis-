from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

import torch
import yaml

CAPSTONE_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
GRAPH_ANALYSER_ROOT_DEFAULT = CAPSTONE_ROOT_DEFAULT / "Graph_Analyser"
SECTION_GNN_ROOT_DEFAULT = str(CAPSTONE_ROOT_DEFAULT / "section_GNN")

KNOWN_BUCKETS = (
    "family_matrimonial",
    "fin_fraud",
    "land_property",
    "motor_accidents",
    "sexual_offences",
)
CROSS_BUCKET_DATASET = "cross_bucket_total_dataset"
CROSS_BUCKET_ALIASES = ("cross_bucket", CROSS_BUCKET_DATASET)


def ensure_section_gnn_on_path(section_gnn_root: str | Path) -> None:
    root = str(section_gnn_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def infer_bucket_from_case_id(case_id: str) -> str | None:
    text = str(case_id)
    for bucket in KNOWN_BUCKETS:
        if text.startswith(f"{bucket}__"):
            return bucket
    if "__" in text:
        prefix = text.split("__", 1)[0]
        if prefix:
            return prefix
    return None


def infer_bucket_from_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    for bucket in KNOWN_BUCKETS:
        if re.search(rf"(^|[/_]){re.escape(bucket)}([/_]|$)", text):
            return bucket
    return None


def is_cross_bucket_scope(scope: str | None) -> bool:
    return str(scope or "") in CROSS_BUCKET_ALIASES


def normalise_scope(scope: str) -> str:
    if is_cross_bucket_scope(scope):
        return CROSS_BUCKET_DATASET
    if scope in KNOWN_BUCKETS:
        return scope
    allowed = ", ".join((*KNOWN_BUCKETS, *CROSS_BUCKET_ALIASES))
    raise ValueError(f"Unknown bucket/scope {scope!r}. Expected one of: {allowed}")


def _timed_run_dir(section_gnn_root: Path, scope: str) -> Path:
    if is_cross_bucket_scope(scope):
        run_name = CROSS_BUCKET_DATASET
    else:
        run_name = f"{scope}_timed_mistral"
    return section_gnn_root / "data" / "timed_bucket_runs" / run_name


def _model_run_dir(section_gnn_root: Path, scope: str) -> Path:
    if is_cross_bucket_scope(scope):
        run_name = CROSS_BUCKET_DATASET
    else:
        run_name = f"{scope}_timed_mistral"
    return section_gnn_root / "outputs" / "timed_bucket_runs" / run_name


def _model_name_for(scope: str, cfg: dict[str, Any]) -> str:
    if cfg.get("model_name"):
        return str(cfg["model_name"])
    variant = str(cfg.get("model_variant", "party_args_preamble_lr_decay"))
    if is_cross_bucket_scope(scope):
        return f"cross_bucket_{variant}_kfold"
    return f"{scope}_{variant}_kfold"


def _resolve_graph_cache(section_gnn_root: Path, scope: str, graph_variant: str) -> Path:
    scope = normalise_scope(scope)
    graph_cache_dir = (
        _timed_run_dir(section_gnn_root, scope)
        / "graph_cache"
    )
    if is_cross_bucket_scope(scope):
        stem = "cross_bucket"
        candidates = [
            graph_cache_dir / f"case_star_global_graph_{stem}_{graph_variant}.reasoning_focused.pt",
            graph_cache_dir / f"case_star_{stem}_{graph_variant}.reasoning_focused.pt",
            graph_cache_dir / f"case_star_global_graph_{CROSS_BUCKET_DATASET}.reasoning_focused.pt",
        ]
    else:
        candidates = [
            graph_cache_dir / f"case_star_global_graph_{scope}_{graph_variant}.reasoning_focused.pt",
            graph_cache_dir / f"case_star_{scope}_{graph_variant}.reasoning_focused.pt",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = sorted(p.name for p in graph_cache_dir.glob("*.pt")) if graph_cache_dir.exists() else []
    raise FileNotFoundError(
        "Could not resolve graph_cache for "
        f"bucket/scope={scope!r}, graph_variant={graph_variant!r}. "
        f"Checked: {[str(p) for p in candidates]}. "
        f"Available graph caches: {available[:20]}"
    )


def _resolve_checkpoint_dir(section_gnn_root: Path, scope: str, cfg: dict[str, Any]) -> Path:
    scope = normalise_scope(scope)
    model_name = _model_name_for(scope, cfg)
    fold = str(cfg.get("fold", "fold_00"))
    checkpoint_dir = (
        _model_run_dir(section_gnn_root, scope)
        / "models"
        / model_name
        / "kfold"
        / fold
    )
    if not (checkpoint_dir / "model.pt").exists():
        raise FileNotFoundError(
            "Could not resolve checkpoint_dir for "
            f"bucket/scope={scope!r}, model_name={model_name!r}, fold={fold!r}: {checkpoint_dir}"
        )
    return checkpoint_dir


def _guard_path_bucket(
    *,
    cfg_key: str,
    path: str | Path,
    expected_bucket: str,
    allow_cross_bucket: bool,
) -> None:
    text = str(path)
    if "cross_bucket" in text and (
        not allow_cross_bucket or not is_cross_bucket_scope(expected_bucket)
    ):
        raise ValueError(
            f"{cfg_key} points at a cross-bucket artifact: {path}. "
            "Use bucket: cross_bucket_total_dataset with allow_cross_bucket: true "
            "for models intentionally trained on the mixed dataset."
        )
    found_bucket = infer_bucket_from_path(path)
    if found_bucket and found_bucket != expected_bucket:
        raise ValueError(
            f"{cfg_key} appears to belong to bucket {found_bucket!r}, "
            f"but config bucket is {expected_bucket!r}: {path}"
        )


def resolve_bucket_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.setdefault("section_gnn_root", SECTION_GNN_ROOT_DEFAULT)
    bucket = out.get("bucket")
    allow_cross_bucket = bool(out.get("allow_cross_bucket", False))
    section_gnn_root = Path(out["section_gnn_root"])

    if bucket:
        bucket = normalise_scope(str(bucket))
        if is_cross_bucket_scope(bucket) and not allow_cross_bucket:
            raise ValueError(
                "Cross-bucket analysis is supported, but it must be explicit: "
                "set bucket: cross_bucket_total_dataset and allow_cross_bucket: true."
            )
        out["bucket"] = bucket
        graph_variant = str(out.get("graph_variant", "party_args_preamble"))
        if not out.get("graph_cache"):
            out["graph_cache"] = str(_resolve_graph_cache(section_gnn_root, bucket, graph_variant))
        if not out.get("checkpoint_dir"):
            out["checkpoint_dir"] = str(_resolve_checkpoint_dir(section_gnn_root, bucket, out))
        if not out.get("cleaned_case_dir"):
            out["cleaned_case_dir"] = str(
                _timed_run_dir(section_gnn_root, bucket)
                / "processed"
                / "cleaned_cases"
            )
        if not out.get("output_root"):
            model_variant = str(out.get("model_variant", "party_args_preamble_lr_decay"))
            fold = str(out.get("fold", "fold_00"))
            out["output_root"] = str(
                GRAPH_ANALYSER_ROOT_DEFAULT / "outputs" / f"{bucket}_{model_variant}_{fold}"
            )
        _guard_path_bucket(
            cfg_key="graph_cache",
            path=out["graph_cache"],
            expected_bucket=bucket,
            allow_cross_bucket=allow_cross_bucket,
        )
        _guard_path_bucket(
            cfg_key="checkpoint_dir",
            path=out["checkpoint_dir"],
            expected_bucket=bucket,
            allow_cross_bucket=allow_cross_bucket,
        )
        if out.get("cleaned_case_dir"):
            _guard_path_bucket(
                cfg_key="cleaned_case_dir",
                path=out["cleaned_case_dir"],
                expected_bucket=bucket,
                allow_cross_bucket=allow_cross_bucket,
            )
    else:
        for key in ("graph_cache", "checkpoint_dir", "cleaned_case_dir"):
            if "cross_bucket" in str(out.get(key, "")) and not allow_cross_bucket:
                raise ValueError(
                    f"{key} points at a cross-bucket artifact, but no explicit bucket "
                    "or allow_cross_bucket flag is set."
                )
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return resolve_bucket_config(raw)


def validate_case_ids_bucket(
    case_ids: list[str] | tuple[str, ...],
    expected_bucket: str | None,
    context: str,
) -> None:
    if not expected_bucket or is_cross_bucket_scope(expected_bucket):
        return
    wrong = []
    missing = 0
    for case_id in case_ids:
        found = infer_bucket_from_case_id(str(case_id))
        if found is None or found not in KNOWN_BUCKETS:
            missing += 1
        elif found != expected_bucket:
            wrong.append(str(case_id))
    if wrong:
        examples = ", ".join(wrong[:5])
        raise ValueError(
            f"{context} contains case ids outside bucket {expected_bucket!r}. "
            f"wrong_bucket_count={len(wrong)}, title_only_or_unknown_prefix_count={missing}. "
            f"Examples: {examples}"
        )


def validate_graph_bucket(data: Any, expected_bucket: str | None, context: str) -> None:
    case_ids = list(getattr(data["case"], "case_id", []))
    validate_case_ids_bucket(case_ids, expected_bucket, context)


def load_graph_cache(
    graph_cache_path: str | Path,
    expected_bucket: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    blob = torch.load(str(graph_cache_path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        data = blob["data"]
        metadata = blob.get("metadata", {})
    else:
        data = blob
        metadata = {}
    validate_graph_bucket(data, expected_bucket, f"graph cache {graph_cache_path}")
    return data, metadata


def infer_model_cfg_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    hidden_dim = None
    num_layers = 0
    num_heads = 4
    for key, value in state_dict.items():
        if key.startswith("input_projections.") and key.endswith(".weight"):
            hidden_dim = int(value.shape[0])
        if key.startswith("convs."):
            idx = int(key.split(".")[1])
            num_layers = max(num_layers, idx + 1)
    cfg = {
        "architecture": "hgt",
        "hidden_dim": hidden_dim or 128,
        "num_layers": num_layers or 2,
        "num_heads": num_heads,
        "dropout": 0.2,
        "mlp_hidden_dim": hidden_dim or 128,
    }
    return cfg


def build_and_load_hgt(
    data: Any,
    checkpoint_dir: str | Path,
    model_cfg: dict[str, Any] | None,
    section_gnn_root: str | Path = SECTION_GNN_ROOT_DEFAULT,
    device: torch.device | str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    ensure_section_gnn_on_path(section_gnn_root)
    # section_GNN exposes its own `src` package on sys.path (not this project's).
    import importlib

    HeteroLegalOutcomeGNN = importlib.import_module("src.models.hetero_gnn").HeteroLegalOutcomeGNN  # noqa: WPS433

    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = dict(model_cfg or {})
    inferred = infer_model_cfg_from_state_dict(state_dict)
    for key, value in inferred.items():
        cfg.setdefault(key, value)

    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    out_dim = 2
    y = getattr(data["case"], "y", None)
    if y is not None:
        uniq = y.unique().tolist()
        out_dim = max(2, int(max(uniq)) + 1)

    model = HeteroLegalOutcomeGNN(
        metadata=data.metadata(),
        input_dims=input_dims,
        out_dim=out_dim,
        cfg=cfg,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading HGT state_dict: {missing[:8]}")
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in state_dict (model cfg mismatch?): {unexpected[:8]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    return model, cfg
