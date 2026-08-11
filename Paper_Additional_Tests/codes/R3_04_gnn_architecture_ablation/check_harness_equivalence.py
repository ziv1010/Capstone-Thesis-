#!/usr/bin/env python3
"""Prove the ablation harness is equivalent to the paper run -- without retraining.

The paper's HGT results are treated as fixed: nothing is re-run. So equivalence
has to be established statically, before any GPU time is spent. Four checks:

1. SPLIT      -- recomputing StratifiedKFold(5, shuffle=True, random_state=42)
                 plus the default_rng(42+fold) validation carve-out reproduces
                 the train/val/test membership recorded in every fold's
                 predictions.csv, exactly.
2. CONFIG     -- each arch_<name>.yaml differs from the paper config in exactly
                 two keys: model.architecture and paths.outputs_dir.
3. SCAFFOLD   -- ArchLegalOutcomeGNN(architecture="hgt"), built from graph
                 metadata read off disk, has exactly the same state_dict keys
                 and parameter shapes as the saved fold_00/model.pt.
4. GRAPH      -- the graph cache this ablation will load is the same file the
                 reference run recorded in kfold_summary.json.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_SECTION_GNN = _REPO / "section_GNN"
_V2_SCRIPTS = _SECTION_GNN / "runs_v2" / "party_args_lr_decay" / "scripts"
for _path in (str(_HERE), str(_V2_SCRIPTS), str(_SECTION_GNN)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from arch_gnn import ArchLegalOutcomeGNN  # noqa: E402
from kfold_cv_v2 import _make_fold_masks  # noqa: E402  -- the paper's own carve-out

PAPER_CONFIG = (
    _SECTION_GNN
    / "ablations/entity_resolved_data/configs/section/cross_bucket_total_dataset/config.yaml"
)
REFERENCE_KFOLD = (
    _SECTION_GNN
    / "outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models"
    / "ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold"
)
GRAPH_CACHE_DIR = (
    _SECTION_GNN
    / "data/ablations/entity_resolved_data/cross_bucket_total_dataset/graph_cache/section"
)
GRAPH_CACHE = GRAPH_CACHE_DIR / "case_star_entity_resolved_cross_bucket_section_sep_lr_decay.reasoning_focused.pt"

K = 5
BASE_SEED = 42
VAL_FRACTION = 0.1

_failures: list[str] = []


def _report(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _failures.append(name)


# ----------------------------------------------------------------- 1. splits


def check_splits() -> None:
    print("\n1. SPLIT EQUIVALENCE")
    frames = {
        fold: pd.read_csv(REFERENCE_KFOLD / f"fold_{fold:02d}" / "predictions.csv")
        for fold in range(K)
    }
    reference = frames[0]
    n_cases = len(reference)
    y_all = reference["target_index"].to_numpy()

    order_ok = all((frames[f]["case_id"].values == reference["case_id"].values).all() for f in range(K))
    _report("case_id order identical across folds", order_ok, f"{n_cases} rows")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=BASE_SEED)
    fold_splits = list(skf.split(np.arange(n_cases), y_all))

    for fold in range(K):
        train_fold_idx, test_fold_idx = fold_splits[fold]
        rng = np.random.default_rng(BASE_SEED + fold)
        train_mask, val_mask, test_mask = _make_fold_masks(
            n_cases, train_fold_idx, test_fold_idx, VAL_FRACTION, rng
        )
        recorded = frames[fold]["split"].to_numpy()
        matches = {
            "train": bool((train_mask.numpy() == (recorded == "train")).all()),
            "val": bool((val_mask.numpy() == (recorded == "val")).all()),
            "test": bool((test_mask.numpy() == (recorded == "test")).all()),
        }
        sizes = f"train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
        _report(f"fold {fold} membership reproduced", all(matches.values()), sizes)

    test_sets = [set(frames[f].loc[frames[f]["split"] == "test", "case_id"]) for f in range(K)]
    overlaps = sum(
        len(test_sets[a] & test_sets[b]) for a in range(K) for b in range(a + 1, K)
    )
    union = set().union(*test_sets)
    _report(
        "test folds partition the corpus",
        overlaps == 0 and len(union) == n_cases,
        f"pairwise overlap={overlaps}, union={len(union)}/{n_cases}",
    )


# ----------------------------------------------------------------- 2. configs


def _flatten(node: object, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def check_configs() -> None:
    print("\n2. CONFIG EQUIVALENCE (only model.architecture + paths.outputs_dir may differ)")
    paper = _flatten(yaml.safe_load(PAPER_CONFIG.read_text()))
    allowed = {"model.architecture", "paths.outputs_dir"}
    configs = sorted((_HERE / "configs").glob("arch_*.yaml"))
    if not configs:
        _report("configs present", False, "no configs/arch_*.yaml -- run make_configs.py first")
        return
    for path in configs:
        variant = _flatten(yaml.safe_load(path.read_text()))
        differing = {
            key
            for key in set(paper) | set(variant)
            if paper.get(key, "<<missing>>") != variant.get(key, "<<missing>>")
        }
        # The width-matched controls are allowed to change hidden_dim as well.
        extra_allowed = allowed | ({"model.hidden_dim"} if "_wide" in path.stem else set())
        _report(
            path.name,
            differing <= extra_allowed,
            f"differs in {sorted(differing)}",
        )


# --------------------------------------------------------------- 3. scaffold


def _graph_feature_dims() -> tuple[int, int]:
    """Read feature widths from the metadata sidecar without loading 6.3 GB."""
    head = GRAPH_CACHE_DIR.joinpath("graph_metadata_section_sep.json").read_bytes()[:4096].decode(
        "utf-8", errors="ignore"
    )
    feature_dim = re.search(r'"feature_dim"\s*:\s*(\d+)', head)
    case_dim = re.search(r'"case_feature_dim"\s*:\s*(\d+)', head)
    if not feature_dim or not case_dim:
        raise RuntimeError("could not read feature_dim/case_feature_dim from graph metadata")
    return int(feature_dim.group(1)), int(case_dim.group(1))


def _graph_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    relations = json.loads(GRAPH_CACHE_DIR.joinpath("relation_mappings_section_sep.json").read_text())
    edge_types = [tuple(rel.split("|")) for rel in relations]
    node_types: list[str] = []
    for src, _, dst in edge_types:
        for node_type in (src, dst):
            if node_type not in node_types:
                node_types.append(node_type)
    return sorted(node_types), edge_types


def check_scaffold() -> None:
    print("\n3. MODEL-SCAFFOLD EQUIVALENCE (no training)")
    cfg = yaml.safe_load(PAPER_CONFIG.read_text())
    node_types, edge_types = _graph_metadata()
    feature_dim, case_feature_dim = _graph_feature_dims()
    input_dims = {
        node_type: (case_feature_dim if node_type == "case" else feature_dim) for node_type in node_types
    }
    _report(
        "graph metadata read from sidecars",
        len(node_types) == 17 and len(edge_types) == 42,
        f"{len(node_types)} node types, {len(edge_types)} relations, "
        f"feature_dim={feature_dim}, case_feature_dim={case_feature_dim}",
    )

    model = ArchLegalOutcomeGNN(
        metadata=(node_types, edge_types),
        input_dims=input_dims,
        out_dim=len(cfg["labels"]["class_order_binary"]),
        cfg=cfg["model"],
    )
    built = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    # The delegate adds a "_delegate." prefix; strip it to compare against the checkpoint.
    built = {key.removeprefix("_delegate."): value for key, value in built.items()}

    checkpoint = torch.load(REFERENCE_KFOLD / "fold_00" / "model.pt", map_location="cpu", weights_only=True)
    saved = {key: tuple(value.shape) for key, value in checkpoint.items()}

    _report(
        "state_dict keys match saved fold_00/model.pt",
        set(built) == set(saved),
        f"{len(built)} built vs {len(saved)} saved; "
        f"missing={sorted(set(saved) - set(built))[:3]} extra={sorted(set(built) - set(saved))[:3]}",
    )
    mismatched = {key for key in set(built) & set(saved) if built[key] != saved[key]}
    _report(
        "parameter shapes match saved fold_00/model.pt",
        not mismatched,
        f"{len(set(built) & set(saved))} shared tensors, {len(mismatched)} shape mismatches",
    )
    n_params = sum(v.numel() for v in checkpoint.values())
    print(f"         reference HGT parameter count: {n_params:,}")


# ------------------------------------------------------------------ 4. graph


def check_graph() -> None:
    print("\n4. GRAPH IDENTITY")
    summary = json.loads((REFERENCE_KFOLD / "kfold_summary.json").read_text())
    recorded = Path(summary["graph_cache"])
    exists = GRAPH_CACHE.exists()
    same = exists and recorded.exists() and GRAPH_CACHE.samefile(recorded)
    stat = GRAPH_CACHE.stat() if exists else None
    _report(
        "ablation loads the reference run's graph cache",
        same,
        f"{GRAPH_CACHE.name} ({stat.st_size / 1e9:.2f} GB)" if stat else "missing",
    )
    agg = summary["aggregate"]
    print(
        f"         reference HGT (not re-run): acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}  "
        f"macro_f1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}"
    )


# ---------------------------------------------------------- 5. completed runs


def check_completed_runs() -> None:
    """Post-hoc: every finished ablation must have trained on the reference folds."""
    runs = sorted((_HERE / "outputs" / "models").glob("arch_*_kfold/kfold"))
    if not runs:
        return
    print("\n5. COMPLETED RUNS USED THE REFERENCE SPLITS")
    reference = {
        fold: pd.read_csv(REFERENCE_KFOLD / f"fold_{fold:02d}" / "predictions.csv")[
            ["case_id", "split", "target_index"]
        ]
        for fold in range(K)
    }
    for directory in runs:
        name = directory.parent.name
        same_split = same_labels = True
        sizes = set()
        for fold in range(K):
            path = directory / f"fold_{fold:02d}" / "predictions.csv"
            if not path.exists():
                same_split = same_labels = False
                break
            frame = pd.read_csv(path)[["case_id", "split", "target_index"]]
            counts = frame["split"].value_counts()
            sizes.add(f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}")
            merged = frame.merge(reference[fold], on="case_id", suffixes=("", "_ref"))
            same_split &= len(merged) == len(frame) and bool((merged["split"] == merged["split_ref"]).all())
            same_labels &= bool((merged["target_index"] == merged["target_index_ref"]).all())
        _report(
            f"{name} matches reference folds",
            same_split and same_labels,
            f"train/val/test {sorted(sizes)}",
        )


def main() -> None:
    print("Harness equivalence checks (R3-04) -- no training, no writes to section_GNN/outputs")
    check_splits()
    check_configs()
    check_scaffold()
    check_graph()
    check_completed_runs()
    print()
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        sys.exit(1)
    print("All checks passed. The ablation harness reproduces the paper's split, config and model scaffold.")


if __name__ == "__main__":
    main()
