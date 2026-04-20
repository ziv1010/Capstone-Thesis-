#!/usr/bin/env python3
"""
Cross-bucket generalisation evaluation for the Hetero-GNN.

Two modes
---------
pairwise      Train on every source bucket (full data, no held-out test set).
              For each trained model, run a forward pass on every other bucket
              and report accuracy.  Produces a 5×5 matrix (diagonal = skipped).

leave_one_out For each target bucket T, merge the remaining 4 source buckets
              into a single graph, train on that merged graph, and evaluate on
              the full target bucket T.

Because the GNN's input projection dimensions depend only on the feature encoder
(same BAAI/bge-m3 + scalar config across all buckets), a model trained on bucket A
can process bucket B's graph without any architecture changes.

Multi-GPU (parallel sources / targets)
---------------------------------------
Pairwise – one source per GPU:

    CUDA_VISIBLE_DEVICES=0 python cross_bucket_eval.py \\
        --configs c1.yaml c2.yaml c3.yaml c4.yaml c5.yaml \\
        --mode pairwise --source family_matrimonial_timed_mistral &

    CUDA_VISIBLE_DEVICES=1 python cross_bucket_eval.py \\
        --configs c1.yaml c2.yaml c3.yaml c4.yaml c5.yaml \\
        --mode pairwise --source fin_fraud_timed_mistral &
    ...

Leave-one-out – one target per GPU:

    CUDA_VISIBLE_DEVICES=0 python cross_bucket_eval.py \\
        --configs c1.yaml c2.yaml c3.yaml c4.yaml c5.yaml \\
        --mode leave_one_out --target family_matrimonial_timed_mistral &
    ...

When all parallel jobs finish, combine outputs:

    python cross_bucket_eval.py \\
        --configs c1.yaml c2.yaml c3.yaml c4.yaml c5.yaml \\
        --mode pairwise --summarise-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.training.metrics import compute_metrics, save_confusion_matrix_plot
from src.training.train import train_model
from src.utils.io import dump_json, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-bucket evaluation for the Hetero-GNN.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--configs", nargs="+", required=True,
        help="YAML config files, one per bucket",
    )
    p.add_argument(
        "--mode", choices=["pairwise", "leave_one_out"], default="pairwise",
        help="Evaluation mode (default: pairwise)",
    )
    p.add_argument(
        "--source",
        help="[pairwise] Run only this source bucket name (project.name in YAML). "
             "Omit to run all sources.",
    )
    p.add_argument(
        "--target",
        help="[leave_one_out] Hold out only this target bucket name. "
             "Omit to run all targets.",
    )
    p.add_argument(
        "--run-name", default="cross_bucket",
        help="Output sub-directory name under each bucket's outputs_dir",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of source data reserved as internal val when training "
             "on a full bucket (default 0.1)",
    )
    p.add_argument(
        "--summarise-only", action="store_true",
        help="Skip training; load already-saved pair results and print summary table.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _load_bundle(cfg: dict) -> dict:
    paths_cfg = cfg.get("paths", {})
    graph_path = Path(paths_cfg["graph_cache_dir"]) / str(cfg["graph"]["cache_name"])
    bundle = torch.load(graph_path, map_location="cpu", weights_only=False)
    return bundle


def _bool_mask(n: int, indices: np.ndarray) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[indices] = True
    return m


def _merge_bundles(bundles: list[dict], val_fraction: float, seed: int) -> tuple[HeteroData, list[str]]:
    """
    Merge multiple graph bundles into a single HeteroData.

    All source cases are placed in the train set; a small stratified fraction
    is held out as internal val for best-checkpoint selection during training.
    No test cases exist in the merged graph (the test bucket is always separate).
    """
    data_list = [b["data"] for b in bundles]
    label_names = list(bundles[0]["metadata"]["label_names"])

    # Collect all node and edge types across all bundles
    all_node_types: list[str] = []
    for d in data_list:
        for nt in d.node_types:
            if nt not in all_node_types:
                all_node_types.append(nt)

    all_edge_types: list[tuple[str, str, str]] = []
    for d in data_list:
        for et in d.edge_types:
            if et not in all_edge_types:
                all_edge_types.append(et)

    # Cumulative node counts per bundle (for edge-index offset)
    cum_counts: dict[str, list[int]] = {nt: [0] for nt in all_node_types}
    for d in data_list:
        for nt in all_node_types:
            prev = cum_counts[nt][-1]
            n = d[nt].x.shape[0] if nt in d.node_types else 0
            cum_counts[nt].append(prev + n)

    merged = HeteroData()

    # Merge node features
    for nt in all_node_types:
        x_parts = [d[nt].x for d in data_list if nt in d.node_types]
        if x_parts:
            merged[nt].x = torch.cat(x_parts, dim=0)
        # Merge node_id lists (non-case nodes)
        if nt != "case":
            nid_parts: list[Any] = []
            for d in data_list:
                if nt in d.node_types and hasattr(d[nt], "node_id"):
                    nid_parts.extend(d[nt].node_id)
            if nid_parts:
                merged[nt].node_id = nid_parts

    # Merge case-specific tensor attributes
    for attr in ("y",):
        parts = [getattr(d["case"], attr) for d in data_list if hasattr(d["case"], attr)]
        if parts:
            setattr(merged["case"], attr, torch.cat(parts, dim=0))

    # Merge case list attributes
    for attr in ("case_id", "file_name", "raw_label"):
        parts_list: list[Any] = []
        for d in data_list:
            if hasattr(d["case"], attr):
                parts_list.extend(getattr(d["case"], attr))
        if parts_list:
            setattr(merged["case"], attr, parts_list)

    # Assign train/val masks over all merged cases (no test in merged graph)
    n_total = int(merged["case"].y.shape[0])
    y_all = merged["case"].y.numpy()
    rng = np.random.default_rng(seed)
    if val_fraction > 0:
        n_val = max(1, int(n_total * val_fraction))
        perm = rng.permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        val_idx = np.array([], dtype=np.int64)
        train_idx = np.arange(n_total)

    merged["case"].train_mask = _bool_mask(n_total, train_idx)
    merged["case"].val_mask = _bool_mask(n_total, val_idx)
    merged["case"].test_mask = torch.zeros(n_total, dtype=torch.bool)

    # Merge edge indices (with per-bundle node offsets)
    for et in all_edge_types:
        src_type, _, dst_type = et
        ei_parts = []
        for bundle_idx, d in enumerate(data_list):
            if et not in d.edge_types:
                continue
            ei = d[et].edge_index.clone()
            ei[0] += cum_counts[src_type][bundle_idx]
            ei[1] += cum_counts[dst_type][bundle_idx]
            ei_parts.append(ei)
        if ei_parts:
            merged[et].edge_index = torch.cat(ei_parts, dim=1)

    return merged, label_names


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _train_on_full_bucket(
    bundle: dict,
    cfg: dict,
    logger: Any,
    val_fraction: float,
) -> tuple[HeteroLegalOutcomeGNN, dict]:
    """Train GNN on all cases of a single bucket (no held-out test set)."""
    data = deepcopy(bundle["data"])
    label_names = list(bundle["metadata"]["label_names"])
    n_cases = int(data["case"].y.shape[0])
    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    n_val = max(1, int(n_cases * val_fraction)) if val_fraction > 0 else 0
    perm = rng.permutation(n_cases)
    val_idx = perm[:n_val] if n_val else np.array([], dtype=np.int64)
    train_idx = perm[n_val:]

    data["case"].train_mask = _bool_mask(n_cases, train_idx)
    data["case"].val_mask = _bool_mask(n_cases, val_idx)
    data["case"].test_mask = torch.zeros(n_cases, dtype=torch.bool)

    set_global_seed(seed)
    result = train_model(
        data=data,
        label_names=label_names,
        cfg=cfg,
        seed=seed,
        logger=logger,
    )

    # Reconstruct model from saved state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dims = {nt: int(bundle["data"][nt].x.shape[1]) for nt in bundle["data"].node_types}
    model = HeteroLegalOutcomeGNN(
        metadata=bundle["data"].metadata(),
        input_dims=input_dims,
        out_dim=len(label_names),
        cfg=cfg.get("model", {}),
    ).to(device)
    model.load_state_dict(result["model_state_dict"])
    model.eval()

    return model, result["metrics"]


def _train_on_merged(
    source_bundles: list[dict],
    source_cfg: dict,
    logger: Any,
    val_fraction: float,
) -> tuple[HeteroLegalOutcomeGNN, dict, list[str]]:
    """Merge multiple source buckets and train a single GNN on the merged graph."""
    seed = int(source_cfg.get("project", {}).get("seed", 42))
    merged_data, label_names = _merge_bundles(source_bundles, val_fraction=val_fraction, seed=seed)

    logger.info(
        "Merged %d source bundles | total_cases=%d  train=%d  val=%d",
        len(source_bundles),
        int(merged_data["case"].y.shape[0]),
        int(merged_data["case"].train_mask.sum().item()),
        int(merged_data["case"].val_mask.sum().item()),
    )

    set_global_seed(seed)
    result = train_model(
        data=merged_data,
        label_names=label_names,
        cfg=source_cfg,
        seed=seed,
        logger=logger,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dims = {nt: int(merged_data[nt].x.shape[1]) for nt in merged_data.node_types}
    model = HeteroLegalOutcomeGNN(
        metadata=merged_data.metadata(),
        input_dims=input_dims,
        out_dim=len(label_names),
        cfg=source_cfg.get("model", {}),
    ).to(device)
    model.load_state_dict(result["model_state_dict"])
    model.eval()

    return model, result["metrics"], label_names


def _eval_on_target(
    model: HeteroLegalOutcomeGNN,
    target_bundle: dict,
    label_names: list[str],
    logger: Any,
) -> tuple[dict, np.ndarray]:
    """
    Run model forward pass on target bucket's graph and evaluate on ALL its cases.

    Edge types not present in the model's metadata are filtered out so the GNN
    does not crash when source and target bucket topologies differ slightly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = deepcopy(target_bundle["data"]).to(device)

    # Filter to edge types the model was trained on
    model_edge_set = set(tuple(et) for et in model.metadata[1])
    filtered_edge_index_dict = {
        et: ei for et, ei in data.edge_index_dict.items()
        if tuple(et) in model_edge_set
    }

    with torch.no_grad():
        logits, hidden = model(data.x_dict, filtered_edge_index_dict)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

    y_true = data["case"].y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_proba = probs.cpu().numpy()

    metrics = compute_metrics(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba, label_names=label_names,
    )
    case_embeddings = hidden["case"].cpu().numpy()

    logger.info(
        "  → target accuracy=%.4f  macro_f1=%.4f  n=%d",
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        int(metrics.get("n_samples", 0)),
    )
    return metrics, case_embeddings


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _bucket_name(cfg: dict) -> str:
    return str(cfg.get("project", {}).get("name", "unknown"))


def _print_matrix(results: dict[str, dict[str, dict]], bucket_names: list[str]) -> None:
    """Print a cross-bucket accuracy matrix to stdout."""
    col_w = max(len(n) for n in bucket_names) + 2
    header = f"{'SOURCE \\ TARGET':<{col_w}}" + "".join(f"{n:>{col_w}}" for n in bucket_names)
    print("\n" + header)
    print("-" * len(header))
    for src in bucket_names:
        row = f"{src:<{col_w}}"
        for tgt in bucket_names:
            if src == tgt:
                row += f"{'—':>{col_w}}"
            else:
                acc = results.get(src, {}).get(tgt, {}).get("accuracy", float("nan"))
                row += f"{acc:>{col_w}.3f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load all configs
    configs = [load_yaml(c) for c in args.configs]
    bucket_names = [_bucket_name(cfg) for cfg in configs]

    # Use the first config's outputs_dir for the shared summary
    first_paths = configs[0].get("paths", {})
    shared_out = ensure_dir(
        Path(first_paths["outputs_dir"]).parent / "cross_bucket" / args.run_name
    )
    log_dir = ensure_dir(shared_out / "logs")
    logger = configure_logger(f"cross_bucket_{args.run_name}", log_dir=log_dir)

    logger.info("Buckets: %s", bucket_names)
    logger.info("Mode: %s", args.mode)

    # ── Summarise-only shortcut ────────────────────────────────────────────
    if args.summarise_only:
        results: dict[str, dict[str, dict]] = {}
        for src in bucket_names:
            results[src] = {}
            for tgt in bucket_names:
                if src == tgt:
                    continue
                p = shared_out / src / tgt / "eval_metrics.json"
                if p.exists():
                    results[src][tgt] = json.loads(p.read_text())
        _print_matrix(results, bucket_names)
        dump_json(results, shared_out / "cross_bucket_matrix.json")
        return

    # ── Pairwise mode ─────────────────────────────────────────────────────
    if args.mode == "pairwise":
        sources_to_run = (
            [i for i, n in enumerate(bucket_names) if n == args.source]
            if args.source else list(range(len(configs)))
        )
        if not sources_to_run:
            raise ValueError(f"Source bucket '{args.source}' not found in configs.")

        all_results: dict[str, dict[str, dict]] = {}

        for src_idx in sources_to_run:
            src_name = bucket_names[src_idx]
            src_cfg = configs[src_idx]
            logger.info("══ Source: %s ══", src_name)

            # Load source graph and train on full bucket
            src_bundle = _load_bundle(src_cfg)
            logger.info("Training on full source bucket …")
            model, src_train_metrics = _train_on_full_bucket(
                bundle=src_bundle,
                cfg=src_cfg,
                logger=logger,
                val_fraction=args.val_fraction,
            )

            src_out = ensure_dir(shared_out / src_name)
            torch.save(model.state_dict(), src_out / "model.pt")
            dump_json(src_train_metrics, src_out / "source_train_metrics.json")

            all_results[src_name] = {}

            for tgt_idx, tgt_cfg in enumerate(configs):
                if tgt_idx == src_idx:
                    continue
                tgt_name = bucket_names[tgt_idx]
                logger.info("  Evaluating on target: %s", tgt_name)

                tgt_bundle = _load_bundle(tgt_cfg)
                label_names = list(src_bundle["metadata"]["label_names"])

                try:
                    eval_metrics, embeddings = _eval_on_target(
                        model=model,
                        target_bundle=tgt_bundle,
                        label_names=label_names,
                        logger=logger,
                    )
                except Exception as exc:
                    logger.warning("  Eval failed for %s→%s: %s", src_name, tgt_name, exc)
                    eval_metrics = {"error": str(exc)}
                    embeddings = np.array([])

                pair_out = ensure_dir(src_out / tgt_name)
                dump_json(eval_metrics, pair_out / "eval_metrics.json")
                if embeddings.size:
                    np.save(pair_out / "target_embeddings.npy", embeddings)

                confusion = eval_metrics.get("confusion_matrix", [])
                if confusion:
                    save_confusion_matrix_plot(
                        confusion=confusion,
                        label_names=label_names,
                        output_path=pair_out / "confusion_matrix.png",
                        title=f"Train={src_name}\nTest={tgt_name}",
                    )

                all_results[src_name][tgt_name] = {
                    "accuracy": float(eval_metrics.get("accuracy", float("nan"))),
                    "macro_f1": float(eval_metrics.get("macro_f1", float("nan"))),
                    "n_samples": int(eval_metrics.get("n_samples", 0)),
                }

        dump_json(all_results, shared_out / "cross_bucket_matrix.json")
        _print_matrix(all_results, bucket_names)

    # ── Leave-one-out mode ────────────────────────────────────────────────
    elif args.mode == "leave_one_out":
        targets_to_run = (
            [i for i, n in enumerate(bucket_names) if n == args.target]
            if args.target else list(range(len(configs)))
        )
        if not targets_to_run:
            raise ValueError(f"Target bucket '{args.target}' not found in configs.")

        loo_results: list[dict] = []

        for tgt_idx in targets_to_run:
            tgt_name = bucket_names[tgt_idx]
            tgt_cfg = configs[tgt_idx]
            logger.info("══ Target (held-out): %s ══", tgt_name)

            # Source: all other buckets
            src_indices = [i for i in range(len(configs)) if i != tgt_idx]
            src_bundles = [_load_bundle(configs[i]) for i in src_indices]
            src_names = [bucket_names[i] for i in src_indices]
            logger.info("  Source buckets: %s", src_names)

            # Use first source cfg as template (same model/training params)
            base_cfg = deepcopy(configs[src_indices[0]])
            base_cfg.setdefault("project", {})["name"] = f"loo_excl_{tgt_name}"

            model, merged_train_metrics, label_names = _train_on_merged(
                source_bundles=src_bundles,
                source_cfg=base_cfg,
                logger=logger,
                val_fraction=args.val_fraction,
            )

            tgt_bundle = _load_bundle(tgt_cfg)
            try:
                eval_metrics, embeddings = _eval_on_target(
                    model=model,
                    target_bundle=tgt_bundle,
                    label_names=label_names,
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("  Eval failed for LOO target %s: %s", tgt_name, exc)
                eval_metrics = {"error": str(exc)}
                embeddings = np.array([])

            loo_out = ensure_dir(shared_out / "leave_one_out" / tgt_name)
            torch.save(model.state_dict(), loo_out / "model.pt")
            dump_json(merged_train_metrics, loo_out / "merged_train_metrics.json")
            dump_json(eval_metrics, loo_out / "eval_metrics.json")
            if embeddings.size:
                np.save(loo_out / "target_embeddings.npy", embeddings)

            confusion = eval_metrics.get("confusion_matrix", [])
            if confusion:
                save_confusion_matrix_plot(
                    confusion=confusion,
                    label_names=label_names,
                    output_path=loo_out / "confusion_matrix.png",
                    title=f"LOO: train on {len(src_names)} buckets\nTest={tgt_name}",
                )

            row = {
                "target": tgt_name,
                "sources": src_names,
                "accuracy": float(eval_metrics.get("accuracy", float("nan"))),
                "macro_f1": float(eval_metrics.get("macro_f1", float("nan"))),
                "n_samples": int(eval_metrics.get("n_samples", 0)),
            }
            loo_results.append(row)
            logger.info(
                "LOO target=%s | accuracy=%.4f  macro_f1=%.4f",
                tgt_name, row["accuracy"], row["macro_f1"],
            )

        dump_json(loo_results, shared_out / "leave_one_out_results.json")
        print("\nLeave-One-Out Results")
        print("-" * 60)
        for row in loo_results:
            print(f"  Target={row['target']:40s}  acc={row['accuracy']:.4f}  f1={row['macro_f1']:.4f}")
        if loo_results:
            accs = [r["accuracy"] for r in loo_results if not np.isnan(r["accuracy"])]
            f1s = [r["macro_f1"] for r in loo_results if not np.isnan(r["macro_f1"])]
            print(f"\n  Mean accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"  Mean macro_f1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")

    logger.info("All outputs written to %s", shared_out)


if __name__ == "__main__":
    main()
