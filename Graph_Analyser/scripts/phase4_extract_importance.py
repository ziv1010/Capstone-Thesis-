#!/usr/bin/env python
"""Phase 4: For each (selected) test case, extract the most important
Statutes, Provisions, Precedents, and Argument-role nodes using the trained
heterogeneous PGExplainer, plus top-k similar training cases via FAISS.

Produces one JSON bundle per test case: the "evidence" that Phase 5 hands to
the LLM.

Multi-GPU usage (8 GPUs example):
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python scripts/phase4_extract_importance.py \
        --rank $i --world-size 8 &
  done
  wait
  python scripts/phase4_extract_importance.py --merge-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyser.loader import build_and_load_hgt, load_config, load_graph_cache  # noqa: E402
from analyser.hetero_pg_explainer import HeteroEdgeMaskerConfig, HeteroPGExplainer  # noqa: E402
from analyser.hetero_utils import (  # noqa: E402
    build_index_to_text,
    get_split_indices,
    is_self_match,
    load_case_text,
    load_fold_splits,
    owning_case_of_arg_node,
)


TARGET_ENTITY_NODE_TYPES = (
    "statute",
    "provision",
    "precedent",
)

ARGUMENT_NODE_TYPES = (
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--case-indices",
        type=str,
        default=None,
        help="Optional comma-separated list of case node indices. Overrides config top_n.",
    )
    parser.add_argument("--rank",       type=int, default=0, help="Worker rank (0-indexed)")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of parallel workers")
    parser.add_argument("--merge-only", action="store_true", help="Skip extraction; just merge partial manifests")
    return parser.parse_args()


def collect_k_hop_case_subgraph(
    data,
    case_node_index: int,
    num_hops: int,
) -> dict[str, set[int]]:
    """GPU-vectorised undirected BFS from a single case node.

    Uses boolean mask tensors instead of Python sets so all expansion ops
    are single GPU index operations rather than per-node Python loops.
    """
    device = next(iter(data.edge_index_dict.values())).device
    visited: dict[str, torch.Tensor] = {
        nt: torch.zeros(data[nt].num_nodes, dtype=torch.bool, device=device)
        for nt in data.node_types
    }
    visited["case"][case_node_index] = True

    for _ in range(num_hops):
        for (src_type, _, dst_type), edge_index in data.edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            active = visited[src_type][src]
            if active.any():
                visited[dst_type][dst[active]] = True
            active = visited[dst_type][dst]
            if active.any():
                visited[src_type][src[active]] = True

    return {
        nt: set(v.nonzero(as_tuple=False).view(-1).tolist())
        for nt, v in visited.items()
        if v.any()
    }


def per_node_max_edge_score(
    data,
    edge_importance: dict[tuple[str, str, str], torch.Tensor],
    reachable_nodes: dict[str, set[int]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """GPU-vectorised max-edge-score aggregation per reachable node.

    Uses scatter_reduce (amax) on GPU instead of a per-edge Python loop,
    then tracks which edge type produced the winning score per node.
    """
    device = next(iter(data.edge_index_dict.values())).device

    reachable_masks: dict[str, torch.Tensor] = {}
    for nt in data.node_types:
        mask = torch.zeros(data[nt].num_nodes, dtype=torch.bool, device=device)
        node_set = reachable_nodes.get(nt)
        if node_set:
            mask[torch.tensor(sorted(node_set), dtype=torch.long, device=device)] = True
        reachable_masks[nt] = mask

    max_scores: dict[str, torch.Tensor] = {
        nt: torch.full((data[nt].num_nodes,), -1.0, device=device)
        for nt in data.node_types
    }
    best_labels: dict[str, dict[int, str]] = {nt: {} for nt in data.node_types}

    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        scores = edge_importance.get(edge_type)
        if scores is None or scores.numel() == 0:
            continue
        src_type, relation, dst_type = edge_type
        edge_label = f"{src_type}|{relation}|{dst_type}"
        src, dst = edge_index[0], edge_index[1]
        scores_dev = scores.to(device)

        for node_type, node_indices in ((src_type, src), (dst_type, dst)):
            reach = reachable_masks[node_type][node_indices]
            if not reach.any():
                continue
            f_idx = node_indices[reach]
            f_sc  = scores_dev[reach]
            updated = max_scores[node_type].clone()
            updated.scatter_reduce_(0, f_idx, f_sc, reduce="amax", include_self=True)
            improved = (updated > max_scores[node_type]).nonzero(as_tuple=False).view(-1)
            if improved.numel():
                for idx in improved.tolist():
                    best_labels[node_type][idx] = edge_label
                max_scores[node_type] = updated

    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for nt, node_set in reachable_nodes.items():
        lbl_map = best_labels.get(nt, {})
        scores_nt = max_scores[nt]
        for idx in node_set:
            sc = float(scores_nt[idx].item())
            lbl = lbl_map.get(idx)
            if sc > -1.0 and lbl:
                out[nt][idx] = {"score": sc, "edge_type": lbl}
    return out


def top_k_from_node_scores(
    node_scores: dict[int, dict[str, Any]],
    index_to_text: dict[int, str],
    k: int,
    exclude_texts_matching: str | None = None,
    raw_index_to_text: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, info in node_scores.items():
        display_text = index_to_text.get(idx, f"<unknown_index_{idx}>")
        if exclude_texts_matching:
            if is_self_match(exclude_texts_matching, display_text):
                continue
            if raw_index_to_text is not None:
                raw_key = raw_index_to_text.get(idx, "")
                owning = owning_case_of_arg_node(raw_key)
                if owning and is_self_match(exclude_texts_matching, owning):
                    continue
        items.append(
            {
                "node_index": idx,
                "text": display_text,
                "importance": info["score"],
                "edge_type": info["edge_type"],
            }
        )
    items.sort(key=lambda item: item["importance"], reverse=True)
    return items[:k]


def build_raw_index_maps(metadata: dict[str, Any]) -> dict[str, dict[int, str]]:
    node_mappings = metadata.get("node_mappings", {})
    out: dict[str, dict[int, str]] = {}
    for node_type, mapping in node_mappings.items():
        if isinstance(mapping, dict):
            out[node_type] = {int(v): k for k, v in mapping.items()}
        else:
            out[node_type] = {i: key for i, key in enumerate(mapping)}
    return out


def extract_similar_case_argument_context(
    bundle_arg_top_nodes: dict[str, list[dict[str, Any]]],
    raw_index_maps: dict[str, dict[int, str]],
    cleaned_case_dir: Path,
    case_id_to_file_name: dict[str, str],
    max_entries: int = 4,
    per_section_chars: int = 600,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    flat: list[tuple[str, dict[str, Any]]] = []
    for arg_type, items in bundle_arg_top_nodes.items():
        for item in items:
            flat.append((arg_type, item))
    flat.sort(key=lambda pair: pair[1]["importance"], reverse=True)
    for arg_type, item in flat:
        raw_key = raw_index_maps.get(arg_type, {}).get(item["node_index"], "")
        owning_case = owning_case_of_arg_node(raw_key)
        if not owning_case or owning_case in seen:
            continue
        seen.add(owning_case)
        file_name = case_id_to_file_name.get(owning_case)
        section_text = ""
        if file_name:
            texts = load_case_text(cleaned_case_dir, file_name, max_chars_per_section=per_section_chars)
            section_text = texts.get(arg_type) or texts.get("arguments", "")
        candidates.append(
            {
                "owning_case": owning_case,
                "argument_section": arg_type,
                "importance": item["importance"],
                "snippet": section_text,
            }
        )
        if len(candidates) >= max_entries:
            break
    return candidates


def merge_manifests(output_root: Path, world_size: int) -> None:
    """Merge per-rank partial manifests into a single manifest.json."""
    all_indices: list[int] = []
    num_hops = 2
    for rank in range(world_size):
        partial = output_root / f"manifest_rank{rank}.json"
        if not partial.exists():
            print(f"[phase4 merge] warning: missing {partial}", flush=True)
            continue
        d = json.loads(partial.read_text())
        all_indices.extend(d["case_indices"])
        num_hops = d.get("num_hops", num_hops)
    all_indices.sort()
    manifest = {"n_cases_explained": len(all_indices), "num_hops": num_hops, "case_indices": all_indices}
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[phase4 merge] manifest.json written — {len(all_indices)} cases total", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    phase12_dir = Path(cfg["output_root"]) / "phase1_2_inference"
    phase3_dir  = Path(cfg["output_root"]) / "phase3_explainer"
    output_root = Path(cfg["output_root"]) / "phase4_explanations"
    (output_root / "cases").mkdir(parents=True, exist_ok=True)

    # ── merge-only mode: just stitch partial manifests ────────────────────
    if args.merge_only:
        merge_manifests(output_root, args.world_size)
        return

    if not (phase12_dir / "summary.json").exists():
        raise FileNotFoundError(f"Phase 1-2 artefacts missing at {phase12_dir}")
    if not (phase3_dir / "explainer.pt").exists():
        raise FileNotFoundError(f"Phase 3 explainer missing at {phase3_dir}")

    rank       = args.rank
    world_size = args.world_size
    device     = torch.device(args.device)

    prefix = f"[phase4 rank={rank}/{world_size}]"
    print(f"{prefix} loading graph cache: {cfg['graph_cache']}", flush=True)
    data, metadata = load_graph_cache(cfg["graph_cache"])
    label_names   = metadata.get("label_names", ["-1", "1"])
    index_to_text = build_index_to_text(metadata)
    raw_index_maps = build_raw_index_maps(metadata)

    cleaned_case_dir_cfg = cfg.get("cleaned_case_dir")
    if cleaned_case_dir_cfg:
        cleaned_case_dir = Path(cleaned_case_dir_cfg)
    else:
        graph_cache_parent = Path(cfg["graph_cache"]).parent
        cleaned_case_dir = graph_cache_parent.parent / "processed" / "cleaned_cases"
    if not cleaned_case_dir.exists():
        print(f"{prefix} warning: cleaned_case_dir not found at {cleaned_case_dir}; target case text will be omitted", flush=True)

    print(f"{prefix} loading HGT on {device}", flush=True)
    model, _ = build_and_load_hgt(
        data=data,
        checkpoint_dir=cfg["checkpoint_dir"],
        model_cfg=cfg.get("model"),
        section_gnn_root=cfg.get("section_gnn_root"),
        device=device,
    )
    data = data.to(device)

    explainer_cfg = cfg.get("explainer", {})
    masker_cfg = HeteroEdgeMaskerConfig(
        hidden_dim=int(explainer_cfg.get("hidden_dim", 64)),
        temperature_start=float(explainer_cfg.get("temperature_start", 5.0)),
        temperature_end=float(explainer_cfg.get("temperature_end", 1.0)),
        coeff_size=float(explainer_cfg.get("coeff_size", 0.01)),
        coeff_ent=float(explainer_cfg.get("coeff_ent", 1.0e-4)),
    )
    explainer = HeteroPGExplainer(model=model, data=data, masker_cfg=masker_cfg, device=device)
    explainer.load(str(phase3_dir / "explainer.pt"))

    print(f"{prefix} scoring all edges with trained explainer...", flush=True)
    edge_importance = {k: v.to(device) for k, v in explainer.edge_importance().items()}
    # keep data on GPU — BFS and scoring now use vectorised GPU ops

    # ── Load phase 1-2 artefacts ──────────────────────────────────────────
    case_embeddings = np.load(phase12_dir / "case_embeddings.npy")
    predictions     = np.load(phase12_dir / "predictions.npy")
    probabilities   = np.load(phase12_dir / "probabilities.npy")
    targets         = np.load(phase12_dir / "targets.npy")
    train_faiss_idx = np.load(phase12_dir / "train_indices.npy")

    import faiss  # noqa: WPS433

    index   = faiss.read_index(str(phase12_dir / "train_faiss.index"))
    metric  = json.loads((phase12_dir / "summary.json").read_text()).get("metric", "cosine")
    top_k_similar = int(cfg.get("faiss", {}).get("top_k_similar_cases", 3))

    case_ids   = list(data["case"].case_id)
    file_names = list(data["case"].file_name)
    raw_labels = list(data["case"].raw_label)

    fold_predictions_csv = Path(cfg["checkpoint_dir"]) / "predictions.csv"
    if fold_predictions_csv.exists():
        split_indices = load_fold_splits(str(fold_predictions_csv), data)
    else:
        split_indices = get_split_indices(data)
    test_indices = split_indices.get("test", [])

    # ── Select cases: test split only ────────────────────────────────────
    all_indices = list(range(len(case_ids)))
    if args.case_indices:
        selected = [int(x) for x in args.case_indices.split(",") if x.strip()]
    else:
        top_n = int(cfg.get("extraction", {}).get("top_n_test_cases", 100))
        if top_n <= 0:
            selected = all_indices  # all splits: train + val + test
        elif top_n >= len(test_indices):
            selected = test_indices  # all test cases
        else:
            confidences = probabilities[test_indices].max(axis=1)
            order = np.argsort(-confidences)
            selected = [int(test_indices[i]) for i in order[:top_n]]

    # ── Shard across workers ──────────────────────────────────────────────
    my_cases = selected[rank::world_size]
    print(f"{prefix} shard {rank}/{world_size}: {len(my_cases)}/{len(selected)} cases", flush=True)

    num_hops        = int(cfg.get("model", {}).get("num_layers", 2)) or 2
    top_k_statutes  = int(cfg.get("extraction", {}).get("top_k_statutes", 5))
    top_k_provisions = int(cfg.get("extraction", {}).get("top_k_provisions", 5))
    top_k_precedents = int(cfg.get("extraction", {}).get("top_k_precedents", 5))
    top_k_arguments = int(cfg.get("extraction", {}).get("top_k_arguments", 3))
    top_k_similar_arg_contexts = int(cfg.get("extraction", {}).get("top_k_similar_arg_contexts", 4))
    target_arg_char_budget     = int(cfg.get("extraction", {}).get("target_case_text_chars", 1200))
    similar_arg_snippet_chars  = int(cfg.get("extraction", {}).get("similar_case_arg_chars", 600))

    case_id_to_file_name = {cid: fn for cid, fn in zip(case_ids, file_names)}

    my_indices: list[int] = []

    for i, case_idx in enumerate(my_cases):
        out_path = output_root / "cases" / f"case_{case_idx}.json"
        if out_path.exists():
            my_indices.append(int(case_idx))
            if (i + 1) % 100 == 0:
                print(f"{prefix} skipped (cached) {i + 1}/{len(my_cases)}", flush=True)
            continue

        target_case_id   = case_ids[case_idx]
        target_file_name = file_names[case_idx]

        reachable  = collect_k_hop_case_subgraph(data=data, case_node_index=case_idx, num_hops=num_hops)
        node_scores = per_node_max_edge_score(data=data, edge_importance=edge_importance, reachable_nodes=reachable)

        top_nodes: dict[str, list[dict[str, Any]]] = {}
        for ent_type, k in [
            ("statute",   top_k_statutes),
            ("provision", top_k_provisions),
            ("precedent", top_k_precedents),
        ]:
            top_nodes[ent_type] = top_k_from_node_scores(
                node_scores.get(ent_type, {}),
                index_to_text.get(ent_type, {}),
                k=k,
                exclude_texts_matching=target_case_id,
                raw_index_to_text=raw_index_maps.get(ent_type, {}),
            )

        raw_arg_top_nodes: dict[str, list[dict[str, Any]]] = {}
        for arg_type in ARGUMENT_NODE_TYPES:
            raw_arg_top_nodes[arg_type] = top_k_from_node_scores(
                node_scores.get(arg_type, {}),
                index_to_text.get(arg_type, {}),
                k=top_k_arguments,
                exclude_texts_matching=target_case_id,
                raw_index_to_text=raw_index_maps.get(arg_type, {}),
            )

        similar_case_argument_context = extract_similar_case_argument_context(
            bundle_arg_top_nodes=raw_arg_top_nodes,
            raw_index_maps=raw_index_maps,
            cleaned_case_dir=cleaned_case_dir,
            case_id_to_file_name=case_id_to_file_name,
            max_entries=top_k_similar_arg_contexts,
            per_section_chars=similar_arg_snippet_chars,
        )

        target_case_text = load_case_text(
            cleaned_case_dir,
            target_file_name,
            max_chars_per_section=target_arg_char_budget,
        )

        query = case_embeddings[case_idx:case_idx + 1].astype(np.float32).copy()
        if metric == "cosine":
            faiss.normalize_L2(query)
        distances, neighbours = index.search(query, top_k_similar)
        similar_cases: list[dict[str, Any]] = []
        for rank_nb, (nb_row_idx, dist) in enumerate(zip(neighbours[0].tolist(), distances[0].tolist())):
            if nb_row_idx < 0:
                continue
            case_node_idx = int(train_faiss_idx[nb_row_idx])
            neighbour_case_id = case_ids[case_node_idx]
            if is_self_match(target_case_id, neighbour_case_id):
                continue
            similar_cases.append(
                {
                    "rank": len(similar_cases) + 1,
                    "case_node_index": case_node_idx,
                    "case_id": neighbour_case_id,
                    "file_name": file_names[case_node_idx],
                    "raw_label": raw_labels[case_node_idx],
                    "target_label": label_names[int(targets[case_node_idx])],
                    "predicted_label": label_names[int(predictions[case_node_idx])],
                    "similarity": float(dist),
                }
            )

        bundle = {
            "case_node_index": int(case_idx),
            "case_id": target_case_id,
            "file_name": target_file_name,
            "raw_label": raw_labels[case_idx],
            "target_label": label_names[int(targets[case_idx])],
            "predicted_label": label_names[int(predictions[case_idx])],
            "confidence": float(probabilities[case_idx].max()),
            "class_probabilities": {
                label_names[c]: float(probabilities[case_idx][c]) for c in range(probabilities.shape[1])
            },
            "num_hops": num_hops,
            "target_case_text": target_case_text,
            "top_nodes": {
                "statute":   top_nodes["statute"],
                "provision": top_nodes["provision"],
                "precedent": top_nodes["precedent"],
            },
            "similar_case_argument_context": similar_case_argument_context,
            "similar_training_cases": similar_cases,
            "_raw_argument_top_nodes": raw_arg_top_nodes,
        }
        with open(out_path, "w") as f:
            json.dump(bundle, f, indent=2)
        my_indices.append(int(case_idx))

        if (i + 1) % 50 == 0 or i == len(my_cases) - 1:
            print(f"{prefix} processed {i + 1}/{len(my_cases)}", flush=True)

    # Write partial manifest for this rank
    partial_manifest = {
        "rank": rank,
        "n_cases_explained": len(my_indices),
        "num_hops": num_hops,
        "case_indices": my_indices,
    }
    with open(output_root / f"manifest_rank{rank}.json", "w") as f:
        json.dump(partial_manifest, f, indent=2)
    print(f"{prefix} done. partial manifest written.", flush=True)

    # If single-process run, write the final manifest directly
    if world_size == 1:
        merge_manifests(output_root, world_size=1)


if __name__ == "__main__":
    main()
