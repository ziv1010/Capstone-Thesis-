#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pattern_explainability_common as common


DEFAULT_EXPLANATION_DIR = (
    common.APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
)
DEFAULT_IDENTITY_TYPES = (
    "judge",
    "petitioner",
    "defence_lawyer",
    "petitioner_lawyer",
    "lawyer",
    "court",
    "respondent",
)


@dataclass
class IdentityTypeData:
    identity_type: str
    case_idx: np.ndarray
    group_idx: np.ndarray
    group_keys: list[str]
    group_display_names: list[str]
    relations: list[str]
    raw_edge_count: int
    raw_node_count: int


@dataclass
class ScoreBundle:
    scores: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    known_eval_links: int
    eval_links: int
    train_links: int
    train_identities: int
    eval_identities: int
    overlap_identities: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether identity names alone carry held-out outcome signal. "
            "This is a shortcut/leakage diagnostic, not a retraining ablation."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--identity-types",
        default=",".join(DEFAULT_IDENTITY_TYPES),
        help="Comma-separated identity/evidence node types to audit.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument(
        "--group-by",
        choices=("name", "node"),
        default="name",
        help=(
            "name groups case-specific entity nodes by normalized displayed identity; "
            "node audits raw graph node IDs."
        ),
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=5.0,
        help="Beta smoothing strength for train-label identity priors.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="Number of label permutations for p-values. Use 0 to skip.",
    )
    parser.add_argument(
        "--permutation-mode",
        choices=("domain", "global"),
        default="domain",
        help="domain shuffles train labels within domain buckets; global shuffles all train labels.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k-identities", type=int, default=50)
    parser.add_argument(
        "--min-train-support",
        type=int,
        default=2,
        help="Minimum train cases for identities shown in the top-skew table.",
    )
    parser.add_argument(
        "--eval-case-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit for evaluated cases.",
    )
    return parser.parse_args()


def parse_identity_types(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def log_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(-(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)).mean())


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean((scores - y_true) ** 2))


def accuracy_at_half(y_true: np.ndarray, scores: np.ndarray) -> float:
    pred = np.asarray(scores) >= 0.5
    return float((pred == np.asarray(y_true, dtype=bool)).mean())


def balanced_accuracy_at_half(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    pred = (np.asarray(scores) >= 0.5).astype(np.int8)
    pos = y_true == 1
    neg = ~pos
    recalls: list[float] = []
    if pos.any():
        recalls.append(float((pred[pos] == 1).mean()))
    if neg.any():
        recalls.append(float((pred[neg] == 0).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def metric_dict(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": roc_auc(y_true, scores),
        "accuracy_at_0_5": accuracy_at_half(y_true, scores),
        "balanced_accuracy_at_0_5": balanced_accuracy_at_half(y_true, scores),
        "brier_score": brier_score(y_true, scores),
        "log_loss": log_loss(y_true, scores),
    }


def collect_identity_edges(
    data: Any,
    identity_type: str,
    group_by: str,
) -> IdentityTypeData | None:
    case_chunks: list[np.ndarray] = []
    node_chunks: list[np.ndarray] = []
    relations: list[str] = []
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if src_type != "case" or dst_type != identity_type or relation.startswith("rev_"):
            continue
        edge_index = data[edge_type].edge_index.cpu().numpy()
        if edge_index.shape[1] == 0:
            continue
        case_chunks.append(edge_index[0].astype(np.int64, copy=False))
        node_chunks.append(edge_index[1].astype(np.int64, copy=False))
        relations.append(relation)

    if not case_chunks:
        return None

    case_idx = np.concatenate(case_chunks)
    node_idx = np.concatenate(node_chunks)
    raw_edge_count = int(len(case_idx))
    raw_node_count = int(data[identity_type].num_nodes)

    case_node_pairs = np.unique(np.column_stack([case_idx, node_idx]), axis=0)
    case_idx = case_node_pairs[:, 0].astype(np.int64, copy=False)
    node_idx = case_node_pairs[:, 1].astype(np.int64, copy=False)

    node_to_group = np.full(raw_node_count, -1, dtype=np.int64)
    key_to_group: dict[str, int] = {}
    group_keys: list[str] = []
    group_display_names: list[str] = []
    node_ids = data[identity_type].node_id

    for node in np.unique(node_idx):
        node_int = int(node)
        raw_id = str(node_ids[node_int])
        display = common.display_node_id(identity_type, raw_id, max_len=240)
        if group_by == "name":
            key = common.canonical_feature_name(identity_type, raw_id)
        else:
            key = f"{identity_type}:{node_int}"
        group = key_to_group.get(key)
        if group is None:
            group = len(group_keys)
            key_to_group[key] = group
            group_keys.append(key)
            group_display_names.append(display)
        node_to_group[node_int] = group

    group_idx = node_to_group[node_idx]
    valid = group_idx >= 0
    case_idx = case_idx[valid]
    group_idx = group_idx[valid]

    # Count each normalized identity once per case; repeated case-specific nodes with
    # the same displayed name should not get extra voting weight.
    case_group_pairs = np.unique(np.column_stack([case_idx, group_idx]), axis=0)
    return IdentityTypeData(
        identity_type=identity_type,
        case_idx=case_group_pairs[:, 0].astype(np.int64, copy=False),
        group_idx=case_group_pairs[:, 1].astype(np.int64, copy=False),
        group_keys=group_keys,
        group_display_names=group_display_names,
        relations=relations,
        raw_edge_count=raw_edge_count,
        raw_node_count=raw_node_count,
    )


def counts_for_train_labels(
    type_data: IdentityTypeData,
    y_by_case: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_link_mask = train_mask[type_data.case_idx]
    train_groups = type_data.group_idx[train_link_mask]
    train_cases_for_links = type_data.case_idx[train_link_mask]
    n_groups = len(type_data.group_keys)
    total = np.bincount(train_groups, minlength=n_groups).astype(np.float64)
    positive = np.bincount(
        train_groups,
        weights=y_by_case[train_cases_for_links],
        minlength=n_groups,
    ).astype(np.float64)
    return positive, total


def score_eval_cases(
    type_data: IdentityTypeData,
    positive_counts: np.ndarray,
    total_counts: np.ndarray,
    eval_mask: np.ndarray,
    eval_case_to_row: np.ndarray,
    n_eval_cases: int,
    prior_positive_rate: float,
    smoothing_alpha: float,
) -> ScoreBundle:
    eval_link_mask = eval_mask[type_data.case_idx]
    eval_cases_for_links = type_data.case_idx[eval_link_mask]
    eval_groups = type_data.group_idx[eval_link_mask]
    eval_rows = eval_case_to_row[eval_cases_for_links]
    valid_eval = eval_rows >= 0
    eval_rows = eval_rows[valid_eval]
    eval_groups = eval_groups[valid_eval]

    train_support_for_eval_link = total_counts[eval_groups]
    known_link_mask = train_support_for_eval_link > 0
    known_rows = eval_rows[known_link_mask]
    known_groups = eval_groups[known_link_mask]
    weights = total_counts[known_groups]
    smoothed_rates = (
        positive_counts[known_groups] + smoothing_alpha * prior_positive_rate
    ) / (total_counts[known_groups] + smoothing_alpha)

    numerator = np.bincount(
        known_rows,
        weights=weights * smoothed_rates,
        minlength=n_eval_cases,
    ).astype(np.float64)
    denominator = np.bincount(known_rows, weights=weights, minlength=n_eval_cases).astype(
        np.float64
    )
    scores = np.full(n_eval_cases, prior_positive_rate, dtype=np.float64)
    known_cases = denominator > 0
    scores[known_cases] = numerator[known_cases] / denominator[known_cases]

    train_link_mask = total_counts[type_data.group_idx] > 0
    train_links = int(train_link_mask.sum())
    eval_links = int(len(eval_groups))
    unique_eval_groups = np.unique(eval_groups)
    return ScoreBundle(
        scores=scores,
        numerator=numerator,
        denominator=denominator,
        known_eval_links=int(known_link_mask.sum()),
        eval_links=eval_links,
        train_links=train_links,
        train_identities=int((total_counts > 0).sum()),
        eval_identities=int(len(unique_eval_groups)),
        overlap_identities=int((total_counts[unique_eval_groups] > 0).sum()),
    )


def combine_score_bundles(
    bundles: list[ScoreBundle],
    n_eval_cases: int,
    prior_positive_rate: float,
) -> ScoreBundle:
    numerator = np.zeros(n_eval_cases, dtype=np.float64)
    denominator = np.zeros(n_eval_cases, dtype=np.float64)
    for bundle in bundles:
        numerator += bundle.numerator
        denominator += bundle.denominator
    scores = np.full(n_eval_cases, prior_positive_rate, dtype=np.float64)
    known_cases = denominator > 0
    scores[known_cases] = numerator[known_cases] / denominator[known_cases]
    return ScoreBundle(
        scores=scores,
        numerator=numerator,
        denominator=denominator,
        known_eval_links=sum(bundle.known_eval_links for bundle in bundles),
        eval_links=sum(bundle.eval_links for bundle in bundles),
        train_links=sum(bundle.train_links for bundle in bundles),
        train_identities=sum(bundle.train_identities for bundle in bundles),
        eval_identities=sum(bundle.eval_identities for bundle in bundles),
        overlap_identities=sum(bundle.overlap_identities for bundle in bundles),
    )


def domain_prior_scores(
    case_meta: pd.DataFrame,
    y_by_case: np.ndarray,
    train_cases: np.ndarray,
    eval_cases: np.ndarray,
    prior_positive_rate: float,
    smoothing_alpha: float,
) -> np.ndarray:
    train_domains = case_meta.loc[train_cases, "domain_bucket"].astype(str).to_numpy()
    eval_domains = case_meta.loc[eval_cases, "domain_bucket"].astype(str).to_numpy()
    domain_counts: dict[str, list[float]] = {}
    for case_idx, domain in zip(train_cases, train_domains, strict=False):
        counts = domain_counts.setdefault(domain, [0.0, 0.0])
        counts[0] += 1.0
        counts[1] += float(y_by_case[int(case_idx)])

    scores = np.full(len(eval_cases), prior_positive_rate, dtype=np.float64)
    for row, domain in enumerate(eval_domains):
        count, positive = domain_counts.get(domain, [0.0, 0.0])
        if count > 0:
            scores[row] = (positive + smoothing_alpha * prior_positive_rate) / (
                count + smoothing_alpha
            )
    return scores


def permuted_train_labels(
    y_by_case: np.ndarray,
    train_cases: np.ndarray,
    train_domains: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = y_by_case.copy()
    if mode == "global":
        shuffled[train_cases] = rng.permutation(y_by_case[train_cases])
        return shuffled

    for domain in np.unique(train_domains):
        domain_cases = train_cases[train_domains == domain]
        if len(domain_cases) > 1:
            shuffled[domain_cases] = rng.permutation(y_by_case[domain_cases])
    return shuffled


def permutation_auc_table(
    type_data_by_name: dict[str, IdentityTypeData],
    observed_bundles: dict[str, ScoreBundle],
    y_by_case: np.ndarray,
    y_eval: np.ndarray,
    train_cases: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    eval_case_to_row: np.ndarray,
    n_eval_cases: int,
    train_domains: np.ndarray,
    prior_positive_rate: float,
    smoothing_alpha: float,
    permutations: int,
    permutation_mode: str,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    if permutations <= 0:
        return {
            name: {
                "permutation_auc_mean": None,
                "permutation_auc_std": None,
                "permutation_auc_p_value": None,
            }
            for name in [*type_data_by_name, "combined"]
        }

    rng = np.random.default_rng(seed)
    auc_values: dict[str, list[float]] = {name: [] for name in type_data_by_name}
    auc_values["combined"] = []

    for _ in range(permutations):
        shuffled_y = permuted_train_labels(
            y_by_case=y_by_case,
            train_cases=train_cases,
            train_domains=train_domains,
            mode=permutation_mode,
            rng=rng,
        )
        perm_bundles: list[ScoreBundle] = []
        for name, type_data in type_data_by_name.items():
            positive_counts, total_counts = counts_for_train_labels(
                type_data=type_data,
                y_by_case=shuffled_y,
                train_mask=train_mask,
            )
            bundle = score_eval_cases(
                type_data=type_data,
                positive_counts=positive_counts,
                total_counts=total_counts,
                eval_mask=eval_mask,
                eval_case_to_row=eval_case_to_row,
                n_eval_cases=n_eval_cases,
                prior_positive_rate=prior_positive_rate,
                smoothing_alpha=smoothing_alpha,
            )
            auc_values[name].append(roc_auc(y_eval, bundle.scores))
            perm_bundles.append(bundle)
        combined = combine_score_bundles(perm_bundles, n_eval_cases, prior_positive_rate)
        auc_values["combined"].append(roc_auc(y_eval, combined.scores))

    rows: dict[str, dict[str, float | None]] = {}
    for name, values in auc_values.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        observed_auc = roc_auc(y_eval, observed_bundles[name].scores)
        if arr.size == 0 or not math.isfinite(observed_auc):
            p_value = None
            mean_auc = None
            std_auc = None
        else:
            p_value = float((1.0 + np.sum(arr >= observed_auc)) / (arr.size + 1.0))
            mean_auc = float(arr.mean())
            std_auc = float(arr.std(ddof=0))
        rows[name] = {
            "permutation_auc_mean": mean_auc,
            "permutation_auc_std": std_auc,
            "permutation_auc_p_value": p_value,
        }
    return rows


def load_counterfactual_summary(explanation_dir: Path) -> dict[str, dict[str, float | int]]:
    path = explanation_dir / "leakage_sensitivity_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    result: dict[str, dict[str, float | int]] = {}
    for row in df.to_dict("records"):
        evidence_type = str(row.get("evidence_type", ""))
        if not evidence_type:
            continue
        result[evidence_type] = {
            "counterfactual_n_groups": int(row.get("n_groups", 0) or 0),
            "counterfactual_mean_abs_delta_pred_proba": safe_float(
                row.get("mean_abs_delta_pred_proba")
            ),
            "counterfactual_flip_rate": safe_float(row.get("flip_rate")),
        }
    return result


def summary_row(
    name: str,
    bundle: ScoreBundle,
    y_eval: np.ndarray,
    prior_scores: np.ndarray,
    domain_scores: np.ndarray,
    train_cases: np.ndarray,
    eval_cases: np.ndarray,
    train_positive_rate: float,
    eval_positive_rate: float,
    type_data: IdentityTypeData | None,
    group_by: str,
    permutation_stats: dict[str, float | None],
    counterfactual_stats: dict[str, float | int],
) -> dict[str, Any]:
    metrics = metric_dict(y_eval, bundle.scores)
    prior_metrics = metric_dict(y_eval, prior_scores)
    domain_metrics = metric_dict(y_eval, domain_scores)
    known_case_count = int((bundle.denominator > 0).sum())
    row: dict[str, Any] = {
        "identity_scope": name,
        "group_by": group_by,
        "relations": "|".join(type_data.relations) if type_data else "",
        "n_train_cases": int(len(train_cases)),
        "n_eval_cases": int(len(eval_cases)),
        "train_positive_rate": train_positive_rate,
        "eval_positive_rate": eval_positive_rate,
        "n_train_identity_links": bundle.train_links,
        "n_eval_identity_links": bundle.eval_links,
        "n_train_identities": bundle.train_identities,
        "n_eval_identities": bundle.eval_identities,
        "n_overlap_identities": bundle.overlap_identities,
        "eval_identity_overlap_share": (
            bundle.overlap_identities / bundle.eval_identities if bundle.eval_identities else 0.0
        ),
        "known_eval_links": bundle.known_eval_links,
        "known_eval_link_share": (
            bundle.known_eval_links / bundle.eval_links if bundle.eval_links else 0.0
        ),
        "known_eval_cases": known_case_count,
        "known_eval_case_share": known_case_count / len(eval_cases) if len(eval_cases) else 0.0,
        "identity_auc_roc": metrics["auc_roc"],
        "identity_accuracy_at_0_5": metrics["accuracy_at_0_5"],
        "identity_balanced_accuracy_at_0_5": metrics["balanced_accuracy_at_0_5"],
        "identity_brier_score": metrics["brier_score"],
        "prior_brier_score": prior_metrics["brier_score"],
        "domain_brier_score": domain_metrics["brier_score"],
        "identity_brier_delta_vs_prior": metrics["brier_score"] - prior_metrics["brier_score"],
        "identity_brier_delta_vs_domain": metrics["brier_score"] - domain_metrics["brier_score"],
        "identity_log_loss": metrics["log_loss"],
        "prior_log_loss": prior_metrics["log_loss"],
        "domain_log_loss": domain_metrics["log_loss"],
        "identity_log_loss_delta_vs_prior": metrics["log_loss"] - prior_metrics["log_loss"],
        "identity_log_loss_delta_vs_domain": metrics["log_loss"] - domain_metrics["log_loss"],
        "raw_edge_count": type_data.raw_edge_count if type_data else "",
        "raw_node_count": type_data.raw_node_count if type_data else "",
    }
    row.update(permutation_stats)
    row.update(counterfactual_stats)
    return row


def top_skew_rows(
    type_data: IdentityTypeData,
    positive_counts: np.ndarray,
    total_counts: np.ndarray,
    eval_mask: np.ndarray,
    train_positive_rate: float,
    smoothing_alpha: float,
    min_train_support: int,
    top_k: int,
) -> list[dict[str, Any]]:
    eval_link_mask = eval_mask[type_data.case_idx]
    eval_groups = type_data.group_idx[eval_link_mask]
    eval_counts = np.bincount(eval_groups, minlength=len(type_data.group_keys)).astype(np.int64)
    eligible = np.where((total_counts >= min_train_support) & (eval_counts > 0))[0]
    if eligible.size == 0:
        return []
    smoothed = (positive_counts + smoothing_alpha * train_positive_rate) / (
        total_counts + smoothing_alpha
    )
    raw_rate = np.divide(
        positive_counts,
        total_counts,
        out=np.full_like(positive_counts, np.nan, dtype=np.float64),
        where=total_counts > 0,
    )
    order = sorted(
        eligible.tolist(),
        key=lambda idx: (abs(smoothed[idx] - train_positive_rate), total_counts[idx]),
        reverse=True,
    )[:top_k]
    rows = []
    for idx in order:
        rows.append(
            {
                "identity_type": type_data.identity_type,
                "identity_key": type_data.group_keys[idx],
                "identity_name": type_data.group_display_names[idx],
                "train_support": int(total_counts[idx]),
                "train_positive": int(positive_counts[idx]),
                "train_negative": int(total_counts[idx] - positive_counts[idx]),
                "train_positive_rate": float(raw_rate[idx]),
                "smoothed_positive_rate": float(smoothed[idx]),
                "smoothed_lift_from_train_prior": float(smoothed[idx] - train_positive_rate),
                "eval_support": int(eval_counts[idx]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.explanation_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = common.infer_artifact_paths(
        explanation_dir=args.explanation_dir,
        graph_cache=args.graph_cache,
        predictions_csv=args.predictions_csv,
    )
    data, _ = common.load_graph(paths["graph_cache"])
    predictions = common.load_predictions(paths["predictions_csv"], expected_cases=int(data["case"].num_nodes))
    case_meta = common.build_case_metadata(data, predictions)

    identity_types = parse_identity_types(args.identity_types)
    train_cases = case_meta.loc[
        case_meta["split"].astype(str) == args.train_split, "case_index"
    ].to_numpy(dtype=np.int64)
    eval_cases = case_meta.loc[
        case_meta["split"].astype(str) == args.eval_split, "case_index"
    ].to_numpy(dtype=np.int64)
    if args.eval_case_limit is not None:
        eval_cases = eval_cases[: args.eval_case_limit]
    if train_cases.size == 0:
        raise ValueError(f"No train cases found for split={args.train_split!r}.")
    if eval_cases.size == 0:
        raise ValueError(f"No eval cases found for split={args.eval_split!r}.")

    n_cases = int(data["case"].num_nodes)
    train_mask = np.zeros(n_cases, dtype=bool)
    eval_mask = np.zeros(n_cases, dtype=bool)
    train_mask[train_cases] = True
    eval_mask[eval_cases] = True
    eval_case_to_row = np.full(n_cases, -1, dtype=np.int64)
    eval_case_to_row[eval_cases] = np.arange(len(eval_cases), dtype=np.int64)

    target_labels = case_meta["target_label"].astype(str).to_numpy()
    y_by_case = (target_labels == str(args.positive_label)).astype(np.float64)
    y_eval = y_by_case[eval_cases].astype(np.int8)
    train_positive_rate = float(y_by_case[train_cases].mean())
    eval_positive_rate = float(y_eval.mean())
    prior_scores = np.full(len(eval_cases), train_positive_rate, dtype=np.float64)
    domain_scores = domain_prior_scores(
        case_meta=case_meta,
        y_by_case=y_by_case,
        train_cases=train_cases,
        eval_cases=eval_cases,
        prior_positive_rate=train_positive_rate,
        smoothing_alpha=args.smoothing_alpha,
    )

    type_data_by_name: dict[str, IdentityTypeData] = {}
    for identity_type in identity_types:
        if identity_type not in data.node_types:
            continue
        type_data = collect_identity_edges(data, identity_type, args.group_by)
        if type_data is not None:
            type_data_by_name[identity_type] = type_data
    if not type_data_by_name:
        raise ValueError("No requested identity edge types were found in the graph.")

    observed_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_bundles: dict[str, ScoreBundle] = {}
    type_bundles: list[ScoreBundle] = []
    for name, type_data in type_data_by_name.items():
        positive_counts, total_counts = counts_for_train_labels(type_data, y_by_case, train_mask)
        observed_counts[name] = (positive_counts, total_counts)
        bundle = score_eval_cases(
            type_data=type_data,
            positive_counts=positive_counts,
            total_counts=total_counts,
            eval_mask=eval_mask,
            eval_case_to_row=eval_case_to_row,
            n_eval_cases=len(eval_cases),
            prior_positive_rate=train_positive_rate,
            smoothing_alpha=args.smoothing_alpha,
        )
        observed_bundles[name] = bundle
        type_bundles.append(bundle)
    observed_bundles["combined"] = combine_score_bundles(
        type_bundles,
        n_eval_cases=len(eval_cases),
        prior_positive_rate=train_positive_rate,
    )

    train_domains = case_meta.loc[train_cases, "domain_bucket"].astype(str).to_numpy()
    permutation_stats = permutation_auc_table(
        type_data_by_name=type_data_by_name,
        observed_bundles=observed_bundles,
        y_by_case=y_by_case,
        y_eval=y_eval,
        train_cases=train_cases,
        train_mask=train_mask,
        eval_mask=eval_mask,
        eval_case_to_row=eval_case_to_row,
        n_eval_cases=len(eval_cases),
        train_domains=train_domains,
        prior_positive_rate=train_positive_rate,
        smoothing_alpha=args.smoothing_alpha,
        permutations=args.permutations,
        permutation_mode=args.permutation_mode,
        seed=args.seed,
    )

    counterfactual_summary = load_counterfactual_summary(args.explanation_dir)
    summary_rows = []
    for name, bundle in observed_bundles.items():
        type_data = type_data_by_name.get(name)
        summary_rows.append(
            summary_row(
                name=name,
                bundle=bundle,
                y_eval=y_eval,
                prior_scores=prior_scores,
                domain_scores=domain_scores,
                train_cases=train_cases,
                eval_cases=eval_cases,
                train_positive_rate=train_positive_rate,
                eval_positive_rate=eval_positive_rate,
                type_data=type_data,
                group_by=args.group_by,
                permutation_stats=permutation_stats.get(name, {}),
                counterfactual_stats=counterfactual_summary.get(name, {}),
            )
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "identity_shortcut_summary.csv", index=False)

    top_rows: list[dict[str, Any]] = []
    for name, type_data in type_data_by_name.items():
        positive_counts, total_counts = observed_counts[name]
        top_rows.extend(
            top_skew_rows(
                type_data=type_data,
                positive_counts=positive_counts,
                total_counts=total_counts,
                eval_mask=eval_mask,
                train_positive_rate=train_positive_rate,
                smoothing_alpha=args.smoothing_alpha,
                min_train_support=args.min_train_support,
                top_k=args.top_k_identities,
            )
        )
    pd.DataFrame(top_rows).to_csv(output_dir / "identity_shortcut_top_skewed_identities.csv", index=False)

    case_rows = case_meta.loc[eval_cases].copy()
    case_rows["identity_score_combined"] = observed_bundles["combined"].scores
    case_rows["identity_known_combined"] = observed_bundles["combined"].denominator > 0
    case_rows["identity_train_support_combined"] = observed_bundles["combined"].denominator
    case_rows["domain_baseline_score"] = domain_scores
    case_rows["global_prior_score"] = prior_scores
    for name, bundle in observed_bundles.items():
        if name == "combined":
            continue
        case_rows[f"identity_score_{name}"] = bundle.scores
        case_rows[f"identity_known_{name}"] = bundle.denominator > 0
        case_rows[f"identity_train_support_{name}"] = bundle.denominator
    case_rows.to_csv(output_dir / "identity_shortcut_case_scores.csv", index=False)

    manifest = {
        "graph_cache": str(paths["graph_cache"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "explanation_dir": str(args.explanation_dir),
        "output_dir": str(output_dir),
        "identity_types": list(type_data_by_name),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "positive_label": str(args.positive_label),
        "group_by": args.group_by,
        "smoothing_alpha": args.smoothing_alpha,
        "permutations": args.permutations,
        "permutation_mode": args.permutation_mode,
        "seed": args.seed,
        "outputs": {
            "identity_shortcut_summary": str(output_dir / "identity_shortcut_summary.csv"),
            "identity_shortcut_case_scores": str(output_dir / "identity_shortcut_case_scores.csv"),
            "identity_shortcut_top_skewed_identities": str(
                output_dir / "identity_shortcut_top_skewed_identities.csv"
            ),
        },
        "interpretation": (
            "If identity-only AUC/log-loss improves over the global or domain baseline, "
            "the identity names encode held-out outcome signal. This is evidence of shortcut "
            "risk, not proof of data leakage. Compare these rows with the joined "
            "counterfactual flip-rate columns to distinguish model reliance from dataset "
            "correlation."
        ),
    }
    write_json(output_dir / "identity_shortcut_manifest.json", manifest)

    print(f"[identity-audit] wrote {output_dir / 'identity_shortcut_summary.csv'}")
    print(f"[identity-audit] wrote {output_dir / 'identity_shortcut_case_scores.csv'}")
    print(f"[identity-audit] wrote {output_dir / 'identity_shortcut_top_skewed_identities.csv'}")


if __name__ == "__main__":
    main()
