#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
SECTION_GNN_ROOT = REPO_ROOT / "section_GNN"
if str(SECTION_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SECTION_GNN_ROOT))

DEFAULT_EXPLANATION_DIR = APP_ROOT / "outputs/target_fold_00_8gpu"
DEFAULT_PATTERN_OUTPUT_DIR = APP_ROOT / "outputs/pattern_why"
DEFAULT_MODEL_PATH = (
    SECTION_GNN_ROOT
    / "outputs/timed_bucket_runs/cross_bucket_total_dataset/models/"
    / "ablation_section_sep_enc_cross_bucket_kfold/kfold/fold_00/model.pt"
)
DEFAULT_CONFIG_PATH = (
    SECTION_GNN_ROOT
    / "ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml"
)
DEFAULT_GRAPH_CACHE = (
    SECTION_GNN_ROOT
    / "data/timed_bucket_runs/cross_bucket_total_dataset/graph_cache/"
    / "case_star_cross_bucket_section_sep_enc.reasoning_focused.pt"
)
DEFAULT_PREDICTIONS_CSV = DEFAULT_MODEL_PATH.parent / "predictions.csv"

SECTION_NODE_TYPES = {
    "preamble",
    "facts",
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
}
STRUCTURAL_FEATURE_TYPES = {"statute", "provision", "precedent", "judge", "court"}
LEGAL_AUTHORITY_TYPES = {"statute", "provision", "precedent"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def first_summary_value(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    if value not in (None, ""):
        return value
    for shard in summary.get("shard_summaries", []) or []:
        value = shard.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_artifact_paths(
    explanation_dir: Path,
    graph_cache: Path | None = None,
    predictions_csv: Path | None = None,
    config: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Path]:
    summary = read_json(explanation_dir / "run_summary.json")
    return {
        "graph_cache": Path(graph_cache or first_summary_value(summary, "graph_cache") or DEFAULT_GRAPH_CACHE),
        "predictions_csv": Path(
            predictions_csv
            or first_summary_value(summary, "predictions_csv")
            or DEFAULT_PREDICTIONS_CSV
        ),
        "config": Path(config or first_summary_value(summary, "config") or DEFAULT_CONFIG_PATH),
        "model_path": Path(model_path or first_summary_value(summary, "model_path") or DEFAULT_MODEL_PATH),
    }


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def normalise_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return str(value)


def display_node_id(node_type: str, node_id: str, max_len: int = 180) -> str:
    marker = f"::{node_type}::"
    if marker in node_id:
        label = node_id.split(marker, 1)[1]
    elif node_id.startswith(f"{node_type}::"):
        label = node_id.split("::", 1)[1]
    elif "::" in node_id:
        label = node_id.rsplit("::", 1)[-1]
    else:
        label = node_id
    label = " ".join(str(label).split())
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def canonical_feature_name(node_type: str, node_id: str) -> str:
    text = display_node_id(node_type, node_id, max_len=1000).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def domain_bucket(case_id: str) -> str:
    if "__" in case_id:
        return case_id.split("__", 1)[0]
    return "unknown"


def load_graph(graph_cache: Path) -> Any:
    require_file(graph_cache, "graph cache")
    bundle = torch.load(graph_cache, map_location="cpu", weights_only=False)
    return bundle["data"], bundle.get("metadata", {})


def load_predictions(predictions_csv: Path, expected_cases: int | None = None) -> pd.DataFrame:
    require_file(predictions_csv, "predictions CSV")
    predictions = pd.read_csv(predictions_csv)
    if expected_cases is not None and len(predictions) != expected_cases:
        raise ValueError(
            f"Predictions rows ({len(predictions)}) do not match graph cases ({expected_cases})."
        )
    return predictions


def build_case_metadata(data: Any, predictions: pd.DataFrame) -> pd.DataFrame:
    n_cases = int(data["case"].num_nodes)
    rows = pd.DataFrame({"case_index": np.arange(n_cases, dtype=np.int64)})
    if "case_id" in predictions.columns:
        rows["case_id"] = predictions["case_id"].astype(str).to_numpy()
    elif hasattr(data["case"], "case_id"):
        rows["case_id"] = [str(value) for value in data["case"].case_id]
    else:
        rows["case_id"] = [display_node_id("case", str(value)) for value in data["case"].node_id]

    if "file_name" in predictions.columns:
        rows["file_name"] = predictions["file_name"].astype(str).to_numpy()
    elif hasattr(data["case"], "file_name"):
        rows["file_name"] = [str(value) for value in data["case"].file_name]
    else:
        rows["file_name"] = ""

    rows["split"] = predictions.get("split", pd.Series([""] * n_cases)).astype(str).to_numpy()
    if "target_label" in predictions.columns:
        rows["target_label"] = predictions["target_label"].map(normalise_label).to_numpy()
    else:
        rows["target_label"] = [normalise_label(value) for value in data["case"].y.tolist()]
    if "pred_label" in predictions.columns:
        rows["pred_label"] = predictions["pred_label"].map(normalise_label).to_numpy()
    else:
        rows["pred_label"] = ""
    rows["confidence"] = pd.to_numeric(
        predictions.get("confidence", pd.Series([np.nan] * n_cases)),
        errors="coerce",
    ).to_numpy()
    rows["correct"] = rows["target_label"].astype(str) == rows["pred_label"].astype(str)
    rows["domain_bucket"] = rows["case_id"].map(domain_bucket)
    return rows


def select_case_indices(
    case_meta: pd.DataFrame,
    split: str = "all",
    case_limit: int | None = None,
) -> np.ndarray:
    if split == "all":
        selected = case_meta["case_index"].to_numpy(dtype=np.int64)
    else:
        selected = case_meta.loc[case_meta["split"].astype(str) == split, "case_index"].to_numpy(
            dtype=np.int64
        )
    if case_limit is not None:
        selected = selected[:case_limit]
    if selected.size == 0:
        raise ValueError(f"No cases selected for split={split!r}.")
    return selected


def _node_ids_for_types(data: Any, node_types: set[str]) -> dict[str, list[str]]:
    return {
        node_type: [str(value) for value in data[node_type].node_id]
        for node_type in node_types
        if node_type in data.node_types
    }


def build_case_feature_matrix(
    data: Any,
    case_indices: np.ndarray,
    feature_types: set[str] | None = None,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    feature_types = set(feature_types or STRUCTURAL_FEATURE_TYPES)
    case_indices = np.asarray(case_indices, dtype=np.int64)
    case_to_row = {int(case_idx): row for row, case_idx in enumerate(case_indices.tolist())}
    node_ids = _node_ids_for_types(data, feature_types)

    feature_to_col: dict[tuple[str, str], int] = {}
    feature_records: list[dict[str, Any]] = []
    row_indices: list[int] = []
    col_indices: list[int] = []

    def feature_col(node_type: str, node_idx: int, relation: str) -> int:
        raw_id = node_ids[node_type][int(node_idx)]
        name = canonical_feature_name(node_type, raw_id)
        key = (node_type, name)
        existing = feature_to_col.get(key)
        if existing is not None:
            return existing
        col = len(feature_records)
        feature_to_col[key] = col
        feature_records.append(
            {
                "feature_index": col,
                "feature_type": node_type,
                "feature_name": name,
                "evidence_global_index": int(node_idx),
                "evidence_id": raw_id,
                "first_relation": relation,
            }
        )
        return col

    def add_feature(case_idx: int, node_type: str, node_idx: int, relation: str) -> None:
        row = case_to_row.get(int(case_idx))
        if row is None or node_type not in feature_types:
            return
        row_indices.append(row)
        col_indices.append(feature_col(node_type, int(node_idx), relation))

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type == "case" and dst_type in feature_types:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for case_idx, dst_idx in zip(edge_index[0], edge_index[1], strict=False):
                add_feature(int(case_idx), dst_type, int(dst_idx), relation)

    section_to_cases: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type != "case" or dst_type not in SECTION_NODE_TYPES:
            continue
        edge_index = data[edge_type].edge_index.cpu().numpy()
        section_map = section_to_cases[dst_type]
        for case_idx, section_idx in zip(edge_index[0], edge_index[1], strict=False):
            if int(case_idx) in case_to_row:
                section_map[int(section_idx)].append(int(case_idx))

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type not in SECTION_NODE_TYPES or dst_type not in feature_types:
            continue
        if not relation.startswith("cites_"):
            continue
        section_map = section_to_cases.get(src_type)
        if not section_map:
            continue
        edge_index = data[edge_type].edge_index.cpu().numpy()
        for section_idx, dst_idx in zip(edge_index[0], edge_index[1], strict=False):
            for case_idx in section_map.get(int(section_idx), []):
                add_feature(case_idx, dst_type, int(dst_idx), relation)

    if not feature_records:
        raise ValueError("No structural case features were extracted from the graph.")

    matrix = sp.coo_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (np.asarray(row_indices, dtype=np.int64), np.asarray(col_indices, dtype=np.int64)),
        ),
        shape=(len(case_indices), len(feature_records)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()

    feature_df = pd.DataFrame(feature_records)
    counts = np.asarray(matrix.getnnz(axis=0)).astype(np.int64)
    feature_df["corpus_case_count"] = counts
    feature_df["idf"] = np.log((len(case_indices) + 1.0) / (counts + 1.0)) + 1.0
    return matrix, feature_df


def save_case_feature_artifacts(
    output_dir: Path,
    matrix: sp.csr_matrix,
    feature_df: pd.DataFrame,
    case_indices: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "case_feature_matrix.npz"
    feature_path = output_dir / "case_feature_metadata.csv"
    case_path = output_dir / "case_feature_case_index.csv"
    sp.save_npz(matrix_path, matrix)
    feature_df.to_csv(feature_path, index=False)
    pd.DataFrame({"row_index": np.arange(len(case_indices)), "case_index": case_indices}).to_csv(
        case_path,
        index=False,
    )
    return {
        "case_feature_matrix": str(matrix_path),
        "case_feature_metadata": str(feature_path),
        "case_feature_case_index": str(case_path),
    }


def load_case_feature_artifacts(
    matrix_path: Path,
    feature_metadata_path: Path,
    case_index_path: Path,
) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    require_file(matrix_path, "case feature matrix")
    require_file(feature_metadata_path, "case feature metadata")
    require_file(case_index_path, "case feature case index")
    matrix = sp.load_npz(matrix_path).tocsr()
    feature_df = pd.read_csv(feature_metadata_path)
    case_rows = pd.read_csv(case_index_path)
    if matrix.shape[1] != len(feature_df) or matrix.shape[0] != len(case_rows):
        raise ValueError("Case feature artifacts have inconsistent shapes.")
    return matrix, feature_df, case_rows


def parse_gpu_list(gpus: str) -> list[int]:
    if gpus == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("--gpus all requires CUDA, but CUDA is not available.")
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("--gpus all found zero CUDA devices.")
        return list(range(count))
    parsed = [int(part.strip()) for part in gpus.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--gpus must be 'all' or a comma-separated GPU list.")
    return parsed
