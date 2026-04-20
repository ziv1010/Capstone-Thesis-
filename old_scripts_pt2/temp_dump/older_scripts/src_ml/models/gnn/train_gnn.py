from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from src_ml.common.io import apply_splits, build_or_load_splits, load_cases_dataframe
from src_ml.common.labels import derive_model_labels
from src_ml.common.metrics import compute_classification_metrics
from src_ml.common.sklearn_compat import make_logistic_regression
from src_ml.common.serialization import save_joblib, save_json, save_torch, save_yaml
from src_ml.common.text_utils import safe_list, safe_text
from src_ml.models.gnn.gnn_model import is_pyg_available, train_gcn_with_pyg
from src_ml.models.gnn.graph_builder import build_case_entity_graph
from src_ml.models.text.embedder import load_or_compute_embeddings


def _build_texts(df: pd.DataFrame, text_mode: str) -> list[str]:
    texts: list[str] = []
    for row in df.to_dict(orient="records"):
        if text_mode == "input_text_only":
            text = safe_text(row.get("ml_input_text"))
        elif text_mode == "facts_only":
            text = safe_text(row.get("facts_text"))
        else:
            parts = [
                safe_text(row.get("facts_text")),
                safe_text(row.get("arguments_petitioner")),
                safe_text(row.get("arguments_respondent")),
            ]
            text = "\n\n".join([p for p in parts if p]).strip()
            if not text:
                text = safe_text(row.get("ml_input_text"))
        texts.append(text)
    return texts


def _fallback_features(df: pd.DataFrame, case_emb: np.ndarray) -> tuple[sparse.csr_matrix, DictVectorizer]:
    feature_dicts: list[dict[str, float]] = []
    for row in df.to_dict(orient="records"):
        feats: dict[str, float] = {}
        court = safe_text(row.get("court"))
        if court:
            feats[f"court={court}"] = 1.0
        year = row.get("year")
        if year is not None and str(year) != "nan":
            feats[f"year={int(year)}"] = 1.0
        case_type = safe_text(row.get("case_type"))
        if case_type:
            feats[f"case_type={case_type}"] = 1.0

        statutes = safe_list(row.get("statutes"))
        provisions = safe_list(row.get("provisions"))
        precedents = safe_list(row.get("precedents"))
        feats["count_statutes"] = float(len(statutes))
        feats["count_provisions"] = float(len(provisions))
        feats["count_precedents"] = float(len(precedents))

        for s in statutes:
            feats[f"statute={s}"] = 1.0
        for p in provisions:
            feats[f"provision={p}"] = 1.0
        for r in precedents:
            feats[f"precedent={r}"] = 1.0

        feature_dicts.append(feats)

    vec = DictVectorizer(sparse=True)
    X_struct = vec.fit_transform(feature_dicts)
    X_emb = sparse.csr_matrix(case_emb)
    X = sparse.hstack([X_emb, X_struct], format="csr")
    return X, vec


def _extract_confidence(proba: np.ndarray | None) -> np.ndarray | None:
    if proba is None:
        return None
    if proba.ndim == 1:
        return proba
    return np.max(proba, axis=1)


def run_gnn_pipeline(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    out_root = Path(config["outputs"]["root"])

    gnn_cfg = config["gnn"]
    run_name = str(gnn_cfg.get("run_name", "gnn_run"))
    model_dir = out_root / "models" / "gnn" / run_name
    result_dir = out_root / "results" / "gnn"
    split_path = out_root / "splits" / "split_ids.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    df = load_cases_dataframe(
        path=dataset_cfg["jsonl_path"],
        limit=dataset_cfg.get("limit"),
        chunk_size=int(dataset_cfg.get("chunk_size", 4096)),
    )
    if df.empty:
        raise ValueError("No rows loaded for gnn pipeline")

    labeled_df, label_to_id, id_to_label = derive_model_labels(df, config["labels"], logger=logger)
    label_names = [id_to_label[idx] for idx in sorted(id_to_label)]

    split_cfg = config["splits"]
    split_ids = build_or_load_splits(
        df=labeled_df,
        split_path=split_path,
        seed=int(split_cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train", 0.7)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        test_ratio=float(split_cfg.get("test", 0.15)),
        force_rebuild=bool(split_cfg.get("force_rebuild", False)),
        logger=logger,
    )

    work_df = apply_splits(labeled_df, split_ids)
    work_df = work_df.sort_values("case_id").reset_index(drop=True)

    text_mode = str(config["text_model"].get("text_mode", "facts_plus_args"))
    texts = _build_texts(work_df, text_mode=text_mode)

    case_emb = load_or_compute_embeddings(
        case_ids=work_df["case_id"].astype(str).tolist(),
        texts=texts,
        embed_cfg=config["text_model"]["embedder"],
        cache_dir=config["text_model"]["cache_dir"],
        namespace=f"gnn_{run_name}_{text_mode}",
        logger=logger,
    )

    y = work_df["y"].astype(int).values
    split_series = work_df["split"].astype(str).tolist()

    use_pyg = bool(gnn_cfg.get("use_pyg", True)) and is_pyg_available()
    logger.info("GNN mode | use_pyg=%s", use_pyg)

    pred_df: pd.DataFrame
    metrics: dict[str, Any] = {
        "run_name": run_name,
        "use_pyg": use_pyg,
        "labels": label_names,
    }

    if use_pyg:
        records = work_df.to_dict(orient="records")
        graph = build_case_entity_graph(
            records=records,
            case_embeddings=case_emb,
            add_case_case_edges=bool(gnn_cfg.get("add_case_case_edges", True)),
            seed=int(split_cfg.get("seed", 42)),
        )

        pyg_result = train_gcn_with_pyg(
            x=graph.x,
            edge_index=graph.edge_index,
            y_case=y,
            case_node_indices=graph.case_node_indices,
            split_name_by_case=split_series,
            cfg=gnn_cfg,
            seed=int(split_cfg.get("seed", 42)),
            logger=logger,
        )

        y_pred_all = pyg_result.y_pred_case
        y_proba_all = pyg_result.y_proba_case
        conf_all = _extract_confidence(y_proba_all)

        frames: list[pd.DataFrame] = []
        for split in ("train", "val", "test"):
            mask = work_df["split"].values == split
            metrics[f"{split}_metrics"] = compute_classification_metrics(
                y_true=y[mask],
                y_pred=y_pred_all[mask],
                labels=list(range(len(label_names))),
                label_names=label_names,
                y_proba=y_proba_all[mask],
            )
            frames.append(
                pd.DataFrame(
                    {
                        "case_id": work_df.loc[mask, "case_id"].astype(str).values,
                        "split": split,
                        "y_true": y[mask],
                        "y_pred": y_pred_all[mask],
                        "y_pred_name": [label_names[int(i)] for i in y_pred_all[mask]],
                        "prob_or_confidence": conf_all[mask] if conf_all is not None else np.nan,
                    }
                )
            )
        pred_df = pd.concat(frames, ignore_index=True)
        save_torch({"model_state": pyg_result.model_state}, model_dir / "model.pt")

    else:
        X, struct_vec = _fallback_features(work_df, case_emb)
        train_mask = work_df["split"].values == "train"

        clf = make_logistic_regression(
            max_iter=int(gnn_cfg.get("fallback_max_iter", 2000)),
            C=float(gnn_cfg.get("fallback_C", 1.0)),
            n_jobs=int(gnn_cfg.get("fallback_n_jobs", 1)),
            random_state=int(split_cfg.get("seed", 42)),
        )
        clf.fit(X[train_mask], y[train_mask])

        y_pred_all = clf.predict(X)
        y_proba_all = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None
        conf_all = _extract_confidence(y_proba_all)

        frames: list[pd.DataFrame] = []
        for split in ("train", "val", "test"):
            mask = work_df["split"].values == split
            metrics[f"{split}_metrics"] = compute_classification_metrics(
                y_true=y[mask],
                y_pred=y_pred_all[mask],
                labels=list(range(len(label_names))),
                label_names=label_names,
                y_proba=y_proba_all[mask] if y_proba_all is not None else None,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "case_id": work_df.loc[mask, "case_id"].astype(str).values,
                        "split": split,
                        "y_true": y[mask],
                        "y_pred": y_pred_all[mask],
                        "y_pred_name": [label_names[int(i)] for i in y_pred_all[mask]],
                        "prob_or_confidence": conf_all[mask] if conf_all is not None else np.nan,
                    }
                )
            )

        pred_df = pd.concat(frames, ignore_index=True)
        save_joblib(
            {"classifier": clf, "vectorizer": struct_vec, "label_to_id": label_to_id},
            model_dir / "fallback_model.joblib",
        )
        save_torch({"type": "fallback_logreg"}, model_dir / "model.pt")

    metrics_path = result_dir / f"{run_name}_metrics.json"
    preds_path = result_dir / f"{run_name}_preds.csv"
    save_json(metrics, metrics_path)
    pred_df.to_csv(preds_path, index=False)
    save_yaml(config, model_dir / "run_config_snapshot.yaml")

    summary = {
        "run_name": run_name,
        "use_pyg": use_pyg,
        "metrics_path": str(metrics_path),
        "preds_path": str(preds_path),
        "n_rows": int(len(work_df)),
    }
    save_json(summary, model_dir / "summary.json")
    logger.info("GNN pipeline complete | run_name=%s", run_name)
    return summary
