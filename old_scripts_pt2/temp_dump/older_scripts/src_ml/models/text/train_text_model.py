from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_ml.common.io import apply_splits, build_or_load_splits, load_cases_dataframe
from src_ml.common.labels import derive_model_labels
from src_ml.common.metrics import compute_classification_metrics
from src_ml.common.serialization import save_joblib, save_json, save_yaml
from src_ml.common.text_utils import join_text_fields, safe_text
from src_ml.models.text.classifier import train_text_classifier
from src_ml.models.text.embedder import load_or_compute_embeddings


def _build_texts(df: pd.DataFrame, text_mode: str) -> list[str]:
    rows: list[str] = []
    for row in df.to_dict(orient="records"):
        if text_mode == "facts_only":
            text = safe_text(row.get("facts_text"))
        elif text_mode == "args_only":
            text = join_text_fields(row, ["arguments_petitioner", "arguments_respondent"])
        elif text_mode == "input_text_only":
            text = safe_text(row.get("ml_input_text"))
        else:
            text = join_text_fields(
                row,
                ["facts_text", "arguments_petitioner", "arguments_respondent"],
            )
        if not text:
            text = safe_text(row.get("ml_input_text"))
        rows.append(text)
    return rows


def _extract_confidence(proba: np.ndarray | None) -> np.ndarray | None:
    if proba is None:
        return None
    if proba.ndim == 1:
        return proba
    return np.max(proba, axis=1)


def run_text_pipeline(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    out_root = Path(config["outputs"]["root"])

    text_cfg = config["text_model"]
    run_name = str(text_cfg.get("run_name", "text_run"))

    model_dir = out_root / "models" / "text" / run_name
    result_dir = out_root / "results" / "text"
    split_path = out_root / "splits" / "split_ids.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    df = load_cases_dataframe(
        path=dataset_cfg["jsonl_path"],
        limit=dataset_cfg.get("limit"),
        chunk_size=int(dataset_cfg.get("chunk_size", 4096)),
    )
    if df.empty:
        raise ValueError("No rows loaded for text pipeline")

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

    text_mode = str(text_cfg.get("text_mode", "facts_plus_args"))
    texts = _build_texts(work_df, text_mode=text_mode)
    work_df["model_text"] = texts

    emb = load_or_compute_embeddings(
        case_ids=work_df["case_id"].astype(str).tolist(),
        texts=texts,
        embed_cfg=text_cfg["embedder"],
        cache_dir=text_cfg["cache_dir"],
        namespace=f"{run_name}_{text_mode}",
        logger=logger,
    )

    split_masks = {
        split: (work_df["split"].values == split)
        for split in ("train", "val", "test")
    }

    X_train = emb[split_masks["train"]]
    y_train = work_df.loc[split_masks["train"], "y"].astype(int).values

    clf = train_text_classifier(
        X_train=X_train,
        y_train=y_train,
        cfg=text_cfg["classifier"],
        seed=int(split_cfg.get("seed", 42)),
    )

    metrics: dict[str, Any] = {
        "run_name": run_name,
        "text_mode": text_mode,
        "labels": label_names,
    }
    pred_frames: list[pd.DataFrame] = []

    for split in ("train", "val", "test"):
        mask = split_masks[split]
        X = emb[mask]
        y_true = work_df.loc[mask, "y"].astype(int).values
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

        metrics[f"{split}_metrics"] = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(range(len(label_names))),
            label_names=label_names,
            y_proba=y_proba,
        )

        conf = _extract_confidence(y_proba)
        pred_frames.append(
            pd.DataFrame(
                {
                    "case_id": work_df.loc[mask, "case_id"].astype(str).values,
                    "split": split,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_pred_name": [label_names[int(i)] for i in y_pred],
                    "prob_or_confidence": conf if conf is not None else np.nan,
                }
            )
        )

    preds_df = pd.concat(pred_frames, ignore_index=True)

    save_joblib(
        {
            "classifier": clf,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "text_mode": text_mode,
            "embedder_config": text_cfg["embedder"],
        },
        model_dir / "classifier.joblib",
    )
    np.save(model_dir / "embeddings.npy", emb)

    metrics_path = result_dir / f"{run_name}_metrics.json"
    preds_path = result_dir / f"{run_name}_preds.csv"
    save_json(metrics, metrics_path)
    preds_df.to_csv(preds_path, index=False)
    save_yaml(config, model_dir / "run_config_snapshot.yaml")

    summary = {
        "run_name": run_name,
        "n_rows": int(len(work_df)),
        "metrics_path": str(metrics_path),
        "preds_path": str(preds_path),
    }
    save_json(summary, model_dir / "summary.json")
    logger.info("Text pipeline complete | run_name=%s", run_name)
    return summary
