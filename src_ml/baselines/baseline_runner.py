from __future__ import annotations

from pathlib import Path
from typing import Any

from src_ml.baselines.structured_classifier import train_structured_classifier
from src_ml.baselines.tfidf_logreg import train_tfidf_logreg
from src_ml.common.io import apply_splits, build_or_load_splits, load_cases_dataframe
from src_ml.common.labels import derive_model_labels
from src_ml.common.serialization import save_joblib, save_json, save_yaml
from src_ml.common.text_utils import join_text_fields, safe_text


def _build_baseline_text_column(df, text_mode: str) -> list[str]:
    rows = []
    for row in df.to_dict(orient="records"):
        if text_mode == "input_text_only":
            text = safe_text(row.get("ml_input_text"))
        elif text_mode == "facts_only":
            text = safe_text(row.get("facts_text"))
        elif text_mode == "args_only":
            text = join_text_fields(
                row,
                fields=["arguments_petitioner", "arguments_respondent"],
            )
        else:
            text = join_text_fields(
                row,
                fields=["facts_text", "arguments_petitioner", "arguments_respondent"],
            )

        if not text:
            text = safe_text(row.get("ml_input_text"))
        rows.append(text)
    return rows


def run_baselines(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    out_root = Path(config["outputs"]["root"])

    models_dir = out_root / "models" / "baselines"
    results_dir = out_root / "results" / "baselines"
    split_path = out_root / "splits" / "split_ids.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_cases_dataframe(
        path=dataset_cfg["jsonl_path"],
        limit=dataset_cfg.get("limit"),
        chunk_size=int(dataset_cfg.get("chunk_size", 4096)),
    )
    if df.empty:
        raise ValueError("No rows loaded for baselines")

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
    text_mode = str(config["baselines"].get("text_mode", "facts_plus_args"))
    work_df["baseline_text"] = _build_baseline_text_column(work_df, text_mode=text_mode)

    logger.info("Running TF-IDF baseline | rows=%d", len(work_df))
    tfidf_artifacts, tfidf_metrics, tfidf_preds = train_tfidf_logreg(
        df=work_df,
        text_col="baseline_text",
        label_col="y",
        split_col="split",
        tfidf_cfg=config["baselines"]["tfidf"],
        model_cfg={**config["baselines"]["logreg"], "seed": split_cfg.get("seed", 42)},
        label_names=label_names,
    )

    save_joblib(tfidf_artifacts, models_dir / "tfidf_logreg.joblib")
    tfidf_preds.to_csv(results_dir / "tfidf_logreg_preds.csv", index=False)
    save_json(tfidf_metrics, results_dir / "tfidf_logreg_metrics.json")

    logger.info("Running structured baseline | rows=%d", len(work_df))
    structured_artifacts, structured_metrics, structured_preds = train_structured_classifier(
        df=work_df,
        label_col="y",
        split_col="split",
        model_cfg={**config["baselines"]["logreg"], "seed": split_cfg.get("seed", 42)},
        label_names=label_names,
    )

    save_joblib(structured_artifacts, models_dir / "structured_logreg.joblib")
    structured_preds.to_csv(results_dir / "structured_logreg_preds.csv", index=False)
    save_json(structured_metrics, results_dir / "structured_logreg_metrics.json")

    save_yaml(config, results_dir / "run_config_snapshot.yaml")

    summary = {
        "n_rows": int(len(work_df)),
        "n_labels": int(len(label_to_id)),
        "labels": label_names,
        "artifacts": {
            "tfidf_model": str(models_dir / "tfidf_logreg.joblib"),
            "structured_model": str(models_dir / "structured_logreg.joblib"),
            "split_ids": str(split_path),
        },
    }
    save_json(summary, results_dir / "summary.json")
    logger.info("Baselines complete")
    return summary
