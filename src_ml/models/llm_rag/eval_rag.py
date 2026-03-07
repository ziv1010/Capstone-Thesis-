from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_ml.common.io import apply_splits, build_or_load_splits, load_cases_dataframe
from src_ml.common.labels import build_label_mapper, derive_model_labels
from src_ml.common.metrics import compute_classification_metrics
from src_ml.common.serialization import save_json, save_yaml
from src_ml.common.text_utils import safe_text
from src_ml.models.llm_rag.faiss_index import (
    build_faiss_index,
    l2_normalize,
    save_faiss_artifacts,
    search_faiss,
)
from src_ml.models.llm_rag.prompt_builder import build_rag_prompt
from src_ml.models.llm_rag.rag_predictor import LocalHTTPLLMClient, predict_with_llm_json
from src_ml.models.text.embedder import load_or_compute_embeddings


def _build_query_texts(df: pd.DataFrame, mode: str) -> list[str]:
    texts: list[str] = []
    for row in df.to_dict(orient="records"):
        if mode == "input_text_only":
            txt = safe_text(row.get("ml_input_text"))
        elif mode == "facts_only":
            txt = safe_text(row.get("facts_text"))
        else:
            parts = [
                safe_text(row.get("facts_text")),
                safe_text(row.get("arguments_petitioner")),
                safe_text(row.get("arguments_respondent")),
            ]
            txt = "\n\n".join([p for p in parts if p]).strip()
            if not txt:
                txt = safe_text(row.get("ml_input_text"))
        texts.append(txt)
    return texts


def _majority_vote_label(retrieved_items: list[dict[str, Any]], default_label: str) -> str:
    counts: dict[str, int] = {}
    for item in retrieved_items:
        name = str(item.get("y_name", ""))
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return default_label
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def run_rag_pipeline(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    out_root = Path(config["outputs"]["root"])

    rag_cfg = config["rag"]
    faiss_cfg = rag_cfg["faiss"]
    llm_cfg = rag_cfg["llm"]

    result_dir = out_root / "results" / "rag"
    rag_root = out_root / "rag"
    faiss_dir = rag_root / "faiss"
    split_path = out_root / "splits" / "split_ids.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    df = load_cases_dataframe(
        path=dataset_cfg["jsonl_path"],
        limit=dataset_cfg.get("limit"),
        chunk_size=int(dataset_cfg.get("chunk_size", 4096)),
    )
    if df.empty:
        raise ValueError("No rows loaded for rag pipeline")

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

    text_mode = str(rag_cfg.get("query_text_mode", "input_text_only"))
    texts = _build_query_texts(work_df, mode=text_mode)

    embed_cfg = config["text_model"]["embedder"]
    emb = load_or_compute_embeddings(
        case_ids=work_df["case_id"].astype(str).tolist(),
        texts=texts,
        embed_cfg=embed_cfg,
        cache_dir=config["text_model"]["cache_dir"],
        namespace=f"rag_{text_mode}",
        logger=logger,
    )

    train_mask = work_df["split"].values == "train"
    test_mask = work_df["split"].values == "test"

    train_df = work_df[train_mask].reset_index(drop=True)
    test_df = work_df[test_mask].reset_index(drop=True)
    train_emb = emb[train_mask]
    test_emb = emb[test_mask]

    train_metadata = []
    for rec in train_df.to_dict(orient="records"):
        snippet = safe_text(rec.get("facts_text"))
        if not snippet:
            snippet = safe_text(rec.get("ml_input_text"))
        train_metadata.append(
            {
                "case_id": str(rec.get("case_id")),
                "court": rec.get("court"),
                "year": rec.get("year"),
                "ml_input_text": rec.get("ml_input_text"),
                "facts_text": rec.get("facts_text"),
                "arguments_petitioner": rec.get("arguments_petitioner"),
                "arguments_respondent": rec.get("arguments_respondent"),
                "outcome_label": rec.get("outcome_label"),
                "outcome_winner": rec.get("outcome_winner"),
                "y_name": rec.get("y_name"),
                "snippet": snippet,
            }
        )

    index, train_case_ids, train_metadata = build_faiss_index(
        embeddings=train_emb,
        case_ids=train_df["case_id"].astype(str).tolist(),
        metadata=train_metadata,
        index_type=str(faiss_cfg.get("index_type", "flat_ip")),
    )

    save_faiss_artifacts(
        index=index,
        index_path=faiss_dir / "index.faiss",
        case_ids=train_case_ids,
        metadata=train_metadata,
        case_ids_path=faiss_dir / "train_case_ids.json",
        metadata_path=faiss_dir / "train_metadata.json",
    )

    mapper = build_label_mapper(config["labels"])

    llm_client = None
    if bool(llm_cfg.get("enabled", False)):
        llm_client = LocalHTTPLLMClient(
            endpoint_url=str(llm_cfg.get("endpoint_url", "")),
            model_name=str(llm_cfg.get("model_name", "")),
            timeout_sec=int(llm_cfg.get("timeout_sec", 120)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
        )

    top_k = int(rag_cfg.get("top_k", 5))
    snippet_max_chars = int(rag_cfg.get("snippet_max_chars", 1200))

    pred_rows: list[dict[str, Any]] = []
    audit_path = result_dir / "retrieval_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for idx, rec in enumerate(test_df.to_dict(orient="records")):
            qvec = l2_normalize(test_emb[idx : idx + 1])[0]
            distances, indices = search_faiss(index, qvec, top_k=top_k + 2)

            retrieved_items: list[dict[str, Any]] = []
            retrieved_case_ids: list[str] = []
            for rank, (score, pos) in enumerate(zip(distances, indices), start=1):
                if int(pos) < 0:
                    continue
                item = train_metadata[int(pos)]
                if str(item.get("case_id")) == str(rec.get("case_id")):
                    continue
                retrieved_items.append({**item, "score": float(score), "rank": rank})
                retrieved_case_ids.append(str(item.get("case_id")))
                if len(retrieved_items) >= top_k:
                    break

            prompt = build_rag_prompt(
                query_record=rec,
                retrieved_items=retrieved_items,
                label_names=label_names,
                snippet_max_chars=snippet_max_chars,
            )

            parsed: dict[str, Any] = {}
            raw_response = ""
            if llm_client is not None:
                try:
                    parsed, raw_response = predict_with_llm_json(llm_client, prompt=prompt, retry_on_parse_error=True)
                except Exception as exc:
                    raw_response = f"LLM_ERROR: {repr(exc)}"

            if parsed:
                mapped = mapper.map_from_record(
                    {
                        "decision": parsed.get("pred_label"),
                        "outcome.winner": parsed.get("pred_winner"),
                        "outcome.label": parsed.get("pred_label"),
                    }
                )
                if mapped is None:
                    mapped = _majority_vote_label(retrieved_items, default_label=label_names[0])
                conf = parsed.get("confidence")
                try:
                    confidence = float(conf)
                except Exception:
                    confidence = np.nan
            else:
                mapped = _majority_vote_label(retrieved_items, default_label=label_names[0])
                confidence = np.nan

            y_pred = label_to_id.get(mapped)
            if y_pred is None:
                y_pred = int(rec.get("y", 0))

            row = {
                "case_id": str(rec.get("case_id")),
                "split": "test",
                "y_true": int(rec.get("y")),
                "y_pred": int(y_pred),
                "y_pred_name": id_to_label[int(y_pred)],
                "prob_or_confidence": confidence,
            }
            pred_rows.append(row)

            audit = {
                "case_id": str(rec.get("case_id")),
                "retrieved_case_ids": retrieved_case_ids,
                "prompt": prompt,
                "raw_llm_response": raw_response,
                "parsed_prediction": parsed,
            }
            audit_file.write(json.dumps(audit, ensure_ascii=False) + "\n")

    pred_df = pd.DataFrame(pred_rows)
    y_true = pred_df["y_true"].astype(int).values
    y_pred = pred_df["y_pred"].astype(int).values

    metrics = {
        "run_name": str(rag_cfg.get("run_name", "rag_run")),
        "labels": label_names,
        "test_metrics": compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(range(len(label_names))),
            label_names=label_names,
            y_proba=None,
        ),
    }

    preds_path = result_dir / "preds.csv"
    metrics_path = result_dir / "metrics.json"
    pred_df.to_csv(preds_path, index=False)
    save_json(metrics, metrics_path)
    save_yaml(config, result_dir / "run_config_snapshot.yaml")

    summary = {
        "n_train_indexed": int(len(train_df)),
        "n_test_predicted": int(len(test_df)),
        "preds_path": str(preds_path),
        "metrics_path": str(metrics_path),
        "audit_path": str(audit_path),
    }
    save_json(summary, result_dir / "summary.json")
    logger.info("RAG pipeline complete | train=%d test=%d", len(train_df), len(test_df))
    return summary
