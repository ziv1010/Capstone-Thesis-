#!/usr/bin/env python3
"""Consolidate every R3-03 / R3-04 result into one JSON for the dashboard.

Pulls from six places, none of which are re-run:
  * R3_03_non_llm_baselines/outputs/baselines_summary.json      (classical models)
  * R3_03_non_llm_baselines/outputs/tfidf_input_audit.json      (leakage audit)
  * R3_03_non_llm_baselines/outputs/run_baselines.log           (pre-sanitizer numbers)
  * R3_04_gnn_architecture_ablation/outputs/models/*/kfold/      (GNN variants)
  * section_GNN/outputs/.../kfold_summary.json                   (published HGT)
  * model comparison/outputs/*/metrics.json                      (InLegalLlama)

Also computes, for every model with per-case predictions, an exact McNemar test
against the HGT over the pooled folds. Because the five test folds partition the
corpus, pooling gives exactly one held-out prediction per case per model --
71,813 paired observations, not 5 fold means.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_SECTION_GNN = _REPO / "section_GNN"
REFERENCE_KFOLD = (
    _SECTION_GNN
    / "outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models"
    / "ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold"
)
R3_03 = _HERE / "R3_03_non_llm_baselines" / "outputs"
R3_04 = _HERE / "R3_04_gnn_architecture_ablation" / "outputs"
LLM_RUNS = {
    "InLegalLlama (CPT)": "lnlproc_inlegalllama_entity_resolved_section_sep_lr_decay_cross_bucket_fold00_test",
    "InLegalLlama (SFT)": "lnlproc_inlegalllama_sft_pred_only_original_prompt_cross_bucket_fold00",
}

DISPLAY = {
    "B0_majority": ("Majority class", "trivial", "predicts the 60.8% positive class for every case"),
    "B1_stratified": ("Stratified random", "trivial", "samples from the training class prior"),
    "B2_logreg_entity": ("Logistic Regression", "classical", "the GNN's own 12 case scalars + entity counts + role histogram"),
    "B5_xgboost_entity": ("XGBoost", "classical", "entity counts + top-4000 canonical authority counts"),
    "B6_xgboost_tfidf_svd": ("XGBoost", "classical", "sanitized TF-IDF reduced to 256 SVD components"),
    "B4_logreg_tfidf": ("Logistic Regression", "classical", "sanitized TF-IDF, 300k word 1-2 grams"),
    "B3_svm_tfidf": ("Linear SVM", "classical", "sanitized TF-IDF, 300k word 1-2 grams"),
    "arch_mlp_kfold": ("MLP (no message passing)", "gnn", "same case features, all edges removed"),
    "arch_gcn_kfold": ("GCN", "gnn", "type-collapsed graph, no relation weights"),
    "arch_sage_kfold": ("GraphSAGE", "gnn", "type-collapsed graph, no relation weights"),
    "arch_gat_kfold": ("GAT", "gnn", "type-collapsed graph, untyped attention"),
    "arch_rgcn_kfold": ("R-GCN", "gnn", "one SAGE operator per relation, no attention"),
    "arch_hgat_kfold": ("Relational GAT", "gnn", "one GAT operator per relation"),
    "hgt": ("HGT", "hgt", "the published model: typed attention over 17 node types, 42 relations"),
}


def _pooled(kfold_dir: Path) -> pd.DataFrame | None:
    """One held-out prediction per case, pooled over the five disjoint test folds."""
    frames = []
    for fold in range(5):
        path = kfold_dir / f"fold_{fold:02d}" / "predictions.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        if "split" in frame.columns:
            frame = frame.loc[frame["split"] == "test"]
        frames.append(frame[["case_id", "target_index", "pred_index"]])
    pooled = pd.concat(frames, ignore_index=True).sort_values("case_id").reset_index(drop=True)
    return None if pooled["case_id"].duplicated().any() else pooled


def _mcnemar(reference_ok: np.ndarray, model_ok: np.ndarray) -> dict:
    ref_only = int(np.sum(reference_ok & ~model_ok))
    model_only = int(np.sum(~reference_ok & model_ok))
    n = ref_only + model_only
    return {
        "hgt_only_correct": ref_only,
        "model_only_correct": model_only,
        "n_discordant": n,
        "p_value": float(binomtest(ref_only, n, 0.5).pvalue) if n else 1.0,
        "n_paired": int(len(model_ok)),
    }


def _pre_sanitizer_numbers() -> dict[str, float]:
    """Accuracy before the final TF-IDF sanitizer, scraped from the first sweep's log."""
    log = R3_03 / "run_baselines.log"
    if not log.exists():
        return {}
    text = log.read_text()
    tail = text.split("=== summary ===")[-1]
    found = {}
    for match in re.finditer(r"^\s+(B\w+)\s+acc=([\d.]+)", tail, re.MULTILINE):
        found[match.group(1)] = float(match.group(2))
    return found


def main() -> None:
    models: list[dict] = []

    reference = json.loads((REFERENCE_KFOLD / "kfold_summary.json").read_text())
    hgt_aggregate = reference["aggregate"]
    hgt_folds = [f["test_accuracy"] for f in reference["folds"]]
    hgt_fold_f1 = np.array([f["test_macro_f1"] for f in reference["folds"]])
    pooled_reference = _pooled(REFERENCE_KFOLD)
    reference_ok = (pooled_reference["pred_index"] == pooled_reference["target_index"]).to_numpy()

    models.append(
        {
            "id": "hgt",
            "aggregate": hgt_aggregate,
            "fold_accuracy": hgt_folds,
            "fold_macro_f1": hgt_fold_f1.tolist(),
            "n_parameters": 2_011_508,
            "rerun": False,
            "significance": None,
        }
    )

    # --- classical baselines
    baselines = json.loads((R3_03 / "baselines_summary.json").read_text())
    pre_sanitizer = _pre_sanitizer_numbers()
    for name, entry in baselines["baselines"].items():
        if not entry.get("valid", True):
            continue  # the raw-text oracle is a diagnostic, not a reported baseline
        folds = entry["folds"]
        record = {
            "id": name,
            "aggregate": entry["aggregate"],
            "fold_accuracy": [folds[str(f)]["accuracy"] for f in range(5) if str(f) in folds],
            "fold_macro_f1": [folds[str(f)]["macro_f1"] for f in range(5) if str(f) in folds],
            "n_features": next((folds[k].get("n_features") for k in folds if folds[k].get("n_features")), None),
            "rerun": True,
            "accuracy_before_sanitizer": pre_sanitizer.get(name),
        }
        pooled = _pooled(R3_03 / "per_fold" / name)
        if pooled is not None and pooled["case_id"].equals(pooled_reference["case_id"]):
            model_ok = (pooled["pred_index"] == pooled["target_index"]).to_numpy()
            record["significance"] = _mcnemar(reference_ok, model_ok)
            record["pooled_accuracy"] = float(model_ok.mean())
        models.append(record)

    # --- GNN architecture variants
    for directory in sorted((R3_04 / "models").glob("arch_*_kfold/kfold")):
        name = directory.parent.name
        summary = json.loads((directory / "kfold_summary.json").read_text())
        info_path = directory / "arch_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        record = {
            "id": name,
            "aggregate": summary["aggregate"],
            "fold_accuracy": [f["test_accuracy"] for f in summary["folds"]],
            "fold_macro_f1": [f["test_macro_f1"] for f in summary["folds"]],
            "n_parameters": int(info.get("n_parameters", 0)) or None,
            "properties": {k: info.get(k) for k in ("graph", "relation_aware", "attention") if k in info},
            "rerun": True,
        }
        pooled = _pooled(directory)
        if pooled is not None and pooled["case_id"].equals(pooled_reference["case_id"]):
            model_ok = (pooled["pred_index"] == pooled["target_index"]).to_numpy()
            record["significance"] = _mcnemar(reference_ok, model_ok)
            record["pooled_accuracy"] = float(model_ok.mean())
            record["significance"]["paired_t_p"] = float(
                ttest_rel(hgt_fold_f1, np.array(record["fold_macro_f1"])).pvalue
            )
        models.append(record)

    for record in models:
        label, family, note = DISPLAY.get(record["id"], (record["id"], "other", ""))
        record["label"] = label
        record["family"] = family
        record["note"] = note

    # --- generative LLM baselines (single fold, for context)
    llm = []
    for label, directory in LLM_RUNS.items():
        path = _REPO / "model comparison" / "outputs" / directory / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        total = metrics.get("total_rows") or metrics.get("n_rows")
        parsed = metrics.get("parsed_rows")
        llm.append(
            {
                "label": label,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "selective_accuracy": metrics.get("selective_accuracy"),
                "selective_macro_f1": metrics.get("selective_macro_f1"),
                "parse_rate": (parsed / total) if (parsed and total) else None,
                "note": "full denominator; unparseable generations count as incorrect abstentions; fold 0 test set",
            }
        )

    payload = {
        "hgt_reference": {
            "accuracy": hgt_aggregate["accuracy_mean"],
            "accuracy_std": hgt_aggregate["accuracy_std"],
            "macro_f1": hgt_aggregate["macro_f1_mean"],
            "macro_f1_std": hgt_aggregate["macro_f1_std"],
            "roc_auc": hgt_aggregate.get("roc_auc_mean"),
            "rerun": False,
            "source": str(REFERENCE_KFOLD / "kfold_summary.json"),
        },
        "n_cases": 71813,
        "class_balance": {"positive": 43686, "negative": 28127},
        "models": sorted(models, key=lambda r: -r["aggregate"]["accuracy_mean"]),
        "llm_baselines": llm,
        "leakage_audit": json.loads((R3_03 / "tfidf_input_audit.json").read_text()),
        "section_ablation": (
            json.loads((R3_03 / "section_ablation.json").read_text())
            if (R3_03 / "section_ablation.json").exists()
            else None
        ),
        "fold_sizes": {"train": [51705, 51706], "val": 5745, "test": [14362, 14363]},
    }
    out = _HERE / "all_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  ({len(models)} models, {len(llm)} LLM rows)")
    for record in payload["models"]:
        sig = record.get("significance") or {}
        print(
            f"  {record['label']:<26} {record['family']:<10} "
            f"acc={record['aggregate']['accuracy_mean']:.4f} "
            f"p={sig.get('p_value', float('nan')):.2e}"
        )


if __name__ == "__main__":
    main()
