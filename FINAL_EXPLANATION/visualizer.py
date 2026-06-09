#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import scipy.sparse as sp


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
STATIC_DIR = APP_ROOT / "visualizer_static"


TABLE_FILES = {
    "case_summary": "case_summary.csv",
    "case_counterfactual_groups": "case_counterfactual_groups.csv",
    "case_top_explanations": "case_top_explanations.csv",
    "typed_path_importance": "typed_path_importance.csv",
    "relation_type_importance": "relation_type_importance.csv",
    "evidence_type_importance": "evidence_type_importance.csv",
    "leakage_sensitivity_summary": "leakage_sensitivity_summary.csv",
    "identity_shortcut_summary": "identity_shortcut_summary.csv",
    "identity_shortcut_case_scores": "identity_shortcut_case_scores.csv",
    "identity_shortcut_top_skewed_identities": "identity_shortcut_top_skewed_identities.csv",
    "mask_sensitivity_summary": "mask_sensitivity_summary.csv",
    "mask_sensitivity_by_domain": "mask_sensitivity_by_domain.csv",
    "mask_sensitivity_top_domain_drops": "mask_sensitivity_top_domain_drops.csv",
    "mask_sensitivity_hub_authorities": "mask_sensitivity_hub_authorities.csv",
    "mask_sensitivity_case_predictions": "mask_sensitivity_case_predictions.csv",
    "attention_counterfactual_overlap": "attention_counterfactual_overlap.csv",
    "connected_case_label_distribution": "connected_case_label_distribution.csv",
    "faithfulness_curves": "faithfulness_curves.csv",
    "faithfulness_curve_summary": "faithfulness_curve_summary.csv",
    "faithfulness_auc_by_case": "faithfulness_auc_by_case.csv",
    "faithfulness_auc_summary": "faithfulness_auc_summary.csv",
    "prediction_bucket_cases": "prediction_bucket_cases.csv",
    "prediction_bucket_summary": "prediction_bucket_summary.csv",
    "prediction_bucket_evidence_types": "prediction_bucket_evidence_types.csv",
}

FULL_GRAPH_TABLE_FILES = {
    "resolution_sweep_summary": "resolution_sweep_summary.csv",
    "resolution_pairwise_alignment": "resolution_pairwise_alignment.csv",
    "community_lineage_chains": "community_lineage_chains.csv",
    "community_lineage_pairs": "community_lineage_pairs.csv",
    "full_graph_node_communities_long": "full_graph_node_communities_long.csv",
    "full_graph_node_communities_wide": "full_graph_node_communities_wide.csv",
    "authority_role_summary_res_1_00": "authority_role_summary_res_1.00.csv",
    "authority_role_classification_res_1_00": "authority_role_classification_res_1.00.csv",
}

PATTERN_TABLE_FILES = {
    "case_communities": "case_communities.csv",
    "community_profiles": "community_profiles.csv",
    "community_feature_profiles": "community_feature_profiles.csv",
    "community_success_failure": "community_success_failure.csv",
    "evidence_label_skew": "evidence_label_skew.csv",
    "case_top_explanations_with_skew": "case_top_explanations_with_skew.csv",
    "counterfactual_neighborhoods": "counterfactual_neighborhoods.csv",
    "counterfactual_neighborhood_feature_differences": "counterfactual_neighborhood_feature_differences.csv",
    "case_embedding_clusters": "case_embedding_clusters.csv",
    "embedding_cluster_profiles": "embedding_cluster_profiles.csv",
    "community_embedding_splits": "community_embedding_splits.csv",
    "structural_embedding_alignment": "structural_embedding_alignment.csv",
}


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in records]


def relation_note(relation: str | None) -> str | None:
    if not relation:
        return None
    relation = str(relation)
    if relation.startswith("rev_"):
        original = relation.removeprefix("rev_")
        return f"{relation} is the reverse graph edge for {original}; it lets HGT pass messages back along the original relation."
    return f"{relation} is an original typed graph relation."


def path_notes(path_family: str | None) -> list[str]:
    if not path_family:
        return []
    parts = str(path_family).split("->")
    notes = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            continue
        note = relation_note(part)
        if note:
            notes.append(note)
    return notes


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        out = int(value)
        return out
    except (TypeError, ValueError):
        return None


class ExplanationStore:
    def __init__(
        self,
        output_dir: Path,
        pattern_dir: Path | None = None,
        full_graph_dir: Path | None = None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.pattern_dir = self._resolve_pattern_dir(pattern_dir)
        self.full_graph_dir = self._resolve_full_graph_dir(full_graph_dir)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._pattern_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._full_graph_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._feature_cache: tuple[tuple[float, float, float], sp.csr_matrix, pd.DataFrame, pd.DataFrame] | None = None

    def _related_output_dir(self, suffix: str) -> Path:
        name = self.output_dir.name
        if name.endswith("_fold00"):
            stem = name[: -len("_fold00")]
            return self.output_dir.parent / f"{stem}_{suffix}"
        return self.output_dir.parent / f"{name}_{suffix}"

    def _resolve_pattern_dir(self, pattern_dir: Path | None) -> Path:
        candidates: list[Path] = []
        if pattern_dir is not None:
            candidates.append(pattern_dir)
        candidates.extend(
            [
                self.output_dir,
                self._related_output_dir("pattern_why"),
                self.output_dir.parent / "pattern_why",
                APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why",
                APP_ROOT / "outputs/pattern_why",
            ]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if (resolved / "community_profiles.csv").exists():
                return resolved
        return (pattern_dir or (self.output_dir.parent / "pattern_why")).resolve()

    def _resolve_full_graph_dir(self, full_graph_dir: Path | None) -> Path:
        candidates: list[Path] = []
        if full_graph_dir is not None:
            candidates.append(full_graph_dir)
        candidates.extend(
            [
                self._related_output_dir("full_graph"),
                self.output_dir.parent / "pattern_why_full_graph",
                APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph",
                APP_ROOT / "outputs/pattern_why_full_graph",
            ]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if (resolved / "full_graph_node_communities_long.csv").exists():
                return resolved
        return (full_graph_dir or (self.output_dir.parent / "pattern_why_full_graph")).resolve()

    def load_full_graph_csv(self, filename: str) -> pd.DataFrame:
        path = self.full_graph_dir / filename
        if not path.exists():
            return pd.DataFrame()
        mtime = path.stat().st_mtime
        cached = self._full_graph_cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_csv(path, low_memory=False)
        self._full_graph_cache[filename] = (mtime, df)
        return df

    def full_graph_resolutions(self) -> list[float]:
        sweep = self.load_full_graph_csv("resolution_sweep_summary.csv")
        if sweep.empty or "resolution" not in sweep.columns:
            return []
        return sorted(float(v) for v in sweep["resolution"].dropna().unique().tolist())

    def full_graph_pick_resolution(self, requested: float | None) -> float | None:
        available = self.full_graph_resolutions()
        if not available:
            return None
        if requested is None:
            preferred = [r for r in available if abs(r - 1.0) < 1e-6]
            return preferred[0] if preferred else available[len(available) // 2]
        return min(available, key=lambda r: abs(r - float(requested)))

    @staticmethod
    def _resolution_suffix(resolution: float) -> str:
        return f"res_{resolution:.2f}"

    def table_path(self, table_name: str) -> Path:
        if table_name in TABLE_FILES:
            return self.output_dir / TABLE_FILES[table_name]
        if table_name in PATTERN_TABLE_FILES:
            return self.pattern_dir / PATTERN_TABLE_FILES[table_name]
        if table_name in FULL_GRAPH_TABLE_FILES:
            return self.full_graph_dir / FULL_GRAPH_TABLE_FILES[table_name]
        raise KeyError(f"Unknown table: {table_name}")

    def load_table(self, table_name: str) -> pd.DataFrame:
        path = self.table_path(table_name)
        if not path.exists():
            return pd.DataFrame()
        if table_name in PATTERN_TABLE_FILES:
            return self.load_pattern_table(table_name)
        if table_name in FULL_GRAPH_TABLE_FILES:
            return self.load_full_graph_csv(FULL_GRAPH_TABLE_FILES[table_name])
        mtime = path.stat().st_mtime
        cached = self._cache.get(table_name)
        if cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_csv(path, low_memory=False)
        self._cache[table_name] = (mtime, df)
        return df

    def load_pattern_table(self, table_name: str) -> pd.DataFrame:
        if table_name not in PATTERN_TABLE_FILES:
            raise KeyError(f"Unknown pattern table: {table_name}")
        path = self.pattern_dir / PATTERN_TABLE_FILES[table_name]
        if not path.exists():
            return pd.DataFrame()
        mtime = path.stat().st_mtime
        cached = self._pattern_cache.get(table_name)
        if cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_csv(path, low_memory=False)
        self._pattern_cache[table_name] = (mtime, df)
        return df

    def load_feature_artifacts(self) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame] | None:
        matrix_path = self.pattern_dir / "case_feature_matrix.npz"
        feature_path = self.pattern_dir / "case_feature_metadata.csv"
        case_path = self.pattern_dir / "case_feature_case_index.csv"
        if not matrix_path.exists() or not feature_path.exists() or not case_path.exists():
            return None
        mtimes = (matrix_path.stat().st_mtime, feature_path.stat().st_mtime, case_path.stat().st_mtime)
        if self._feature_cache and self._feature_cache[0] == mtimes:
            return self._feature_cache[1], self._feature_cache[2], self._feature_cache[3]
        matrix = sp.load_npz(matrix_path).tocsr()
        features = pd.read_csv(feature_path, low_memory=False)
        case_rows = pd.read_csv(case_path, low_memory=False)
        self._feature_cache = (mtimes, matrix, features, case_rows)
        return matrix, features, case_rows

    def run_summary(self) -> dict[str, Any]:
        return read_json(self.output_dir / "run_summary.json")

    def manifest(self) -> dict[str, Any]:
        return read_json(self.output_dir / "manifest.json")

    def status(self) -> dict[str, Any]:
        files = {}
        for table_name, filename in TABLE_FILES.items():
            path = self.output_dir / filename
            files[table_name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "rows": file_line_count(path),
            }
        pattern_files = {}
        for table_name, filename in PATTERN_TABLE_FILES.items():
            path = self.pattern_dir / filename
            pattern_files[table_name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "rows": file_line_count(path),
            }
        full_graph_files = {}
        for table_name, filename in FULL_GRAPH_TABLE_FILES.items():
            path = self.full_graph_dir / filename
            full_graph_files[table_name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "rows": file_line_count(path),
            }
        return {
            "output_dir": str(self.output_dir),
            "pattern_dir": str(self.pattern_dir),
            "full_graph_dir": str(self.full_graph_dir),
            "manifest_exists": (self.output_dir / "manifest.json").exists(),
            "run_summary_exists": (self.output_dir / "run_summary.json").exists(),
            "files": files,
            "pattern_available": (self.pattern_dir / "community_profiles.csv").exists(),
            "pattern_files": pattern_files,
            "full_graph_available": (self.full_graph_dir / "resolution_sweep_summary.csv").exists(),
            "full_graph_files": full_graph_files,
        }

    # ─────────────────────────────────────────────────────────────────
    # OVERVIEW (legacy compat) and EXPERIMENT-SCOPED ENDPOINTS
    # ─────────────────────────────────────────────────────────────────

    def overview(self) -> dict[str, Any]:
        case_summary = self.load_table("case_summary")
        paths = self.load_table("typed_path_importance")
        evidence = self.load_table("evidence_type_importance")
        relations = self.load_table("relation_type_importance")
        leakage = self.load_table("leakage_sensitivity_summary")
        attention = self.load_table("attention_counterfactual_overlap")
        run_summary = self.run_summary()

        metrics: dict[str, Any] = {
            "output_dir": str(self.output_dir),
            "processed_cases": int(run_summary.get("processed_cases", len(case_summary))),
            "processed_groups": int(run_summary.get("processed_groups", 0)),
            "failed_cases": len(run_summary.get("failed_cases", [])),
        }
        if not case_summary.empty:
            work = case_summary.copy()
            for column in ("max_abs_delta_pred_proba", "n_prediction_flips", "baseline_pred_proba"):
                if column in work:
                    work[column] = pd.to_numeric(work[column], errors="coerce")
            if {"target_label", "saved_pred_label"}.issubset(work.columns):
                metrics["saved_accuracy"] = float((work["target_label"].astype(str) == work["saved_pred_label"].astype(str)).mean())
            if {"target_label", "baseline_pred_label"}.issubset(work.columns):
                metrics["baseline_accuracy"] = float((work["target_label"].astype(str) == work["baseline_pred_label"].astype(str)).mean())
            if "max_abs_delta_pred_proba" in work:
                metrics["mean_top_abs_delta"] = float(work["max_abs_delta_pred_proba"].mean())
                metrics["median_top_abs_delta"] = float(work["max_abs_delta_pred_proba"].median())
            if "n_prediction_flips" in work:
                metrics["cases_with_flips"] = int((work["n_prediction_flips"] > 0).sum())

        overlap = None
        if not attention.empty and "counterfactual_attention_overlap" in attention:
            overlap_values = pd.to_numeric(attention["counterfactual_attention_overlap"], errors="coerce")
            overlap = float(overlap_values.mean())
        metrics["mean_attention_overlap"] = overlap

        return {
            "metrics": metrics,
            "top_paths": self._records(paths, 15),
            "top_evidence_types": self._records(evidence, 15),
            "top_relations": self._records(relations, 15),
            "leakage": self._records(leakage, 15),
            "status": self.status(),
        }

    def experiment_overview(self) -> dict[str, Any]:
        """Headline numbers + pipeline status for the landing tab."""
        case_summary = self.load_table("case_summary")
        run_summary = self.run_summary()
        pattern = self.pattern_overview()
        comm = self.load_pattern_table("community_profiles")
        align = self.load_pattern_table("structural_embedding_alignment")
        attn = self.load_table("attention_counterfactual_overlap")
        auc = self.load_table("faithfulness_auc_summary")
        leakage = self.load_table("leakage_sensitivity_summary")
        identity_shortcuts = self.load_table("identity_shortcut_summary")
        mask_sensitivity = self.load_table("mask_sensitivity_summary")

        n_cases = int(run_summary.get("processed_cases", len(case_summary)))
        n_groups = int(run_summary.get("processed_groups", 0))

        accuracy = None
        flip_rate = None
        if not case_summary.empty and {"target_label", "baseline_pred_label"}.issubset(case_summary.columns):
            accuracy = float(
                (case_summary["target_label"].astype(str) == case_summary["baseline_pred_label"].astype(str)).mean()
            )
        if not case_summary.empty and "n_prediction_flips" in case_summary:
            flip_rate = float((pd.to_numeric(case_summary["n_prediction_flips"], errors="coerce").fillna(0) > 0).mean())

        attention_overlap = None
        if not attn.empty and "counterfactual_attention_overlap" in attn:
            attention_overlap = safe_float(pd.to_numeric(attn["counterfactual_attention_overlap"], errors="coerce").mean())

        cf_suff_auc = cf_comp_auc = att_comp_auc = rnd_comp_auc = None
        if not auc.empty:
            for _, row in auc.iterrows():
                ranker = str(row.get("ranker", ""))
                if ranker == "counterfactual":
                    cf_suff_auc = safe_float(row.get("mean_sufficiency_auc"))
                    cf_comp_auc = safe_float(row.get("mean_comprehensiveness_auc"))
                elif ranker == "attention":
                    att_comp_auc = safe_float(row.get("mean_comprehensiveness_auc"))
                elif ranker == "random":
                    rnd_comp_auc = safe_float(row.get("mean_comprehensiveness_auc"))

        comp_lift_random = None
        if cf_comp_auc is not None and rnd_comp_auc is not None and rnd_comp_auc > 0:
            comp_lift_random = cf_comp_auc / rnd_comp_auc

        n_communities = int(len(comm)) if not comm.empty else 0
        nmi = ari = noise_rate = None
        if not align.empty:
            row = align.iloc[0]
            nmi = safe_float(row.get("normalized_mutual_info_all"))
            ari = safe_float(row.get("adjusted_rand_all"))
            noise_rate = safe_float(row.get("noise_rate"))

        leakage_total_share = None
        if not leakage.empty and "evidence_type" in leakage and "mean_abs_delta_pred_proba" in leakage:
            judges_courts_sum = pd.to_numeric(
                leakage[leakage["evidence_type"].astype(str).isin(["judge", "court", "petitioner", "respondent", "lawyer"])][
                    "mean_abs_delta_pred_proba"
                ],
                errors="coerce",
            ).sum()
            total_sum = pd.to_numeric(leakage["mean_abs_delta_pred_proba"], errors="coerce").sum()
            if total_sum and total_sum > 0:
                leakage_total_share = float(judges_courts_sum / total_sum)

        identity_shortcut_auc = None
        identity_shortcut_scope = None
        identity_shortcut_domain_delta = None
        if not identity_shortcuts.empty and "identity_auc_roc" in identity_shortcuts.columns:
            shortcut_rows = identity_shortcuts[
                identity_shortcuts.get("identity_scope", pd.Series(dtype=str)).astype(str) != "combined"
            ].copy()
            if not shortcut_rows.empty:
                shortcut_rows["identity_auc_roc"] = pd.to_numeric(
                    shortcut_rows["identity_auc_roc"], errors="coerce"
                )
                best = shortcut_rows.sort_values("identity_auc_roc", ascending=False).head(1)
                if not best.empty:
                    row = best.iloc[0]
                    identity_shortcut_scope = str(row.get("identity_scope"))
                    identity_shortcut_auc = safe_float(row.get("identity_auc_roc"))
                    identity_shortcut_domain_delta = safe_float(
                        row.get("identity_log_loss_delta_vs_domain")
                    )

        identity_mask_drop = None
        identity_mask_flip_rate = None
        identity_mask_confidence_drop = None
        hub_top50_accuracy_drop = None
        hub_top50_flip_rate = None
        if not mask_sensitivity.empty and "mask_name" in mask_sensitivity.columns:
            masks = mask_sensitivity.copy()
            for column in ("accuracy_drop", "macro_f1_drop", "confidence_drop", "flip_rate"):
                if column in masks:
                    masks[column] = pd.to_numeric(masks[column], errors="coerce")
            all_identity = masks[masks["mask_name"].astype(str) == "no_all_identities"]
            if not all_identity.empty:
                row = all_identity.iloc[0]
                identity_mask_drop = safe_float(row.get("accuracy_drop"))
                identity_mask_flip_rate = safe_float(row.get("flip_rate"))
                identity_mask_confidence_drop = safe_float(row.get("confidence_drop"))
            top50 = masks[masks["mask_name"].astype(str) == "remove_top_50_hubs"]
            if not top50.empty:
                row = top50.iloc[0]
                hub_top50_accuracy_drop = safe_float(row.get("accuracy_drop"))
                hub_top50_flip_rate = safe_float(row.get("flip_rate"))

        return {
            "headline": {
                "n_cases": n_cases,
                "n_groups": n_groups,
                "accuracy": accuracy,
                "flip_rate": flip_rate,
                "attention_overlap": attention_overlap,
                "cf_sufficiency_auc": cf_suff_auc,
                "cf_comprehensiveness_auc": cf_comp_auc,
                "att_comprehensiveness_auc": att_comp_auc,
                "rnd_comprehensiveness_auc": rnd_comp_auc,
                "comp_lift_vs_random": comp_lift_random,
                "n_communities": n_communities,
                "embedding_nmi": nmi,
                "embedding_ari": ari,
                "embedding_noise_rate": noise_rate,
                "identity_evidence_share": leakage_total_share,
                "identity_shortcut_scope": identity_shortcut_scope,
                "identity_shortcut_auc": identity_shortcut_auc,
                "identity_shortcut_log_loss_delta_vs_domain": identity_shortcut_domain_delta,
                "identity_mask_accuracy_drop": identity_mask_drop,
                "identity_mask_confidence_drop": identity_mask_confidence_drop,
                "identity_mask_flip_rate": identity_mask_flip_rate,
                "hub_top50_accuracy_drop": hub_top50_accuracy_drop,
                "hub_top50_flip_rate": hub_top50_flip_rate,
            },
            "status": self.status(),
            "pattern_available": pattern.get("available", False),
            "output_dir": str(self.output_dir),
            "pattern_dir": str(self.pattern_dir),
        }

    # ── EXP 1: HGT case embedding extraction ───────────────────────────
    def exp_embeddings(self) -> dict[str, Any]:
        case_summary = self.load_table("case_summary")
        manifest = read_json(self.pattern_dir / "hgt_case_embeddings_manifest.json")
        clusters = self.load_pattern_table("case_embedding_clusters")
        align = self.load_pattern_table("structural_embedding_alignment")
        cluster_profiles = self.load_pattern_table("embedding_cluster_profiles")

        # Confidence histogram (from baseline_pred_proba)
        hist = []
        if not case_summary.empty and "baseline_pred_proba" in case_summary:
            probs = pd.to_numeric(case_summary["baseline_pred_proba"], errors="coerce").dropna()
            bins = np.linspace(0.5, 1.0, 11)
            counts, edges = np.histogram(probs, bins=bins)
            hist = [
                {"bucket": f"{edges[i]:.2f}–{edges[i+1]:.2f}", "n_cases": int(counts[i])}
                for i in range(len(counts))
            ]

        accuracy_by_split = []
        if not case_summary.empty and {"split", "target_label", "baseline_pred_label"}.issubset(case_summary.columns):
            tmp = case_summary.copy()
            tmp["correct"] = (tmp["target_label"].astype(str) == tmp["baseline_pred_label"].astype(str)).astype(int)
            for split, group in tmp.groupby("split"):
                accuracy_by_split.append({
                    "split": str(split),
                    "n_cases": int(len(group)),
                    "accuracy": float(group["correct"].mean()),
                    "mean_confidence": float(pd.to_numeric(group["baseline_pred_proba"], errors="coerce").mean()),
                })

        align_row = clean_records(align.head(1).to_dict("records"))[0] if not align.empty else None
        manifest_meta = {
            "embedding_dim": manifest.get("embedding_dim"),
            "n_cases": manifest.get("n_cases"),
            "n_classes": manifest.get("n_classes"),
            "model_path": manifest.get("model_path"),
            "config_path": manifest.get("config_path"),
        }

        findings = []
        if manifest_meta.get("n_cases") and manifest_meta.get("embedding_dim"):
            findings.append(
                f"Extracted {int(manifest_meta['n_cases']):,} per-case HGT embeddings of dimension {int(manifest_meta['embedding_dim'])}; reused by every downstream analysis."
            )
        if hist:
            high_conf = sum(b["n_cases"] for b in hist if float(b["bucket"].split("–")[0]) >= 0.8)
            total = sum(b["n_cases"] for b in hist)
            if total:
                findings.append(
                    f"{high_conf/total:.0%} of cases get a confidence ≥ 0.80 — the model is decisive on most of the corpus."
                )
        if accuracy_by_split:
            for entry in accuracy_by_split:
                findings.append(
                    f"{entry['split'].title()} split: {entry['accuracy']:.1%} accuracy across {entry['n_cases']:,} cases (mean confidence {entry['mean_confidence']:.2f})."
                )
        if align_row:
            nmi = align_row.get("normalized_mutual_info_all")
            if nmi is not None:
                qual = "weak" if nmi < 0.1 else ("partial" if nmi < 0.3 else "strong")
                findings.append(
                    f"Embedding-vs-structural alignment NMI = {nmi:.3f} ({qual}). HGT embeddings are not just topology — they encode information beyond shared-authority neighbourhoods."
                )
        return {
            "manifest": manifest_meta,
            "confidence_histogram": hist,
            "accuracy_by_split": accuracy_by_split,
            "alignment_row": align_row,
            "n_clusters": int(len(cluster_profiles)) if not cluster_profiles.empty else 0,
            "n_assigned_cases": int(len(clusters)) if not clusters.empty else 0,
            "findings": findings,
        }

    # ── EXP 2: Counterfactual masking ─────────────────────────────────
    def exp_counterfactual(self) -> dict[str, Any]:
        case_summary = self.load_table("case_summary")
        evidence = self.load_table("evidence_type_importance")
        relations = self.load_table("relation_type_importance")
        paths = self.load_table("typed_path_importance")
        leakage = self.load_table("leakage_sensitivity_summary")
        attention = self.load_table("attention_counterfactual_overlap")
        skew = self.load_pattern_table("evidence_label_skew")

        findings: list[str] = []

        # Top evidence types and their share
        if not evidence.empty and "sum_abs_delta_pred_proba" in evidence:
            ev = evidence.copy()
            ev["sum_abs_delta_pred_proba"] = pd.to_numeric(ev["sum_abs_delta_pred_proba"], errors="coerce").fillna(0)
            ev = ev.sort_values("sum_abs_delta_pred_proba", ascending=False)
            total_imp = ev["sum_abs_delta_pred_proba"].sum()
            if total_imp > 0:
                top3 = ev.head(3)
                share = top3["sum_abs_delta_pred_proba"].sum() / total_imp
                names = ", ".join(str(n) for n in top3["evidence_type"].tolist())
                findings.append(f"Top 3 evidence types ({names}) account for {share:.0%} of total counterfactual importance.")

        # Path family with most flips
        if not paths.empty and "flip_rate" in paths:
            p = paths.copy()
            p["flip_rate"] = pd.to_numeric(p["flip_rate"], errors="coerce").fillna(0)
            p["n_cases"] = pd.to_numeric(p.get("n_cases", 0), errors="coerce").fillna(0)
            p_filt = p[p["n_cases"] >= 100].sort_values("flip_rate", ascending=False)
            if not p_filt.empty:
                row = p_filt.iloc[0]
                findings.append(
                    f"`{row['path_family']}` has the highest flip rate ({row['flip_rate']:.1%}) among well-supported paths — masking it changes the predicted class most often."
                )

        # Cases with flips
        if not case_summary.empty and "n_prediction_flips" in case_summary:
            flips = pd.to_numeric(case_summary["n_prediction_flips"], errors="coerce").fillna(0)
            with_flips = int((flips > 0).sum())
            findings.append(f"{with_flips:,} of {len(case_summary):,} cases ({with_flips/max(1,len(case_summary)):.0%}) have at least one evidence group whose removal flips the prediction.")

        # Attention vs counterfactual overlap
        if not attention.empty and "counterfactual_attention_overlap" in attention:
            mean_overlap = pd.to_numeric(attention["counterfactual_attention_overlap"], errors="coerce").mean()
            if pd.notna(mean_overlap):
                judgement = "low" if mean_overlap < 0.3 else ("modest" if mean_overlap < 0.6 else "strong")
                findings.append(
                    f"Mean top-k attention/counterfactual overlap is {mean_overlap:.0%} ({judgement}). Attention rankings often disagree with what actually changes the prediction."
                )

        # Identity-leakage share
        if not leakage.empty and "evidence_type" in leakage and "mean_abs_delta_pred_proba" in leakage:
            identity = leakage[leakage["evidence_type"].astype(str).isin(["judge", "court", "petitioner", "respondent", "defence_lawyer", "petitioner_lawyer", "lawyer"])]
            if not identity.empty:
                top_identity = identity.assign(
                    _val=pd.to_numeric(identity["mean_abs_delta_pred_proba"], errors="coerce").fillna(0)
                ).sort_values("_val", ascending=False).iloc[0]
                findings.append(
                    f"Identity nodes show meaningful sensitivity: `{top_identity['evidence_type']}` mean importance {safe_float(top_identity['mean_abs_delta_pred_proba']):.3f}, flip rate {safe_float(top_identity['flip_rate']):.1%}. Audit for leakage."
                )

        # Discriminative evidence count from skew
        if not skew.empty and "skew_class" in skew:
            disc_n = int((skew["skew_class"].astype(str) == "label_discriminative").sum())
            findings.append(f"{disc_n:,} evidence nodes have statistically significant label skew (BH-corrected) — these are the nuggets that actually push toward one outcome.")

        return {
            "evidence_types": self._records(evidence, 25),
            "relation_types": self._records(relations, 30),
            "path_families": self._records(paths, 30),
            "leakage": self._records(leakage, 25),
            "attention_overlap_summary": {
                "n_cases": int(len(attention)) if not attention.empty else 0,
                "mean": safe_float(pd.to_numeric(attention.get("counterfactual_attention_overlap", pd.Series()), errors="coerce").mean()) if not attention.empty else None,
                "median": safe_float(pd.to_numeric(attention.get("counterfactual_attention_overlap", pd.Series()), errors="coerce").median()) if not attention.empty else None,
            },
            "findings": findings,
        }

    # ── EXP 3: Identity shortcut audit ────────────────────────────────
    def exp_identity_shortcuts(self) -> dict[str, Any]:
        summary = self.load_table("identity_shortcut_summary")
        case_scores = self.load_table("identity_shortcut_case_scores")
        skewed = self.load_table("identity_shortcut_top_skewed_identities")
        mask_summary = self.load_table("mask_sensitivity_summary")
        mask_domains = self.load_table("mask_sensitivity_top_domain_drops")
        manifest = read_json(self.output_dir / "identity_shortcut_manifest.json")

        if summary.empty and mask_summary.empty:
            return {
                "available": False,
                "manifest": manifest,
                "summary": [],
                "case_scores": [],
                "top_skewed_identities": [],
                "mask_summary": [],
                "mask_domain_drops": [],
                "findings": [
                    "Identity shortcut and mask-sensitivity audit outputs were not found. Run `run_scripts/run_identity_shortcut_audit.sh --permutations 100` and `run_scripts/run_mask_sensitivity_audit.sh` for this output directory."
                ],
            }

        work = summary.copy()
        for column in (
            "identity_auc_roc",
            "known_eval_case_share",
            "eval_identity_overlap_share",
            "identity_log_loss_delta_vs_domain",
            "identity_brier_delta_vs_domain",
            "permutation_auc_p_value",
            "counterfactual_mean_abs_delta_pred_proba",
            "counterfactual_flip_rate",
        ):
            if column in work:
                work[column] = pd.to_numeric(work[column], errors="coerce")

        non_combined = work[
            work.get("identity_scope", pd.Series(dtype=str)).astype(str) != "combined"
        ].copy()
        top_auc = pd.Series(dtype=object)
        top_counterfactual = pd.Series(dtype=object)
        if not non_combined.empty and "identity_auc_roc" in non_combined:
            top_auc = non_combined.sort_values("identity_auc_roc", ascending=False).iloc[0]
        if not non_combined.empty and "counterfactual_flip_rate" in non_combined:
            top_counterfactual = non_combined.sort_values(
                "counterfactual_flip_rate", ascending=False, na_position="last"
            ).iloc[0]

        findings: list[str] = []
        if not top_auc.empty:
            delta = safe_float(top_auc.get("identity_log_loss_delta_vs_domain"))
            auc_value = safe_float(top_auc.get("identity_auc_roc"))
            delta_text = (
                f" and beats the domain baseline by {-delta:.3f} log-loss"
                if delta is not None and delta < 0
                else ""
            )
            findings.append(
                f"`{top_auc.get('identity_scope')}` has the strongest identity-only held-out signal "
                f"(AUC {auc_value:.3f}{delta_text})."
                if auc_value is not None
                else f"`{top_auc.get('identity_scope')}` has the strongest identity-only held-out signal."
            )
        if not top_counterfactual.empty:
            flip_value = safe_float(top_counterfactual.get("counterfactual_flip_rate"))
            findings.append(
                f"`{top_counterfactual.get('identity_scope')}` has the highest counterfactual flip rate "
                f"({flip_value:.1%}); compare this with its identity-only AUC to separate model reliance from broad shortcut predictiveness."
                if flip_value is not None
                else f"`{top_counterfactual.get('identity_scope')}` has the highest counterfactual flip rate; compare this with its identity-only AUC to separate model reliance from broad shortcut predictiveness."
            )
        combined = work[work.get("identity_scope", pd.Series(dtype=str)).astype(str) == "combined"]
        if not combined.empty:
            row = combined.iloc[0]
            coverage_value = safe_float(row.get("known_eval_case_share"))
            auc_value = safe_float(row.get("identity_auc_roc"))
            if coverage_value is not None and auc_value is not None:
                findings.append(
                    f"Combined known identity coverage reaches {coverage_value:.1%} of eval cases, "
                    f"with identity-only AUC {auc_value:.3f}."
                )
        if "permutation_auc_p_value" in non_combined and non_combined["permutation_auc_p_value"].notna().any():
            significant = int((non_combined["permutation_auc_p_value"] <= 0.05).sum())
            findings.append(
                f"{significant} identity scopes beat the domain-preserving permutation null at p≤0.05."
            )

        identity_mask_rows = pd.DataFrame()
        domain_drop_rows = pd.DataFrame()
        if not mask_summary.empty:
            masks = mask_summary.copy()
            if "mask_family" in masks.columns:
                masks = masks[masks["mask_family"].astype(str) == "identity"].copy()
            for column in (
                "n_cases",
                "masked_edge_share",
                "baseline_accuracy",
                "masked_accuracy",
                "accuracy_drop",
                "baseline_macro_f1",
                "masked_macro_f1",
                "macro_f1_drop",
                "confidence_drop",
                "mean_confidence_drop",
                "flip_rate",
            ):
                if column in masks:
                    masks[column] = pd.to_numeric(masks[column], errors="coerce")
            if not masks.empty:
                all_identity = masks[masks["mask_name"].astype(str) == "no_all_identities"]
                if not all_identity.empty:
                    row = all_identity.iloc[0]
                    acc_drop = safe_float(row.get("accuracy_drop")) or 0.0
                    f1_drop = safe_float(row.get("macro_f1_drop")) or 0.0
                    flip_rate = safe_float(row.get("flip_rate")) or 0.0
                    findings.append(
                        f"Full identity masking drops accuracy by {acc_drop:.2%}, macro-F1 by {f1_drop:.2%}, and flips {flip_rate:.1%} of test predictions."
                    )
                strongest = masks.sort_values("accuracy_drop", ascending=False, na_position="last").iloc[0]
                strongest_drop = safe_float(strongest.get("accuracy_drop")) or 0.0
                findings.append(
                    f"Among separate inference masks, `{strongest.get('mask_name')}` has the largest accuracy drop ({strongest_drop:.2%})."
                )
                keep_mask_cols = [
                    "mask_name",
                    "n_cases",
                    "masked_edge_share",
                    "baseline_accuracy",
                    "masked_accuracy",
                    "accuracy_drop",
                    "baseline_macro_f1",
                    "masked_macro_f1",
                    "macro_f1_drop",
                    "confidence_drop",
                    "mean_confidence_drop",
                    "flip_rate",
                ]
                identity_mask_rows = masks[[c for c in keep_mask_cols if c in masks.columns]].sort_values(
                    "accuracy_drop", ascending=False, na_position="last"
                )

        if not mask_domains.empty:
            domains = mask_domains.copy()
            if "mask_family" in domains.columns:
                domains = domains[domains["mask_family"].astype(str) == "identity"].copy()
            for column in (
                "n_cases",
                "accuracy_drop",
                "macro_f1_drop",
                "confidence_drop",
                "mean_confidence_drop",
                "flip_rate",
            ):
                if column in domains:
                    domains[column] = pd.to_numeric(domains[column], errors="coerce")
            keep_domain_cols = [
                "mask_name",
                "domain_bucket",
                "n_cases",
                "accuracy_drop",
                "macro_f1_drop",
                "confidence_drop",
                "mean_confidence_drop",
                "flip_rate",
            ]
            domain_drop_rows = domains[[c for c in keep_domain_cols if c in domains.columns]].sort_values(
                ["mask_name", "accuracy_drop"], ascending=[True, False], na_position="last"
            )

        cases = pd.DataFrame()
        if not case_scores.empty:
            cases = case_scores.copy()
            for column in (
                "identity_score_combined",
                "domain_baseline_score",
                "global_prior_score",
                "identity_train_support_combined",
                "confidence",
            ):
                if column in cases:
                    cases[column] = pd.to_numeric(cases[column], errors="coerce")
            if {"identity_score_combined", "domain_baseline_score"}.issubset(cases.columns):
                cases["identity_score_gap_vs_domain"] = (
                    cases["identity_score_combined"] - cases["domain_baseline_score"]
                )
                cases["abs_identity_score_gap_vs_domain"] = cases[
                    "identity_score_gap_vs_domain"
                ].abs()
                cases = cases.sort_values(
                    ["abs_identity_score_gap_vs_domain", "identity_train_support_combined"],
                    ascending=[False, False],
                    na_position="last",
                )
            keep_case_cols = [
                "case_index",
                "case_id",
                "target_label",
                "pred_label",
                "confidence",
                "correct",
                "domain_bucket",
                "identity_score_combined",
                "domain_baseline_score",
                "identity_score_gap_vs_domain",
                "identity_train_support_combined",
                "identity_known_combined",
            ]
            cases = cases[[col for col in keep_case_cols if col in cases.columns]].head(40)

        skew_rows = pd.DataFrame()
        if not skewed.empty:
            skew_rows = skewed.copy()
            for column in (
                "train_support",
                "train_positive_rate",
                "smoothed_positive_rate",
                "smoothed_lift_from_train_prior",
                "eval_support",
            ):
                if column in skew_rows:
                    skew_rows[column] = pd.to_numeric(skew_rows[column], errors="coerce")
            if "smoothed_lift_from_train_prior" in skew_rows:
                skew_rows = skew_rows.assign(
                    __sort=skew_rows["smoothed_lift_from_train_prior"].abs()
                ).sort_values(["__sort", "train_support"], ascending=[False, False]).drop(
                    columns=["__sort"]
                )
            keep_skew_cols = [
                "identity_type",
                "identity_name",
                "train_support",
                "train_positive",
                "train_negative",
                "train_positive_rate",
                "smoothed_positive_rate",
                "smoothed_lift_from_train_prior",
                "eval_support",
            ]
            skew_rows = skew_rows[[col for col in keep_skew_cols if col in skew_rows.columns]].head(60)

        if {"identity_auc_roc", "counterfactual_flip_rate"}.issubset(work.columns):
            summary_sorted = work.sort_values(
                ["identity_auc_roc", "counterfactual_flip_rate"],
                ascending=[False, False],
                na_position="last",
            )
        else:
            summary_sorted = work
        return {
            "available": True,
            "manifest": manifest,
            "summary": self._records(summary_sorted, 50),
            "case_scores": clean_records(cases.to_dict("records")) if not cases.empty else [],
            "top_skewed_identities": clean_records(skew_rows.to_dict("records")) if not skew_rows.empty else [],
            "mask_summary": clean_records(identity_mask_rows.to_dict("records")) if not identity_mask_rows.empty else [],
            "mask_domain_drops": clean_records(domain_drop_rows.to_dict("records")) if not domain_drop_rows.empty else [],
            "findings": findings,
        }

    # ── EXP 3: Faithfulness validation ────────────────────────────────
    def exp_faithfulness(self) -> dict[str, Any]:
        auc = self.load_table("faithfulness_auc_summary")
        curve = self.load_table("faithfulness_curve_summary")
        bucket = self.load_table("prediction_bucket_summary")
        bucket_evidence = self.load_table("prediction_bucket_evidence_types")

        findings: list[str] = []
        per_ranker = {}
        if not auc.empty and "ranker" in auc:
            for _, row in auc.iterrows():
                per_ranker[str(row["ranker"])] = {
                    "n_cases": safe_int(row.get("n_cases")),
                    "mean_sufficiency_auc": safe_float(row.get("mean_sufficiency_auc")),
                    "mean_comprehensiveness_auc": safe_float(row.get("mean_comprehensiveness_auc")),
                    "median_sufficiency_auc": safe_float(row.get("median_sufficiency_auc")),
                    "median_comprehensiveness_auc": safe_float(row.get("median_comprehensiveness_auc")),
                }
        cf = per_ranker.get("counterfactual", {})
        att = per_ranker.get("attention", {})
        rnd = per_ranker.get("random", {})
        if cf.get("mean_comprehensiveness_auc") and rnd.get("mean_comprehensiveness_auc"):
            ratio = cf["mean_comprehensiveness_auc"] / rnd["mean_comprehensiveness_auc"]
            findings.append(
                f"Counterfactual ranking is {ratio:.1f}× better than random at comprehensiveness (mean AUC {cf['mean_comprehensiveness_auc']:.3f} vs {rnd['mean_comprehensiveness_auc']:.3f}). Removing top-k counterfactual evidence drops probability fastest."
            )
        if cf.get("mean_sufficiency_auc") is not None and att.get("mean_sufficiency_auc") is not None:
            diff = cf["mean_sufficiency_auc"] - att["mean_sufficiency_auc"]
            sign = "above" if diff > 0 else "below"
            findings.append(
                f"On sufficiency AUC, counterfactual ({cf['mean_sufficiency_auc']:.3f}) sits {abs(diff):.3f} {sign} attention ({att['mean_sufficiency_auc']:.3f}). The compact top-k explanation preserves the prediction."
            )

        # Buckets
        if not bucket.empty and "confidence_bucket" in bucket:
            for _, row in bucket.iterrows():
                if str(row.get("confidence_bucket")) == "high_confidence_correct":
                    purity = safe_float(row.get("mean_evidence_purity"))
                    if purity is not None:
                        findings.append(
                            f"High-confidence correct cases ({safe_int(row.get('n_cases')):,}) have mean evidence purity {purity:.0%} — explanations are concentrated, not scattered."
                        )
                if str(row.get("confidence_bucket")) == "high_confidence_wrong":
                    n = safe_int(row.get("n_cases"))
                    purity = safe_float(row.get("mean_evidence_purity"))
                    if n and purity is not None:
                        findings.append(
                            f"High-confidence WRONG cases ({n:,}) still show high purity ({purity:.0%}) — when the model is confidently wrong, it's because a misleading evidence neighbourhood looked clean."
                        )

        return {
            "auc_summary": self._records(auc, 10),
            "curve_summary": self._records(curve, 500),
            "bucket_summary": self._records(bucket, 20),
            "bucket_evidence_types": self._records(bucket_evidence, 100),
            "per_ranker": per_ranker,
            "findings": findings,
        }

    # ── EXP 4: Legal communities ──────────────────────────────────────
    def exp_communities(self) -> dict[str, Any]:
        communities = self.load_pattern_table("community_profiles")
        success = self.load_pattern_table("community_success_failure")
        if communities.empty:
            return {"available": False, "findings": []}

        c = communities.copy()
        for col in ("size", "accuracy", "mean_confidence", "high_confidence_wrong_n"):
            if col in c:
                c[col] = pd.to_numeric(c[col], errors="coerce")

        findings: list[str] = []
        n_total = int(len(c))
        size_total = int(c["size"].sum()) if "size" in c else None
        weighted_acc = None
        if {"size", "accuracy"}.issubset(c.columns):
            w = c["size"].fillna(0)
            a = c["accuracy"]
            if w.sum() > 0 and a.notna().any():
                weighted_acc = float((a.fillna(0) * w).sum() / w.sum())
                findings.append(
                    f"Detected {n_total:,} structural communities covering {size_total:,} cases. Size-weighted accuracy across communities is {weighted_acc:.1%}."
                )

        # Largest community signature
        if "size" in c:
            biggest = c.sort_values("size", ascending=False).iloc[0]
            findings.append(
                f"Largest community ({safe_int(biggest['community_id'])}, n={safe_int(biggest['size'])}) is dominated by the {biggest.get('dominant_domain_bucket', '?')} domain and label {biggest.get('dominant_label', '?')}."
            )

        # Failure modes
        if "high_confidence_wrong_n" in c:
            confident_wrong_total = int(c["high_confidence_wrong_n"].fillna(0).sum())
            findings.append(
                f"{confident_wrong_total:,} cases sit in communities flagged as 'confident wrong' — these are the model's structural failure modes."
            )
            big_failure = c[c["size"].fillna(0) >= 50].sort_values("high_confidence_wrong_n", ascending=False)
            if not big_failure.empty:
                row = big_failure.iloc[0]
                findings.append(
                    f"Worst large failure community: id {safe_int(row['community_id'])} (n={safe_int(row['size'])}, accuracy {safe_float(row.get('accuracy')):.1%})."
                )

        # Top by size
        top = c.sort_values("size", ascending=False).head(15)
        risky = c.copy()
        if "high_confidence_wrong_n" in risky:
            risky = risky[risky["size"].fillna(0) >= 20].sort_values("high_confidence_wrong_n", ascending=False).head(15)

        domain_counts = []
        if "dominant_domain_bucket" in c:
            for domain, group in c.groupby("dominant_domain_bucket"):
                domain_counts.append({
                    "domain": str(domain),
                    "n_communities": int(len(group)),
                    "n_cases": int(group["size"].sum()) if "size" in group else None,
                })
            domain_counts.sort(key=lambda r: r.get("n_cases") or 0, reverse=True)

        return {
            "available": True,
            "n_communities": n_total,
            "n_cases_in_communities": size_total,
            "weighted_accuracy": weighted_acc,
            "top_communities": self._records(top, 15),
            "risky_communities": self._records(risky, 15),
            "domain_counts": domain_counts,
            "findings": findings,
        }

    # ── EXP 5: Embedding clusters & alignment ─────────────────────────
    def exp_embedding_clusters(self) -> dict[str, Any]:
        clusters = self.load_pattern_table("embedding_cluster_profiles")
        align = self.load_pattern_table("structural_embedding_alignment")
        splits = self.load_pattern_table("community_embedding_splits")

        findings: list[str] = []
        align_row = clean_records(align.head(1).to_dict("records"))[0] if not align.empty else None
        if align_row:
            ari = align_row.get("adjusted_rand_all")
            nmi = align_row.get("normalized_mutual_info_all")
            v = align_row.get("v_measure_all")
            noise = align_row.get("noise_rate")
            n_emb = align_row.get("n_embedding_clusters_including_noise")
            n_struct = align_row.get("n_structural_communities")
            if n_emb is not None and n_struct is not None:
                findings.append(
                    f"HDBSCAN finds {int(n_emb)} embedding clusters versus {int(n_struct)} Leiden structural communities — embeddings collapse much of the topology."
                )
            if noise is not None:
                findings.append(
                    f"HDBSCAN noise rate is {float(noise):.1%}: most cases land in dense embedding regions, the rest are diffuse."
                )
            if nmi is not None and ari is not None:
                qual = "weak" if nmi < 0.1 else ("partial" if nmi < 0.3 else "strong")
                v_str = f"{float(v):.3f}" if v is not None else "—"
                findings.append(
                    f"Embedding↔structural alignment: NMI {float(nmi):.3f}, ARI {float(ari):.3f}, V-measure {v_str} ({qual}). Outcome-relevant signal goes beyond shared-authority topology."
                )

        if not clusters.empty:
            c = clusters.copy()
            for col in ("size", "accuracy", "label_1_rate", "label_-1_rate"):
                if col in c:
                    c[col] = pd.to_numeric(c[col], errors="coerce")
            if {"label_1_rate", "label_-1_rate"}.issubset(c.columns):
                c["purity"] = c[["label_1_rate", "label_-1_rate"]].max(axis=1)
                pure = c[(c["size"].fillna(0) >= 50) & (c["purity"] >= 0.9)]
                findings.append(
                    f"{int(len(pure))} embedding clusters of n≥50 are outcome-pure (>90% one label). These are tight pockets where the HGT representation already separates outcomes."
                )

        return {
            "alignment_row": align_row,
            "clusters": self._records(clusters, 30),
            "splits": self._records(splits, 30),
            "flow": self.community_embedding_flow(splits),
            "findings": findings,
        }

    @staticmethod
    def _cluster_label(cluster_id: Any) -> str:
        text = str(cluster_id).strip()
        return "Noise" if text == "-1" else f"Cluster {text}"

    @staticmethod
    def _parse_count_list(text: Any) -> list[tuple[str, int]]:
        if text is None or pd.isna(text):
            return []
        pairs: list[tuple[str, int]] = []
        for part in str(text).split("|"):
            item = part.strip()
            if not item:
                continue
            match = re.match(r"^(.*?)\s*\((\d+)\)\s*$", item)
            if match:
                pairs.append((match.group(1).strip(), int(match.group(2))))
        return pairs

    def community_embedding_flow(self, splits: pd.DataFrame, limit: int = 14) -> dict[str, Any]:
        if splits.empty or "top_embedding_clusters" not in splits:
            return {"available": False, "links": [], "source_totals": [], "target_totals": []}

        work = splits.copy()
        for column in ("size", "community_id"):
            if column in work:
                work[column] = pd.to_numeric(work[column], errors="coerce")
        if "size" in work:
            work = work.sort_values("size", ascending=False, na_position="last")
        selected = work.head(limit)

        links: list[dict[str, Any]] = []
        source_totals: dict[str, dict[str, Any]] = {}
        target_totals: dict[str, dict[str, Any]] = {}
        linked_cases = 0
        for _, row in selected.iterrows():
            community_id = safe_int(row.get("community_id"))
            if community_id is None:
                continue
            source_id = f"community:{community_id}"
            source_label = f"Community {community_id}"
            size = safe_int(row.get("size")) or 0
            source_totals[source_id] = {
                "id": source_id,
                "label": source_label,
                "community_id": community_id,
                "value": size,
            }
            for cluster_id, count in self._parse_count_list(row.get("top_embedding_clusters")):
                target_id = f"cluster:{cluster_id}"
                target_label = self._cluster_label(cluster_id)
                links.append({
                    "source": source_id,
                    "source_label": source_label,
                    "target": target_id,
                    "target_label": target_label,
                    "value": int(count),
                    "community_id": community_id,
                    "cluster_id": str(cluster_id),
                })
                linked_cases += int(count)
                if target_id not in target_totals:
                    target_totals[target_id] = {
                        "id": target_id,
                        "label": target_label,
                        "cluster_id": str(cluster_id),
                        "value": 0,
                    }
                target_totals[target_id]["value"] += int(count)

        total_cases = safe_int(pd.to_numeric(work.get("size", pd.Series(dtype=float)), errors="coerce").sum()) or None
        selected_cases = safe_int(pd.to_numeric(selected.get("size", pd.Series(dtype=float)), errors="coerce").sum()) or None
        return {
            "available": bool(links),
            "links": links,
            "source_totals": sorted(source_totals.values(), key=lambda row: row.get("value") or 0, reverse=True),
            "target_totals": sorted(target_totals.values(), key=lambda row: row.get("value") or 0, reverse=True),
            "shown_communities": int(len(source_totals)),
            "selected_cases": selected_cases,
            "linked_cases": linked_cases,
            "total_cases": total_cases,
        }

    # ── EXP 6: Counterfactual neighborhoods ───────────────────────────
    def exp_neighborhoods(self) -> dict[str, Any]:
        neigh = self.load_pattern_table("counterfactual_neighborhoods")
        diffs = self.load_pattern_table("counterfactual_neighborhood_feature_differences")

        findings: list[str] = []
        if neigh.empty:
            return {"available": False, "findings": findings}

        n = len(neigh)
        cosine = pd.to_numeric(neigh.get("cosine_similarity", pd.Series()), errors="coerce").dropna()
        if not cosine.empty:
            findings.append(
                f"For {n:,} test cases the closest opposite-label training case has mean cosine {cosine.mean():.3f} (median {cosine.median():.3f}). Even the boundary cases are close to ‘the other side’."
            )
            close_cases = int((cosine >= 0.95).sum())
            findings.append(
                f"{close_cases:,} cases sit at cosine ≥ 0.95 to an opposite-label case — these are decision-boundary candidates worth manual review."
            )

        if not diffs.empty and "feature_type" in diffs and "side" in diffs:
            for side_label, side_key in (("query-only", "query_only"), ("opposite-only", "opposite_only")):
                sub = diffs[diffs["side"].astype(str) == side_key]
                if sub.empty:
                    continue
                top_types = sub["feature_type"].value_counts().head(3)
                if not top_types.empty:
                    line = ", ".join(f"{t} ({n_})" for t, n_ in top_types.items())
                    findings.append(
                        f"Most common {side_label} feature types in the contrastive evidence: {line}."
                    )

        # Top opposite-pair sample (highest cosine, label discordant)
        sample = neigh.copy()
        if "cosine_similarity" in sample:
            sample["cosine_similarity"] = pd.to_numeric(sample["cosine_similarity"], errors="coerce")
            sample = sample.sort_values("cosine_similarity", ascending=False)

        return {
            "available": True,
            "n_pairs": int(n),
            "cosine_summary": {
                "mean": safe_float(cosine.mean()) if not cosine.empty else None,
                "median": safe_float(cosine.median()) if not cosine.empty else None,
                "p95": safe_float(cosine.quantile(0.95)) if not cosine.empty else None,
            },
            "top_pairs": self._records(sample, 50),
            "feature_type_summary": self._diff_feature_summary(diffs),
            "contrast_graph": self.opposite_graph(
                safe_int(sample.iloc[0].get("case_index")) if not sample.empty else None
            ),
            "findings": findings,
        }

    @staticmethod
    def _feature_rows(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
        if df.empty:
            return []
        work = df.copy()
        if "rank" in work:
            work["rank"] = pd.to_numeric(work["rank"], errors="coerce")
            work = work.sort_values("rank", na_position="last")
        columns = [
            "rank",
            "feature_index",
            "feature_type",
            "feature_name",
            "idf",
            "corpus_case_count",
            "skew_class",
            "skew_direction",
            "log_odds_vs_base",
            "g_test_q_value_bh",
        ]
        work = work[[column for column in columns if column in work.columns]]
        return clean_records(work.head(limit).to_dict("records"))

    def _top_shared_features(self, case_index: int, opposite_case_index: int, limit: int = 8) -> list[dict[str, Any]]:
        artifacts = self.load_feature_artifacts()
        if artifacts is None:
            return []
        matrix, features, case_rows = artifacts
        row_map = self._case_to_feature_row(case_rows)
        left = row_map.get(int(case_index))
        right = row_map.get(int(opposite_case_index))
        if left is None or right is None or left >= matrix.shape[0] or right >= matrix.shape[0]:
            return []

        left_cols = matrix.getrow(left).indices
        right_cols = matrix.getrow(right).indices
        shared = np.intersect1d(left_cols, right_cols, assume_unique=False)
        if len(shared) == 0:
            return []

        meta = features.copy()
        if "feature_index" in meta:
            meta["_feature_index"] = pd.to_numeric(meta["feature_index"], errors="coerce")
            meta = meta.dropna(subset=["_feature_index"]).set_index("_feature_index", drop=False)
        else:
            meta["_feature_index"] = np.arange(len(meta))
            meta = meta.set_index("_feature_index", drop=False)

        rows: list[dict[str, Any]] = []
        for feature_index in shared:
            if feature_index not in meta.index:
                continue
            row = meta.loc[feature_index]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append({
                "rank": len(rows) + 1,
                "feature_index": int(feature_index),
                "feature_type": row.get("feature_type"),
                "feature_name": row.get("feature_name"),
                "idf": safe_float(row.get("idf")) or 0.0,
                "corpus_case_count": safe_int(row.get("corpus_case_count")),
                "side": "shared",
            })
        rows.sort(key=lambda row: (row.get("idf") or 0, -(row.get("corpus_case_count") or 0)), reverse=True)
        for idx, row in enumerate(rows[:limit], start=1):
            row["rank"] = idx
        return clean_records(rows[:limit])

    def opposite_graph(self, case_index: int | None) -> dict[str, Any]:
        neigh = self.load_pattern_table("counterfactual_neighborhoods")
        diffs = self.load_pattern_table("counterfactual_neighborhood_feature_differences")
        if neigh.empty:
            return {"available": False, "reason": "counterfactual_neighborhoods.csv not found"}

        work = neigh.copy()
        if "cosine_similarity" in work:
            work["cosine_similarity"] = pd.to_numeric(work["cosine_similarity"], errors="coerce")
        if case_index is None or int(case_index) < 0:
            work = work.sort_values("cosine_similarity", ascending=False, na_position="last")
            row_df = work.head(1)
        else:
            row_df = work[pd.to_numeric(work.get("case_index", pd.Series(dtype=float)), errors="coerce") == int(case_index)]
            if row_df.empty:
                return {"available": False, "reason": f"case_index {case_index} has no opposite-neighborhood row"}
        if row_df.empty:
            return {"available": False, "reason": "No opposite-neighborhood rows available"}

        row = row_df.iloc[0]
        query_idx = safe_int(row.get("case_index"))
        opposite_idx = safe_int(row.get("nearest_opposite_case_index"))
        diff_rows = pd.DataFrame()
        if not diffs.empty and query_idx is not None and "case_index" in diffs:
            diff_rows = diffs[pd.to_numeric(diffs["case_index"], errors="coerce") == int(query_idx)].copy()
            if opposite_idx is not None and "nearest_opposite_case_index" in diff_rows:
                diff_rows = diff_rows[
                    pd.to_numeric(diff_rows["nearest_opposite_case_index"], errors="coerce") == int(opposite_idx)
                ]

        query_only = diff_rows[diff_rows["side"].astype(str) == "query_only"] if not diff_rows.empty and "side" in diff_rows else pd.DataFrame()
        opposite_only = diff_rows[diff_rows["side"].astype(str) == "opposite_only"] if not diff_rows.empty and "side" in diff_rows else pd.DataFrame()
        shared = self._top_shared_features(query_idx, opposite_idx, 8) if query_idx is not None and opposite_idx is not None else []

        return {
            "available": True,
            "summary": clean_records(row_df.head(1).to_dict("records"))[0],
            "shared_features": shared,
            "query_only_features": self._feature_rows(query_only, 8),
            "opposite_only_features": self._feature_rows(opposite_only, 8),
        }

    @staticmethod
    def _case_to_feature_row(case_rows: pd.DataFrame) -> dict[int, int]:
        if case_rows.empty or "case_index" not in case_rows or "row_index" not in case_rows:
            return {}
        return {
            int(case): int(row)
            for case, row in zip(
                pd.to_numeric(case_rows["case_index"], errors="coerce"),
                pd.to_numeric(case_rows["row_index"], errors="coerce"),
            )
            if pd.notna(case) and pd.notna(row)
        }

    @staticmethod
    def _diff_feature_summary(diffs: pd.DataFrame) -> list[dict[str, Any]]:
        if diffs.empty or "side" not in diffs or "feature_type" not in diffs:
            return []
        rows = []
        for (side, ft), grp in diffs.groupby(["side", "feature_type"]):
            rows.append({
                "side": str(side),
                "feature_type": str(ft),
                "n": int(len(grp)),
            })
        rows.sort(key=lambda r: r["n"], reverse=True)
        return rows[:30]

    # ── Aggregate cross-experiment trends ──────────────────────────────
    def exp_aggregate(self) -> dict[str, Any]:
        case = self.load_table("case_summary")
        bucket = self.load_table("prediction_bucket_summary")
        leakage = self.load_table("leakage_sensitivity_summary")
        identity_shortcuts = self.load_table("identity_shortcut_summary")
        mask_sensitivity = self.load_table("mask_sensitivity_summary")
        evidence = self.load_table("evidence_type_importance")
        comm = self.load_pattern_table("community_profiles")
        align = self.load_pattern_table("structural_embedding_alignment")
        auc = self.load_table("faithfulness_auc_summary")

        trends: list[str] = []

        # Confidence vs accuracy
        if not case.empty and {"baseline_pred_proba", "target_label", "baseline_pred_label"}.issubset(case.columns):
            tmp = case.copy()
            tmp["correct"] = (tmp["target_label"].astype(str) == tmp["baseline_pred_label"].astype(str)).astype(int)
            tmp["bucket"] = pd.cut(pd.to_numeric(tmp["baseline_pred_proba"], errors="coerce"), bins=[0.5, 0.7, 0.85, 1.0], include_lowest=True)
            for b, group in tmp.groupby("bucket", observed=True):
                if len(group) > 0:
                    trends.append(f"Cases with prediction-prob in {b}: n={len(group):,}, accuracy {group['correct'].mean():.1%}.")

        # Cross-link: is the most label-discriminative evidence type the same as the one with most importance?
        if not evidence.empty and "sum_abs_delta_pred_proba" in evidence:
            top = evidence.assign(_v=pd.to_numeric(evidence["sum_abs_delta_pred_proba"], errors="coerce").fillna(0)).sort_values("_v", ascending=False).iloc[0]
            trends.append(
                f"Across the corpus, `{top['evidence_type']}` carries the most aggregated counterfactual importance ({safe_float(top['_v']):.0f} cumulative)."
            )

        if not identity_shortcuts.empty and {"identity_scope", "identity_auc_roc"}.issubset(identity_shortcuts.columns):
            ids = identity_shortcuts[identity_shortcuts["identity_scope"].astype(str) != "combined"].copy()
            if not ids.empty:
                ids["identity_auc_roc"] = pd.to_numeric(ids["identity_auc_roc"], errors="coerce")
                row = ids.sort_values("identity_auc_roc", ascending=False).iloc[0]
                trends.append(
                    f"Identity shortcut audit: `{row['identity_scope']}` alone reaches AUC {safe_float(row['identity_auc_roc']):.3f}; this is shortcut risk, not automatic proof of leakage."
                )

        if not mask_sensitivity.empty and {"mask_name", "accuracy_drop", "flip_rate"}.issubset(mask_sensitivity.columns):
            masks = mask_sensitivity.copy()
            masks["accuracy_drop"] = pd.to_numeric(masks["accuracy_drop"], errors="coerce")
            masks["flip_rate"] = pd.to_numeric(masks["flip_rate"], errors="coerce")
            all_identity = masks[masks["mask_name"].astype(str) == "no_all_identities"]
            if not all_identity.empty:
                row = all_identity.iloc[0]
                acc_drop = safe_float(row.get("accuracy_drop")) or 0.0
                flip_rate = safe_float(row.get("flip_rate")) or 0.0
                trends.append(
                    f"Post-hoc all-identity masking drops accuracy by {acc_drop:.2%} and flips {flip_rate:.1%} of test predictions."
                )
            hub50 = masks[masks["mask_name"].astype(str) == "remove_top_50_hubs"]
            if not hub50.empty:
                row = hub50.iloc[0]
                acc_drop = safe_float(row.get("accuracy_drop")) or 0.0
                flip_rate = safe_float(row.get("flip_rate")) or 0.0
                trends.append(
                    f"Removing the top 50 hub authorities barely changes performance (accuracy drop {acc_drop:.2%}, flip rate {flip_rate:.1%}), so broad legal hubs are structural context rather than a brittle shortcut."
                )

        # Communities accuracy vs corpus accuracy
        if not comm.empty and {"size", "accuracy"}.issubset(comm.columns):
            w = pd.to_numeric(comm["size"], errors="coerce").fillna(0)
            a = pd.to_numeric(comm["accuracy"], errors="coerce")
            if w.sum() > 0 and a.notna().any():
                wacc = float((a.fillna(0) * w).sum() / w.sum())
                trends.append(f"Community-weighted accuracy ({wacc:.1%}) is the corpus baseline disaggregated by structural neighbourhood.")

        # NMI vs accuracy: does HGT outperform pure structure?
        if not align.empty:
            row = align.iloc[0]
            nmi = safe_float(row.get("normalized_mutual_info_all"))
            if nmi is not None and nmi < 0.2:
                trends.append(
                    f"Low NMI ({nmi:.2f}) plus high HGT accuracy means the model has learned outcome-relevant signal beyond pure citation topology — it earns its keep over a structural baseline."
                )

        return {"trends": trends}

    # ─────────────────────────────────────────────────────────────────
    # Existing endpoints (kept for backward compat)
    # ─────────────────────────────────────────────────────────────────

    def validation(self) -> dict[str, Any]:
        auc_summary = self.load_table("faithfulness_auc_summary")
        curve_summary = self.load_table("faithfulness_curve_summary")
        bucket_summary = self.load_table("prediction_bucket_summary")
        bucket_evidence = self.load_table("prediction_bucket_evidence_types")
        return {
            "available": not auc_summary.empty or not bucket_summary.empty,
            "auc_summary": self._records(auc_summary, 50),
            "curve_summary": self._records(curve_summary, 500),
            "bucket_summary": self._records(bucket_summary, 20),
            "bucket_evidence_types": self._records(bucket_evidence, 100),
        }

    def pattern_overview(self) -> dict[str, Any]:
        communities = self.load_pattern_table("community_profiles")
        community_cases = self.load_pattern_table("case_communities")
        success = self.load_pattern_table("community_success_failure")
        skew = self.load_pattern_table("evidence_label_skew")
        neighborhoods = self.load_pattern_table("counterfactual_neighborhoods")
        clusters = self.load_pattern_table("embedding_cluster_profiles")
        alignment = self.load_pattern_table("structural_embedding_alignment")
        available = not communities.empty
        if not available:
            return {"available": False, "status": self.status()}

        metrics: dict[str, Any] = {
            "pattern_dir": str(self.pattern_dir),
            "n_cases": int(len(community_cases)) if not community_cases.empty else None,
            "n_communities": int(len(communities)),
            "n_neighborhood_comparisons": int(len(neighborhoods)) if not neighborhoods.empty else None,
            "n_skew_rows": int(len(skew)) if not skew.empty else None,
            "n_embedding_clusters": int(len(clusters)) if not clusters.empty else None,
        }
        if "size" in communities:
            metrics["largest_community_size"] = int(pd.to_numeric(communities["size"], errors="coerce").max())
            metrics["median_community_size"] = float(pd.to_numeric(communities["size"], errors="coerce").median())
        if {"size", "accuracy"}.issubset(communities.columns):
            weights = pd.to_numeric(communities["size"], errors="coerce").fillna(0)
            acc = pd.to_numeric(communities["accuracy"], errors="coerce")
            if weights.sum() > 0 and acc.notna().any():
                metrics["weighted_community_accuracy"] = float((acc.fillna(0) * weights).sum() / weights.sum())
        if "high_confidence_wrong_n" in communities:
            metrics["high_confidence_wrong_n"] = int(pd.to_numeric(communities["high_confidence_wrong_n"], errors="coerce").fillna(0).sum())
        if "skew_class" in skew:
            metrics["label_discriminative_evidence_n"] = int((skew["skew_class"].astype(str) == "label_discriminative").sum())
        if not alignment.empty:
            for column in ("noise_rate", "normalized_mutual_info_all", "v_measure_all", "adjusted_rand_all"):
                if column in alignment:
                    metrics[column] = clean_value(alignment.iloc[0][column])

        top_communities = communities.copy()
        if "size" in top_communities:
            top_communities = top_communities.assign(__sort=pd.to_numeric(top_communities["size"], errors="coerce")).sort_values("__sort", ascending=False).drop(columns=["__sort"])

        risky = success.copy() if not success.empty else communities.copy()
        if not risky.empty:
            sort_cols = [col for col in ("high_confidence_wrong_n", "mean_confidence", "size") if col in risky.columns]
            if sort_cols:
                for col in sort_cols:
                    risky[col] = pd.to_numeric(risky[col], errors="coerce")
                risky = risky.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

        discriminative = skew.copy()
        if not discriminative.empty:
            if "skew_class" in discriminative:
                discriminative = discriminative[discriminative["skew_class"].astype(str) == "label_discriminative"]
            for col in ("g_test_q_value_bh", "support_train_n", "log_odds_vs_base"):
                if col in discriminative:
                    discriminative[col] = pd.to_numeric(discriminative[col], errors="coerce")
            if {"g_test_q_value_bh", "support_train_n"}.issubset(discriminative.columns):
                discriminative = discriminative.sort_values(["g_test_q_value_bh", "support_train_n"], ascending=[True, False])

        return {
            "available": True,
            "metrics": metrics,
            "top_communities": self._records(top_communities, 25),
            "risky_communities": self._records(risky, 25),
            "discriminative_evidence": self._records(discriminative, 50),
            "embedding_clusters": self._records(clusters, 25),
            "alignment": self._records(alignment, 1),
            "status": self.status(),
        }

    def communities(self, params: dict[str, list[str]]) -> dict[str, Any]:
        df = self.load_pattern_table("community_profiles")
        if df.empty:
            return {"rows": [], "total": 0, "page": 1, "limit": 50}
        work = df.copy()
        query = first(params, "q", "").strip().lower()
        if query:
            mask = pd.Series(False, index=work.index)
            for column in work.columns:
                if work[column].dtype == object:
                    mask = mask | work[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            work = work[mask]
        domain = first(params, "domain", "")
        if domain and "dominant_domain_bucket" in work:
            work = work[work["dominant_domain_bucket"].astype(str) == domain]
        label = first(params, "label", "")
        if label and "dominant_label" in work:
            work = work[work["dominant_label"].astype(str) == label]
        sort_col = first(params, "sort", "size")
        descending = first(params, "dir", "desc") != "asc"
        if sort_col in work:
            sort_values = pd.to_numeric(work[sort_col], errors="coerce")
            if bool(sort_values.notna().any()):
                work = work.assign(__sort=sort_values).sort_values("__sort", ascending=not descending, na_position="last").drop(columns=["__sort"])
            else:
                work = work.sort_values(sort_col, ascending=not descending, na_position="last")
        page = max(1, int(first(params, "page", "1")))
        limit = min(200, max(10, int(first(params, "limit", "50"))))
        total = int(len(work))
        rows = work.iloc[(page - 1) * limit : (page - 1) * limit + limit]
        columns = [
            "community_id",
            "size",
            "dominant_label",
            "dominant_domain_bucket",
            "accuracy",
            "mean_confidence",
            "high_confidence_wrong_n",
            "label_-1_rate",
            "label_1_rate",
            "top_provisions",
            "top_statutes",
            "top_precedents",
        ]
        rows = rows[[column for column in columns if column in rows.columns]]
        return {"rows": clean_records(rows.to_dict("records")), "total": total, "page": page, "limit": limit}

    def community_detail(self, community_id: int) -> dict[str, Any]:
        profiles = self.load_pattern_table("community_profiles")
        cases = self.load_pattern_table("case_communities")
        features = self.load_pattern_table("community_feature_profiles")
        success = self.load_pattern_table("community_success_failure")
        splits = self.load_pattern_table("community_embedding_splits")

        profile_row = None
        if not profiles.empty and "community_id" in profiles:
            match = profiles[pd.to_numeric(profiles["community_id"], errors="coerce") == int(community_id)]
            if not match.empty:
                profile_row = clean_records(match.head(1).to_dict("records"))[0]

        feature_rows = pd.DataFrame()
        if not features.empty and "community_id" in features:
            feature_rows = features[pd.to_numeric(features["community_id"], errors="coerce") == int(community_id)].copy()
            if "enrichment" in feature_rows:
                feature_rows["enrichment"] = pd.to_numeric(feature_rows["enrichment"], errors="coerce")
                feature_rows = feature_rows.sort_values(["enrichment", "case_count_in_community"], ascending=[False, False], na_position="last")

        case_rows = pd.DataFrame()
        if not cases.empty and "community_id" in cases:
            case_rows = cases[pd.to_numeric(cases["community_id"], errors="coerce") == int(community_id)].copy()
            if "pagerank" in case_rows:
                case_rows["pagerank"] = pd.to_numeric(case_rows["pagerank"], errors="coerce")
                case_rows = case_rows.sort_values("pagerank", ascending=False, na_position="last")

        success_row = None
        if not success.empty and "community_id" in success:
            match = success[pd.to_numeric(success["community_id"], errors="coerce") == int(community_id)]
            if not match.empty:
                success_row = clean_records(match.head(1).to_dict("records"))[0]

        split_row = None
        if not splits.empty and "community_id" in splits:
            match = splits[pd.to_numeric(splits["community_id"], errors="coerce") == int(community_id)]
            if not match.empty:
                split_row = clean_records(match.head(1).to_dict("records"))[0]

        case_columns = [
            "case_index",
            "case_id",
            "split",
            "target_label",
            "pred_label",
            "confidence",
            "pagerank",
        ]
        return {
            "profile": profile_row,
            "success": success_row,
            "embedding_split": split_row,
            "features": clean_records(feature_rows.head(100).to_dict("records")) if not feature_rows.empty else [],
            "representative_cases": clean_records(case_rows[[col for col in case_columns if col in case_rows.columns]].head(50).to_dict("records")) if not case_rows.empty else [],
        }

    def cases(self, params: dict[str, list[str]]) -> dict[str, Any]:
        df = self.load_table("case_summary")
        if df.empty:
            return {"rows": [], "total": 0, "page": 1, "limit": 50}

        work = df.copy()
        query = first(params, "q", "").strip().lower()
        if query:
            mask = pd.Series(False, index=work.index)
            for column in ("case_id", "top_evidence_name", "top_path_family", "top_group_id"):
                if column in work:
                    mask = mask | work[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            work = work[mask]

        target = first(params, "target", "")
        if target and "target_label" in work:
            work = work[work["target_label"].astype(str) == target]

        pred = first(params, "pred", "")
        if pred and "baseline_pred_label" in work:
            work = work[work["baseline_pred_label"].astype(str) == pred]

        correctness = first(params, "correct", "")
        if correctness and {"target_label", "baseline_pred_label"}.issubset(work.columns):
            is_correct = work["target_label"].astype(str) == work["baseline_pred_label"].astype(str)
            work = work[is_correct] if correctness == "correct" else work[~is_correct]

        sort_col = first(params, "sort", "max_abs_delta_pred_proba")
        descending = first(params, "dir", "desc") != "asc"
        if sort_col in work:
            sort_values = pd.to_numeric(work[sort_col], errors="coerce")
            if bool(sort_values.notna().any()):
                work = work.assign(__sort=sort_values).sort_values("__sort", ascending=not descending, na_position="last")
                work = work.drop(columns=["__sort"])
            else:
                work = work.sort_values(sort_col, ascending=not descending, na_position="last")

        page = max(1, int(first(params, "page", "1")))
        limit = min(200, max(10, int(first(params, "limit", "50"))))
        total = int(len(work))
        start = (page - 1) * limit
        rows = work.iloc[start : start + limit]
        columns = [
            "case_index",
            "case_id",
            "target_label",
            "saved_pred_label",
            "baseline_pred_label",
            "baseline_pred_proba",
            "max_abs_delta_pred_proba",
            "top_evidence_type",
            "top_evidence_name",
            "top_path_family",
            "n_prediction_flips",
        ]
        rows = rows[[column for column in columns if column in rows.columns]]
        return {"rows": clean_records(rows.to_dict("records")), "total": total, "page": page, "limit": limit}

    def _connected_cases_for_case(self, case_index: int, limit: int = 18) -> tuple[list[dict[str, Any]], int]:
        artifacts = self.load_feature_artifacts()
        if artifacts is None:
            return [], 0
        matrix, _, case_rows = artifacts
        row_map = self._case_to_feature_row(case_rows)
        row_index = row_map.get(int(case_index))
        if row_index is None or row_index >= matrix.shape[0]:
            return [], 0

        feature_indices = matrix.getrow(row_index).indices
        if len(feature_indices) == 0:
            return [], 0

        overlaps = np.asarray(matrix[:, feature_indices].sum(axis=1)).ravel()
        overlaps[row_index] = 0
        connected_row_indices = np.flatnonzero(overlaps > 0)
        if len(connected_row_indices) == 0:
            return [], 0

        top_rows = connected_row_indices[np.argsort(overlaps[connected_row_indices])[::-1]][:limit]
        row_to_case = {
            int(row): int(case)
            for case, row in zip(
                pd.to_numeric(case_rows["case_index"], errors="coerce"),
                pd.to_numeric(case_rows["row_index"], errors="coerce"),
            )
            if pd.notna(case) and pd.notna(row)
        }
        top_case_indices = [row_to_case.get(int(feature_row)) for feature_row in top_rows]
        top_case_indices = [int(idx) for idx in top_case_indices if idx is not None]

        community_rows = self.load_pattern_table("case_communities")
        profiles: dict[int, dict[str, Any]] = {}
        if not community_rows.empty and "case_index" in community_rows:
            community_match = community_rows[
                pd.to_numeric(community_rows["case_index"], errors="coerce").isin(top_case_indices)
            ]
            for _, row in community_match.iterrows():
                idx = safe_int(row.get("case_index"))
                if idx is not None:
                    profiles[idx] = {key: clean_value(value) for key, value in row.items()}

        summary_rows = self.load_table("case_summary")
        if not summary_rows.empty and "case_index" in summary_rows:
            summary_match = summary_rows[
                pd.to_numeric(summary_rows["case_index"], errors="coerce").isin(top_case_indices)
            ]
            for _, row in summary_match.iterrows():
                idx = safe_int(row.get("case_index"))
                if idx is None:
                    continue
                profiles.setdefault(idx, {})
                for key in ("case_id", "split", "target_label", "baseline_pred_label", "baseline_pred_proba"):
                    if key in row:
                        profiles[idx][key] = clean_value(row.get(key))

        connected: list[dict[str, Any]] = []
        for feature_row in top_rows:
            idx = row_to_case.get(int(feature_row))
            if idx is None:
                continue
            profile = profiles.get(idx, {})
            connected.append({
                "case_index": idx,
                "case_id": profile.get("case_id"),
                "split": profile.get("split"),
                "target_label": profile.get("target_label"),
                "pred_label": profile.get("pred_label", profile.get("baseline_pred_label")),
                "confidence": profile.get("confidence", profile.get("baseline_pred_proba")),
                "community_id": profile.get("community_id"),
                "domain_bucket": profile.get("domain_bucket"),
                "shared_feature_count": int(overlaps[int(feature_row)]),
            })
        return clean_records(connected), int(len(connected_row_indices))

    def local_case_graph(
        self,
        case_index: int,
        summary: dict[str, Any] | None,
        top_rows: pd.DataFrame,
    ) -> dict[str, Any]:
        all_groups = self.load_table("case_counterfactual_groups")
        if not all_groups.empty and "case_index" in all_groups:
            groups = all_groups[pd.to_numeric(all_groups["case_index"], errors="coerce") == int(case_index)].copy()
        else:
            groups = top_rows.copy()
        if groups.empty:
            return {
                "available": False,
                "summary": {"total_groups": 0, "total_paths": 0, "connected_case_count": 0},
                "paths": [],
                "evidence": [],
                "connected_cases": [],
            }

        for column in ("group_rank_abs", "abs_delta_pred_proba", "delta_pred_proba", "attention_score"):
            if column in groups:
                groups[column] = pd.to_numeric(groups[column], errors="coerce")
        if "abs_delta_pred_proba" in groups:
            groups = groups.sort_values(["abs_delta_pred_proba", "group_rank_abs"], ascending=[False, True], na_position="last")
        elif "group_rank_abs" in groups:
            groups = groups.sort_values("group_rank_abs", na_position="last")

        path_rows: list[dict[str, Any]] = []
        if "path_family" in groups:
            for path, part in groups.groupby("path_family", dropna=False):
                path_text = str(path) if path is not None and not pd.isna(path) else "unknown path"
                importance = safe_float(part.get("abs_delta_pred_proba", pd.Series(dtype=float)).max()) or 0.0
                path_rows.append({
                    "id": f"path:{path_text}",
                    "path_family": path_text,
                    "label": path_text,
                    "group_count": int(len(part)),
                    "importance": importance,
                    "mean_attention": safe_float(pd.to_numeric(part.get("attention_score", pd.Series(dtype=float)), errors="coerce").mean()),
                })
        path_rows.sort(key=lambda row: (row.get("importance") or 0, row.get("group_count") or 0), reverse=True)

        evidence_columns = [
            "group_rank_abs",
            "group_id",
            "group_kind",
            "evidence_type",
            "evidence_global_index",
            "evidence_id",
            "evidence_name",
            "path_family",
            "relation_types",
            "masked_edge_count",
            "prediction_flipped",
            "delta_pred_proba",
            "abs_delta_pred_proba",
            "attention_score",
            "support_train_n",
            "support_positive_rate",
            "support_negative_rate",
        ]
        visible_evidence = groups[[column for column in evidence_columns if column in groups.columns]].head(32).copy()
        evidence_records: list[dict[str, Any]] = []
        for idx, row in visible_evidence.iterrows():
            path_text = row.get("path_family")
            path_text = str(path_text) if path_text is not None and not pd.isna(path_text) else "unknown path"
            group_id = row.get("group_id") if "group_id" in visible_evidence else idx
            record = {key: clean_value(value) for key, value in row.items()}
            record["id"] = f"evidence:{group_id}"
            record["path_id"] = f"path:{path_text}"
            record["label"] = record.get("evidence_name") or record.get("evidence_type") or "evidence"
            record["importance"] = safe_float(record.get("abs_delta_pred_proba")) or 0.0
            evidence_records.append(record)

        connected_cases, connected_count = self._connected_cases_for_case(case_index, limit=18)
        return {
            "available": True,
            "summary": {
                "case_index": int(case_index),
                "case_id": summary.get("case_id") if summary else None,
                "total_groups": int(len(groups)),
                "total_paths": int(len(path_rows)),
                "shown_paths": int(min(len(path_rows), 18)),
                "shown_evidence": int(len(evidence_records)),
                "connected_case_count": connected_count,
                "shown_connected_cases": int(len(connected_cases)),
            },
            "paths": clean_records(path_rows[:18]),
            "evidence": clean_records(evidence_records),
            "connected_cases": connected_cases,
        }

    def case_detail(self, case_index: int) -> dict[str, Any]:
        case_summary = self.load_table("case_summary")
        top = self.load_table("case_top_explanations")
        attention = self.load_table("attention_counterfactual_overlap")
        case_communities = self.load_pattern_table("case_communities")
        neighborhoods = self.load_pattern_table("counterfactual_neighborhoods")
        clusters = self.load_pattern_table("case_embedding_clusters")
        if case_summary.empty:
            return {"summary": None, "top_explanations": [], "attention": None, "pattern": None}

        summary_rows = case_summary[case_summary["case_index"].astype(int) == int(case_index)]
        summary = clean_records(summary_rows.to_dict("records"))[0] if not summary_rows.empty else None

        top_rows = pd.DataFrame()
        if not top.empty and "case_index" in top:
            top_rows = top[top["case_index"].astype(int) == int(case_index)].copy()
            if "group_rank_abs" in top_rows:
                top_rows["group_rank_abs"] = pd.to_numeric(top_rows["group_rank_abs"], errors="coerce")
                top_rows = top_rows.sort_values("group_rank_abs")

        attention_row = None
        if not attention.empty and "case_index" in attention:
            match = attention[attention["case_index"].astype(int) == int(case_index)]
            if not match.empty:
                attention_row = clean_records(match.to_dict("records"))[0]

        pattern: dict[str, Any] = {}
        if not case_communities.empty and "case_index" in case_communities:
            match = case_communities[pd.to_numeric(case_communities["case_index"], errors="coerce") == int(case_index)]
            if not match.empty:
                pattern["community"] = clean_records(match.head(1).to_dict("records"))[0]
        if not neighborhoods.empty and "case_index" in neighborhoods:
            match = neighborhoods[pd.to_numeric(neighborhoods["case_index"], errors="coerce") == int(case_index)]
            if not match.empty:
                pattern["nearest_opposite"] = clean_records(match.head(1).to_dict("records"))[0]
        if not clusters.empty and "case_index" in clusters:
            match = clusters[pd.to_numeric(clusters["case_index"], errors="coerce") == int(case_index)]
            if not match.empty:
                pattern["embedding_cluster"] = clean_records(match.head(1).to_dict("records"))[0]

        return {
            "summary": summary,
            "top_explanations": clean_records(top_rows.head(50).to_dict("records")),
            "attention": attention_row,
            "pattern": pattern or None,
            "local_graph": self.local_case_graph(int(case_index), summary, top_rows),
        }

    def evidence_detail(self, params: dict[str, list[str]]) -> dict[str, Any]:
        evidence_type = first(params, "evidence_type", "")
        evidence_global_index = first(params, "evidence_global_index", "")
        evidence_id = first(params, "evidence_id", "")
        evidence_name = first(params, "evidence_name", "")
        relation_types = first(params, "relation_types", "")
        path_family = first(params, "path_family", "")

        support = self.load_table("connected_case_label_distribution")
        top = self.load_table("case_top_explanations")

        support_row = None
        if not support.empty:
            support_match = support.copy()
            if evidence_type and "evidence_type" in support_match:
                support_match = support_match[support_match["evidence_type"].astype(str) == evidence_type]
            if evidence_global_index and evidence_global_index not in {"nan", "None"} and "evidence_global_index" in support_match:
                support_match = support_match[
                    pd.to_numeric(support_match["evidence_global_index"], errors="coerce")
                    == pd.to_numeric(pd.Series([evidence_global_index]), errors="coerce").iloc[0]
                ]
            elif evidence_id and "evidence_id" in support_match:
                support_match = support_match[support_match["evidence_id"].astype(str) == evidence_id]
            elif evidence_name and "evidence_name" in support_match:
                support_match = support_match[support_match["evidence_name"].astype(str) == evidence_name]
            if not support_match.empty:
                support_row = clean_records(support_match.head(1).to_dict("records"))[0]

        case_rows = pd.DataFrame()
        if not top.empty:
            case_rows = top.copy()
            if evidence_type and "evidence_type" in case_rows:
                case_rows = case_rows[case_rows["evidence_type"].astype(str) == evidence_type]
            if evidence_global_index and evidence_global_index not in {"nan", "None"} and "evidence_global_index" in case_rows:
                case_rows = case_rows[
                    pd.to_numeric(case_rows["evidence_global_index"], errors="coerce")
                    == pd.to_numeric(pd.Series([evidence_global_index]), errors="coerce").iloc[0]
                ]
            elif evidence_id and "evidence_id" in case_rows:
                case_rows = case_rows[case_rows["evidence_id"].astype(str) == evidence_id]
            elif evidence_name and "evidence_name" in case_rows:
                case_rows = case_rows[case_rows["evidence_name"].astype(str) == evidence_name]
            if relation_types and "relation_types" in case_rows:
                tokens = [token for token in relation_types.split("|") if token]
                if tokens and case_rows.empty:
                    relation_mask = pd.Series(False, index=top.index)
                    for token in tokens:
                        relation_mask = relation_mask | top["relation_types"].fillna("").astype(str).str.contains(token, regex=False)
                    case_rows = top[relation_mask]
            if "abs_delta_pred_proba" in case_rows:
                case_rows["abs_delta_pred_proba"] = pd.to_numeric(case_rows["abs_delta_pred_proba"], errors="coerce")
                case_rows = case_rows.sort_values("abs_delta_pred_proba", ascending=False, na_position="last")

        notes = []
        for token in relation_types.split("|"):
            if token:
                pieces = token.split("__")
                if len(pieces) == 3:
                    note = relation_note(pieces[1])
                    if note:
                        notes.append(f"{pieces[0]} -> {pieces[1]} -> {pieces[2]}: {note}")
        if not notes:
            notes.extend(path_notes(path_family))
        deduped_notes = list(dict.fromkeys(notes))

        columns = [
            "case_index",
            "case_id",
            "target_label",
            "baseline_pred_label",
            "evidence_type",
            "evidence_name",
            "path_family",
            "delta_pred_proba",
            "abs_delta_pred_proba",
            "support_train_n",
            "support_positive_rate",
        ]
        if not case_rows.empty:
            case_rows = case_rows[[column for column in columns if column in case_rows.columns]]

        return {
            "evidence_type": evidence_type,
            "evidence_global_index": evidence_global_index,
            "evidence_id": evidence_id,
            "evidence_name": evidence_name,
            "relation_types": relation_types,
            "path_family": path_family,
            "relation_notes": deduped_notes,
            "support": support_row,
            "cases": clean_records(case_rows.head(50).to_dict("records")) if not case_rows.empty else [],
        }

    def generic_table(self, table_name: str, params: dict[str, list[str]]) -> dict[str, Any]:
        df = self.load_table(table_name)
        if df.empty:
            return {"rows": [], "total": 0}
        work = df.copy()
        query = first(params, "q", "").strip().lower()
        if query:
            mask = pd.Series(False, index=work.index)
            for column in work.columns:
                if work[column].dtype == object:
                    mask = mask | work[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            work = work[mask]
        type_filter = first(params, "type", "")
        if type_filter and "evidence_type" in work:
            work = work[work["evidence_type"].astype(str) == type_filter]
        sort_col = first(params, "sort", "")
        if sort_col and sort_col in work:
            sort_values = pd.to_numeric(work[sort_col], errors="coerce")
            ascending = first(params, "dir", "desc") == "asc"
            if bool(sort_values.notna().any()):
                work = work.assign(__sort=sort_values).sort_values("__sort", ascending=ascending, na_position="last")
                work = work.drop(columns=["__sort"])
            else:
                work = work.sort_values(sort_col, ascending=ascending, na_position="last")
        limit = min(500, max(10, int(first(params, "limit", "100"))))
        return {"rows": clean_records(work.head(limit).to_dict("records")), "total": int(len(work))}

    # ── EXP 7: Full-graph Leiden communities ──────────────────────────
    def exp_full_graph_communities(self, requested_resolution: float | None) -> dict[str, Any]:
        sweep = self.load_full_graph_csv("resolution_sweep_summary.csv")
        manifest = read_json(self.full_graph_dir / "full_graph_manifest.json")
        if sweep.empty:
            return {
                "available": False,
                "manifest": manifest,
                "findings": [],
                "resolutions": [],
            }
        resolution = self.full_graph_pick_resolution(requested_resolution)
        if resolution is None:
            return {"available": False, "manifest": manifest, "findings": [], "resolutions": []}
        suffix = self._resolution_suffix(resolution)
        profiles = self.load_full_graph_csv(f"full_graph_community_profiles_{suffix}.csv")
        predictions = self.load_full_graph_csv(f"full_graph_community_predictions_{suffix}.csv")
        boundary = self.load_full_graph_csv(f"full_graph_boundary_cases_{suffix}.csv")
        authorities_table = self.load_full_graph_csv(
            f"full_graph_community_authorities_{suffix}.csv"
        )

        sweep_records = clean_records(sweep.sort_values("resolution").to_dict("records"))

        findings: list[str] = []
        if not profiles.empty:
            n_comm = int(profiles.shape[0])
            n_cases = int(profiles["n_cases"].sum()) if "n_cases" in profiles else 0
            weighted_acc = None
            if {"n_cases", "accuracy"}.issubset(profiles.columns):
                w = profiles["n_cases"].fillna(0)
                a = pd.to_numeric(profiles["accuracy"], errors="coerce")
                if w.sum() > 0 and a.notna().any():
                    weighted_acc = float((a.fillna(0) * w).sum() / w.sum())
            findings.append(
                f"At resolution {resolution:.2f}, the full graph splits into "
                f"{n_comm:,} communities covering {n_cases:,} cases."
                + (f" Size-weighted accuracy {weighted_acc:.1%}." if weighted_acc is not None else "")
            )
            big = profiles.sort_values("n_cases", ascending=False).iloc[0]
            findings.append(
                f"Largest community #{safe_int(big.get('community_id'))} has "
                f"{safe_int(big.get('n_cases'))} cases — dominant domain "
                f"{big.get('dominant_domain_bucket', '?')}, label {big.get('dominant_label', '?')}, "
                f"accuracy {safe_float(big.get('accuracy')) or 0:.1%}."
            )
        if not predictions.empty and "high_conf_wrong_n" in predictions.columns:
            risky = predictions.copy()
            risky["n_cases"] = pd.to_numeric(risky.get("n_cases", 0), errors="coerce").fillna(0)
            risky = risky[risky["n_cases"] >= 20].sort_values("high_conf_wrong_n", ascending=False)
            if not risky.empty:
                row = risky.iloc[0]
                findings.append(
                    f"Worst confident-wrong community at this resolution: #{safe_int(row.get('community_id'))} "
                    f"with {safe_int(row.get('high_conf_wrong_n'))} confident-wrong cases "
                    f"(accuracy {safe_float(row.get('accuracy')) or 0:.1%})."
                )
        if not boundary.empty:
            findings.append(
                f"{int(boundary.shape[0]):,} cases are flagged as boundary cases — their authority "
                f"neighbors span multiple communities (high normalized entropy)."
            )

        top_profiles = profiles.sort_values("n_cases", ascending=False).head(20) if not profiles.empty else profiles

        risky_predictions = pd.DataFrame()
        if not predictions.empty:
            rp = predictions.copy()
            rp["n_cases"] = pd.to_numeric(rp.get("n_cases", 0), errors="coerce").fillna(0)
            rp = rp[rp["n_cases"] >= 20]
            if "high_conf_wrong_n" in rp:
                risky_predictions = rp.sort_values("high_conf_wrong_n", ascending=False).head(20)

        # Build a community → predictions lookup so the frontend can render
        # both panels as cards without a second join.
        prediction_by_id: dict[int, dict[str, Any]] = {}
        if not predictions.empty and "community_id" in predictions.columns:
            for record in clean_records(predictions.to_dict("records")):
                cid = record.get("community_id")
                if cid is not None:
                    prediction_by_id[int(cid)] = record

        # Per-community top authorities, keyed by community_id, so each
        # community card can render an "evidence pill" list.
        authorities_by_id: dict[int, list[dict[str, Any]]] = {}
        if not authorities_table.empty and "community_id" in authorities_table.columns:
            keep_cols = [
                c
                for c in [
                    "community_id",
                    "feature_type",
                    "feature_rank",
                    "feature_name",
                    "case_count_in_community",
                    "community_rate",
                ]
                if c in authorities_table.columns
            ]
            cleaned = clean_records(authorities_table[keep_cols].to_dict("records"))
            for record in cleaned:
                cid = record.get("community_id")
                if cid is None:
                    continue
                authorities_by_id.setdefault(int(cid), []).append(record)
            for cid, rows in authorities_by_id.items():
                rows.sort(
                    key=lambda r: (
                        str(r.get("feature_type") or ""),
                        int(r.get("feature_rank") or 999),
                    )
                )

        boundary_top = pd.DataFrame()
        if not boundary.empty:
            cols = [
                "case_index",
                "case_id",
                "case_community",
                "n_authorities",
                "n_communities_touched",
                "neighborhood_normalized_entropy",
                "split",
                "target_label",
                "pred_label",
                "confidence",
                "correct",
                "domain_bucket",
            ]
            keep = [c for c in cols if c in boundary.columns]
            boundary_top = boundary.sort_values(
                "neighborhood_normalized_entropy", ascending=False
            )[keep].head(25)

        return {
            "available": True,
            "manifest": manifest,
            "current_resolution": resolution,
            "resolutions": self.full_graph_resolutions(),
            "sweep_summary": sweep_records,
            "n_communities": int(profiles.shape[0]) if not profiles.empty else 0,
            "n_cases_in_communities": int(profiles["n_cases"].sum()) if "n_cases" in profiles else 0,
            "top_profiles": self._records(top_profiles, 20),
            "risky_predictions": self._records(risky_predictions, 20),
            "boundary_cases": self._records(boundary_top, 25),
            "predictions_by_community": prediction_by_id,
            "authorities_by_community": authorities_by_id,
            "findings": findings,
        }

    # ── EXP 8: Community hierarchy (broad → specific) ─────────────────
    def exp_community_hierarchy(self) -> dict[str, Any]:
        manifest = read_json(self.full_graph_dir / "community_hierarchy_manifest.json")
        pairwise = self.load_full_graph_csv("resolution_pairwise_alignment.csv")
        chains = self.load_full_graph_csv("community_lineage_chains.csv")
        pairs = self.load_full_graph_csv("community_lineage_pairs.csv")

        if pairwise.empty and chains.empty and pairs.empty:
            return {"available": False, "manifest": manifest, "findings": []}

        findings: list[str] = []
        if not pairwise.empty:
            adjacent = pairwise.sort_values(
                ["coarse_resolution", "fine_resolution"]
            ).copy()
            top_drift = adjacent.sort_values("normalized_mutual_info").iloc[0]
            findings.append(
                f"Lowest hierarchical alignment is between resolutions "
                f"{safe_float(top_drift['coarse_resolution']):.2f} and "
                f"{safe_float(top_drift['fine_resolution']):.2f} (NMI "
                f"{safe_float(top_drift['normalized_mutual_info']):.3f}) — that is where "
                f"the partition reorganizes most when refining."
            )
        if not chains.empty:
            big_leaves = chains.sort_values("leaf_size", ascending=False).head(1)
            row = big_leaves.iloc[0]
            findings.append(
                f"Largest leaf community at the finest resolution ({safe_float(row.get('leaf_resolution')):.2f}) "
                f"has {safe_int(row.get('leaf_size'))} cases."
            )

        chains_sorted = chains.copy()
        if "leaf_size" in chains_sorted.columns:
            chains_sorted = chains_sorted.sort_values("leaf_size", ascending=False)

        # Sample: per coarse community, list a few fine-community children with sizes.
        children = pd.DataFrame()
        if not pairs.empty and {"parent_community", "child_community"}.issubset(pairs.columns):
            children = (
                pairs.sort_values(
                    ["coarse_resolution", "fine_resolution", "parent_community", "overlap_n"],
                    ascending=[True, True, True, False],
                )
                .groupby(["coarse_resolution", "fine_resolution", "parent_community"], as_index=False)
                .head(5)
            )

        return {
            "available": True,
            "manifest": manifest,
            "pairwise": clean_records(pairwise.to_dict("records")),
            "lineage_chains": self._records(chains_sorted, 50),
            "lineage_pairs": self._records(children, 200),
            "findings": findings,
        }

    # ── EXP 9: Bridge / hub / core authorities ────────────────────────
    def exp_bridge_hub(self, requested_resolution: float | None) -> dict[str, Any]:
        manifest_files = sorted(self.full_graph_dir.glob("bridge_hub_manifest_res_*.json"))
        manifest = read_json(manifest_files[0]) if manifest_files else {}
        resolution = self.full_graph_pick_resolution(requested_resolution)
        if resolution is None:
            return {"available": False, "manifest": manifest, "findings": [], "resolutions": []}
        suffix = self._resolution_suffix(resolution)
        roles = self.load_full_graph_csv(f"authority_role_classification_{suffix}.csv")
        role_summary = self.load_full_graph_csv(f"authority_role_summary_{suffix}.csv")
        bridge_pairs = self.load_full_graph_csv(f"bridge_authority_pairs_{suffix}.csv")
        mask_summary = self.load_table("mask_sensitivity_summary")
        hub_authorities = self.load_table("mask_sensitivity_hub_authorities")
        per_resolution_manifest = read_json(self.full_graph_dir / f"bridge_hub_manifest_{suffix}.json")
        if per_resolution_manifest:
            manifest = per_resolution_manifest

        if roles.empty:
            return {
                "available": False,
                "manifest": manifest,
                "findings": [],
                "resolutions": self.full_graph_resolutions(),
                "current_resolution": resolution,
            }

        findings: list[str] = []
        role_counts = (
            roles["role"].value_counts() if "role" in roles.columns else pd.Series(dtype=int)
        )
        n_total = int(roles.shape[0])
        if not role_counts.empty:
            findings.append(
                f"Classified {n_total:,} legal authorities at resolution {resolution:.2f}: "
                + ", ".join(f"{role} {int(cnt):,}" for role, cnt in role_counts.items())
                + "."
            )
        if "role" in roles.columns:
            hubs = roles[roles["role"] == "hub"].sort_values("n_cases", ascending=False)
            if not hubs.empty:
                hub = hubs.iloc[0]
                findings.append(
                    f"Top hub authority: {hub.get('feature_type', '?')} \"{hub.get('feature_name', '?')}\" "
                    f"appears in {safe_int(hub.get('n_cases'))} cases across "
                    f"{safe_int(hub.get('n_communities_touched'))} communities."
                )
            bridges = roles[roles["role"] == "bridge"].sort_values("n_cases", ascending=False)
            if not bridges.empty and not bridge_pairs.empty:
                top_pair = bridge_pairs.iloc[0]
                findings.append(
                    f"Communities #{safe_int(top_pair.get('community_a'))} and "
                    f"#{safe_int(top_pair.get('community_b'))} are connected by "
                    f"{safe_int(top_pair.get('n_bridge_authorities'))} bridge authorities — "
                    f"the strongest cross-community link."
                )

        hub_sensitivity = pd.DataFrame()
        if not mask_summary.empty:
            hub_sensitivity = mask_summary.copy()
            if "mask_family" in hub_sensitivity.columns:
                hub_sensitivity = hub_sensitivity[
                    hub_sensitivity["mask_family"].astype(str) == "hub_authority"
                ].copy()
            for column in (
                "top_k_hubs",
                "n_cases",
                "masked_edge_share",
                "accuracy_drop",
                "macro_f1_drop",
                "confidence_drop",
                "mean_confidence_drop",
                "flip_rate",
            ):
                if column in hub_sensitivity:
                    hub_sensitivity[column] = pd.to_numeric(hub_sensitivity[column], errors="coerce")
            if not hub_sensitivity.empty:
                top = hub_sensitivity.sort_values("top_k_hubs", ascending=False).iloc[0]
                acc_drop = safe_float(top.get("accuracy_drop")) or 0.0
                flip_rate = safe_float(top.get("flip_rate")) or 0.0
                findings.append(
                    f"Hub-removal stress test: removing top {safe_int(top.get('top_k_hubs'))} hubs changes accuracy by {acc_drop:.2%} and flips {flip_rate:.1%} of test predictions."
                )

        # Top examples per role
        per_role: dict[str, list[dict[str, Any]]] = {}
        for role in ("core", "bridge", "hub", "rare"):
            if "role" not in roles.columns:
                continue
            subset = roles[roles["role"] == role].copy()
            if subset.empty:
                per_role[role] = []
                continue
            sort_cols = [c for c in ["n_cases", "max_community_share"] if c in subset.columns]
            if sort_cols:
                subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            keep = [
                c
                for c in [
                    "feature_type",
                    "feature_name",
                    "n_cases",
                    "n_test_cases",
                    "n_communities_touched",
                    "home_community",
                    "max_community_share",
                    "normalized_entropy",
                    "model_accuracy_on_citing_cases",
                    "mean_confidence_on_citing_cases",
                    "top_communities",
                ]
                if c in subset.columns
            ]
            per_role[role] = self._records(subset[keep], 25)

        return {
            "available": True,
            "manifest": manifest,
            "current_resolution": resolution,
            "resolutions": self.full_graph_resolutions(),
            "n_authorities": n_total,
            "role_summary": clean_records(role_summary.to_dict("records")),
            "bridge_pairs": clean_records(bridge_pairs.head(30).to_dict("records")),
            "examples_by_role": per_role,
            "hub_sensitivity": clean_records(hub_sensitivity.to_dict("records")) if not hub_sensitivity.empty else [],
            "masked_hub_authorities": self._records(hub_authorities, 50),
            "findings": findings,
        }

    def exp_mask_sensitivity(self) -> dict[str, Any]:
        summary = self.load_table("mask_sensitivity_summary")
        top_domains = self.load_table("mask_sensitivity_top_domain_drops")
        hubs = self.load_table("mask_sensitivity_hub_authorities")
        manifest = read_json(self.output_dir / "mask_sensitivity_manifest.json")

        if summary.empty:
            return {
                "available": False,
                "manifest": manifest,
                "identity_masks": [],
                "hub_masks": [],
                "domain_drops": [],
                "hub_authorities": [],
                "findings": [
                    "Mask sensitivity outputs not found. "
                    "Run the mask sensitivity audit for this output directory."
                ],
            }

        work = summary.copy()
        for col in (
            "n_cases", "masked_edge_share", "baseline_accuracy", "masked_accuracy",
            "accuracy_drop", "baseline_macro_f1", "masked_macro_f1", "macro_f1_drop",
            "confidence_drop", "mean_confidence_drop", "flip_rate",
            "n_masked_edges", "n_original_edges", "n_masked_authorities",
        ):
            if col in work:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        identity_masks = pd.DataFrame()
        hub_masks = pd.DataFrame()
        if "mask_family" in work.columns:
            identity_masks = work[work["mask_family"].astype(str) == "identity"].sort_values(
                "accuracy_drop", ascending=False, na_position="last"
            )
            hub_masks = work[work["mask_family"].astype(str) == "hub_authority"].sort_values(
                "n_masked_authorities", ascending=True, na_position="last"
            )
        else:
            identity_masks = work

        domain_work = pd.DataFrame()
        if not top_domains.empty:
            domain_work = top_domains.copy()
            for col in ("n_cases", "accuracy_drop", "macro_f1_drop", "confidence_drop", "flip_rate"):
                if col in domain_work:
                    domain_work[col] = pd.to_numeric(domain_work[col], errors="coerce")

        hub_auth_rows = pd.DataFrame()
        if not hubs.empty:
            hub_auth_rows = hubs.copy()
            for col in ("hub_rank", "n_cases", "n_unique_cases"):
                if col in hub_auth_rows:
                    hub_auth_rows[col] = pd.to_numeric(hub_auth_rows[col], errors="coerce")
            if "hub_rank" in hub_auth_rows.columns:
                hub_auth_rows = hub_auth_rows.sort_values("hub_rank", ascending=True, na_position="last")

        findings: list[str] = []
        if not identity_masks.empty:
            all_mask = identity_masks[
                identity_masks.get("mask_name", pd.Series(dtype=str)).astype(str) == "no_all_identities"
            ]
            if not all_mask.empty:
                row = all_mask.iloc[0]
                acc_drop = safe_float(row.get("accuracy_drop")) or 0.0
                flip = safe_float(row.get("flip_rate")) or 0.0
                conf_drop = safe_float(row.get("confidence_drop")) or 0.0
                findings.append(
                    f"Removing all identity nodes drops accuracy by {acc_drop:.2%} and flips "
                    f"{flip:.1%} of test predictions (mean original-class confidence drop {conf_drop:.3f})."
                )
            others = identity_masks[
                identity_masks.get("mask_name", pd.Series(dtype=str)).astype(str) != "no_all_identities"
            ]
            if not others.empty:
                strongest = others.sort_values("accuracy_drop", ascending=False, na_position="last").iloc[0]
                strongest_drop = safe_float(strongest.get("accuracy_drop")) or 0.0
                findings.append(
                    f"Among individual identity masks, `{strongest.get('mask_name')}` causes "
                    f"the largest accuracy drop ({strongest_drop:.2%})."
                )

        if not hub_masks.empty:
            smallest = hub_masks.iloc[0]
            largest = hub_masks.iloc[-1]
            small_k = safe_int(smallest.get("n_masked_authorities")) or "?"
            large_k = safe_int(largest.get("n_masked_authorities")) or "?"
            small_drop = safe_float(smallest.get("accuracy_drop")) or 0.0
            large_drop = safe_float(largest.get("accuracy_drop")) or 0.0
            small_flip = safe_float(smallest.get("flip_rate")) or 0.0
            findings.append(
                f"Removing the top {small_k} hub authorities drops accuracy by {small_drop:.2%} "
                f"(flip rate {small_flip:.1%}); removing the top {large_k} drops accuracy by {large_drop:.2%}."
            )

        keep_mask_cols = [
            "mask_name", "mask_family", "top_k_hubs", "n_cases", "n_masked_authorities",
            "masked_edge_share", "baseline_accuracy", "masked_accuracy", "accuracy_drop",
            "baseline_macro_f1", "masked_macro_f1", "macro_f1_drop",
            "confidence_drop", "mean_confidence_drop", "flip_rate",
        ]
        keep_domain_cols = [
            "mask_name", "mask_family", "domain_bucket", "n_cases",
            "accuracy_drop", "macro_f1_drop", "confidence_drop", "flip_rate",
        ]
        keep_hub_cols = ["hub_rank", "feature_type", "feature_name", "n_cases", "n_unique_cases", "role"]

        def _trim(df: pd.DataFrame, cols: list[str], limit: int = 200) -> list[dict[str, Any]]:
            if df.empty:
                return []
            return clean_records(df[[c for c in cols if c in df.columns]].head(limit).to_dict("records"))

        return {
            "available": True,
            "manifest": manifest,
            "identity_masks": _trim(identity_masks, keep_mask_cols),
            "hub_masks": _trim(hub_masks, keep_mask_cols),
            "domain_drops": _trim(domain_work, keep_domain_cols),
            "hub_authorities": _trim(hub_auth_rows, keep_hub_cols, limit=60),
            "findings": findings,
        }

    @staticmethod
    def _records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return clean_records(df.head(limit).to_dict("records"))


def first(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0]


# ─────────────────────────────────────────────────────────────────
# Multi-run discovery
# ─────────────────────────────────────────────────────────────────

def discover_runs(outputs_root: Path) -> list[dict[str, Any]]:
    """Find sibling output dirs that look like explanation runs (have case_summary or community_profiles)."""
    if not outputs_root.exists():
        return []
    runs = []
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        kinds = []
        if (child / "case_summary.csv").exists():
            kinds.append("counterfactual")
        if (child / "faithfulness_auc_summary.csv").exists():
            kinds.append("faithfulness")
        if (child / "identity_shortcut_summary.csv").exists():
            kinds.append("identity_shortcut")
        if (child / "community_profiles.csv").exists():
            kinds.append("pattern")
        if (child / "embedding_cluster_profiles.csv").exists():
            kinds.append("clusters")
        if (child / "counterfactual_neighborhoods.csv").exists():
            kinds.append("neighborhoods")
        if (child / "resolution_sweep_summary.csv").exists():
            kinds.append("full_graph")
        if not kinds:
            continue
        runs.append({"name": child.name, "path": str(child), "kinds": kinds})
    return runs


class VisualizerHandler(BaseHTTPRequestHandler):
    store: ExplanationStore
    runs: list[dict[str, Any]] = []

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self.serve_static(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self.serve_static(STATIC_DIR / path.removeprefix("/static/"))
            elif path == "/api/status":
                payload = self.store.status()
                payload["runs"] = self.runs
                self.send_json(payload)
            elif path == "/api/overview":
                self.send_json(self.store.overview())
            elif path == "/api/exp_overview":
                self.send_json(self.store.experiment_overview())
            elif path == "/api/exp/embeddings":
                self.send_json(self.store.exp_embeddings())
            elif path == "/api/exp/counterfactual":
                self.send_json(self.store.exp_counterfactual())
            elif path == "/api/exp/identity_shortcuts":
                self.send_json(self.store.exp_identity_shortcuts())
            elif path == "/api/exp/faithfulness":
                self.send_json(self.store.exp_faithfulness())
            elif path == "/api/exp/communities":
                self.send_json(self.store.exp_communities())
            elif path == "/api/exp/embedding_clusters":
                self.send_json(self.store.exp_embedding_clusters())
            elif path == "/api/exp/neighborhoods":
                self.send_json(self.store.exp_neighborhoods())
            elif path == "/api/opposite_graph":
                self.send_json(self.store.opposite_graph(safe_int(first(params, "case_index", "-1"))))
            elif path == "/api/exp/aggregate":
                self.send_json(self.store.exp_aggregate())
            elif path == "/api/exp/full_graph_communities":
                requested = first(params, "resolution", "")
                requested_value = float(requested) if requested else None
                self.send_json(self.store.exp_full_graph_communities(requested_value))
            elif path == "/api/exp/community_hierarchy":
                self.send_json(self.store.exp_community_hierarchy())
            elif path == "/api/exp/bridge_hub":
                requested = first(params, "resolution", "")
                requested_value = float(requested) if requested else None
                self.send_json(self.store.exp_bridge_hub(requested_value))
            elif path == "/api/exp/mask_sensitivity":
                self.send_json(self.store.exp_mask_sensitivity())
            elif path == "/api/cases":
                self.send_json(self.store.cases(params))
            elif path == "/api/case":
                self.send_json(self.store.case_detail(int(first(params, "case_index", "-1"))))
            elif path == "/api/evidence_detail":
                self.send_json(self.store.evidence_detail(params))
            elif path == "/api/validation":
                self.send_json(self.store.validation())
            elif path == "/api/pattern":
                self.send_json(self.store.pattern_overview())
            elif path == "/api/communities":
                self.send_json(self.store.communities(params))
            elif path == "/api/community":
                self.send_json(self.store.community_detail(int(first(params, "community_id", "-1"))))
            elif path == "/api/table":
                table_name = first(params, "name", "")
                self.send_json(self.store.generic_table(table_name, params))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.send_json({"error": repr(exc)}, status=500)

    def serve_static(self, path: Path) -> None:
        resolved = path.resolve()
        if not str(resolved).startswith(str(STATIC_DIR.resolve())) or not resolved.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[visualizer] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local visualizer for FINAL_EXPLANATION outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pattern-dir",
        type=Path,
        default=None,
        help="Directory with pattern_why CSVs. Defaults to sibling outputs/pattern_why when present.",
    )
    parser.add_argument(
        "--full-graph-dir",
        type=Path,
        default=None,
        help="Directory with pattern_why_full_graph CSVs. Defaults to sibling outputs/pattern_why_full_graph.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {args.output_dir}")
    VisualizerHandler.store = ExplanationStore(
        args.output_dir,
        pattern_dir=args.pattern_dir,
        full_graph_dir=args.full_graph_dir,
    )
    VisualizerHandler.runs = discover_runs(args.output_dir.parent)
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    print(f"[visualizer] output_dir={args.output_dir.resolve()}", flush=True)
    print(f"[visualizer] pattern_dir={VisualizerHandler.store.pattern_dir}", flush=True)
    print(f"[visualizer] full_graph_dir={VisualizerHandler.store.full_graph_dir}", flush=True)
    print(f"[visualizer] discovered_runs={[r['name'] for r in VisualizerHandler.runs]}", flush=True)
    print(f"[visualizer] url=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
