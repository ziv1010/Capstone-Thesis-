#!/usr/bin/env python
"""Shared data layer for the case-similarity presentation graphs.

Used by both ``visualizer.py`` (live panels) and
``generate_presentation_figures.py`` (static slide figures) so the browser and
the deck always show the same numbers.

Three things live here:

``CaseNeighborIndex``
    Nearest same-label / opposite-label cases in HGT embedding space plus the
    shared / query-only / other-only evidence sets behind a case pair.

``CounterfactualFactorIndex``
    Per-case counterfactual masking results, keyed so evidence boxes can be
    stamped with the rank and effect size the masking measured.

``case_display_name``
    Turns the internal ``bucket__Title on Date`` case id into a human case name
    plus an Indian Kanoon lookup URL.

Deliberately free of ``torch`` / ``matplotlib`` / HTTP imports: this module is
imported by the web server, which must stay lightweight.  The IDF + label-skew
ranking mirrors ``counterfactual_neighborhoods.top_difference_features``, which
cannot be imported directly because that module pulls in torch.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import scipy.sparse as sp


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PATTERN_DIR = (
    APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why"
)
DEFAULT_EXPLANATION_DIR = (
    APP_ROOT / "outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
)

Pool = Literal["test", "train", "val", "all"]
Order = Literal["evidence", "counterfactual"]
#: Which label decides whether a neighbour counts as "same" or "opposite".
#: ``target`` reproduces the published nearest-opposite-label analysis; ``pred``
#: finds cases the *model* actually decided the other way, which is the only
#: setting where "what made the difference" is a real question — the nearest
#: opposite-true-label case gets the same prediction ~93% of the time.
Match = Literal["target", "pred"]

#: Single-letter codes used on the ego-graph edge labels ("1J, 4P, 1V").
TYPE_INITIALS = {
    "judge": "J",
    "precedent": "P",
    "provision": "V",
    "court": "C",
    "statute": "S",
}

#: Legible names for the five feature types carried by the case-feature matrix.
TYPE_LABELS = {
    "judge": "JUDGE",
    "precedent": "PRECEDENT",
    "provision": "PROVISION",
    "court": "COURT",
    "statute": "STATUTE",
}

INDIAN_KANOON_SEARCH = "https://indiankanoon.org/search/?formInput="

_BUCKET_TITLES = {
    "fin_fraud": "Financial fraud",
    "financial_fraud": "Financial fraud",
    "family_matrimonial": "Family / matrimonial",
    "land_property": "Land / property",
    "motor_accidents": "Motor accidents",
    "sexual_offences": "Sexual offences",
    "food_safety": "Food safety",
}


# ---------------------------------------------------------------------------
# Case names
# ---------------------------------------------------------------------------


def case_display_name(case_id: Any, short_limit: int = 62) -> dict[str, Any]:
    """Split ``bucket__Title on Date`` into displayable parts.

    The corpus was scraped as PDFs named by title only (see ``INPUT_DATA/``), so
    there is no stored document id to link to.  A search URL is the lookup
    affordance instead.
    """
    raw = "" if case_id is None else str(case_id)
    if raw in {"", "nan", "None"}:
        return {
            "case_id": None,
            "bucket": None,
            "bucket_label": None,
            "title": None,
            "short_title": None,
            "search_url": None,
        }
    bucket, _, title = raw.partition("__")
    if not title:
        bucket, title = "", raw
    title = " ".join(title.split())
    short = title if len(title) <= short_limit else title[: max(0, short_limit - 1)].rstrip() + "…"
    # The scraped titles carry a trailing " on <date>"; keep it out of the search
    # string so Indian Kanoon matches on the parties.
    search_terms = re.sub(r"\s+on\s+\d{1,2}\s+\w+,\s*\d{4}\s*$", "", title).strip() or title
    return {
        "case_id": raw,
        "bucket": bucket or None,
        "bucket_label": _BUCKET_TITLES.get(bucket, bucket.replace("_", " ").title() or None) if bucket else None,
        "title": title,
        "short_title": short,
        "search_url": INDIAN_KANOON_SEARCH + quote_plus(search_terms),
    }


def label_badge(label: Any) -> str:
    """``1`` -> ``+1``, ``-1`` -> ``-1`` for node captions."""
    text = str(label).strip()
    if text in {"", "nan", "None"}:
        return "?"
    if text.startswith("-"):
        return text
    return f"+{text}"


def _clean(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean(value) for key, value in row.items()}


# ---------------------------------------------------------------------------
# Nearest-neighbour + evidence sets
# ---------------------------------------------------------------------------


class CaseNeighborIndex:
    """Cosine nearest neighbours over the frozen HGT case embeddings.

    Brute force over the 71,813 x 64 embedding matrix takes ~12 ms per query, so
    no ANN structure or precomputed neighbour table is needed.
    """

    def __init__(self, pattern_dir: Path | str | None = None) -> None:
        self.pattern_dir = Path(pattern_dir or DEFAULT_PATTERN_DIR).resolve()
        self._loaded = False
        self._skew: dict[tuple[str, int], dict[str, Any]] | None = None

    # -- loading ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return (
            (self.pattern_dir / "hgt_case_embeddings.npz").exists()
            and (self.pattern_dir / "case_feature_matrix.npz").exists()
            and (self.pattern_dir / "case_feature_metadata.csv").exists()
            and (self.pattern_dir / "case_feature_case_index.csv").exists()
        )

    def load(self) -> None:
        if self._loaded:
            return
        if not self.available:
            raise FileNotFoundError(
                f"Pattern artifacts (hgt_case_embeddings.npz, case_feature_*) not found in {self.pattern_dir}"
            )
        payload = np.load(self.pattern_dir / "hgt_case_embeddings.npz", allow_pickle=True)
        embeddings = payload["embeddings"].astype(np.float32, copy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = embeddings / norms
        self.case_indices = payload["case_indices"].astype(np.int64)
        self.split = payload["split"].astype(str)
        self.target_label = payload["target_label"].astype(str)
        self.pred_label = payload["pred_label"].astype(str)
        self.confidence = payload["confidence"].astype(float)
        self.case_id = payload["case_id"].astype(str)
        self.file_name = payload["file_name"].astype(str)

        self.matrix = sp.load_npz(self.pattern_dir / "case_feature_matrix.npz").tocsr()
        features = pd.read_csv(self.pattern_dir / "case_feature_metadata.csv", low_memory=False)
        self.feature_type = features["feature_type"].astype(str).to_numpy()
        self.feature_name = features["feature_name"].astype(str).to_numpy()
        self.feature_idf = pd.to_numeric(features["idf"], errors="coerce").fillna(0.0).to_numpy()
        self.feature_count = (
            pd.to_numeric(features["corpus_case_count"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        )
        self.feature_gidx = (
            pd.to_numeric(features["evidence_global_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        )

        case_rows = pd.read_csv(self.pattern_dir / "case_feature_case_index.csv")
        matrix_case_indices = case_rows["case_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(matrix_case_indices, self.case_indices):
            # Fall back to an explicit map rather than assuming row == case_index;
            # counterfactual_neighborhoods.py hard-requires equality, we do not.
            self._matrix_row = {
                int(case): int(row)
                for case, row in zip(matrix_case_indices, case_rows["row_index"].to_numpy(dtype=np.int64))
            }
        else:
            self._matrix_row = None
        self._embedding_row = {int(case): row for row, case in enumerate(self.case_indices)}
        self._loaded = True

    # -- lookups ----------------------------------------------------------
    def has_case(self, case_index: int) -> bool:
        self.load()
        return int(case_index) in self._embedding_row

    def _row(self, case_index: int) -> int:
        self.load()
        row = self._embedding_row.get(int(case_index))
        if row is None:
            raise KeyError(f"case_index {case_index} is not in the embedding table")
        return row

    def _matrix_row_for(self, case_index: int) -> int | None:
        self.load()
        if self._matrix_row is None:
            return self._embedding_row.get(int(case_index))
        return self._matrix_row.get(int(case_index))

    def case_meta(self, case_index: int) -> dict[str, Any]:
        row = self._row(case_index)
        meta = case_display_name(self.case_id[row])
        meta.update(
            {
                "case_index": int(self.case_indices[row]),
                "file_name": str(self.file_name[row]),
                "split": str(self.split[row]),
                "target_label": str(self.target_label[row]),
                "pred_label": str(self.pred_label[row]),
                "confidence": float(self.confidence[row]),
                "label_badge": label_badge(self.target_label[row]),
                "pred_badge": label_badge(self.pred_label[row]),
                "correct": str(self.target_label[row]) == str(self.pred_label[row]),
            }
        )
        return meta

    def _pool_rows(self, pool: Pool) -> np.ndarray:
        self.load()
        if pool == "all":
            return np.arange(len(self.case_indices), dtype=np.int64)
        return np.where(self.split == pool)[0].astype(np.int64)

    def nearest(
        self,
        case_index: int,
        *,
        same_label: bool,
        k: int = 3,
        pool: Pool = "test",
        match: Match = "target",
    ) -> list[tuple[int, float]]:
        """Top-``k`` cosine neighbours with the same / opposite label.

        ``match="target"`` compares true labels (the published analysis);
        ``match="pred"`` compares what the model actually predicted.
        """
        row = self._row(case_index)
        labels = self.pred_label if match == "pred" else self.target_label
        label = labels[row]
        candidates = self._pool_rows(pool)
        if candidates.size == 0:
            return []
        mask = (labels[candidates] == label) if same_label else (labels[candidates] != label)
        candidates = candidates[mask]
        candidates = candidates[candidates != row]
        if candidates.size == 0:
            return []
        scores = self.embeddings[candidates] @ self.embeddings[row]
        take = min(int(k), candidates.size)
        order = np.argpartition(-scores, take - 1)[:take]
        order = order[np.argsort(-scores[order])]
        return [(int(self.case_indices[candidates[i]]), float(scores[i])) for i in order]

    # -- evidence sets ----------------------------------------------------
    def feature_indices(self, case_index: int) -> np.ndarray:
        row = self._matrix_row_for(case_index)
        if row is None or row >= self.matrix.shape[0]:
            return np.empty(0, dtype=np.int64)
        return self.matrix.indices[self.matrix.indptr[row] : self.matrix.indptr[row + 1]].astype(np.int64)

    def shared_type_counts(self, left: int, right: int) -> Counter:
        shared = np.intersect1d(self.feature_indices(left), self.feature_indices(right), assume_unique=False)
        if shared.size == 0:
            return Counter()
        return Counter(self.feature_type[shared].tolist())

    @staticmethod
    def type_count_label(counts: Counter, top_n: int = 3) -> str:
        """``Counter({'precedent': 41, 'provision': 19})`` -> ``"41P, 19V"``."""
        if not counts:
            return ""
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        return ", ".join(f"{n}{TYPE_INITIALS.get(t, t[:1].upper())}" for t, n in ordered)

    # -- ranking ----------------------------------------------------------
    def _skew_lookup(self) -> dict[tuple[str, int], dict[str, Any]]:
        if self._skew is not None:
            return self._skew
        path = self.pattern_dir / "evidence_label_skew.csv"
        if not path.exists():
            self._skew = {}
            return self._skew
        skew = pd.read_csv(
            path,
            usecols=[
                "evidence_type",
                "evidence_global_index",
                "skew_class",
                "skew_direction",
                "log_odds_vs_base",
                "g_test_q_value_bh",
            ],
            low_memory=False,
        )
        gidx = pd.to_numeric(skew["evidence_global_index"], errors="coerce")
        skew = skew[gidx.notna()]
        gidx = gidx[gidx.notna()].astype(np.int64)
        self._skew = {
            (str(t), int(g)): {
                "skew_class": c,
                "skew_direction": d,
                "log_odds_vs_base": o,
                "g_test_q_value_bh": q,
            }
            for t, g, c, d, o, q in zip(
                skew["evidence_type"].astype(str),
                gidx,
                skew["skew_class"],
                skew["skew_direction"],
                skew["log_odds_vs_base"],
                skew["g_test_q_value_bh"],
            )
        }
        return self._skew

    def feature_rows(self, indices: Iterable[int], *, with_skew: bool = True) -> list[dict[str, Any]]:
        """Feature-metadata rows for the given feature-matrix columns."""
        skew = self._skew_lookup() if with_skew else {}
        rows: list[dict[str, Any]] = []
        for feature_index in indices:
            i = int(feature_index)
            ftype = str(self.feature_type[i])
            gidx = int(self.feature_gidx[i])
            entry = skew.get((ftype, gidx), {})
            rows.append(
                clean_row(
                    {
                        "feature_index": i,
                        "feature_type": ftype,
                        "feature_name": str(self.feature_name[i]),
                        "evidence_type": ftype,
                        "evidence_name": str(self.feature_name[i]),
                        "evidence_global_index": gidx,
                        "idf": float(self.feature_idf[i]),
                        "corpus_case_count": int(self.feature_count[i]),
                        "skew_class": entry.get("skew_class", ""),
                        "skew_direction": entry.get("skew_direction", ""),
                        "log_odds_vs_base": entry.get("log_odds_vs_base"),
                        "g_test_q_value_bh": entry.get("g_test_q_value_bh"),
                    }
                )
            )
        return rows

    @staticmethod
    def _rank_key(row: dict[str, Any]) -> tuple[float, float]:
        """Label-discriminative skew first, then IDF.

        Mirrors ``counterfactual_neighborhoods.feature_rank_key`` so the live
        panels order evidence the same way the published CSVs do.
        """
        skew_score = 0.0
        if str(row.get("skew_class")) == "label_discriminative":
            try:
                skew_score = abs(float(row.get("log_odds_vs_base")))
            except (TypeError, ValueError):
                skew_score = 0.0
        return (skew_score, float(row.get("idf") or 0.0))

    def contrast_sets(
        self,
        query_case: int,
        other_case: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """``(query_only, shared, other_only)`` evidence rows in evidence-rank order.

        Not truncated: ``contrast_graph`` annotates these with counterfactual
        results first, so that it can optionally re-rank by counterfactual
        importance before cutting the columns down to the visible rows.
        """
        q = self.feature_indices(query_case)
        o = self.feature_indices(other_case)
        q_set, o_set = set(q.tolist()), set(o.tolist())

        query_only = self.feature_rows(sorted(q_set - o_set))
        other_only = self.feature_rows(sorted(o_set - q_set))
        shared = self.feature_rows(sorted(q_set & o_set))
        query_only.sort(key=self._rank_key, reverse=True)
        other_only.sort(key=self._rank_key, reverse=True)
        # Shared evidence is ranked by IDF (rarest first) as in
        # ExplanationStore._top_shared_features; skew is a contrastive signal and
        # says nothing useful about evidence both cases carry.
        shared.sort(key=lambda row: (float(row.get("idf") or 0.0), -(row.get("corpus_case_count") or 0)), reverse=True)
        for side, rows in (("query_only", query_only), ("shared", shared), ("other_only", other_only)):
            for row in rows:
                row["side"] = side
        return query_only, shared, other_only

    def pair_counts(self, query_case: int, other_case: int) -> dict[str, int]:
        q = set(self.feature_indices(query_case).tolist())
        o = set(self.feature_indices(other_case).tolist())
        return {
            "shared_feature_count": len(q & o),
            "query_only_feature_count": len(q - o),
            "other_only_feature_count": len(o - q),
        }

    def cosine(self, left: int, right: int) -> float:
        return float(self.embeddings[self._row(left)] @ self.embeddings[self._row(right)])

    def cosine_to_cases(self, case_index: int, others: Iterable[int]) -> dict[int, float]:
        """Cosine of ``case_index`` against each of ``others`` in one pass.

        Cases missing from the embedding table are simply left out of the
        result, so callers can treat a missing key as "no similarity known".
        """
        self.load()
        query = self.embeddings[self._row(case_index)]
        rows: list[int] = []
        cases: list[int] = []
        for other in others:
            row = self._embedding_row.get(int(other))
            if row is None:
                continue
            rows.append(int(row))
            cases.append(int(other))
        if not rows:
            return {}
        scores = self.embeddings[np.asarray(rows, dtype=np.int64)] @ query
        return {case: float(score) for case, score in zip(cases, scores)}

    def nearest_any(
        self,
        case_index: int,
        *,
        k: int = 1,
        pool: Pool = "all",
    ) -> list[tuple[int, float]]:
        """Top-``k`` cosine neighbours regardless of label.

        ``nearest()`` always splits the pool by same / opposite label, which
        makes it unable to answer "which case is simply the most similar one?".
        """
        row = self._row(case_index)
        candidates = self._pool_rows(pool)
        candidates = candidates[candidates != row]
        if candidates.size == 0:
            return []
        scores = self.embeddings[candidates] @ self.embeddings[row]
        take = min(int(k), candidates.size)
        order = np.argpartition(-scores, take - 1)[:take]
        order = order[np.argsort(-scores[order])]
        return [(int(self.case_indices[candidates[i]]), float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Counterfactual factors
# ---------------------------------------------------------------------------

_FACTOR_COLUMNS = [
    "case_index",
    "group_rank_abs",
    "cf_evidence_rank",
    "evidence_type",
    "evidence_global_index",
    "evidence_name",
    "delta_pred_proba",
    "abs_delta_pred_proba",
    "prediction_flipped",
]

#: Rank at or below which a factor is treated as a "main driving factor" and
#: highlighted on the slide.
TOP_FACTOR_RANK = 3


class CounterfactualFactorIndex:
    """Per-case counterfactual masking results, ready to stamp onto evidence boxes.

    Counterfactual explanations were only produced for the 14,363 **test** cases,
    so ``annotate`` marks rows for any other case as ``cf_available=False``
    rather than silently showing them as uninfluential.
    """

    def __init__(self, explanation_dir: Path | str | None = None) -> None:
        self.explanation_dir = Path(explanation_dir or DEFAULT_EXPLANATION_DIR).resolve()
        self._by_case: dict[int, list[dict[str, Any]]] | None = None
        self._covered: set[int] = set()

    @property
    def sidecar_path(self) -> Path:
        return self.explanation_dir / "case_counterfactual_factor_index.csv"

    @property
    def source_path(self) -> Path:
        return self.explanation_dir / "case_counterfactual_groups.csv"

    @property
    def available(self) -> bool:
        return self.sidecar_path.exists() or self.source_path.exists()

    def _frame(self) -> pd.DataFrame:
        if self.sidecar_path.exists():
            return pd.read_csv(self.sidecar_path, low_memory=False)
        if not self.source_path.exists():
            return pd.DataFrame(columns=_FACTOR_COLUMNS)
        # Fallback: the 390 MB table, read with usecols so we never pull the
        # whole thing into memory.  Run build_counterfactual_factor_index.py to
        # avoid this path.
        frame = pd.read_csv(
            self.source_path,
            usecols=[
                "case_index",
                "group_rank_abs",
                "group_kind",
                "evidence_type",
                "evidence_global_index",
                "evidence_name",
                "delta_pred_proba",
                "abs_delta_pred_proba",
                "prediction_flipped",
            ],
            low_memory=False,
        )
        frame = frame[frame["group_kind"].astype(str) != "relation_type"].drop(columns=["group_kind"])
        frame["group_rank_abs"] = pd.to_numeric(frame["group_rank_abs"], errors="coerce")
        frame = frame.sort_values(["case_index", "group_rank_abs"], na_position="last", kind="mergesort")
        frame["cf_evidence_rank"] = frame.groupby("case_index").cumcount() + 1
        return frame

    def load(self) -> None:
        if self._by_case is not None:
            return
        frame = self._frame()
        by_case: dict[int, list[dict[str, Any]]] = {}
        if not frame.empty:
            frame["case_index"] = pd.to_numeric(frame["case_index"], errors="coerce")
            frame = frame[frame["case_index"].notna()]
            for case_index, part in frame.groupby(frame["case_index"].astype(np.int64), sort=False):
                by_case[int(case_index)] = [clean_row(row) for row in part.to_dict("records")]
        self._by_case = by_case
        self._covered = set(by_case)

    def covers(self, case_index: int) -> bool:
        self.load()
        return int(case_index) in self._covered

    def factors(self, case_index: int) -> list[dict[str, Any]]:
        self.load()
        return self._by_case.get(int(case_index), [])

    def _keyed(self, case_index: int) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, str], dict]]:
        by_gidx: dict[tuple[str, int], dict] = {}
        by_name: dict[tuple[str, str], dict] = {}
        for row in self.factors(case_index):
            etype = str(row.get("evidence_type") or "").lower()
            gidx = row.get("evidence_global_index")
            # evidence_global_index is unique per *type*, not globally
            # (judge:13 and court:13 are different entities), so the type must
            # be part of the key.
            if gidx is not None:
                by_gidx.setdefault((etype, int(gidx)), row)
            name = str(row.get("evidence_name") or "").strip().lower()
            if name:
                by_name.setdefault((etype, name), row)
        return by_gidx, by_name

    @staticmethod
    def _stamp(target: dict[str, Any], factor: dict[str, Any] | None, available: bool) -> dict[str, Any]:
        if not available:
            target.update(
                {
                    "cf_available": False,
                    "cf_rank": None,
                    "cf_evidence_rank": None,
                    "cf_delta": None,
                    "cf_abs_delta": None,
                    "cf_flips": False,
                    "cf_direction": None,
                    "cf_top": False,
                }
            )
            return target
        if factor is None:
            target.update(
                {
                    "cf_available": True,
                    "cf_rank": None,
                    "cf_evidence_rank": None,
                    "cf_delta": None,
                    "cf_abs_delta": None,
                    "cf_flips": False,
                    "cf_direction": None,
                    "cf_top": False,
                }
            )
            return target
        delta = factor.get("delta_pred_proba")
        rank = factor.get("cf_evidence_rank")
        target.update(
            {
                "cf_available": True,
                "cf_rank": _clean(factor.get("group_rank_abs")),
                "cf_evidence_rank": _clean(rank),
                "cf_delta": _clean(delta),
                "cf_abs_delta": _clean(factor.get("abs_delta_pred_proba")),
                "cf_flips": bool(factor.get("prediction_flipped")),
                # delta_pred_proba = baseline - masked, so a positive delta means
                # removing the evidence lowered confidence: the evidence supports
                # the decision.  Negative means it argued against it.
                "cf_direction": None if delta is None else ("supports" if float(delta) > 0 else "opposes"),
                "cf_top": bool(rank is not None and int(rank) <= TOP_FACTOR_RANK),
            }
        )
        return target

    def annotate(self, case_index: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stamp ``cf_*`` fields onto evidence rows belonging to ``case_index``."""
        available = self.covers(case_index)
        by_gidx, by_name = self._keyed(case_index) if available else ({}, {})
        for row in rows:
            etype = str(row.get("feature_type") or row.get("evidence_type") or "").lower()
            gidx = row.get("evidence_global_index")
            factor = None
            if gidx is not None:
                factor = by_gidx.get((etype, int(gidx)))
            if factor is None:
                name = str(row.get("feature_name") or row.get("evidence_name") or "").strip().lower()
                factor = by_name.get((etype, name)) if name else None
            self._stamp(row, factor, available)
        return rows

    def top_factors(self, case_index: int, limit: int = 8) -> list[dict[str, Any]]:
        """The case's strongest counterfactual factors, including evidence types
        (arguments, petitioner, respondent, lawyer) that never get a box in the
        contrast diagrams."""
        out: list[dict[str, Any]] = []
        for row in self.factors(case_index)[:limit]:
            delta = row.get("delta_pred_proba")
            rank = row.get("cf_evidence_rank")
            out.append(
                {
                    "cf_evidence_rank": _clean(rank),
                    "cf_rank": _clean(row.get("group_rank_abs")),
                    "evidence_type": row.get("evidence_type"),
                    "evidence_name": row.get("evidence_name"),
                    "cf_delta": _clean(delta),
                    "cf_abs_delta": _clean(row.get("abs_delta_pred_proba")),
                    "cf_flips": bool(row.get("prediction_flipped")),
                    "cf_direction": None if delta is None else ("supports" if float(delta) > 0 else "opposes"),
                    "cf_top": bool(rank is not None and int(rank) <= TOP_FACTOR_RANK),
                    "has_box": str(row.get("evidence_type") or "") in TYPE_INITIALS,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Assembled graph payloads
# ---------------------------------------------------------------------------


def ego_graph(
    neighbors: CaseNeighborIndex,
    factors: CounterfactualFactorIndex | None,
    case_index: int,
    *,
    k_same: int = 3,
    k_opposite: int = 3,
    pool: Pool = "test",
    match: Match = "target",
) -> dict[str, Any]:
    """The 'case among its most similar cases' network behind the first slide."""
    if not neighbors.available:
        return {"available": False, "reason": "Embedding / feature artifacts not found."}
    if not neighbors.has_case(case_index):
        return {"available": False, "reason": f"case_index {case_index} is not in the embedding table."}

    center = neighbors.case_meta(case_index)
    if factors is not None:
        center["top_factors"] = factors.top_factors(case_index, limit=5)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for side, same_label, k in (("same", True, k_same), ("opposite", False, k_opposite)):
        for neighbor_index, cosine in neighbors.nearest(
            case_index, same_label=same_label, k=k, pool=pool, match=match
        ):
            meta = neighbors.case_meta(neighbor_index)
            meta["side"] = side
            if factors is not None:
                meta["cf_available"] = factors.covers(neighbor_index)
                meta["top_factors"] = factors.top_factors(neighbor_index, limit=3)
            nodes.append(meta)
            counts = neighbors.shared_type_counts(case_index, neighbor_index)
            edges.append(
                {
                    "source": int(case_index),
                    "target": int(neighbor_index),
                    "side": side,
                    "cosine": cosine,
                    "shared_counts": {str(t): int(n) for t, n in counts.items()},
                    "shared_label": neighbors.type_count_label(counts),
                    "shared_total": int(sum(counts.values())),
                }
            )

    return {
        "available": True,
        "pool": pool,
        "match": match,
        "center": center,
        "nodes": nodes,
        "edges": edges,
        "k_same": int(k_same),
        "k_opposite": int(k_opposite),
    }


def _order_column(rows: list[dict[str, Any]], order: Order, limit: int) -> list[dict[str, Any]]:
    """Cut a contrast column down to the visible rows and number them."""
    if order == "counterfactual":
        # Rows the masking never scored sink to the bottom rather than sorting as
        # a zero effect, which would read as "measured and found unimportant".
        rows = sorted(
            rows,
            key=lambda row: (
                row.get("cf_abs_delta") is not None,
                float(row.get("cf_abs_delta") or 0.0),
                float(row.get("idf") or 0.0),
            ),
            reverse=True,
        )
    out = rows[:limit]
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def contrast_graph(
    neighbors: CaseNeighborIndex,
    factors: CounterfactualFactorIndex | None,
    case_index: int,
    *,
    side: Literal["same", "opposite"] = "opposite",
    pool: Pool = "test",
    other_case: int | None = None,
    limit: int = 8,
    order: Order = "evidence",
    match: Match = "target",
) -> dict[str, Any]:
    """Query case vs its nearest same-label or opposite-label case.

    ``other_case`` pins the comparison case (used to keep the published paper
    figure on its original pair); otherwise the neighbour is resolved live from
    ``pool``.

    ``order`` picks which evidence reaches the visible columns: ``"evidence"``
    keeps the published IDF + label-skew ranking, ``"counterfactual"`` promotes
    the evidence the masking actually found to drive the decision.
    """
    if not neighbors.available:
        return {"available": False, "reason": "Embedding / feature artifacts not found."}
    if not neighbors.has_case(case_index):
        return {"available": False, "reason": f"case_index {case_index} is not in the embedding table."}

    if other_case is None:
        matches = neighbors.nearest(
            case_index, same_label=(side == "same"), k=1, pool=pool, match=match
        )
        if not matches:
            field = "predicted" if match == "pred" else "true"
            return {
                "available": False,
                "reason": f"No case with the {side} {field} label near case {case_index} in the {pool} pool.",
            }
        other_case, cosine = matches[0]
    else:
        if not neighbors.has_case(other_case):
            return {"available": False, "reason": f"case_index {other_case} is not in the embedding table."}
        cosine = neighbors.cosine(case_index, other_case)

    query_only, shared, other_only = neighbors.contrast_sets(case_index, other_case)
    if factors is not None:
        factors.annotate(case_index, query_only)
        factors.annotate(case_index, shared)
        factors.annotate(other_case, other_only)
    query_only = _order_column(query_only, order, limit)
    shared = _order_column(shared, order, limit)
    other_only = _order_column(other_only, order, limit)

    query = neighbors.case_meta(case_index)
    other = neighbors.case_meta(other_case)
    if factors is not None:
        query["cf_available"] = factors.covers(case_index)
        query["top_factors"] = factors.top_factors(case_index, limit=6)
        other["cf_available"] = factors.covers(other_case)
        other["top_factors"] = factors.top_factors(other_case, limit=6)

    counts = neighbors.pair_counts(case_index, other_case)
    return {
        "available": True,
        "side": side,
        "pool": pool,
        "order": order,
        "match": match,
        "query": query,
        "other": other,
        "cosine_similarity": cosine,
        "query_only_features": query_only,
        "shared_features": shared,
        "other_only_features": other_only,
        **counts,
        # Kept for the legacy /api/opposite_graph payload shape.
        "opposite_only_features": other_only,
    }


def showcase_ranking(
    neighbors: CaseNeighborIndex,
    factors: CounterfactualFactorIndex | None,
    *,
    limit: int = 25,
    pool: Pool = "test",
    sample: int | None = 2500,
    seed: int = 0,
    match: Match = "target",
) -> list[dict[str, Any]]:
    """Rank cases by how legible their presentation figures would be.

    Scores a case on: how much evidence its nearest neighbours actually share
    (empty edge labels make a dull slide), how many distinct evidence types are
    involved, how close the neighbours are, and whether it has counterfactual
    coverage for the badges.
    """
    if not neighbors.available:
        return []
    neighbors.load()
    candidates = neighbors._pool_rows(pool)
    if candidates.size == 0:
        return []
    if sample is not None and candidates.size > sample:
        rng = np.random.default_rng(seed)
        candidates = rng.choice(candidates, size=sample, replace=False)

    scored: list[dict[str, Any]] = []
    for row in candidates:
        case_index = int(neighbors.case_indices[row])
        if factors is not None and not factors.covers(case_index):
            continue
        pairs = neighbors.nearest(
            case_index, same_label=True, k=3, pool=pool, match=match
        ) + neighbors.nearest(case_index, same_label=False, k=3, pool=pool, match=match)
        if len(pairs) < 4:
            continue
        totals, types, cosines = [], set(), []
        for neighbor_index, cosine in pairs:
            counts = neighbors.shared_type_counts(case_index, neighbor_index)
            totals.append(sum(counts.values()))
            types.update(counts)
            cosines.append(cosine)
        rich_edges = sum(1 for total in totals if total >= 3)
        box_count = int(neighbors.feature_indices(case_index).size)
        score = (
            rich_edges * 3.0
            + len(types) * 1.5
            + min(sum(totals), 60) * 0.05
            + float(np.mean(cosines)) * 2.0
            + min(box_count, 40) * 0.05
        )
        meta = neighbors.case_meta(case_index)
        meta.update(
            {
                "score": round(float(score), 3),
                "rich_edges": int(rich_edges),
                "shared_types": int(len(types)),
                "shared_total": int(sum(totals)),
                "mean_cosine": round(float(np.mean(cosines)), 4),
                "evidence_boxes": box_count,
            }
        )
        scored.append(meta)

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:limit]
