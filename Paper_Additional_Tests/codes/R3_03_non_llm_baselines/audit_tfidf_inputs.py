#!/usr/bin/env python3
"""Audit TF-IDF provenance, HGT alignment, and final leakage sanitization."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from text_sanitization import audit_documents, sanitize_document

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLEANED_CASES = (
    REPO
    / "section_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset"
    / "processed/cleaned_cases"
)
REFERENCE_KFOLD = (
    REPO
    / "section_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models"
    / "ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold"
)
FEATURES = HERE / "outputs/features"
OUTPUT = HERE / "outputs/tfidf_input_audit.json"

HGT_CASE_SECTIONS = ("preamble", "facts", "arguments")
ALLOWED_KEPT_ROLES = {
    "PREAMBLE",
    "FAC",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "STA",
}
FORBIDDEN_DECISION_ROLES = {"ANALYSIS", "ISSUE", "NONE", "RATIO", "RLC", "RPC"}


def normalized(text: str) -> str:
    return " ".join(str(text or "").split())


def main() -> None:
    index = json.loads((FEATURES / "case_index.json").read_text())
    case_ids = index["case_ids"]
    labels = index["labels"]
    paths = sorted(CLEANED_CASES.glob("*.json"))
    with gzip.open(FEATURES / "text.jsonl.gz", "rt", encoding="utf-8") as handle:
        cached_documents = [json.loads(line) for line in handle]

    if not (len(paths) == len(case_ids) == len(labels) == len(cached_documents)):
        raise AssertionError(
            f"length mismatch: paths={len(paths)}, ids={len(case_ids)}, "
            f"labels={len(labels)}, texts={len(cached_documents)}"
        )

    kept_roles: Counter[str] = Counter()
    dropped_roles: Counter[str] = Counter()
    source_hash = hashlib.sha256()
    sanitized_hash = hashlib.sha256()
    sanitized_documents: list[str] = []

    for row, (path, case_id, label, cached) in enumerate(
        zip(paths, case_ids, labels, cached_documents, strict=True)
    ):
        case = json.loads(path.read_text())
        if path.stem != case_id or str(case["case_id"]) != case_id:
            raise AssertionError(f"row {row}: case ID/order mismatch")
        if str(case["raw_label"]) != str(label):
            raise AssertionError(f"row {row}: label mismatch for {case_id}")

        texts = case.get("texts") or {}
        builder_document = "\n".join(str(texts.get(section) or "") for section in HGT_CASE_SECTIONS)
        if cached != builder_document:
            raise AssertionError(f"row {row}: TF-IDF cache differs from cleaned HGT sections")

        # case_star_builder.py uses the same sections and omits empty strings;
        # normalize whitespace to ignore only its newline separator choice.
        hgt_case_node_document = "\n\n".join(
            str(texts.get(section) or "") for section in HGT_CASE_SECTIONS if texts.get(section)
        )
        if normalized(cached) != normalized(hgt_case_node_document):
            raise AssertionError(f"row {row}: TF-IDF content differs from HGT case-node content")

        audit = case.get("leakage_audit") or {}
        row_kept = Counter(audit.get("kept_sentence_role_counts") or {})
        row_dropped = Counter(audit.get("dropped_sentence_role_counts") or {})
        unexpected = set(row_kept) - ALLOWED_KEPT_ROLES
        if unexpected:
            raise AssertionError(f"row {row}: unexpected retained roles {sorted(unexpected)}")
        kept_roles.update(row_kept)
        dropped_roles.update(row_dropped)

        clean = sanitize_document(cached)
        sanitized_documents.append(clean)
        source_hash.update(cached.encode("utf-8"))
        source_hash.update(b"\0")
        sanitized_hash.update(clean.encode("utf-8"))
        sanitized_hash.update(b"\0")

    sanitization = audit_documents(cached_documents, sanitized_documents)

    # The exact HGT prediction artifacts are the split authority. Check every
    # fold against the same ordered corpus and labels.
    position = {case_id: row for row, case_id in enumerate(case_ids)}
    fold_audit = {}
    test_sets = []
    for fold in range(5):
        frame = pd.read_csv(REFERENCE_KFOLD / f"fold_{fold:02d}/predictions.csv")
        if len(frame) != len(case_ids) or set(frame["case_id"]) != set(case_ids):
            raise AssertionError(f"fold {fold}: case membership differs from TF-IDF/HGT corpus")
        rows = frame["case_id"].map(position)
        recorded_labels = frame["raw_label"].astype(str).tolist()
        expected_labels = [str(labels[int(row)]) for row in rows]
        if recorded_labels != expected_labels:
            raise AssertionError(f"fold {fold}: labels differ from HGT predictions.csv")
        split_counts = frame["split"].value_counts().to_dict()
        tests = set(frame.loc[frame["split"] == "test", "case_id"])
        test_sets.append(tests)
        fold_audit[str(fold)] = {key: int(value) for key, value in split_counts.items()}

    overlap = sum(
        len(test_sets[left] & test_sets[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )
    test_union = set().union(*test_sets)
    if overlap or len(test_union) != len(case_ids):
        raise AssertionError(
            f"HGT test folds do not partition corpus: overlap={overlap}, union={len(test_union)}"
        )

    result = {
        "status": "PASS",
        "n_cases": len(case_ids),
        "source": str(CLEANED_CASES),
        "hgt_case_sections": list(HGT_CASE_SECTIONS),
        "source_documents_sha256": source_hash.hexdigest(),
        "sanitized_documents_sha256": sanitized_hash.hexdigest(),
        "source_equals_hgt_case_node_content": True,
        "case_order_and_labels_equal_hgt_predictions": True,
        "allowed_retained_roles": sorted(ALLOWED_KEPT_ROLES),
        "forbidden_decision_roles": sorted(FORBIDDEN_DECISION_ROLES),
        "kept_role_counts": dict(sorted(kept_roles.items())),
        "dropped_role_counts": dict(sorted(dropped_roles.items())),
        "unexpected_retained_role_count": 0,
        "sanitization": sanitization,
        "fold_split_counts": fold_audit,
        "test_fold_overlap": overlap,
        "test_fold_union": len(test_union),
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
