#!/usr/bin/env python3
"""Compare decision labels/text between model output CSV and ground-truth CSV.

Default inputs:
- outputs/cases.csv
- Ground_TruCapstone-Thesis-/scriptsth/openai_extracted (1).csv

Produces:
- outputs/decision_compare_all.csv
- outputs/decision_compare_right.csv
- outputs/decision_compare_wrong.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return NORM_RE.sub(" ", text).strip()


def normalize_label_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return NORM_RE.sub("_", text).strip("_")


def tokenize(value: str) -> list[str]:
    return [tok for tok in value.split() if tok]


def token_sort(value: str) -> str:
    return " ".join(sorted(tokenize(value)))


def token_set(value: str) -> set[str]:
    return set(tokenize(value))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def canonical_decision(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    label = normalize_label_token(raw)
    text = normalize_text(raw)

    exact_aliases = {
        "for_the_appellant": "for_appellant",
        "in_favour_of_appellant": "for_appellant",
        "in_favor_of_appellant": "for_appellant",
        "in_favour_of_petitioner": "for_appellant",
        "in_favor_of_petitioner": "for_appellant",
        "for_the_petitioner": "for_appellant",
        "for_petitioner": "for_appellant",
        "for_the_applicant": "for_appellant",
        "for_applicant": "for_appellant",
        "in_favour_of_applicant": "for_appellant",
        "in_favor_of_applicant": "for_appellant",
        "in_favour_of_claimant": "for_appellant",
        "in_favor_of_claimant": "for_appellant",
        "against_the_appellant": "against_appellant",
        "against_the_petitioner": "against_appellant",
        "against_petitioner": "against_appellant",
        "against_the_applicant": "against_appellant",
        "against_applicant": "against_appellant",
        "in_favour_of_respondent": "against_appellant",
        "in_favor_of_respondent": "against_appellant",
        "in_favour_of_state": "against_appellant",
        "in_favor_of_state": "against_appellant",
        "allowed": "for_appellant",
        "allow": "for_appellant",
        "granted": "for_appellant",
        "grant": "for_appellant",
        "rejected": "against_appellant",
        "reject": "against_appellant",
        "refused": "against_appellant",
        "refuse": "against_appellant",
        "denied": "against_appellant",
        "deny": "against_appellant",
        "adjourned": "delayed",
        "deferred": "delayed",
        "postponed": "delayed",
        "disposed_of": "disposed",
        "disposed_off": "disposed",
        "quashed_proceedings": "quashed",
        "partly_allow": "partly_allowed",
        "partly_granted": "partly_allowed",
        "stay_granted": "stayed",
        "interim_stay": "stayed",
        "granted_bail_with_conditions": "for_appellant",
        "suspended_fine_pending_appeal": "suspended_sentence",
    }
    if label in exact_aliases:
        return exact_aliases[label]

    canonical = {
        "for_appellant",
        "against_appellant",
        "dismissed",
        "delayed",
        "withdrawn",
        "disposed",
        "quashed",
        "remanded",
        "partly_allowed",
        "stayed",
        "suspended_sentence",
        "modified_sentence",
        "closed",
        "interim_order",
        "issued_notice",
        "procedural_order",
        "granted_interim_relief",
        "other",
    }
    if label in canonical:
        return label

    if any(phrase in text for phrase in ("partly allowed", "allowed in part", "partly granted")):
        return "partly_allowed"

    if any(
        phrase in text
        for phrase in (
            "in favour of appellant",
            "in favor of appellant",
            "in favour of petitioner",
            "in favor of petitioner",
            "in favour of the petitioner",
            "in favor of the petitioner",
            "in favour of the appellant",
            "in favor of the appellant",
            "in favour of applicant",
            "in favor of applicant",
            "for the appellant",
            "for the petitioner",
            "petition allowed",
            "appeal allowed",
            "bail granted",
            "granted bail",
            "granted bail with conditions",
            "application allowed",
            "writ allowed",
            "relief granted",
            "quashing the proceedings",
            "proceedings quashed",
            "stay of proceedings",
            "stay further proceedings",
            "suspended sentence",
            "suspended fine sentence",
            "liberty in favour of the petitioner",
            "liberty in favour of petitioner",
            "granting liberty to the petitioner",
            "granting liberty to petitioner",
        )
    ):
        return "for_appellant"

    if any(
        phrase in text
        for phrase in (
            "in favour of respondent",
            "in favor of respondent",
            "in favour of state",
            "in favor of state",
            "against the appellant",
            "against the petitioner",
            "petition rejected",
            "appeal rejected",
            "bail rejected",
            "application rejected",
            "relief denied",
        )
    ):
        return "against_appellant"

    if any(tok in text for tok in ("dismissed", "dismissal")):
        return "dismissed"
    if any(tok in text for tok in ("disposed", "disposed of", "dispose of", "disposed off", "disposal")):
        return "disposed"
    if any(tok in text for tok in ("quashed", "set aside")):
        return "quashed"
    if any(tok in text for tok in ("withdrawn", "withdraw")):
        return "withdrawn"
    if any(tok in text for tok in ("adjourned", "deferred", "postponed", "stand over")):
        return "delayed"
    if any(tok in text for tok in ("remanded", "remand")):
        return "remanded"
    if any(tok in text for tok in ("suspended sentence", "suspended fine sentence")):
        return "suspended_sentence"
    if any(tok in text for tok in ("conviction upheld", "sentence modified", "modified sentence")):
        return "modified_sentence"
    if any(tok in text for tok in ("stayed", "stay granted", "interim stay", "stay of proceedings")):
        return "stayed"

    return label


def infer_winner_side(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    if any(token in text for token in ("appellant", "petitioner", "applicant", "claimant", "assessee", "accused")):
        return "for_appellant"
    if any(token in text for token in ("respondent", "state", "defendant", "prosecution", "union", "complainant")):
        return "against_appellant"
    return ""


def decision_family(label: str) -> str:
    if label in {
        "for_appellant",
        "quashed",
        "partly_allowed",
        "stayed",
        "granted_interim_relief",
        "interim_order",
        "suspended_sentence",
    }:
        return "favourable"
    if label in {"against_appellant", "dismissed"}:
        return "against"
    if label in {"modified_sentence"}:
        return "mixed"
    if label in {"disposed", "closed", "delayed", "remanded", "withdrawn", "issued_notice", "procedural_order"}:
        return "procedural"
    return ""


def infer_winner_side_from_parties(row: dict[str, str]) -> str:
    winner_text = normalize_text(row.get("outcome_winner"))
    if not winner_text:
        return ""

    petitioner_text = normalize_text(row.get("petitioner_applicant"))
    respondent_text = normalize_text(row.get("respondent_state_defendant"))

    if petitioner_text and (winner_text in petitioner_text or petitioner_text in winner_text):
        return "for_appellant"
    if respondent_text and (winner_text in respondent_text or respondent_text in winner_text):
        return "against_appellant"

    winner_tokens = token_set(winner_text)
    if not winner_tokens:
        return ""

    petitioner_overlap = len(winner_tokens & token_set(petitioner_text))
    respondent_overlap = len(winner_tokens & token_set(respondent_text))

    if petitioner_overlap > respondent_overlap and petitioner_overlap >= 2:
        return "for_appellant"
    if respondent_overlap > petitioner_overlap and respondent_overlap >= 2:
        return "against_appellant"

    return ""


def resolve_pred_label(row: dict[str, str]) -> tuple[str, str]:
    decision_label = canonical_decision(row.get("decision"))
    if decision_label in {"disposed", "closed", "procedural_order", "issued_notice", "other", ""}:
        winner_side = infer_winner_side(row.get("outcome_winner"))
        if not winner_side:
            winner_side = infer_winner_side_from_parties(row)
        if winner_side:
            return winner_side, "outcome_winner"

        outcome_label = canonical_decision(row.get("outcome_label"))
        if outcome_label and decision_family(outcome_label) in {"favourable", "against", "mixed"}:
            return outcome_label, "outcome_label"

    return decision_label, "decision"


def resolve_truth_label(row: dict[str, str]) -> tuple[str, str]:
    decision_label = canonical_decision(row.get("decision"))
    return decision_label, "decision"


def compare_decisions(
    pred: str,
    truth: str,
    pred_label: str,
    truth_label: str,
    fuzzy_threshold: float,
    jaccard_threshold: float,
) -> tuple[str, str, float, float]:
    pred_norm = normalize_text(pred)
    truth_norm = normalize_text(truth)

    if not pred_norm or not truth_norm:
        return "unscored", "missing_text", 0.0, 0.0

    if pred_label and truth_label and pred_label == truth_label:
        return "right", "canonical_label_match", 1.0, 1.0

    pred_family = decision_family(pred_label)
    truth_family = decision_family(truth_label)
    if pred_family and truth_family and pred_family == truth_family and pred_family in {"favourable", "against", "mixed"}:
        return "right", "semantic_polarity_match", 1.0, 1.0

    label_ratio = seq_ratio(pred_label.replace("_", " "), truth_label.replace("_", " "))
    if pred_label and truth_label and label_ratio >= 0.9:
        return "right", "canonical_fuzzy_match", label_ratio, label_ratio

    if pred_norm == truth_norm:
        return "right", "normalized_exact_match", 1.0, 1.0

    sort_ratio = seq_ratio(token_sort(pred_norm), token_sort(truth_norm))
    pred_set = token_set(pred_norm)
    truth_set = token_set(truth_norm)
    jac = jaccard(pred_set, truth_set)

    if sort_ratio >= fuzzy_threshold:
        return "right", "fuzzy_ratio_match", sort_ratio, jac

    if jac >= jaccard_threshold:
        return "right", "token_overlap_match", sort_ratio, jac

    return "wrong", "no_match", sort_ratio, jac


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare decision column across two CSV files")
    parser.add_argument(
        "--pred-csv",
        default="outputs/cases.csv",
        help="Path to prediction/output CSV (default: outputs/cases.csv)",
    )
    parser.add_argument(
        "--truth-csv",
        default="Ground_Truth/openai_extracted (1).csv",
        help="Path to ground-truth CSV (default: Ground_Truth/openai_extracted (1).csv)",
    )
    parser.add_argument(
        "--out-all",
        default="outputs/decision_compare_all.csv",
        help="Path for full comparison output CSV",
    )
    parser.add_argument(
        "--out-right",
        default="outputs/decision_compare_right.csv",
        help="Path for right-only rows",
    )
    parser.add_argument(
        "--out-wrong",
        default="outputs/decision_compare_wrong.csv",
        help="Path for wrong-only rows",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.84,
        help="Sequence similarity threshold for fuzzy text match (0-1)",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.75,
        help="Token overlap threshold for match (0-1)",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred_csv)
    truth_path = Path(args.truth_csv)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {truth_path}")

    pred_rows = load_csv(pred_path)
    truth_rows = load_csv(truth_path)

    truth_by_file = {row.get("file_name", "").strip(): row for row in truth_rows if row.get("file_name")}

    report_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []

    for pred in pred_rows:
        file_name = (pred.get("file_name") or "").strip()
        pred_decision = (pred.get("decision") or "").strip()
        pred_label, pred_label_source = resolve_pred_label(pred)
        pred_family = decision_family(pred_label)
        gt_row = truth_by_file.get(file_name)

        if gt_row is None:
            row = {
                "file_name": file_name,
                "pred_decision": pred_decision,
                "ground_truth_decision": "",
                "pred_canonical": pred_label,
                "ground_truth_canonical": "",
                "pred_family": pred_family,
                "ground_truth_family": "",
                "pred_label_source": pred_label_source,
                "ground_truth_label_source": "",
                "result": "unscored",
                "reason": "missing_ground_truth_row",
                "fuzzy_score": "",
                "token_overlap": "",
            }
            report_rows.append(row)
            continue

        gt_decision = (gt_row.get("decision") or "").strip()
        gt_label, gt_label_source = resolve_truth_label(gt_row)
        gt_family = decision_family(gt_label)
        result, reason, fuzzy_score, token_overlap = compare_decisions(
            pred=pred_decision,
            truth=gt_decision,
            pred_label=pred_label,
            truth_label=gt_label,
            fuzzy_threshold=args.fuzzy_threshold,
            jaccard_threshold=args.jaccard_threshold,
        )

        row = {
            "file_name": file_name,
            "pred_decision": pred_decision,
            "ground_truth_decision": gt_decision,
            "pred_canonical": pred_label,
            "ground_truth_canonical": gt_label,
            "pred_family": pred_family,
            "ground_truth_family": gt_family,
            "pred_label_source": pred_label_source,
            "ground_truth_label_source": gt_label_source,
            "result": result,
            "reason": reason,
            "fuzzy_score": f"{fuzzy_score:.4f}" if gt_decision else "",
            "token_overlap": f"{token_overlap:.4f}" if gt_decision else "",
        }
        report_rows.append(row)
        if result == "right":
            right_rows.append(row)
        elif result == "wrong":
            wrong_rows.append(row)

    fieldnames = [
        "file_name",
        "pred_decision",
        "ground_truth_decision",
        "pred_canonical",
        "ground_truth_canonical",
        "pred_family",
        "ground_truth_family",
        "pred_label_source",
        "ground_truth_label_source",
        "result",
        "reason",
        "fuzzy_score",
        "token_overlap",
    ]

    out_all = Path(args.out_all)
    out_right = Path(args.out_right)
    out_wrong = Path(args.out_wrong)

    write_csv(out_all, report_rows, fieldnames)
    write_csv(out_right, right_rows, fieldnames)
    write_csv(out_wrong, wrong_rows, fieldnames)

    total = len(report_rows)
    right = len(right_rows)
    wrong = len(wrong_rows)
    unscored = total - right - wrong
    scored = right + wrong
    acc = (right / scored) * 100.0 if scored else 0.0

    print(f"Pred rows: {len(pred_rows)}")
    print(f"Truth rows: {len(truth_rows)}")
    print(f"Compared (scored): {scored}")
    print(f"Right: {right}")
    print(f"Wrong: {wrong}")
    print(f"Unscored: {unscored}")
    print(f"Accuracy over scored rows: {acc:.2f}%")
    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_right}")
    print(f"Wrote: {out_wrong}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
