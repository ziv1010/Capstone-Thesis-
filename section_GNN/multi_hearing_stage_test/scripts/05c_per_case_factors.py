#!/usr/bin/env python3
"""
Step 05c — For each multi-hearing case whose prediction transitioned, build a
per-case "what changed" report combining:

  1. The new sentences at the later stage, grouped by rhetorical role
     (the judge's actual ruling text + supporting argumentation that wasn't
     there at the earlier stage).
  2. The new entities at the later stage (statutes, provisions, precedents,
     judges, lawyers) ranked by a *discriminative score*: how much more
     often that specific entity appears in cases of THIS transition pattern
     than in the matched-control transition pattern.
        - For LOSE -> WIN: contrast against LOSE -> LOSE
        - For WIN -> LOSE: contrast against WIN -> WIN
        - For LOSE -> LOSE: contrast against LOSE -> WIN (i.e. what failed
          to flip the case)
        - For WIN -> WIN: contrast against WIN -> LOSE
  3. A short list of "anchor sentences" — the RPC / RATIO / STA / PRE_RELIED
     sentences from the later stage that mention any top-decisive entity,
     because those are typically where the judge cites the legal hook
     responsible for the outcome.

Inputs (already on disk, none modified):
  - outputs/analysis/per_case_diffs/<case>.json  (added/removed entity diff)
  - outputs/analysis/transition_aggregates.json  (population priors)
  - data/input_jsons/STAGE<N>__<case>.json       (raw stage docs)

Outputs (new):
  - outputs/analysis/per_case_factors/<case>.json   (one JSON per transition case)
  - outputs/analysis/per_case_factors/_index.csv    (compact index across all cases)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

EXP_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/multi_hearing_stage_test")
DIFFS_DIR = EXP_ROOT / "outputs/analysis/per_case_diffs"
AGG_PATH = EXP_ROOT / "outputs/analysis/transition_aggregates.json"
INPUT_JSON_DIR = EXP_ROOT / "data/input_jsons"
OUT_DIR = EXP_ROOT / "outputs/analysis/per_case_factors"

LEGAL_LABELS = ("STATUTE", "PROVISION", "PRECEDENT")
PARTY_LABELS = ("JUDGE", "LAWYER", "COURT")
DECISION_ROLES = ("RPC", "RATIO", "STA", "PRE_RELIED", "PRE_NOT_RELIED")

# Pair each transition with its closest "control" so the discriminative score
# answers: "what makes this transition different from the case where the
# starting prediction was the same but the ending prediction wasn't?"
CONTRAST = {
    "LOSE -> WIN":  "LOSE -> LOSE",
    "LOSE -> LOSE": "LOSE -> WIN",
    "WIN -> LOSE":  "WIN -> WIN",
    "WIN -> WIN":   "WIN -> LOSE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diffs-dir", default=str(DIFFS_DIR))
    p.add_argument("--aggregates", default=str(AGG_PATH))
    p.add_argument("--input-jsons", default=str(INPUT_JSON_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--top-k-factors", type=int, default=8)
    p.add_argument("--max-anchor-sentences", type=int, default=12)
    p.add_argument("--changed-only", action="store_true",
                   help="Only emit reports for cases whose prediction changed across stages.")
    return p.parse_args()


def build_entity_priors(aggregates: dict) -> dict:
    """For each (transition, label, entity) compute a smoothed differential
    frequency: how much more often this entity appears in the target
    transition than in its contrast. Higher = more discriminative for the
    target transition.

    Score = (n_target / N_target) - (n_contrast / N_contrast),
    smoothed by adding 0.5 to numerators and 1 to denominators."""
    by_t = aggregates.get("by_transition", {})
    priors: dict[str, dict[str, dict[str, float]]] = {}
    for transition, contrast_t in CONTRAST.items():
        if transition not in by_t or contrast_t not in by_t:
            continue
        n_t = max(1, by_t[transition]["n_cases"])
        n_c = max(1, by_t[contrast_t]["n_cases"])

        target_top = by_t[transition].get("top_added_entities", {})
        contrast_top = by_t[contrast_t].get("top_added_entities", {})

        # Build a quick lookup of contrast counts.
        contrast_counts: dict[tuple[str, str], int] = {}
        for label, items in contrast_top.items():
            for it in items:
                contrast_counts[(label, it["entity"])] = it["n_cases"]

        scored: dict[str, dict[str, float]] = defaultdict(dict)
        for label, items in target_top.items():
            for it in items:
                ent = it["entity"]
                t_cnt = it["n_cases"]
                c_cnt = contrast_counts.get((label, ent), 0)
                f_t = (t_cnt + 0.5) / (n_t + 1)
                f_c = (c_cnt + 0.5) / (n_c + 1)
                scored[label][ent] = round(f_t - f_c, 4)
        priors[transition] = scored
    return priors


def load_stage_docs(base_case_id: str, n_stages: int) -> list[dict | None]:
    docs: list[dict | None] = []
    for i in range(1, n_stages + 1):
        path = Path(INPUT_JSON_DIR) / f"STAGE{i}__{base_case_id}.json"
        if not path.exists():
            docs.append(None)
            continue
        with open(path) as f:
            docs.append(json.load(f))
    return docs


def collect_role_sentences(doc: dict, roles: tuple[str, ...]) -> list[dict]:
    out = []
    for s in doc.get("sentences", []):
        role = s.get("rhetorical_role")
        if role in roles:
            out.append({
                "rhetorical_role": role,
                "text": s.get("text", "").strip(),
                "entities": [{"text": e.get("text", "").strip(), "label": e.get("label", "")}
                             for e in s.get("entities", [])],
            })
    return out


def text_normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def find_anchor_sentences(later_doc: dict, top_entities: list[tuple[str, str, float]],
                          max_n: int) -> list[dict]:
    """Return decision-bearing sentences from the later stage that mention any
    of the top-decisive entities."""
    if not later_doc:
        return []
    target_strings = {text_normalize(e[1]) for e in top_entities if e[1]}
    anchors = []
    for s in later_doc.get("sentences", []):
        if s.get("rhetorical_role") not in DECISION_ROLES:
            continue
        sent_text = s.get("text", "")
        sent_text_norm = text_normalize(sent_text)
        matched = []
        for ent in s.get("entities", []) or []:
            if text_normalize(ent.get("text", "")) in target_strings:
                matched.append({"text": ent.get("text", "").strip(), "label": ent.get("label", "")})
        # Also do substring fallback in case the sentence references the entity
        # without it being annotated.
        if not matched:
            for tn in target_strings:
                if tn and tn in sent_text_norm and len(tn) >= 5:
                    matched.append({"text": tn, "label": "(substring_match)"})
                    break
        if matched:
            anchors.append({
                "rhetorical_role": s.get("rhetorical_role"),
                "text": sent_text.strip(),
                "matched_entities": matched,
            })
            if len(anchors) >= max_n:
                break
    return anchors


def build_factor_report(case_diff: dict, priors: dict, args) -> dict | None:
    base_case_id = case_diff["base_case_id"]
    transition = case_diff.get("transition", "")
    n_stages = case_diff.get("n_stages", len(case_diff.get("stages", [])))
    stages_meta = case_diff.get("stages", [])
    if n_stages < 2:
        return None
    if args.changed_only and len(set(s["pred_label"] for s in stages_meta)) <= 1:
        return None

    # Synthetic first-to-last entity diff so multi-stage cases reduce to one
    # transition.  We mirror the logic in 05b.
    first, last = stages_meta[0], stages_meta[-1]
    a_ents = first.get("entities_by_label", {}) or {}
    b_ents = last.get("entities_by_label", {}) or {}
    added_by_label: dict[str, list[str]] = {}
    for label in set(a_ents) | set(b_ents):
        diff = sorted(set(b_ents.get(label, [])) - set(a_ents.get(label, [])))
        if diff:
            added_by_label[label] = diff

    # Section sentence delta first->last
    a_sec = first.get("section_sentence_counts", {}) or {}
    b_sec = last.get("section_sentence_counts", {}) or {}
    sec_delta = {role: b_sec.get(role, 0) - a_sec.get(role, 0)
                 for role in set(a_sec) | set(b_sec)
                 if b_sec.get(role, 0) - a_sec.get(role, 0) != 0}

    # Score added entities using the population priors.
    transition_priors = priors.get(transition, {})
    scored: list[tuple[str, str, float]] = []  # (label, entity, score)
    for label, ents in added_by_label.items():
        if label in {"DATE", "CASE_NUMBER", "GPE", "ORG", "OTHER_PERSON",
                     "PETITIONER", "RESPONDENT"}:
            continue
        label_priors = transition_priors.get(label, {})
        for e in ents:
            score = label_priors.get(e, 0.0)
            scored.append((label, e, score))

    # Sort: legal labels first, then by absolute score (most discriminative).
    label_priority = {l: i for i, l in enumerate(LEGAL_LABELS + PARTY_LABELS)}
    scored.sort(key=lambda t: (label_priority.get(t[0], 99), -t[2]))
    top_factors = scored[: args.top_k_factors]

    # Pull the actual stage docs to extract decision-bearing text.
    docs = load_stage_docs(base_case_id, n_stages)
    later_doc = docs[-1]
    earlier_doc = docs[0]

    # New decision-role sentences = (decision sentences in later) minus
    # any sentence with the same normalized text in earlier.
    earlier_sentence_set: set[str] = set()
    if earlier_doc:
        for s in earlier_doc.get("sentences", []):
            if s.get("rhetorical_role") in DECISION_ROLES:
                earlier_sentence_set.add(text_normalize(s.get("text", "")))

    new_decision_sentences: list[dict] = []
    if later_doc:
        for s in later_doc.get("sentences", []):
            if s.get("rhetorical_role") not in DECISION_ROLES:
                continue
            tn = text_normalize(s.get("text", ""))
            if tn and tn not in earlier_sentence_set:
                new_decision_sentences.append({
                    "rhetorical_role": s.get("rhetorical_role"),
                    "text": s.get("text", "").strip(),
                    "entities": [{"text": e.get("text", "").strip(), "label": e.get("label", "")}
                                 for e in s.get("entities", []) or []],
                })

    # Anchor sentences mentioning the top-decisive entities (decision-roles only).
    anchors = find_anchor_sentences(later_doc, top_factors, args.max_anchor_sentences)

    return {
        "base_case_id": base_case_id,
        "category": case_diff.get("category", ""),
        "outcome_split": case_diff.get("outcome_split", ""),
        "transition": transition,
        "n_stages": n_stages,
        "true_label": case_diff.get("true_label"),
        "section_sentence_delta_first_to_last": sec_delta,
        "top_decisive_factors": [
            {"label": l, "entity": e, "discriminative_score": s}
            for (l, e, s) in top_factors
        ],
        "new_decision_role_sentences": new_decision_sentences[:args.max_anchor_sentences],
        "anchor_sentences": anchors,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.aggregates) as f:
        aggregates = json.load(f)
    priors = build_entity_priors(aggregates)

    index_rows = []
    written = 0
    skipped_no_change = 0
    for diff_path in sorted(Path(args.diffs_dir).glob("*.json")):
        with open(diff_path) as f:
            case_diff = json.load(f)
        report = build_factor_report(case_diff, priors, args)
        if report is None:
            skipped_no_change += 1
            continue
        with open(out_dir / f"{report['base_case_id']}.json", "w") as f:
            json.dump(report, f, indent=2)
        written += 1

        top_str = "; ".join(f"{f['label']}:{f['entity'][:40]}" for f in report["top_decisive_factors"][:3])
        index_rows.append({
            "base_case_id": report["base_case_id"],
            "category": report["category"],
            "transition": report["transition"],
            "true_label": report["true_label"],
            "n_stages": report["n_stages"],
            "n_top_factors": len(report["top_decisive_factors"]),
            "n_new_decision_sentences": len(report["new_decision_role_sentences"]),
            "n_anchor_sentences": len(report["anchor_sentences"]),
            "top_factors_preview": top_str,
        })

    if index_rows:
        with open(out_dir / "_index.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)

    print(f"[05c] wrote {written} per-case factor reports -> {out_dir}")
    if args.changed_only:
        print(f"[05c] skipped {skipped_no_change} cases with no prediction change")
    print(f"[05c] index: {out_dir / '_index.csv'}")


if __name__ == "__main__":
    main()
