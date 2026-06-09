#!/usr/bin/env python3
"""
Step 05 — Group per-stage predictions by base case and analyse what changed
between stages.

Outputs (under outputs/analysis/):
- stage_transitions.csv     one row per case, with stage1..N predictions and
                            transition string (e.g. "LOSE -> WIN")
- transition_counts.json    counts of each transition pattern
- per_case_diffs/           one JSON per case detailing what changed between
                            consecutive stages (new entities, sentence-count
                            deltas per section, etc.)
- summary.json              high-level stats (final-prediction accuracy,
                            fraction of cases that changed prediction)
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = EXP_ROOT / "data/input_jsons"

PRETTY = {"-1": "LOSE", "1": "WIN"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=str(EXP_ROOT / "outputs/inference/predictions.csv"))
    p.add_argument("--manifest", default=str(EXP_ROOT / "outputs/stage_manifest.json"))
    p.add_argument("--out-dir", default=str(EXP_ROOT / "outputs/analysis"))
    return p.parse_args()


def load_input_doc(stage_filename: str) -> dict:
    path = INPUT_DIR / stage_filename
    with open(path) as f:
        return json.load(f)


def collect_entities(doc: dict) -> dict:
    """Aggregate {label: set(text)} entity mentions across all sentences."""
    by_label: dict[str, set[str]] = defaultdict(set)
    for s in doc.get("sentences", []):
        for ent in s.get("entities", []):
            label = ent.get("label", "OTHER")
            text = (ent.get("text", "") or "").strip()
            if text:
                by_label[label].add(text)
    return {k: sorted(v) for k, v in by_label.items()}


def section_sentence_counts(doc: dict) -> dict:
    counts = Counter()
    for s in doc.get("sentences", []):
        role = s.get("rhetorical_role", "NONE")
        counts[role] += 1
    return dict(counts)


def diff_sets(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict:
    """For each entity label, list entities new in b vs a, and dropped in b vs a."""
    out = {}
    labels = set(a) | set(b)
    for label in sorted(labels):
        sa = set(a.get(label, []))
        sb = set(b.get(label, []))
        added = sorted(sb - sa)
        removed = sorted(sa - sb)
        if added or removed:
            out[label] = {"added": added, "removed": removed}
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    diffs_dir = out_dir / "per_case_diffs"
    if diffs_dir.exists():
        shutil.rmtree(diffs_dir)
    diffs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    df = df.sort_values(["base_case_id", "stage_index"]).reset_index(drop=True)
    with open(args.manifest) as f:
        manifest = json.load(f)
    case_lookup = {c["base_case_id"]: c for c in manifest["cases"]}

    transition_rows = []
    transition_counts: Counter = Counter()
    final_pred_correct = 0
    final_pred_total = 0
    cases_with_change = 0

    for base_id, grp in df.groupby("base_case_id", sort=False):
        grp = grp.sort_values("stage_index")
        stages = grp[["stage_index", "pred_label", "target_label", "confidence"]].to_dict("records")
        manifest_row = case_lookup.get(base_id, {})

        labels = [PRETTY.get(str(s["pred_label"]), str(s["pred_label"])) for s in stages]
        transition_str = " -> ".join(labels)
        transition_counts[transition_str] += 1

        true_label = PRETTY.get(str(stages[-1]["target_label"]), str(stages[-1]["target_label"]))
        final_pred_label = labels[-1]
        if true_label and final_pred_label:
            final_pred_total += 1
            if true_label == final_pred_label:
                final_pred_correct += 1

        if len(set(labels)) > 1:
            cases_with_change += 1

        row = {
            "base_case_id": base_id,
            "category": manifest_row.get("category", ""),
            "outcome_split": manifest_row.get("outcome_split", ""),
            "n_stages": len(stages),
            "true_label": true_label,
            "transition": transition_str,
            "final_pred_correct": (true_label == final_pred_label),
            "changed_prediction": (len(set(labels)) > 1),
        }
        for i, s in enumerate(stages, start=1):
            row[f"stage{i}_pred"] = PRETTY.get(str(s["pred_label"]), str(s["pred_label"]))
            row[f"stage{i}_conf"] = round(float(s["confidence"]), 4)
        transition_rows.append(row)

        per_stage_entities = []
        per_stage_sections = []
        per_stage_dates = []
        ok = True
        for s_meta in manifest_row.get("stages", []):
            try:
                doc = load_input_doc(s_meta["input_filename"])
            except Exception as exc:
                ok = False
                per_stage_entities.append({"_error": str(exc)})
                per_stage_sections.append({})
                per_stage_dates.append(s_meta.get("date", ""))
                continue
            per_stage_entities.append(collect_entities(doc))
            per_stage_sections.append(section_sentence_counts(doc))
            per_stage_dates.append(s_meta.get("date", ""))

        consecutive_diffs = []
        if ok and len(per_stage_entities) >= 2:
            for i in range(len(per_stage_entities) - 1):
                a = per_stage_entities[i]
                b = per_stage_entities[i + 1]
                sa = per_stage_sections[i]
                sb = per_stage_sections[i + 1]
                section_delta = {role: sb.get(role, 0) - sa.get(role, 0)
                                 for role in set(sa) | set(sb)
                                 if sa.get(role, 0) != sb.get(role, 0)}
                consecutive_diffs.append({
                    "from_stage": i + 1,
                    "to_stage": i + 2,
                    "from_pred": labels[i],
                    "to_pred": labels[i + 1],
                    "from_date": per_stage_dates[i],
                    "to_date": per_stage_dates[i + 1],
                    "entity_diff": diff_sets(a, b),
                    "section_sentence_count_delta": section_delta,
                })

        per_case_payload = {
            "base_case_id": base_id,
            "category": manifest_row.get("category", ""),
            "outcome_split": manifest_row.get("outcome_split", ""),
            "n_stages": len(stages),
            "true_label": true_label,
            "transition": transition_str,
            "stages": [
                {
                    "stage_index": stages[i]["stage_index"],
                    "date": per_stage_dates[i] if i < len(per_stage_dates) else "",
                    "pred_label": labels[i],
                    "confidence": round(float(stages[i]["confidence"]), 4),
                    "section_sentence_counts": per_stage_sections[i] if i < len(per_stage_sections) else {},
                    "entities_by_label": per_stage_entities[i] if i < len(per_stage_entities) else {},
                }
                for i in range(len(stages))
            ],
            "consecutive_diffs": consecutive_diffs,
        }
        with open(diffs_dir / f"{base_id}.json", "w") as f:
            json.dump(per_case_payload, f, indent=2)

    transitions_df = pd.DataFrame(transition_rows)
    transitions_df.to_csv(out_dir / "stage_transitions.csv", index=False)
    with open(out_dir / "transition_counts.json", "w") as f:
        json.dump(dict(transition_counts.most_common()), f, indent=2)

    summary = {
        "n_cases": len(transition_rows),
        "n_stages_distribution": dict(Counter(r["n_stages"] for r in transition_rows)),
        "transition_counts": dict(transition_counts.most_common()),
        "final_pred_accuracy": (final_pred_correct / final_pred_total) if final_pred_total else None,
        "cases_with_changed_prediction": cases_with_change,
        "cases_consistent_prediction": len(transition_rows) - cases_with_change,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[05_analyze] cases analysed: {summary['n_cases']}")
    print(f"[05_analyze] final-prediction accuracy: {summary['final_pred_accuracy']}")
    print(f"[05_analyze] cases that changed prediction across stages: {cases_with_change}")
    print(f"[05_analyze] top transitions: {list(transition_counts.most_common(6))}")
    print(f"[05_analyze] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
