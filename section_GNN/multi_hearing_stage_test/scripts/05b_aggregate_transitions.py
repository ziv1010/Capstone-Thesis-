#!/usr/bin/env python3
"""
Step 05b — Aggregate the per-case stage diffs across transition types.

For each transition pattern (LOSE -> WIN, WIN -> LOSE, LOSE -> LOSE,
WIN -> WIN, ...), summarise across all cases:

  - section_sentence_delta: mean / median delta per rhetorical role from the
    earlier stage to the later stage (which sections systematically grow?)
  - entity_label_add_freq: how often each entity-label class sees ADDITIONS
    in the later stage (e.g. did this transition tend to bring in new
    PROVISIONs?)
  - top_added_entities / top_removed_entities: per entity label, the most
    frequent specific entity strings that were added or removed at the
    later stage, restricted to legal-content labels
    (STATUTE, PROVISION, PRECEDENT, JUDGE, LAWYER).

The script also writes a *contrast* view: for transitions that flipped
(LOSE -> WIN, WIN -> LOSE) versus matched stable patterns
(LOSE -> LOSE, WIN -> WIN), it computes the ratio of section / entity-label
add frequencies to highlight what is unusually concentrated in the flip
group.

For multi-stage cases (3+ stages) the diff between the FIRST and LAST stage
is used so every case contributes one (start_pred -> end_pred) bucket.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
DIFFS_DIR = EXP_ROOT / "outputs/analysis/per_case_diffs"
OUT_PATH = EXP_ROOT / "outputs/analysis/transition_aggregates.json"

# Roles ordered roughly chronologically through a court document.
ROLES_OF_INTEREST = ["PREAMBLE", "FAC", "ARG_PETITIONER", "ARG_RESPONDENT",
                     "STA", "PRE_RELIED", "PRE_NOT_RELIED", "RATIO",
                     "RPC", "ANALYSIS", "ISSUE", "RLC", "NONE"]

LEGAL_ENTITY_LABELS = ("STATUTE", "PROVISION", "PRECEDENT", "JUDGE", "LAWYER")
NOISY_LABELS = {"DATE", "CASE_NUMBER", "GPE", "ORG"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diffs-dir", default=str(DIFFS_DIR))
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--top-k", type=int, default=15,
                   help="How many top added/removed entities to report per label.")
    return p.parse_args()


def derive_first_to_last(case: dict) -> dict | None:
    """Build a synthetic 'consecutive_diffs'-style record between the first and
    last stage so multi-stage cases contribute one record each."""
    stages = case.get("stages", [])
    if len(stages) < 2:
        return None
    first, last = stages[0], stages[-1]

    section_delta = {}
    a_sec = first.get("section_sentence_counts", {}) or {}
    b_sec = last.get("section_sentence_counts", {}) or {}
    for role in set(a_sec) | set(b_sec):
        delta = b_sec.get(role, 0) - a_sec.get(role, 0)
        if delta != 0:
            section_delta[role] = delta

    a_ents = first.get("entities_by_label", {}) or {}
    b_ents = last.get("entities_by_label", {}) or {}
    entity_diff: dict[str, dict[str, list[str]]] = {}
    for label in set(a_ents) | set(b_ents):
        sa = set(a_ents.get(label, []))
        sb = set(b_ents.get(label, []))
        added = sorted(sb - sa)
        removed = sorted(sa - sb)
        if added or removed:
            entity_diff[label] = {"added": added, "removed": removed}

    return {
        "from_pred": first.get("pred_label"),
        "to_pred": last.get("pred_label"),
        "section_sentence_count_delta": section_delta,
        "entity_diff": entity_diff,
    }


def aggregate_one_group(records: list[dict], top_k: int) -> dict:
    """records is a list of first-to-last diffs that share the same transition."""
    n = len(records)
    section_deltas: dict[str, list[int]] = defaultdict(list)
    entity_label_add_count: Counter = Counter()
    entity_label_remove_count: Counter = Counter()
    added_by_label: dict[str, Counter] = defaultdict(Counter)
    removed_by_label: dict[str, Counter] = defaultdict(Counter)

    for rec in records:
        for role, delta in rec.get("section_sentence_count_delta", {}).items():
            section_deltas[role].append(delta)
        for label, change in rec.get("entity_diff", {}).items():
            if label in NOISY_LABELS:
                continue
            if change.get("added"):
                entity_label_add_count[label] += 1
                if label in LEGAL_ENTITY_LABELS:
                    for e in change["added"]:
                        added_by_label[label][e] += 1
            if change.get("removed"):
                entity_label_remove_count[label] += 1
                if label in LEGAL_ENTITY_LABELS:
                    for e in change["removed"]:
                        removed_by_label[label][e] += 1

    section_summary = {}
    for role, deltas in section_deltas.items():
        # Pad with zeros for cases that didn't mention this role.
        padded = deltas + [0] * (n - len(deltas))
        section_summary[role] = {
            "mean": round(statistics.mean(padded), 3),
            "median": statistics.median(padded),
            "n_cases_with_change": len(deltas),
            "frac_cases_with_change": round(len(deltas) / n, 3) if n else 0.0,
        }

    section_summary_sorted = dict(
        sorted(section_summary.items(),
               key=lambda kv: (kv[0] not in ROLES_OF_INTEREST, -kv[1]["mean"]))
    )

    label_add_freq = {
        label: {"count": cnt, "frac": round(cnt / n, 3) if n else 0.0}
        for label, cnt in entity_label_add_count.most_common()
    }
    label_remove_freq = {
        label: {"count": cnt, "frac": round(cnt / n, 3) if n else 0.0}
        for label, cnt in entity_label_remove_count.most_common()
    }

    top_added = {
        label: [{"entity": e, "n_cases": c} for e, c in cnts.most_common(top_k)]
        for label, cnts in added_by_label.items()
    }
    top_removed = {
        label: [{"entity": e, "n_cases": c} for e, c in cnts.most_common(top_k)]
        for label, cnts in removed_by_label.items()
    }

    return {
        "n_cases": n,
        "section_sentence_delta": section_summary_sorted,
        "entity_label_add_freq": label_add_freq,
        "entity_label_remove_freq": label_remove_freq,
        "top_added_entities": top_added,
        "top_removed_entities": top_removed,
    }


def contrast(a: dict, b: dict) -> dict:
    """Return the section / entity-label add-frequency ratios of group A over group B,
    so values >> 1 mean the feature is much more common in A than in B."""
    out = {"section_mean_ratio": {}, "entity_label_add_ratio": {}}
    a_sec = a.get("section_sentence_delta", {})
    b_sec = b.get("section_sentence_delta", {})
    for role in set(a_sec) | set(b_sec):
        am = a_sec.get(role, {}).get("mean", 0.0)
        bm = b_sec.get(role, {}).get("mean", 0.0)
        out["section_mean_ratio"][role] = {
            "a_mean": am,
            "b_mean": bm,
            "diff": round(am - bm, 3),
        }
    a_ent = a.get("entity_label_add_freq", {})
    b_ent = b.get("entity_label_add_freq", {})
    for label in set(a_ent) | set(b_ent):
        af = a_ent.get(label, {}).get("frac", 0.0)
        bf = b_ent.get(label, {}).get("frac", 0.0)
        out["entity_label_add_ratio"][label] = {
            "a_frac": af,
            "b_frac": bf,
            "diff": round(af - bf, 3),
        }
    out["section_mean_ratio"] = dict(
        sorted(out["section_mean_ratio"].items(), key=lambda kv: -abs(kv[1]["diff"]))
    )
    out["entity_label_add_ratio"] = dict(
        sorted(out["entity_label_add_ratio"].items(), key=lambda kv: -abs(kv[1]["diff"]))
    )
    return out


def main() -> None:
    args = parse_args()
    diffs_dir = Path(args.diffs_dir)
    files = sorted(diffs_dir.glob("*.json"))

    transition_groups: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        with open(f) as fp:
            case = json.load(fp)
        rec = derive_first_to_last(case)
        if rec is None:
            continue
        key = f"{rec['from_pred']} -> {rec['to_pred']}"
        transition_groups[key].append(rec)

    # Aggregate per group
    aggregates = {}
    for key, recs in transition_groups.items():
        aggregates[key] = aggregate_one_group(recs, top_k=args.top_k)

    # Contrasts
    contrasts = {}
    if "LOSE -> WIN" in aggregates and "LOSE -> LOSE" in aggregates:
        contrasts["LOSE->WIN_vs_LOSE->LOSE"] = contrast(aggregates["LOSE -> WIN"], aggregates["LOSE -> LOSE"])
    if "WIN -> LOSE" in aggregates and "WIN -> WIN" in aggregates:
        contrasts["WIN->LOSE_vs_WIN->WIN"] = contrast(aggregates["WIN -> LOSE"], aggregates["WIN -> WIN"])
    if "LOSE -> WIN" in aggregates and "WIN -> LOSE" in aggregates:
        contrasts["LOSE->WIN_vs_WIN->LOSE"] = contrast(aggregates["LOSE -> WIN"], aggregates["WIN -> LOSE"])

    payload = {
        "n_cases_total": sum(len(v) for v in transition_groups.values()),
        "transition_counts": {k: len(v) for k, v in sorted(transition_groups.items(), key=lambda kv: -len(kv[1]))},
        "by_transition": aggregates,
        "contrasts": contrasts,
        "notes": {
            "diff_mode": "first_stage vs last_stage",
            "noisy_labels_excluded": sorted(NOISY_LABELS),
            "legal_entity_labels": list(LEGAL_ENTITY_LABELS),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    # Console snapshot.
    print(f"[05b] aggregated {payload['n_cases_total']} cases over {len(aggregates)} transition patterns")
    print(f"[05b] counts: {payload['transition_counts']}")
    for tname in ["LOSE -> WIN", "WIN -> LOSE", "LOSE -> LOSE", "WIN -> WIN"]:
        if tname not in aggregates:
            continue
        agg = aggregates[tname]
        print(f"\n--- {tname}  (n={agg['n_cases']}) ---")
        top_secs = list(agg["section_sentence_delta"].items())[:5]
        print("  top section deltas (mean sentences added at later stage):")
        for role, info in top_secs:
            print(f"    {role:<18}  mean={info['mean']:+.2f}  median={info['median']:+}  in {info['frac_cases_with_change']*100:.0f}% of cases")
        print("  most-frequently ADDED entity labels (frac of cases):")
        for label, info in list(agg["entity_label_add_freq"].items())[:6]:
            print(f"    {label:<14}  {info['frac']*100:.0f}%  ({info['count']}/{agg['n_cases']})")

    if "LOSE->WIN_vs_LOSE->LOSE" in contrasts:
        print("\n--- CONTRAST: LOSE->WIN vs LOSE->LOSE  (positive diff = stronger in LOSE->WIN) ---")
        c = contrasts["LOSE->WIN_vs_LOSE->LOSE"]
        print("  section mean delta (top by |diff|):")
        for role, info in list(c["section_mean_ratio"].items())[:6]:
            print(f"    {role:<18}  WIN={info['a_mean']:+.2f}  LOSE={info['b_mean']:+.2f}  diff={info['diff']:+.2f}")
        print("  entity-label add frac (top by |diff|):")
        for label, info in list(c["entity_label_add_ratio"].items())[:6]:
            print(f"    {label:<14}  WIN={info['a_frac']:.2f}  LOSE={info['b_frac']:.2f}  diff={info['diff']:+.2f}")

    print(f"\n[05b] full payload -> {args.out}")


if __name__ == "__main__":
    main()
