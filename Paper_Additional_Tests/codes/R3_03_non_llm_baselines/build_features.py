#!/usr/bin/env python3
"""Build non-LLM baseline features for Reviewer 3 comment R3-03.

Answers: *"...not against simpler baselines such as SVM on TF-IDF, Logistic
Regression on entity counts, or XGBoost. The absolute improvement over a trivial
baseline is not shown."*

Everything is read from the **leakage-controlled cleaned cases** the HGT itself
consumes::

    section_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/
    processed/cleaned_cases/*.json          (71,813 files)

so the resulting numbers ablate the *model*, not the preprocessing. Four caches
are written, all in the same order as the graph's case nodes
(``sorted(glob("*.json"))``, which is how ``load_cleaned_cases`` orders them):

  TXT     preamble + facts + arguments -- the three sections the HGT case node encodes;
          TF-IDF applies a final mask/outcome-token sanitizer when loading this cache
  ENT     the GNN's 12 case scalars + per-entity-type counts + retained-role histogram
  AUTH    per-case canonical statute/provision/precedent counts, vocabulary UNFITTED
          so each fold can select its own top-K from its own training rows
  RAWTXT  (--with-raw) the unfiltered judgment text, for the deliberately-leaked
          oracle row only -- NOT a baseline

Leakage guards (asserted, and recorded in build_report.json):
  * ``raw_label`` is read only as the target; the label mirrors in
    ``metadata.source_label_field/source_label_value/source_decision_label``
    are never read.
  * ``leakage_audit.dropped_sentence_role_counts`` is excluded -- the RPC/RATIO
    counts proxy for decision length. Only ``kept_sentence_role_counts`` is used.
  * Entity fields ``global_case_frequency``/``degree``/``is_shared_node`` are
    excluded: they are corpus-level statistics computed over test cases too.
  * ``[LEAKAGE_MASK]`` occurrences are counted here and removed before TF-IDF
    vectorization by ``text_sanitization.py``.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
CLEANED_CASES = (
    _REPO
    / "section_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset"
    / "processed/cleaned_cases"
)
UPSTREAM_MERGED = (
    _REPO
    / "DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved"
    / "combined_dataset_without_food_safety"
)
INPUT_DATA = _REPO / "INPUT_DATA"
FEATURES_DIR = _HERE / "outputs" / "features"

# --- the GNN's own case-node scalars (features.case_scalar_names in config.yaml)
CASE_SCALARS = [
    "respondent_count",
    "judge_count",
    "lawyer_count",
    "statute_count",
    "provision_count",
    "precedent_count",
    "preamble_length",
    "facts_length",
    "arguments_length",
    "case_year",
    "petition_type_known",
    "petition_type_hash",
]
# --- entity types the reasoning graph keeps (config.yaml include_node_types,
#     minus the case node and the six text-section nodes).
ENTITY_TYPES = [
    "petitioner",
    "respondent",
    "court",
    "judge",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "statute",
    "provision",
    "precedent",
]
# --- rhetorical roles the preprocessing retains (config.yaml *_roles).
RETAINED_ROLES = ["PREAMBLE", "FAC", "ARG_PETITIONER", "ARG_RESPONDENT", "PRE_RELIED", "PRE_NOT_RELIED", "STA"]
AUTHORITY_TYPES = ["statute", "provision", "precedent"]
MASK_TOKEN = "[LEAKAGE_MASK]"

# Never read as features. Enforced by _case_features and audited afterwards.
BLOCKED_METADATA_KEYS = frozenset(
    {"source_label_field", "source_label_value", "source_decision_label"}
)
BLOCKED_ENTITY_KEYS = frozenset({"global_case_frequency", "degree", "is_shared_node"})
BLOCKED_AUDIT_KEYS = frozenset({"dropped_sentence_role_counts"})

FEATURE_NAMES = (
    CASE_SCALARS
    + [f"n_{t}" for t in ENTITY_TYPES]
    + [f"mentions_{t}" for t in ENTITY_TYPES]
    + [f"role_{r}" for r in RETAINED_ROLES]
)


def _case_features(case: dict) -> tuple[np.ndarray, str, dict[str, dict[str, int]], int]:
    """Return (ENT row, TXT document, AUTH counts, surviving mask-token count)."""
    metadata = case.get("metadata") or {}
    texts = case.get("texts") or {}

    values = {name: 0.0 for name in CASE_SCALARS}
    for name in CASE_SCALARS:
        if name == "petition_type_known":
            values[name] = 1.0 if metadata.get("petition_type") else 0.0
        elif name == "petition_type_hash":
            values[name] = float(metadata.get("petition_type_hash") or 0.0)
        else:
            values[name] = float(metadata.get(name) or 0.0)

    distinct: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    authorities: dict[str, dict[str, int]] = {t: {} for t in AUTHORITY_TYPES}
    for entity in case.get("entities") or []:
        entity_type = str(entity.get("entity_type") or "")
        if entity_type not in ENTITY_TYPES:
            continue
        n_mentions = len(entity.get("mentions") or [])
        distinct[entity_type] += 1
        mentions[entity_type] += n_mentions
        if entity_type in AUTHORITY_TYPES:
            name = str(entity.get("canonical_name") or entity.get("raw_name") or "").strip().lower()
            if name:
                bucket = authorities[entity_type]
                bucket[name] = bucket.get(name, 0) + n_mentions

    kept_roles = ((case.get("leakage_audit") or {}).get("kept_sentence_role_counts")) or {}

    row = np.array(
        [values[name] for name in CASE_SCALARS]
        + [float(distinct[t]) for t in ENTITY_TYPES]
        + [float(mentions[t]) for t in ENTITY_TYPES]
        + [float(kept_roles.get(role, 0)) for role in RETAINED_ROLES],
        dtype=np.float32,
    )

    # Exactly what the HGT case node encodes: preamble | facts | arguments.
    document = "\n".join(
        str(texts.get(section) or "") for section in ("preamble", "facts", "arguments")
    )
    return row, document, authorities, document.count(MASK_TOKEN)


def _process_chunk(paths: list[str]) -> tuple[list[str], list[str], list[str], np.ndarray, list[str], int]:
    case_ids, labels, documents, auth_lines = [], [], [], []
    rows = np.zeros((len(paths), len(FEATURE_NAMES)), dtype=np.float32)
    masks = 0
    for i, path in enumerate(paths):
        with open(path, encoding="utf-8") as handle:
            case = json.load(handle)
        row, document, authorities, n_mask = _case_features(case)
        rows[i] = row
        masks += n_mask
        case_ids.append(str(case["case_id"]))
        labels.append(str(case["raw_label"]))
        documents.append(document)
        auth_lines.append(json.dumps(authorities, ensure_ascii=False))
    return case_ids, labels, documents, rows, auth_lines, masks


# ------------------------------------------------------------------ raw text


_MERGED_DATE = re.compile(r"^(?P<day>\d{1,2})_(?P<month>[A-Za-z]+)_(?P<year>\d{4})$")


def _build_text_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for directory in sorted(INPUT_DATA.glob("*_text*")):
        if not directory.is_dir():
            continue
        for name in os.listdir(directory):
            if name.endswith(".txt"):
                index.setdefault(name[:-4], str(directory / name))
    return index


def _resolve_raw_text(case_id: str, index: dict[str, str]) -> tuple[str, str]:
    """Return (text, how). Merged cases concatenate their constituent hearings."""
    file_id = case_id.split("__", 1)[-1]
    path = index.get(file_id)
    if path:
        return Path(path).read_text(encoding="utf-8", errors="ignore"), "direct"

    if file_id.endswith("_MERGED"):
        # Only the filename list is read from upstream -- never a label field.
        upstream = UPSTREAM_MERGED / f"{case_id}.json"
        if upstream.exists():
            merged_from = json.loads(upstream.read_text()).get("_merged_from") or []
            parts = []
            for name in merged_from:
                stem = name[:-5] if name.endswith(".json") else name
                hit = index.get(stem)
                if hit:
                    parts.append(Path(hit).read_text(encoding="utf-8", errors="ignore"))
            if parts:
                return "\n".join(parts), "merged"
    return "", "missing"


def _write_raw_text(case_ids: list[str], out_path: Path) -> dict[str, int]:
    index = _build_text_index()
    stats = Counter()
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        for case_id in case_ids:
            text, how = _resolve_raw_text(case_id, index)
            stats[how] += 1
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    return dict(stats)


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N cases (smoke test)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--with-raw", action="store_true", help="also cache unfiltered raw text (oracle row)")
    parser.add_argument("--out", type=Path, default=FEATURES_DIR)
    args = parser.parse_args()

    # sorted() matches load_cleaned_cases, i.e. the graph's case-node order.
    paths = sorted(glob.glob(str(CLEANED_CASES / "*.json")))
    if args.limit:
        paths = paths[: args.limit]
    print(f"cleaned cases: {len(paths):,}")
    args.out.mkdir(parents=True, exist_ok=True)

    chunks = [paths[i : i + 500] for i in range(0, len(paths), 500)]
    case_ids: list[str] = []
    labels: list[str] = []
    row_blocks: list[np.ndarray] = []
    total_masks = 0

    with gzip.open(args.out / "text.jsonl.gz", "wt", encoding="utf-8") as text_out, gzip.open(
        args.out / "auth.jsonl.gz", "wt", encoding="utf-8"
    ) as auth_out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (ids, labs, docs, rows, auths, masks) in enumerate(
            pool.map(_process_chunk, chunks), start=1
        ):
            case_ids.extend(ids)
            labels.extend(labs)
            row_blocks.append(rows)
            total_masks += masks
            for document in docs:
                text_out.write(json.dumps(document, ensure_ascii=False) + "\n")
            for line in auths:
                auth_out.write(line + "\n")
            if done % 20 == 0 or done == len(chunks):
                print(f"  {done}/{len(chunks)} chunks  ({len(case_ids):,} cases)", flush=True)

    features = np.vstack(row_blocks)
    np.savez_compressed(
        args.out / "ent.npz",
        X=features,
        feature_names=np.array(FEATURE_NAMES),
        case_ids=np.array(case_ids),
        labels=np.array(labels),
    )
    (args.out / "case_index.json").write_text(
        json.dumps({"case_ids": case_ids, "labels": labels}, ensure_ascii=False)
    )

    raw_stats: dict[str, int] = {}
    if args.with_raw:
        print("caching unfiltered raw text (oracle row only) ...")
        raw_stats = _write_raw_text(case_ids, args.out / "rawtext.jsonl.gz")
        print(f"  raw text resolution: {raw_stats}")

    # ------------------------------------------------------------ leakage audit
    blocked_in_features = sorted(
        set(FEATURE_NAMES) & (BLOCKED_METADATA_KEYS | BLOCKED_ENTITY_KEYS | BLOCKED_AUDIT_KEYS)
    )
    assert not blocked_in_features, f"blocked keys leaked into features: {blocked_in_features}"

    report = {
        "n_cases": len(case_ids),
        "label_distribution": dict(Counter(labels)),
        "ent_feature_names": FEATURE_NAMES,
        "ent_shape": list(features.shape),
        "leakage_mask_tokens_surviving_in_text": total_masks,
        "blocked_metadata_keys": sorted(BLOCKED_METADATA_KEYS),
        "blocked_entity_keys": sorted(BLOCKED_ENTITY_KEYS),
        "blocked_audit_keys": sorted(BLOCKED_AUDIT_KEYS),
        "blocked_keys_present_in_features": blocked_in_features,
        "raw_text_resolution": raw_stats,
        "source": str(CLEANED_CASES),
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "ent_feature_names"}, indent=2))


if __name__ == "__main__":
    main()
