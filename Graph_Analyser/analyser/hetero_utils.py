from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

_DATE_PATTERN = re.compile(
    r"\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def normalize_title_for_match(title: str) -> str:
    """Canonicalise a case title / precedent string so we can detect
    self-citation. Lowercases, strips 'on <date>' tails, strips punctuation,
    collapses whitespace.
    """
    if not title:
        return ""
    cleaned = _DATE_PATTERN.sub("", title)
    cleaned = cleaned.translate(_PUNCT_TABLE).lower()
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    # Drop very common tokens that add noise
    return cleaned


_LEGAL_STOP_WORDS = {
    "vs", "v", "state", "of", "the", "and", "another", "others",
    "anr", "ors", "on", "in", "a", "an", "by", "for", "etc",
    "union", "ministry", "department", "govt", "government",
    "nct", "delhi",  # very frequent jurisdictional tokens
}


def _distinctive_tokens(normalised: str) -> set[str]:
    """Return the party-name-ish tokens of a normalised case title —
    everything that isn't a generic legal stop-word.
    """
    return {tok for tok in normalised.split() if tok and tok not in _LEGAL_STOP_WORDS}


def is_self_match(target_case_id: str, candidate_text: str) -> bool:
    """Return True when the candidate (a precedent text or arg-node owning
    case) refers to the target case itself.

    Strategy:
      1. After normalisation, if the strings are identical -> self.
      2. If one is a substring of the other AND the shorter one has enough
         distinctive party tokens -> self.
      3. Otherwise, require high overlap on *distinctive* (non stop-word)
         tokens. Pure legal-phrasing overlap ("vs state of punjab and
         another") is ignored — only party-name matches count.
    """
    if not target_case_id or not candidate_text:
        return False
    target = normalize_title_for_match(target_case_id)
    cand = normalize_title_for_match(candidate_text)
    if not target or not cand:
        return False
    if target == cand:
        return True
    target_distinct = _distinctive_tokens(target)
    cand_distinct = _distinctive_tokens(cand)
    if not target_distinct or not cand_distinct:
        return False
    overlap = target_distinct & cand_distinct
    smaller = min(len(target_distinct), len(cand_distinct))
    # Substring case: e.g. "sanjeev kumar vs state of punjab and another" is
    # fully contained in the full title with the date stripped. Require at
    # least 2 shared distinctive tokens to avoid generic-term false positives.
    if (cand in target or target in cand) and len(overlap) >= 2:
        return True
    # Token-overlap case: require essentially all distinctive tokens of the
    # shorter title to appear in the longer, with at least 2 shared tokens.
    if smaller >= 2 and len(overlap) >= smaller and len(overlap) >= 2:
        return True
    return False


def owning_case_of_arg_node(arg_node_key: str) -> str | None:
    """Argument node keys look like ``case::<case_id>::<arg_type>``.
    Return <case_id> or None if the format is unexpected.
    """
    if not arg_node_key or not arg_node_key.startswith("case::"):
        return None
    parts = arg_node_key.split("::")
    if len(parts) < 3:
        return None
    return parts[1]


def load_case_text(
    cleaned_case_dir: str | Path,
    file_name: str,
    max_chars_per_section: int = 1500,
) -> dict[str, str]:
    """Load facts / arguments / preamble for a case from the preprocessed
    cleaned_cases JSON. Returns {} if the file is missing.
    """
    path = Path(cleaned_case_dir) / file_name
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return {}
    texts = blob.get("texts", {}) or {}
    out: dict[str, str] = {}
    for section in ("preamble", "facts", "arguments", "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments"):
        value = texts.get(section)
        if not value:
            continue
        if len(value) > max_chars_per_section:
            value = value[:max_chars_per_section].rstrip() + "..."
        out[section] = value
    return out


def invert_node_mapping(mapping: dict[str, int] | list[str]) -> dict[int, str]:
    if isinstance(mapping, dict):
        return {int(v): k for k, v in mapping.items()}
    return {i: key for i, key in enumerate(mapping)}


def strip_node_prefix(canonical_key: str, node_type: str) -> str:
    prefix = f"{node_type}::"
    if canonical_key.startswith(prefix):
        return canonical_key[len(prefix):]
    # Case-scoped nodes use either ``case::<case_id>::<arg_type>`` or
    # ``case::<case_id>::<node_type>::<value>``. Keep argument nodes anchored
    # to the owning case, but show the actual value for actors/courts.
    if canonical_key.startswith("case::"):
        parts = canonical_key.split("::")
        if len(parts) >= 4 and parts[-2] == node_type:
            return parts[-1]
        if canonical_key.endswith(f"::{node_type}"):
            return canonical_key[len("case::"):-len(f"::{node_type}")]
    return canonical_key


def build_index_to_text(metadata: dict[str, Any]) -> dict[str, dict[int, str]]:
    node_mappings: dict[str, Any] = metadata.get("node_mappings", {})
    index_to_text: dict[str, dict[int, str]] = {}
    for node_type, mapping in node_mappings.items():
        inv = invert_node_mapping(mapping)
        index_to_text[node_type] = {
            idx: strip_node_prefix(key, node_type) for idx, key in inv.items()
        }
    return index_to_text


def build_case_id_to_index(data) -> dict[str, int]:
    case_ids = list(getattr(data["case"], "case_id", []))
    return {cid: idx for idx, cid in enumerate(case_ids)}


def get_split_indices(data) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for split_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data["case"], split_name, None)
        if mask is None:
            continue
        out[split_name.replace("_mask", "")] = mask.nonzero(as_tuple=False).view(-1).tolist()
    return out


def load_fold_splits(
    predictions_csv_path: str,
    data,
    expected_bucket: str | None = None,
) -> dict[str, list[int]]:
    """Recover the fold-specific train/val/test split from a training-time
    predictions.csv written by section_GNN.

    K-fold checkpoints are trained on a different split than the one baked
    into the graph cache's train_mask/val_mask/test_mask tensors, so we must
    re-derive the split from the fold artefacts.
    """
    import pandas as pd  # kept local to avoid a hard torch-only import

    df = pd.read_csv(predictions_csv_path)
    if expected_bucket:
        from analyser.loader import validate_case_ids_bucket  # local to avoid import cycles

        validate_case_ids_bucket(
            [str(x) for x in df["case_id"].tolist()],
            expected_bucket,
            f"fold predictions {predictions_csv_path}",
        )
    case_ids = list(data["case"].case_id)
    case_id_to_node_idx = {cid: idx for idx, cid in enumerate(case_ids)}
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    unmatched = 0
    for _, row in df.iterrows():
        node_idx = case_id_to_node_idx.get(str(row["case_id"]))
        if node_idx is None:
            unmatched += 1
            continue
        split = str(row["split"])
        if split in splits:
            splits[split].append(int(node_idx))
    if unmatched:
        print(
            f"[load_fold_splits] warning: {unmatched} rows in {predictions_csv_path} "
            "could not be matched to graph cases (skipped)",
            flush=True,
        )
    return splits
