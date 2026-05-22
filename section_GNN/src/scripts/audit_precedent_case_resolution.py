#!/usr/bin/env python3
"""
Audit whether PRECEDENT entity nodes can be resolved back to real case nodes.

This script is intentionally read-only by default. It scans cleaned case JSONs,
matches entity_type="precedent" records against the corpus case IDs, and reports
candidate graph edges that could be added in a later graph-builder change:

  precedent --resolves_to_case--> case
  case      --cites_case---------> case   (derived convenience edge)

No existing graph cache, cleaned case JSON, or training output is modified. Pass
--write to emit a new CSV/JSON report under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_CASE_DIR = (
    REPO_ROOT
    / "data/ablations/entity_resolved_data/cross_bucket_total_dataset/processed/cleaned_cases"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/precedent_case_resolution_audit"

PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
WHITESPACE_RE = re.compile(r"\s+")
DATE_TAIL_RE = re.compile(
    r"\bon\s+\d{1,2}\s+[a-z]+\s*,?\s+\d{4}\b",
    flags=re.IGNORECASE,
)
NUMERIC_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
REPORT_CITATION_RE = re.compile(
    r"\b\d{4}\s+(?:scc|scr|air|acj|cri\s*l\s*j|crl\s*j|all\s*lj|"
    r"mh\s*lj|ilr|klj|rlw|rcr|scale|supreme|indiankanoon)\b.*$",
    flags=re.IGNORECASE,
)
INDIAN_KANOON_RE = re.compile(r"\bindian\s+kanoon\b", flags=re.IGNORECASE)
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "another",
    "anr",
    "anrs",
    "by",
    "for",
    "in",
    "of",
    "ors",
    "others",
    "state",
    "the",
    "through",
    "to",
    "u",
    "up",
    "v",
    "vs",
}
COMMON_PARTY_TOKENS = STOP_TOKENS | {
    "alias",
    "devi",
    "kumar",
    "lal",
    "md",
    "mohd",
    "mr",
    "mrs",
    "ms",
    "ram",
    "shri",
    "singh",
    "smt",
}


@dataclass
class CaseRecord:
    case_id: str
    file_name: str
    raw_label: str | None
    case_year: int | None
    aliases: list[str]


@dataclass
class PrecedentMention:
    source_case_id: str
    source_case_year: int | None
    source_file_name: str
    precedent_raw: str
    precedent_canonical: str
    mention_count: int
    sections: str
    seen_in_arguments: bool
    precedent_node_key: str


@dataclass
class LinkCandidate:
    source_case_id: str
    source_case_year: int | None
    precedent_raw: str
    precedent_canonical: str
    precedent_node_key: str
    target_case_id: str
    target_case_year: int | None
    target_file_name: str
    match_method: str
    match_score: float
    ambiguous: bool
    temporal_relation: str
    mention_count: int
    sections: str
    seen_in_arguments: bool
    proposed_src_type: str
    proposed_relation: str
    proposed_dst_type: str
    proposed_src_key: str
    proposed_dst_key: str
    derived_case_src_key: str
    derived_case_relation: str
    derived_case_dst_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether precedent entity nodes can be mapped to existing case "
            "nodes without mutating existing graph/output files."
        )
    )
    parser.add_argument(
        "--cleaned-case-dir",
        type=Path,
        default=DEFAULT_CLEANED_CASE_DIR,
        help="Directory containing processed/cleaned_cases/*.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config; paths.cleaned_case_dir overrides --cleaned-case-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="New output directory used only when --write is passed.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write a new audit CSV/JSON report. Default is stdout-only dry run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cleaned cases to scan.",
    )
    parser.add_argument(
        "--arguments-only",
        action="store_true",
        help="Only resolve precedents whose entity record was seen in arguments.",
    )
    parser.add_argument(
        "--allow-self-links",
        action="store_true",
        help="Allow a case title detected as a PRECEDENT to resolve to the same case.",
    )
    parser.add_argument(
        "--require-earlier-target",
        action="store_true",
        help="Only keep links where target_case_year < source_case_year when both years exist.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.90,
        help="Minimum fuzzy score for non-exact matches when --enable-fuzzy is used.",
    )
    parser.add_argument(
        "--enable-fuzzy",
        action="store_true",
        help=(
            "Enable fuzzy party-name matching. By default the audit keeps only "
            "exact and prefix-style matches to avoid proposing noisy edges."
        ),
    )
    parser.add_argument(
        "--ambiguity-gap",
        type=float,
        default=0.02,
        help="Mark top match ambiguous if the runner-up is within this score gap.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="Maximum candidate case records to fuzzy-score per precedent.",
    )
    parser.add_argument(
        "--precedent-nodes-shared",
        action="store_true",
        help=(
            "Use precedent::<canonical> as the proposed precedent node key. "
            "Default matches current local-star keys: case::<source>::precedent::<canonical>."
        ),
    )
    parser.add_argument(
        "--keep-indian-kanoon-precedents",
        action="store_true",
        help=(
            "Keep precedent strings containing 'Indian Kanoon'. By default these "
            "are skipped because they are usually scraper/footer self-citations."
        ),
    )
    parser.add_argument(
        "--unmatched-sample-size",
        type=int,
        default=200,
        help="Number of unmatched precedents to include when --write is used.",
    )
    return parser.parse_args()


def load_config_cleaned_case_dir(config_path: Path | None) -> Path | None:
    if config_path is None:
        return None
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("--config requires PyYAML to be installed") from exc

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    value = (cfg.get("paths") or {}).get("cleaned_case_dir")
    return Path(value) if value else None


def strip_category_prefix(value: str) -> str:
    if "__" not in value:
        return value
    prefix, rest = value.split("__", 1)
    if prefix and " " not in prefix and len(prefix) <= 40:
        return rest
    return value


def strip_report_citation(value: str) -> str:
    value = INDIAN_KANOON_RE.sub(" ", value)
    value = REPORT_CITATION_RE.sub(" ", value)
    return value


def normalize_title(value: str, *, strip_date: bool = True) -> str:
    text = str(value or "")
    text = text.removesuffix(".json")
    text = strip_category_prefix(text)
    text = strip_report_citation(text)
    text = text.replace("&", " and ")
    text = re.sub(r"\bversus\b", " vs ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bv[.]?\b", " vs ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+-\s*vs\s+-\s*", " vs ", text, flags=re.IGNORECASE)
    text = text.replace("...", " ")
    if strip_date:
        text = DATE_TAIL_RE.sub(" ", text)
        text = NUMERIC_DATE_RE.sub(" ", text)
    text = text.lower().translate(PUNCT_TRANSLATION)
    return WHITESPACE_RE.sub(" ", text).strip()


def token_set(value: str) -> set[str]:
    return {
        tok
        for tok in normalize_title(value).split()
        if len(tok) > 1 and tok not in STOP_TOKENS
    }


def split_vs(value: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+vs\s+", value, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right:
        return None
    return left, right


def component_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if len(left) >= 4 and right.startswith(left):
        return 0.96
    if len(right) >= 4 and left.startswith(right):
        return 0.94
    left_tokens = set(left.split()) - STOP_TOKENS
    right_tokens = set(right.split()) - STOP_TOKENS
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(sequence, jaccard)


def first_significant_party_token(value: str) -> str | None:
    tokens = [tok for tok in value.split() if len(tok) > 1]
    for token in tokens:
        if token not in COMMON_PARTY_TOKENS:
            return token
    for token in tokens:
        if token not in STOP_TOKENS:
            return token
    return tokens[0] if tokens else None


def anchor_tokens_compatible(left: str, right: str) -> bool:
    left_anchor = first_significant_party_token(left)
    right_anchor = first_significant_party_token(right)
    if not left_anchor or not right_anchor:
        return False
    if left_anchor == right_anchor:
        return True
    if min(len(left_anchor), len(right_anchor)) >= 4:
        return left_anchor.startswith(right_anchor) or right_anchor.startswith(left_anchor)
    return False


def case_node_key(case_id: str) -> str:
    return f"case::{case_id}"


def precedent_node_key(source_case_id: str, canonical: str, shared: bool) -> str:
    if shared:
        return f"precedent::{canonical}"
    return f"case::{source_case_id}::precedent::{canonical}"


def temporal_relation(source_year: int | None, target_year: int | None) -> str:
    if source_year is None or target_year is None:
        return "unknown"
    if target_year < source_year:
        return "target_before_source"
    if target_year == source_year:
        return "same_year"
    return "target_after_source"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


def case_aliases(payload: dict[str, Any], path: Path) -> list[str]:
    values = [
        payload.get("case_id", ""),
        payload.get("file_id", ""),
        payload.get("file_name", ""),
        path.stem,
    ]
    aliases: set[str] = set()
    for value in values:
        if not value:
            continue
        for strip_date in (True, False):
            normalized = normalize_title(str(value), strip_date=strip_date)
            if normalized:
                aliases.add(normalized)
    return sorted(aliases)


def iter_case_paths(cleaned_case_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(cleaned_case_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def collect_cases_and_precedents(
    paths: list[Path],
    *,
    arguments_only: bool,
    precedent_nodes_shared: bool,
    keep_indian_kanoon_precedents: bool,
) -> tuple[list[CaseRecord], list[PrecedentMention], Counter]:
    cases: list[CaseRecord] = []
    precedents: list[PrecedentMention] = []
    skipped = Counter()

    for path in paths:
        try:
            payload = read_json(path)
        except Exception:
            skipped["bad_json"] += 1
            continue

        case_id = str(payload.get("case_id") or path.stem)
        metadata = payload.get("metadata") or {}
        case_year = metadata.get("case_year")
        try:
            case_year = int(case_year) if case_year not in (None, "") else None
        except (TypeError, ValueError):
            case_year = None

        cases.append(
            CaseRecord(
                case_id=case_id,
                file_name=str(payload.get("file_name") or path.name),
                raw_label=payload.get("raw_label"),
                case_year=case_year,
                aliases=case_aliases(payload, path),
            )
        )

        for entity in payload.get("entities") or []:
            if entity.get("entity_type") != "precedent":
                continue
            if arguments_only and not entity.get("seen_in_arguments"):
                skipped["precedent_not_in_arguments"] += 1
                continue
            canonical = str(entity.get("canonical_name") or entity.get("raw_name") or "").strip()
            raw = str(entity.get("raw_name") or canonical).strip()
            if not canonical and not raw:
                skipped["empty_precedent"] += 1
                continue
            if (
                not keep_indian_kanoon_precedents
                and "indian kanoon" in f"{canonical} {raw}".lower()
            ):
                skipped["indian_kanoon_footer_like_precedent"] += 1
                continue
            sections = sorted(
                {
                    str(m.get("section"))
                    for m in (entity.get("mentions") or [])
                    if m.get("section")
                }
            )
            precedents.append(
                PrecedentMention(
                    source_case_id=case_id,
                    source_case_year=case_year,
                    source_file_name=str(payload.get("file_name") or path.name),
                    precedent_raw=raw,
                    precedent_canonical=canonical,
                    mention_count=int(entity.get("local_case_frequency") or 0),
                    sections=";".join(sections),
                    seen_in_arguments=bool(entity.get("seen_in_arguments")),
                    precedent_node_key=precedent_node_key(
                        case_id,
                        canonical,
                        precedent_nodes_shared,
                    ),
                )
            )
    return cases, precedents, skipped


def alias_anchor(value: str) -> str | None:
    parties = split_vs(value)
    if parties:
        return first_significant_party_token(parties[0])
    return first_significant_party_token(value)


def build_indexes(
    cases: list[CaseRecord],
) -> tuple[dict[str, list[int]], dict[str, set[int]], dict[str, set[int]]]:
    exact_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, set[int]] = defaultdict(set)
    anchor_index: dict[str, set[int]] = defaultdict(set)
    for idx, case in enumerate(cases):
        for alias in case.aliases:
            exact_index[alias].append(idx)
            anchor = alias_anchor(alias)
            if anchor:
                anchor_index[anchor].add(idx)
        primary_alias = case.aliases[0] if case.aliases else normalize_title(case.case_id)
        for token in token_set(primary_alias):
            token_index[token].add(idx)
    return exact_index, token_index, anchor_index


def precedent_variants(mention: PrecedentMention) -> list[str]:
    values = {
        mention.precedent_canonical,
        mention.precedent_raw,
        strip_report_citation(mention.precedent_canonical),
        strip_report_citation(mention.precedent_raw),
    }
    variants: set[str] = set()
    for value in values:
        if not value:
            continue
        for strip_date in (True, False):
            normalized = normalize_title(value, strip_date=strip_date)
            if normalized:
                variants.add(normalized)
    return sorted(variants, key=lambda item: (-len(item), item))


def score_alias(precedent: str, alias: str) -> tuple[float, str]:
    if precedent == alias:
        return 1.0, "exact"
    if len(precedent) >= 12 and alias.startswith(precedent):
        return 0.96, "prefix"
    if len(alias) >= 12 and precedent.startswith(alias):
        return 0.94, "reverse_prefix"
    precedent_parties = split_vs(precedent)
    alias_parties = split_vs(alias)
    if precedent_parties and alias_parties:
        p_left, p_right = precedent_parties
        a_left, a_right = alias_parties
        if not anchor_tokens_compatible(p_left, a_left):
            return min(0.84, SequenceMatcher(None, precedent, alias).ratio()), "party_anchor_mismatch"
        left_score = component_similarity(p_left, a_left)
        right_score = component_similarity(p_right, a_right)
        if left_score < 0.80:
            return min(0.84, SequenceMatcher(None, precedent, alias).ratio()), "party_mismatch"
        return (0.72 * left_score) + (0.28 * right_score), "party_fuzzy"
    p_tokens = set(precedent.split())
    a_tokens = set(alias.split())
    jaccard = len(p_tokens & a_tokens) / max(1, len(p_tokens | a_tokens))
    sequence = SequenceMatcher(None, precedent, alias).ratio()
    if jaccard >= 0.80:
        return max(sequence, 0.90 + (jaccard - 0.80) * 0.25), "token_overlap"
    return sequence, "fuzzy"


def score_prefix_alias(precedent: str, alias: str) -> tuple[float, str] | None:
    if len(precedent) >= 12 and alias.startswith(precedent):
        return 0.96, "prefix"
    if len(alias) >= 12 and precedent.startswith(alias):
        return 0.94, "reverse_prefix"
    return None


def candidate_case_indexes(
    variant: str,
    token_index: dict[str, set[int]],
    max_candidates: int,
) -> set[int]:
    tokens = [tok for tok in variant.split() if len(tok) > 1 and tok not in STOP_TOKENS]
    if not tokens:
        return set()
    postings = sorted(
        ((len(token_index.get(tok, ())), tok) for tok in set(tokens)),
        key=lambda item: item[0],
    )
    candidates: set[int] = set()
    for posting_size, token in postings:
        if posting_size == 0:
            continue
        candidates.update(token_index[token])
        if len(candidates) >= max_candidates:
            break
    if len(candidates) > max_candidates:
        return set(sorted(candidates)[:max_candidates])
    return candidates


def prefix_candidate_case_indexes(
    variants: list[str],
    anchor_index: dict[str, set[int]],
    max_candidates: int,
) -> set[int]:
    candidates: set[int] = set()
    for variant in variants:
        anchor = alias_anchor(variant)
        if not anchor:
            continue
        candidates.update(anchor_index.get(anchor, set()))
        if len(candidates) >= max_candidates:
            break
    if len(candidates) > max_candidates:
        return set(sorted(candidates)[:max_candidates])
    return candidates


def passes_filters(
    mention: PrecedentMention,
    target: CaseRecord,
    *,
    allow_self_links: bool,
    require_earlier_target: bool,
) -> bool:
    if not allow_self_links and mention.source_case_id == target.case_id:
        return False
    if require_earlier_target:
        if mention.source_case_year is None or target.case_year is None:
            return False
        if target.case_year >= mention.source_case_year:
            return False
    return True


def resolve_one(
    mention: PrecedentMention,
    cases: list[CaseRecord],
    exact_index: dict[str, list[int]],
    token_index: dict[str, set[int]],
    anchor_index: dict[str, set[int]],
    *,
    allow_self_links: bool,
    require_earlier_target: bool,
    fuzzy_threshold: float,
    enable_fuzzy: bool,
    ambiguity_gap: float,
    max_candidates: int,
) -> tuple[LinkCandidate | None, str]:
    variants = precedent_variants(mention)
    if not variants:
        return None, "no_variants"

    scored: list[tuple[float, str, int, str]] = []
    for variant in variants:
        for case_idx in exact_index.get(variant, []):
            target = cases[case_idx]
            if passes_filters(
                mention,
                target,
                allow_self_links=allow_self_links,
                require_earlier_target=require_earlier_target,
            ):
                scored.append((1.0, "exact", case_idx, variant))

    if not scored:
        if enable_fuzzy:
            candidate_indexes: set[int] = set()
            for variant in variants:
                candidate_indexes.update(
                    candidate_case_indexes(variant, token_index, max_candidates)
                )
        else:
            candidate_indexes = prefix_candidate_case_indexes(
                variants,
                anchor_index,
                max_candidates,
            )
        for case_idx in candidate_indexes:
            target = cases[case_idx]
            if not passes_filters(
                mention,
                target,
                allow_self_links=allow_self_links,
                require_earlier_target=require_earlier_target,
            ):
                continue
            best_score = 0.0
            best_method = "fuzzy"
            for variant in variants:
                for alias in target.aliases:
                    if enable_fuzzy:
                        score, method = score_alias(variant, alias)
                    else:
                        prefix_score = score_prefix_alias(variant, alias)
                        if prefix_score is None:
                            continue
                        score, method = prefix_score
                    if score > best_score:
                        best_score = score
                        best_method = method
            is_prefix_style = best_method in {"prefix", "reverse_prefix"}
            if is_prefix_style or (enable_fuzzy and best_score >= fuzzy_threshold):
                scored.append((best_score, best_method, case_idx, variants[0]))

    if not scored:
        return None, "unmatched"

    scored.sort(key=lambda item: (-item[0], item[2]))
    best_score, best_method, best_idx, _ = scored[0]
    ambiguous = False
    for runner_score, _, runner_idx, _ in scored[1:]:
        if runner_idx != best_idx and (best_score - runner_score) <= ambiguity_gap:
            ambiguous = True
            break

    target = cases[best_idx]
    return (
        LinkCandidate(
            source_case_id=mention.source_case_id,
            source_case_year=mention.source_case_year,
            precedent_raw=mention.precedent_raw,
            precedent_canonical=mention.precedent_canonical,
            precedent_node_key=mention.precedent_node_key,
            target_case_id=target.case_id,
            target_case_year=target.case_year,
            target_file_name=target.file_name,
            match_method=best_method,
            match_score=round(float(best_score), 4),
            ambiguous=ambiguous,
            temporal_relation=temporal_relation(mention.source_case_year, target.case_year),
            mention_count=mention.mention_count,
            sections=mention.sections,
            seen_in_arguments=mention.seen_in_arguments,
            proposed_src_type="precedent",
            proposed_relation="resolves_to_case",
            proposed_dst_type="case",
            proposed_src_key=mention.precedent_node_key,
            proposed_dst_key=case_node_key(target.case_id),
            derived_case_src_key=case_node_key(mention.source_case_id),
            derived_case_relation="cites_case",
            derived_case_dst_key=case_node_key(target.case_id),
        ),
        "matched",
    )


def resolve_all(
    precedents: list[PrecedentMention],
    cases: list[CaseRecord],
    exact_index: dict[str, list[int]],
    token_index: dict[str, set[int]],
    anchor_index: dict[str, set[int]],
    args: argparse.Namespace,
) -> tuple[list[LinkCandidate], list[PrecedentMention], Counter]:
    candidates: list[LinkCandidate] = []
    unmatched: list[PrecedentMention] = []
    status_counts: Counter = Counter()

    for mention in precedents:
        candidate, status = resolve_one(
            mention,
            cases,
            exact_index,
            token_index,
            anchor_index,
            allow_self_links=args.allow_self_links,
            require_earlier_target=args.require_earlier_target,
            fuzzy_threshold=args.fuzzy_threshold,
            enable_fuzzy=args.enable_fuzzy,
            ambiguity_gap=args.ambiguity_gap,
            max_candidates=args.max_candidates,
        )
        status_counts[status] += 1
        if candidate is None:
            unmatched.append(mention)
        else:
            candidates.append(candidate)
            status_counts[f"method:{candidate.match_method}"] += 1
            if candidate.ambiguous:
                status_counts["ambiguous"] += 1
            status_counts[f"temporal:{candidate.temporal_relation}"] += 1
    return candidates, unmatched, status_counts


def write_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    candidates: list[LinkCandidate],
    unmatched: list[PrecedentMention],
    unmatched_sample_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "precedent_case_link_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if candidates:
        with (out_dir / "precedent_case_link_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(candidates[0]).keys()))
            writer.writeheader()
            for row in candidates:
                writer.writerow(asdict(row))

    sample = unmatched[:unmatched_sample_size]
    if sample:
        with (out_dir / "unmatched_precedents_sample.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(sample[0]).keys()))
            writer.writeheader()
            for row in sample:
                writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    cfg_cleaned_case_dir = load_config_cleaned_case_dir(args.config)
    cleaned_case_dir = cfg_cleaned_case_dir or args.cleaned_case_dir
    if not cleaned_case_dir.is_dir():
        raise SystemExit(f"cleaned case dir not found: {cleaned_case_dir}")

    paths = iter_case_paths(cleaned_case_dir, args.limit)
    cases, precedents, skipped = collect_cases_and_precedents(
        paths,
        arguments_only=args.arguments_only,
        precedent_nodes_shared=args.precedent_nodes_shared,
        keep_indian_kanoon_precedents=args.keep_indian_kanoon_precedents,
    )
    exact_index, token_index, anchor_index = build_indexes(cases)
    candidates, unmatched, status_counts = resolve_all(
        precedents,
        cases,
        exact_index,
        token_index,
        anchor_index,
        args,
    )

    unique_source_cases = {p.source_case_id for p in precedents}
    unique_target_cases = {c.target_case_id for c in candidates}
    unique_case_edges = {
        (c.source_case_id, c.target_case_id)
        for c in candidates
        if c.source_case_id != c.target_case_id
    }
    method_counts = Counter(c.match_method for c in candidates)
    temporal_counts = Counter(c.temporal_relation for c in candidates)
    section_counts = Counter()
    for mention in precedents:
        for section in mention.sections.split(";"):
            if section:
                section_counts[section] += 1

    summary = {
        "cleaned_case_dir": str(cleaned_case_dir),
        "dry_run": not args.write,
        "cases_scanned": len(cases),
        "precedent_entity_records": len(precedents),
        "source_cases_with_precedents": len(unique_source_cases),
        "matched_precedent_records": len(candidates),
        "unmatched_precedent_records": len(unmatched),
        "match_rate": round(len(candidates) / len(precedents), 6) if precedents else None,
        "unique_target_cases_resolved": len(unique_target_cases),
        "unique_derived_case_to_case_edges": len(unique_case_edges),
        "ambiguous_matches": sum(1 for c in candidates if c.ambiguous),
        "match_methods": dict(method_counts.most_common()),
        "temporal_relations": dict(temporal_counts.most_common()),
        "precedent_sections": dict(section_counts.most_common()),
        "skipped": dict(skipped),
        "options": {
            "arguments_only": args.arguments_only,
            "allow_self_links": args.allow_self_links,
            "require_earlier_target": args.require_earlier_target,
            "enable_fuzzy": args.enable_fuzzy,
            "fuzzy_threshold": args.fuzzy_threshold,
            "precedent_nodes_shared": args.precedent_nodes_shared,
            "keep_indian_kanoon_precedents": args.keep_indian_kanoon_precedents,
            "limit": args.limit,
        },
    }

    print(json.dumps(summary, indent=2))
    if candidates:
        print("\nTop candidate examples:")
        for candidate in candidates[:10]:
            print(
                f"- {candidate.source_case_id} | {candidate.precedent_canonical!r} "
                f"-> {candidate.target_case_id} "
                f"({candidate.match_method}, score={candidate.match_score})"
            )

    if args.write:
        write_outputs(
            args.out_dir,
            summary,
            candidates,
            unmatched,
            args.unmatched_sample_size,
        )
        print(f"\nWrote new audit files under: {args.out_dir}")
    else:
        print("\nDry run only; pass --write to create new audit files.")


if __name__ == "__main__":
    main()
