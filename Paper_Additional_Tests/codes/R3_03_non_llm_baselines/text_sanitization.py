#!/usr/bin/env python3
"""Deterministic final leakage guard for flat-text baselines.

The source document remains the HGT case node's PREAMBLE + FAC + argument text.
The HGT preprocessing has already removed decision/rationale rhetorical roles,
but retained PREAMBLE sentences can still contain publisher/cause-list outcome
markers such as ``[ALLOWED]``. Phrase masking also leaves an observable
``[LEAKAGE_MASK]`` token.

For TF-IDF only, remove both the mask artifact and operative-outcome vocabulary.
This is intentionally conservative: it may remove references to lower-court
outcomes from facts, but it prevents a sparse lexical model from exploiting
decision words that are not available as legitimate predictive evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

MASK_TOKEN_RE = re.compile(r"\[\s*LEAKAGE_MASK\s*\]|\bleakage_mask\b", re.IGNORECASE)

# Inflected operative-decision words. Case-type terms (appeal, petition, bail,
# compensation, conviction) remain because they describe the legal problem;
# words that state what a court did to them do not.
OUTCOME_TERMS = frozenset(
    """
    acquit acquits acquitted acquitting acquittal
    affirm affirms affirmed affirming affirmation
    allow allows allowed allowing
    aside
    award awards awarded awarding
    close closes closed closing closure
    convict convicts convicted convicting
    decree decreed decreeing
    dismiss dismisses dismissed dismissing dismissal
    dispose disposes disposed disposing disposal
    enhance enhances enhanced enhancing enhancement
    fail fails failed failing failure
    grant grants granted granting
    infructuous
    merit merits meritorious
    modify modifies modified modifying modification
    quash quashes quashed quashing
    reduce reduces reduced reducing reduction
    refuse refuses refused refusing refusal
    reject rejects rejected rejecting rejection
    remand remands remanded remanding
    restore restores restored restoring restoration
    reverse reverses reversed reversing reversal
    set sets setting
    succeed succeeds succeeded succeeding success successful unsuccessful
    uphold upholds upheld upholding
    vacate vacates vacated vacating
    withdraw withdraws withdrew withdrawn withdrawing withdrawal
    """.split()
)

_OUTCOME_ALTERNATION = "|".join(
    re.escape(term) for term in sorted(OUTCOME_TERMS, key=lambda value: (-len(value), value))
)
OUTCOME_TOKEN_RE = re.compile(rf"\b(?:{_OUTCOME_ALTERNATION})\b", re.IGNORECASE)
# Covers both "set aside" and common legal variants such as
# "set the impugned order dated 01.01.2020 aside".
SET_ASIDE_RE = re.compile(
    r"\bset(?:\s+[\w.-]+){0,8}\s+aside\b",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


def sanitize_document(text: str) -> str:
    """Remove observable mask artifacts and direct operative-outcome language."""
    cleaned = MASK_TOKEN_RE.sub(" ", str(text or ""))
    cleaned = OUTCOME_TOKEN_RE.sub(" ", cleaned)
    # Token removal can bring formerly separated words together (for example,
    # "set [outcome word] aside"), so phrase removal must run afterwards.
    cleaned = SET_ASIDE_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def audit_documents(source: Iterable[str], sanitized: Iterable[str]) -> dict[str, int]:
    """Return corpus audit counts and fail if forbidden text survives."""
    counts: Counter[str] = Counter()
    source_count = sanitized_count = 0
    for before, after in zip(source, sanitized, strict=True):
        source_count += 1
        counts["source_mask_tokens"] += len(MASK_TOKEN_RE.findall(before))
        counts["source_outcome_terms"] += len(OUTCOME_TOKEN_RE.findall(before))
        counts["source_set_aside_phrases"] += len(SET_ASIDE_RE.findall(before))
        counts["sanitized_mask_tokens"] += len(MASK_TOKEN_RE.findall(after))
        counts["sanitized_outcome_terms"] += len(OUTCOME_TOKEN_RE.findall(after))
        counts["sanitized_set_aside_phrases"] += len(SET_ASIDE_RE.findall(after))
        counts["characters_removed"] += len(before) - len(after)
        counts["empty_after_sanitization"] += int(not after)
        sanitized_count += 1

    if source_count != sanitized_count:
        raise AssertionError(f"source/sanitized length mismatch: {source_count} != {sanitized_count}")
    forbidden_survivors = (
        counts["sanitized_mask_tokens"]
        + counts["sanitized_outcome_terms"]
        + counts["sanitized_set_aside_phrases"]
    )
    if forbidden_survivors:
        raise AssertionError(f"{forbidden_survivors} forbidden leakage tokens survived sanitization")
    counts["documents"] = source_count
    return dict(counts)


def assert_vocabulary_clean(feature_names: Iterable[str]) -> None:
    """Fail if a fitted TF-IDF vocabulary contains any forbidden feature."""
    bad = []
    for feature in feature_names:
        value = str(feature)
        if MASK_TOKEN_RE.search(value) or OUTCOME_TOKEN_RE.search(value) or SET_ASIDE_RE.search(value):
            bad.append(value)
            if len(bad) == 20:
                break
    if bad:
        raise AssertionError(f"forbidden leakage features entered TF-IDF vocabulary: {bad}")
