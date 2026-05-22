"""
Deterministic precedent resolver based on citation extraction.

Strategy:
  1. Normalise the raw text (lowercase, collapse whitespace, strip punctuation
     except parens/digits/slashes used by citations).
  2. Try to extract one or more citation tokens.
  3. If at least one citation token is extracted, the canonical key is the
     SET of citation tokens, sorted+joined. Two precedents share a canonical
     iff they share at least one identical citation token (and we use a
     transitive merge in the pipeline pass).
  4. If no citation can be extracted, return (None, None) — caller leaves
     the precedent as its own canonical (no fuzzy merging here, per the
     project's high-precision rule).

Citation patterns recognised (case-insensitive):
  -  YYYY  V  SCC  P             "2012 10 scc 303"
  -  YYYY  V  SCR  P
  -  YYYY  V  ACJ  P             "2009 acj 421" (volume optional for some reporters)
  -  YYYY  V  ACC  P
  -  YYYY  V  CRIME  P
  -  YYYY  V  ELT  P
  -  YYYY  V  PLJR  P
  -  YYYY  V  GLH  P
  -  YYYY  V  ADJ  P
  -  YYYY  SUPP  V  SCC  P       "1992 supp 1 scc 335"
  -  AIR  YYYY  COURT  P         "air 1965 sc 444", "air 1995 sc 755"
  -  YYYY  INSC  P                "2022 insc 690"
  -  YYYY  SCC  ONLINE  COURT  P  "2020 scc online sc 98"
  -  YYYY  LAW  SUIT  COURT  P    "2018 law suit sc 303"
  -  (YYYY)  V  SCC  P            parenthesized year is normalised

We deliberately avoid over-fuzzy matching. The user's rule: "even the
smallest difference can have a completely different meaning."
"""
from __future__ import annotations

import re
import string

WHITESPACE_RE = re.compile(r"\s+")
# Light normalization: keep digits, letters, parens, hyphen, slash; replace others with space
KEEP_CHARS = set(string.ascii_lowercase + string.digits + "()-/ ")


def _light_normalize(text: str) -> str:
    s = text.lower()
    s = "".join(ch if ch in KEEP_CHARS else " " for ch in s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    # remove parens around standalone digit groups, e.g. "(2009)" -> "2009", "(3)" -> "3"
    s = re.sub(r"\(\s*(\d+)\s*\)", r"\1", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


# Reporter / court abbreviations recognised in citations
REPORTER_TOKENS = (
    "scc", "scr", "acj", "acc", "crime", "elt", "pljr", "glh", "adj",
    "mlj", "mplj", "mh lj", "ad delhi", "ad", "all", "cal", "del", "delhi",
    "mp", "kar", "mad", "bom", "pat", "raj", "p&h", "ph", "hp", "sik",
    "ker", "guj", "hyd", "jh", "ori", "tn", "up", "ap", "ts", "uk", "ilr",
    "scc cri", "supreme court cases", "scc online sc", "law suit sc", "insc",
    "scale", "ac", "supp",
)
# We compile a single big regex with named alternatives.
# The general shape is: <YEAR> <NUM>? <REPORTER...> <PAGE>
# Below we enumerate the high-confidence patterns one by one.

# These match a citation in the normalized text. Each yields a token like 'scc:2012:10:303'.
CITATION_REGEXES: list[tuple[str, re.Pattern]] = [
    # YYYY  insc  P
    ("insc", re.compile(r"\b(?P<year>\d{4})\s+insc\s+(?P<page>\d+)\b")),
    # AIR YYYY <court> P  (court is 2-4 lowercase tokens)
    ("air", re.compile(r"\bair\s+(?P<year>\d{4})\s+(?P<court>(?:sc|all|cal|del|delhi|mp|kar|mad|bom|pat|raj|p&h|ph|hp|sik|ker|guj|hyd|jh|ori|tn|up|ap|ts|uk|ilr))\s+(?P<page>\d+)\b")),
    # YYYY  V  scc cri  P
    ("scc_cri", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*scc\s+cri\s+(?P<page>\d+)\b")),
    # YYYY  scc cri  P (without vol)
    ("scc_cri", re.compile(r"\b(?P<year>\d{4})\s+scc\s+cri\s+(?P<page>\d+)\b")),
    # YYYY  V  scc  P  (V is volume)
    ("scc", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)\s+scc\s+(?P<page>\d+)\b")),
    # YYYY  V  supreme court cases  P
    ("scc", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)\s+supreme court cases\s+(?P<page>\d+)\b")),
    # YYYY  supp  V  scc  P
    ("scc_supp", re.compile(r"\b(?P<year>\d{4})\s+supp\s+(?P<vol>\d+)\s+scc\s+(?P<page>\d+)\b")),
    # YYYY  scc online <court>  P
    ("scc_online", re.compile(r"\b(?P<year>\d{4})\s+scc online\s+(?P<court>sc|all|cal|del|delhi|mp|kar|mad|bom|pat|raj|hp|ker|guj)\s+(?P<page>\d+)\b")),
    # YYYY  law suit  <court>  P
    ("law_suit", re.compile(r"\b(?P<year>\d{4})\s+law suit\s+(?P<court>sc|all|cal|del|delhi|mp|kar|mad|bom|pat|raj|hp|ker|guj)\s+(?P<page>\d+)\b")),
    # YYYY  V  scr  P
    ("scr", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)\s+scr\s+(?P<page>\d+)\b")),
    # YYYY  V  acj  P
    ("acj", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*acj\s+(?P<page>\d+)\b")),
    # YYYY  acj  P  (without vol)
    ("acj", re.compile(r"\b(?P<year>\d{4})\s+acj\s+(?P<page>\d+)\b")),
    # YYYY  V  acc  P
    ("acc", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*acc\s+(?P<page>\d+)\b")),
    # YYYY  V  elt  P
    ("elt", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*elt\s+(?P<page>\d+)\b")),
    # YYYY  V  pljr  P
    ("pljr", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*pljr\s+(?P<page>\d+)\b")),
    # YYYY  V  glh  P
    ("glh", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*glh\s+(?P<page>\d+)\b")),
    # YYYY  V  adj  P
    ("adj", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*adj\s+(?P<page>\d+)\b")),
    # YYYY  V  ad delhi  P
    ("ad_delhi", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*ad delhi\s+(?P<page>\d+)\b")),
    # YYYY  V  scale  P
    ("scale", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*scale\s+(?P<page>\d+)\b")),
    # YYYY  V  crime  P
    ("crime", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*crime\s+(?P<page>\d+)\b")),
    # YYYY  V  mplj / mlj  P
    ("mplj", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*mplj\s+(?P<page>\d+)\b")),
    ("mlj", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*mlj\s+(?P<page>\d+)\b")),
    # YYYY  V  ilr <court>  P
    ("ilr", re.compile(r"\bilr\s+(?P<year>\d{4})\s+(?P<court>sc|all|cal|del|delhi|mp|kar|mad|bom|pat|raj|hp|ker|guj)\s+(?P<page>\d+)\b")),
    # YYYY  V  ac  P (UK appeals)
    ("ac", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*ac\s+(?P<page>\d+)\b")),
    # foreign style: "1970 ac 467" handled by ac above

    # YYYY  V  acrc  P (e.g. "2022 120 acrc 392")
    ("acrc", re.compile(r"\b(?P<year>\d{4})\s+(?P<vol>\d+)?\s*acrc\s+(?P<page>\d+)\b")),
]


def _make_token(label: str, year: str, vol: str | None, page: str, court: str | None = None) -> str:
    parts = [label, year]
    if court:
        parts.append(court)
    if vol:
        parts.append(vol)
    parts.append(page)
    return ":".join(parts)


def extract_citations(raw_text: str) -> list[str]:
    """Return all citation tokens extracted from the raw text. May be empty."""
    s = _light_normalize(raw_text)
    if not s:
        return []
    # Glue runs like "20063" (typo for "2006 3") that we can't safely fix — leave them.
    found: list[str] = []
    seen: set[str] = set()
    for label, rx in CITATION_REGEXES:
        for m in rx.finditer(s):
            year = m.group("year")
            try:
                vol = m.group("vol")
            except IndexError:
                vol = None
            try:
                court = m.group("court")
            except IndexError:
                court = None
            page = m.group("page")
            tok = _make_token(label, year, vol, page, court)
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


def resolve_precedent(raw_text: str) -> tuple[str | None, str | None, list[str]]:
    """Resolve a precedent. Returns (canonical_id, display_name, citations).

    canonical_id is None when no citation can be extracted (caller keeps
    the precedent as its own node, identified by its normalised raw text).
    Otherwise canonical_id is the sorted+joined citation set, and the
    display_name is the original raw text (a representative is chosen later
    in the pipeline pass).
    """
    citations = extract_citations(raw_text)
    if not citations:
        return None, None, []
    citations_sorted = sorted(set(citations))
    canonical_id = "precedent:" + "|".join(citations_sorted)
    return canonical_id, raw_text.strip(), citations_sorted


# self-test
if __name__ == "__main__":
    cases = [
        "gian singh vs state of punjab 2012 10 scc 303",
        "gian singh v state of punjab 2012 10 scc 303",
        "gian singh vs state of punjab anr 2012 10 scc 303",
        "Gian Singh v. State of Punjab and Another (2012) 10 SCC 303",
        "ratan lal vs state of punjab air 1965 sc 444",
        "rattan lal v state of punjab air 1965 sc 444",
        "satender kumar antil vs central bureau of investigation and ors 2022 insc 690",
        "satendra kumar antil vs cbi",
        "1992 supp 1 scc 335",
        "state of haryana v bhajan lal 1992 supp 1 scc 335",
        "ved prakash vs state of haryana 1981 1 scc 447 air 1981 sc 643",
        "sushila aggarwal vs state nct of delhi 2020 scc online sc 98",
        "sushila aggarwal and others vs state nct of delhi and another 2020 5 scc 1",
        "baker vs willoughby 1970 ac 467",
        "baker v willoughby 1970 ac 467",
        "no citation in this name",
        "pranay sethi",
        "national insurance company ltd vs pranay sethi ors",
        "mangla ram vs oriental insurance co ltd ors 2018 law suit sc 303",
    ]
    for c in cases:
        cid, disp, cites = resolve_precedent(c)
        print(f"{c[:55]!r:60s} -> {cid}")
