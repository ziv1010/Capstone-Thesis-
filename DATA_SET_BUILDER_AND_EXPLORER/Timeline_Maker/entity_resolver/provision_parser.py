"""
Deterministic provision parser.

Goal: collapse string variants like
    "section 482"   "Section 482"   "S. 482"   "Sec. 482"   "u/s 482"   "482"
    "section 498a"  "section 498-a"  "498a"  "498-A"  "S. 498A"
    "article 226"   "Art. 226"   "art 226"

into a canonical id like  section:482  or  section:498a  or  article:226 .

Sub-clauses ARE preserved as part of the id — section 156(3) is canonically
DISTINCT from section 156, and section 4(1) is distinct from section 4. The
user explicitly asked that even small differences be preserved because they
can carry different legal meaning.

Multi-section references separated by '/' (e.g. "section 3/4", "7/16") are
preserved as a sorted comma-joined list:  section:3,4  or  section:7,16

Forms ('section' / 'article' / 'chapter' / 'part' / 'rule' / 'sub-section') are
kept distinct — section:21 != article:21.

If we cannot parse, we return (None, None) and the caller should leave the
entity as-is (its own canonical).
"""
from __future__ import annotations

import re

WHITESPACE_RE = re.compile(r"\s+")
# Replace "Sec.4" -> "Sec 4" by inserting space when a dot sits between alphanumerics.
DOT_BETWEEN_ALNUM = re.compile(r"([A-Za-z0-9])\.([A-Za-z0-9])")
# After form keyword, allow whitespace, dots, hyphens, or nothing as the separator.
SEP = r"[\s\.\-]*"

# Form keywords (longest/specific first to avoid 's' matching inside 'section')
FORM_PATTERNS = [
    (re.compile(rf"^\s*sub[\s\-]*sections?{SEP}", re.I), "subsection"),
    (re.compile(rf"^\s*sub[\s\-]*rules?{SEP}", re.I), "subrule"),
    (re.compile(rf"^\s*sub[\s\-]*clauses?{SEP}", re.I), "subclause"),
    (re.compile(rf"^\s*sections?{SEP}", re.I), "section"),
    (re.compile(rf"^\s*secs\.?{SEP}", re.I), "section"),
    (re.compile(rf"^\s*sec\.?{SEP}", re.I), "section"),
    (re.compile(rf"^\s*ss\.?{SEP}", re.I), "section"),
    (re.compile(rf"^\s*s\.?{SEP}(?=\d)", re.I), "section"),  # require digit after to avoid 's' grabbing words
    (re.compile(r"^\s*u\s*/\s*s\s+", re.I), "section"),
    (re.compile(rf"^\s*under\s+sections?{SEP}", re.I), "section"),
    (re.compile(rf"^\s*articles?{SEP}", re.I), "article"),
    (re.compile(rf"^\s*art\.?{SEP}", re.I), "article"),
    (re.compile(rf"^\s*chapters?{SEP}", re.I), "chapter"),
    (re.compile(rf"^\s*parts?{SEP}", re.I), "part"),
    (re.compile(rf"^\s*rules?{SEP}", re.I), "rule"),
    (re.compile(rf"^\s*orders?{SEP}", re.I), "order"),
    (re.compile(rf"^\s*clauses?{SEP}", re.I), "clause"),
    (re.compile(rf"^\s*schedules?{SEP}", re.I), "schedule"),
    (re.compile(rf"^\s*regulations?{SEP}", re.I), "regulation"),
]

# Roman numeral set for chapter/part forms
ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.I)

# A "number token" — digits with optional sub-letter and optional sub-clause
# Examples: 482 / 498a / 498A / 498-a / 304b / 300-A / 156(3) / 4(1) / 3g(5) / 28-a
NUM_TOKEN_RE = re.compile(
    r"""
    (?P<num>\d+)                       # main number
    (?:[\s\-]*(?P<sub>[a-z]))?         # optional sub-letter (a/b/c..)
    (?:\s*\(\s*(?P<clause>[\d\w]+)\s*\)
        (?:\s*\(\s*(?P<sub2>[\d\w]+)\s*\))? )?  # optional 1-2 sub-clauses in parens
    """,
    re.I | re.X,
)


def _norm_lower_compact(text: str) -> str:
    s = text.lower().strip()
    s = DOT_BETWEEN_ALNUM.sub(r"\1 \2", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s


def _strip_form(text: str) -> tuple[str | None, str]:
    """Strip the leading form keyword (section/article/...). Return (form, rest)."""
    for rx, form in FORM_PATTERNS:
        m = rx.match(text)
        if m:
            return form, text[m.end():].strip()
    # no explicit form keyword — treat bare numbers as 'section' by default
    return None, text


def _format_number_token(num: str, sub: str | None, clause: str | None, sub2: str | None) -> str:
    out = num
    if sub:
        out += sub.lower()
    if clause is not None:
        out += f"({clause})"
        if sub2 is not None:
            out += f"({sub2})"
    return out


def parse_provision(raw_text: str) -> tuple[str | None, str | None]:
    """
    Parse a raw provision mention. Returns (canonical_id, display_name) or
    (None, None) if unparseable.

    canonical_id is of the form  '<form>:<num>[a][(c)][(s)][,<num>...]'
    e.g.:
        'section:482'
        'section:498a'
        'section:156(3)'
        'section:3,4'      (from "section 3/4")
        'article:226'
        'chapter:v'
        'part:iii'
        'subsection:1'
    """
    if not raw_text or not raw_text.strip():
        return None, None
    s = _norm_lower_compact(raw_text)

    form, rest = _strip_form(s)

    # Roman-numeral chapters / parts: "chapter v", "part iii"
    if form in {"chapter", "part"} and ROMAN_RE.match(rest.strip()):
        roman = rest.strip().lower()
        return f"{form}:{roman}", f"{form.capitalize()} {roman.upper()}"

    # Sub-section/sub-rule patterns: "sub-section (1)" -> subsection:1
    if form in {"subsection", "subrule", "subclause"}:
        m = re.match(r"^\s*\(\s*([\d\w]+)\s*\)\s*$", rest)
        if m:
            return f"{form}:{m.group(1).lower()}", f"{form.capitalize()} {m.group(1)}"
        # fallthrough — try the normal numeric path

    # No form found and rest isn't a number — give up
    if form is None:
        # Bare number-like? Default to 'section'
        if re.match(r"^[\d\(]", rest):
            form = "section"
        else:
            return None, None

    # Handle slash-separated multi-section references: "3/4", "7/16/"
    parts = re.split(r"[\\/&]+|\band\b|\sand\s", rest)
    parts = [p.strip(" ,.;:") for p in parts if p.strip(" ,.;:")]

    # Each part should match NUM_TOKEN_RE; if any one doesn't, drop it.
    nums: list[str] = []
    for p in parts:
        # If this looks like 'subsection (1) of section 2', skip — too compound to canonicalize cleanly.
        if " of " in p:
            return None, None
        m = NUM_TOKEN_RE.match(p)
        if not m:
            continue
        token = _format_number_token(
            m.group("num"),
            m.group("sub"),
            m.group("clause"),
            m.group("sub2"),
        )
        nums.append(token)

    if not nums:
        return None, None

    # Sort multi-number references for canonical stability (lexicographic on numeric value first)
    def sort_key(t: str) -> tuple:
        m = re.match(r"^(\d+)(.*)$", t)
        return (int(m.group(1)) if m else 0, m.group(2) if m else t)

    nums_sorted = sorted(nums, key=sort_key)
    canonical_id = f"{form}:" + ",".join(nums_sorted)

    if len(nums_sorted) == 1:
        display = f"{form.capitalize()} {nums_sorted[0]}"
    else:
        display = f"{form.capitalize()}s " + "/".join(nums_sorted)

    return canonical_id, display


# ----- self-test (run: python provision_parser.py) -----
if __name__ == "__main__":
    cases = [
        "section 482",
        "Section 482",
        "S. 482",
        "Sec. 482",
        "u/s 482",
        "482",
        "section 498a",
        "section 498-a",
        "498a",
        "498-A",
        "Sections 498A",
        "ss. 3",
        "section 156(3)",
        "section 156 (3)",
        "section 3/4",
        "section 7/16",
        "section 3(g)",
        "section 3g(5)",
        "section 3g (5)",
        "article 226",
        "art. 226",
        "Article 21",
        "Articles 14",
        "chapter v",
        "Chapter XIV",
        "part iii",
        "rule 3",
        "sub-rule (2)",
        "sub-section (1)",
        "section 28-a",
        "section 304-b",
        "120-b",
        "120b",
        "section 120-b",
        "article 300a",
        "article 300-a",
    ]
    for c in cases:
        cid, disp = parse_provision(c)
        print(f"{c!r:40s} -> {cid!r:30s} {disp!r}")
