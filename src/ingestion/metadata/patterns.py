"""
src.ingestion.metadata.patterns
================================
Compiled regular-expression patterns for Indian legal document parsing.

All patterns are compiled once at module import time with re.compile() so
they are reused across extractions without repeated compilation overhead.

Naming convention
-----------------
* ``RE_<FIELD>``     — matches a single primary field (act name, section, etc.)
* ``RE_<FIELD>_ALT`` — alternative / secondary pattern for the same field
* ``PAT_<GROUP>``    — tuple of patterns that are tried in order

Design notes
------------
* All patterns use verbose mode (re.VERBOSE) for readability and comments.
* Named groups (``(?P<name>...)``) are used wherever the matched sub-value
  will be extracted.
* Patterns are ordered from most-specific to least-specific within each group.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Act Name & Year
# ---------------------------------------------------------------------------

RE_ACT_NAME_FULL = re.compile(
    r"""
    (?:The\s+)?                         # optional leading "The"
    (?P<act_name>
        [A-Z][A-Za-z\s\(\),\-'&]+?     # Act name (title-case words)
        (?:Act|Code|Order|Ordinance|Rules?|Regulations?)  # mandatory keyword
        (?:\s+\(Amendment\))?           # optional Amendment suffix
    )
    ,?\s*
    (?P<year>\d{4})                     # year
    """,
    re.VERBOSE | re.MULTILINE,
)

RE_ACT_NAME_SHORT = re.compile(
    r"""
    (?P<act_name>
        [A-Z][A-Za-z\s\-'&]+?
        (?:Act|Code|Order|Ordinance|Rules?|Regulations?)
    )
    \s*,?\s*
    (?P<year>1[89]\d{2}|20\d{2})       # 1800-2099
    """,
    re.VERBOSE | re.MULTILINE,
)

RE_ACT_NUMBER = re.compile(
    r"""
    Act\s+No\.?\s*
    (?P<act_number>\d+)
    \s+of\s+
    (?P<year>\d{4})
    """,
    re.VERBOSE | re.IGNORECASE,
)

PAT_ACT_NAME = (RE_ACT_NAME_FULL, RE_ACT_NAME_SHORT)

# ---------------------------------------------------------------------------
# Chapter
# ---------------------------------------------------------------------------

RE_CHAPTER_ROMAN = re.compile(
    r"""
    CHAPTER\s+
    (?P<chapter_number>[IVXLCDM]+)      # Roman numeral
    (?:\s*[:\-–]\s*
       (?P<chapter_name>[A-Z][^\n]{3,80}))?  # optional title after colon/dash
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

RE_CHAPTER_ARABIC = re.compile(
    r"""
    CHAPTER\s+
    (?P<chapter_number>\d+)             # Arabic numeral
    (?:\s*[:\-–]\s*
       (?P<chapter_name>[A-Z][^\n]{3,80}))?
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

PAT_CHAPTER = (RE_CHAPTER_ROMAN, RE_CHAPTER_ARABIC)

# ---------------------------------------------------------------------------
# Section / Sub-section Number & Title
# ---------------------------------------------------------------------------

RE_SECTION_WITH_TITLE = re.compile(
    r"""
    ^\s*                                # leading whitespace
    (?P<section_number>
        \d+[A-Z]?                       # e.g. 3, 12A
        (?:\(\d+\))?                    # optional sub: (1)
        (?:\([a-z]\))?                  # optional clause: (a)
    )
    \.?\s+                              # period + space
    (?P<section_title>
        [A-Z][A-Za-z\s,\(\)\-'&]{3,120}  # title (starts uppercase)
    )
    (?:\.\s*—|\s*\.—|\s*—|\.\-)         # em-dash or period-dash typical in Indian acts
    """,
    re.VERBOSE | re.MULTILINE,
)

RE_SECTION_SIMPLE = re.compile(
    r"""
    \bSection\s+
    (?P<section_number>\d+[A-Z]?(?:\(\d+\))?)
    (?:\s+[Oo]f\s+)?
    """,
    re.VERBOSE,
)

RE_SUB_SECTION = re.compile(
    r"""
    ^\s*
    \((?P<sub_section>[1-9]\d?|[a-z]{1,2}|[ivx]+)\)  # (1), (a), (ii)
    \s+
    (?P<text>[A-Z].{0,200})             # content of sub-section
    """,
    re.VERBOSE | re.MULTILINE,
)

PAT_SECTION = (RE_SECTION_WITH_TITLE, RE_SECTION_SIMPLE)

# ---------------------------------------------------------------------------
# Page Number
# ---------------------------------------------------------------------------

RE_PAGE_NUMBER_FOOTER = re.compile(
    r"""
    (?:^|\f)                            # start of line or form-feed
    \s*
    (?:Page\s+)?
    (?P<page_number>\d+)
    \s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

RE_PAGE_NUMBER_HEADER = re.compile(
    r"""
    ^\s*\[?\s*
    (?P<page_number>\d+)
    \s*\]?\s*$
    """,
    re.VERBOSE | re.MULTILINE,
)

PAT_PAGE = (RE_PAGE_NUMBER_FOOTER, RE_PAGE_NUMBER_HEADER)

# ---------------------------------------------------------------------------
# Keywords  (seed vocabulary for Indian immigration law domain)
# ---------------------------------------------------------------------------

# Domain keyword sets — checked with whole-word matching
KEYWORD_SEED_SETS: dict[str, list[str]] = {
    "citizenship": [
        "citizenship", "citizen", "naturalisation", "naturalization",
        "domicile", "stateless", "renunciation", "deprivation",
        "registration of citizens", "citizenship certificate",
    ],
    "immigration": [
        "immigration", "immigrant", "port of entry", "border control",
        "immigration officer", "entry permit", "arrival card",
        "immigration clearance",
    ],
    "emigration": [
        "emigration", "emigrant", "emigration clearance", "ecr",
        "emigration check required", "protected area",
        "foreign employment",
    ],
    "passport": [
        "passport", "travel document", "diplomatic passport",
        "official passport", "emergency certificate",
        "passport authority", "passport officer",
        "passport application",
    ],
    "visa": [
        "visa", "tourist visa", "student visa", "work visa",
        "visa on arrival", "e-visa", "visa extension",
        "multiple entry", "single entry",
    ],
    "foreigners": [
        "foreigner", "foreign national", "alien", "frro",
        "foreigners registration", "residential permit",
        "prohibited area", "restricted area", "deportation",
        "expulsion", "internment",
    ],
    "penalty": [
        "penalty", "offence", "fine", "imprisonment", "punishable",
        "conviction", "sentence", "prosecution",
    ],
    "authority": [
        "central government", "state government", "registrar general",
        "passport authority", "immigration officer",
        "foreigners tribunal",
    ],
}

# Flat set of all seeds for quick O(1) lookup
ALL_KEYWORD_SEEDS: frozenset[str] = frozenset(
    kw for seeds in KEYWORD_SEED_SETS.values() for kw in seeds
)

# Compiled whole-word pattern for each keyword (built lazily on first use)
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {}


def get_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Return a compiled whole-word regex pattern for *keyword*."""
    if keyword not in _KEYWORD_PATTERNS:
        escaped = re.escape(keyword)
        _KEYWORD_PATTERNS[keyword] = re.compile(
            rf"\b{escaped}\b", re.IGNORECASE
        )
    return _KEYWORD_PATTERNS[keyword]


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

RE_DEFINITION_MEANS = re.compile(
    r"""
    (?:^|\n)\s*
    [\u201c\u201d"]?                      # optional opening quote (curly or straight)
    (?P<term>[A-Za-z][A-Za-z\s\-']{1,60}?)  # the defined term
    [\u201c\u201d"]?\s*
    (?:means|includes|shall\s+(?:mean|include)|refers?\s+to)
    \s+
    (?P<definition>[^;\.]{10,500})      # definition body
    [;.]                                # terminator
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

RE_DEFINITION_QUOTE_DASH = re.compile(
    r"""
    (?:^|\n)\s*
    [\u201c\u201d"]
    (?P<term>[A-Za-z][A-Za-z\s\-']{1,60}?)
    [\u201c\u201d"]
    \s*
    [\u2014\-]\s*
    (?P<definition>[^;\.]{10,500})
    [;.]
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

RE_DEFINITION_SECTION_REF = re.compile(
    r"""
    \((?P<ref>[a-z]{1,3}|\d+)\)        # section reference like (a), (ii), (1)
    \s+
    [\u201c\u201d"]?
    (?P<term>[A-Za-z][A-Za-z\s\-']{1,60}?)
    [\u201c\u201d"]?
    \s+means\s+
    (?P<definition>[^;\.]{10,300})
    [;.]
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

PAT_DEFINITION = (RE_DEFINITION_MEANS, RE_DEFINITION_QUOTE_DASH, RE_DEFINITION_SECTION_REF)

# ---------------------------------------------------------------------------
# Authorities
# ---------------------------------------------------------------------------

AUTHORITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "central_government",
        re.compile(r"\bCentral\s+Government\b", re.IGNORECASE),
    ),
    (
        "state_government",
        re.compile(r"\bState\s+Government\b", re.IGNORECASE),
    ),
    (
        "registrar_general",
        re.compile(r"\bRegistrar[- ]General\b", re.IGNORECASE),
    ),
    (
        "passport_authority",
        re.compile(r"\bPassport\s+(?:Authority|Officer|Office)\b", re.IGNORECASE),
    ),
    (
        "immigration_officer",
        re.compile(r"\bImmigration\s+Officer\b", re.IGNORECASE),
    ),
    (
        "frro",
        re.compile(
            r"\b(?:FRRO|Foreigners['\s]+Regional['\s]+Registration['\s]+Officer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tribunal",
        re.compile(r"\bForeigners\s+Tribunal\b", re.IGNORECASE),
    ),
    (
        "court",
        re.compile(
            r"\b(?:High\s+Court|Supreme\s+Court|District\s+Court|Sessions\s+Court)\b",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Penalty
# ---------------------------------------------------------------------------

RE_PENALTY_IMPRISONMENT_FINE = re.compile(
    r"""
    (?P<raw>
        (?:punishable|liable|sentenced?)\s+(?:to|with)\s+
        (?:
            (?:imprisonment[^,\.]{0,80})?       # imprisonment term
            (?:,?\s*(?:or|and)\s*)?
            (?:(?:a\s+)?fine[^,\.]{0,80})?      # fine amount
        )
    )
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)

RE_PENALTY_YEARS = re.compile(
    r"""
    (?P<years>\d+(?:\.\d+)?)\s*year[s]?  # e.g. 3 years
    |
    (?P<months>\d+)\s*month[s]?           # e.g. 6 months
    """,
    re.VERBOSE | re.IGNORECASE,
)

RE_PENALTY_FINE_AMOUNT = re.compile(
    r"""
    (?:Rs\.?|INR|₹)\s*
    (?P<amount>[\d,]+(?:\.\d+)?)         # e.g. 10,000 or 50000.00
    |
    (?P<amount_words>                    # written out amounts
        (?:one|two|three|five|ten|twenty|fifty|hundred|thousand|lakh|crore)
        (?:\s+(?:thousand|lakh|crore|hundred))*
    )
    \s+rupees?
    """,
    re.VERBOSE | re.IGNORECASE,
)

PAT_PENALTY = (RE_PENALTY_IMPRISONMENT_FINE,)

# ---------------------------------------------------------------------------
# Cross References & Related Sections
# ---------------------------------------------------------------------------

RE_SECTION_REF_SAME_ACT = re.compile(
    r"""
    \bsection\s+
    (?P<section>\d+[A-Z]?(?:\s*\(\d+\))?)  # section number with optional sub
    (?:\s+of\s+this\s+Act)?                # optional "of this Act"
    """,
    re.VERBOSE | re.IGNORECASE,
)

RE_CROSS_REF_EXTERNAL = re.compile(
    r"""
    \b(?:section|rule|article|schedule|clause)\s+
    (?P<number>\d+[A-Z]?(?:\s*\(\d+\))?)
    \s+of\s+(?:the\s+)?
    (?P<act_name>
        [A-Z][A-Za-z\s\(\),\-'&]+?
        (?:Act|Code|Order|Ordinance|Rules?|Regulations?)
    )
    (?:,?\s*(?P<year>\d{4}))?
    """,
    re.VERBOSE | re.MULTILINE,
)

RE_SCHEDULE_REF = re.compile(
    r"""
    \b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|
        (?:Eleventh|Twelfth)|
        [A-Z][a-z]+)
    \s+Schedule\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

PAT_CROSS_REF = (RE_CROSS_REF_EXTERNAL, RE_SCHEDULE_REF)
PAT_RELATED_SECTION = (RE_SECTION_REF_SAME_ACT,)
