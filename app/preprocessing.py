import re
import unicodedata
from typing import Optional

# Contract artifacts: page numbers, exhibit references, signature blocks
PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)
EXHIBIT_RE = re.compile(r"\bEXHIBIT\s+[A-Z0-9]+\b", re.IGNORECASE)
SIGNATURE_BLOCK_RE = re.compile(
    r"(IN WITNESS WHEREOF|SIGNATURE PAGE|EXECUTED AS OF).*$",
    re.IGNORECASE | re.DOTALL
)

# Repeated whitespace, tabs, non-breaking spaces
WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Junk characters common in PDF-extracted legal text
JUNK_CHARS_RE = re.compile(r"[^\x00-\x7F\u2018\u2019\u201c\u201d\u2013\u2014]")

# Table of contents patterns (lots of dots, page refs)
TOC_RE = re.compile(r"^.*\.{4,}.*\d+\s*$", re.MULTILINE)

# All-caps section headers — we keep these but normalize them
ALLCAPS_HEADER_RE = re.compile(r"^([A-Z\s]{10,})$", re.MULTILINE)

# Empty parentheticals and defined-term artifacts
EMPTY_PAREN_RE = re.compile(r"\(\s*\)")

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    return text

def remove_contract_artifacts(text: str) -> str:
    text = PAGE_NUMBER_RE.sub("", text)
    text = TOC_RE.sub("", text)
    text = EMPTY_PAREN_RE.sub("", text)
    match = SIGNATURE_BLOCK_RE.search(text)
    if match:
        text = text[:match.start()]
    return text

def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving paragraph breaks."""
    # Replace tabs and multiple spaces with single space
    text = WHITESPACE_RE.sub(" ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2 (paragraph break)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_lii_text(text: str) -> str:
    """
    LII-specific cleaning.
    LII text is already fairly clean (HTML-parsed), mostly needs
    whitespace normalization and removal of nav artifacts.

    Two noise patterns dominate LII Wex pages and must be stripped before
    embedding, otherwise the trailing category cross-link tags
    ("law and economics", "criminal procedure", …) leak into chunks and
    drag retrieval toward category-name matches rather than substantive
    definitions:
      1. "Read more about <topic>"  — closing CTA link
      2. "[Last reviewed in <month> of <year> by the Wex Definitions Team ]"
         followed by a list of category tags through end of doc.
    Each marker appears at most once per doc, so DOTALL-stripping from
    the marker through end-of-text is safe.
    """
    nav_patterns = [
        r"\[Last reviewed.*",          # strips the review tag + trailing categories
        r"Read more about.*",          # strips the CTA + anything after it
        r"Further Reading.*$",
        r"See also:.*",
        r"Retrieved from.*",
        r"Last updated.*\d{4}",
        r"This article.*encyclop.*",
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def clean_cuad_text(text: str) -> str:
    """
    CUAD-specific cleaning.
    CUAD contracts come from PDFs and have heavier artifacts.
    """
    text = normalize_unicode(text)
    text = remove_contract_artifacts(text)
    text = JUNK_CHARS_RE.sub("", text)
    text = normalize_whitespace(text)
    return text


def clean_document(text: str, source: str) -> Optional[str]:
    """
    Master cleaning function. Routes to source-specific cleaner.
    Returns None if the document is too short to be useful after cleaning.

    Min-length threshold is source-dependent: LII Wex glossary entries are
    short by design (e.g. "A/R is the abbreviation for accounts receivable")
    and remain high-precision retrieval targets, so they get a lower bar
    than full CUAD contracts.
    """
    if not text or not text.strip():
        return None

    source_lower = source.lower() if source else ""
    is_lii = "lii" in source_lower or "cornell" in source_lower

    if "cuad" in source_lower:
        cleaned = clean_cuad_text(text)
        min_words = 50
    elif is_lii:
        cleaned = clean_lii_text(text)
        min_words = 8
    else:
        cleaned = normalize_unicode(text)
        cleaned = normalize_whitespace(cleaned)
        min_words = 50

    if len(cleaned.split()) < min_words:
        return None

    return cleaned
