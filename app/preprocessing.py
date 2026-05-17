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
    """
    # Remove common LII nav text that bleeds into content
    nav_patterns = [
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
    """
    if not text or not text.strip():
        return None

    if "CUAD" in source or "cuad" in source.lower():
        cleaned = clean_cuad_text(text)
    elif "LII" in source or "Cornell" in source:
        cleaned = clean_lii_text(text)
    else:
        # Generic fallback
        cleaned = normalize_unicode(text)
        cleaned = normalize_whitespace(cleaned)

    # Reject documents that are too short after cleaning
    if len(cleaned.split()) < 50:
        return None

    return cleaned
