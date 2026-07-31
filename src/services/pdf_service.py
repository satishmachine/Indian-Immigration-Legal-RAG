"""
services.pdf_service
====================
PDF file synchronization, Act-to-PDF resolution, and statutory section page finder.
"""

from __future__ import annotations

import logging
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Map common Act title keywords to actual PDF filenames in Data_Set
ACT_TO_PDF_MAP: dict[str, str] = {
    "citizenship act": "the_citizenship_act_1955.pdf",
    "citizenship act, 1955": "the_citizenship_act_1955.pdf",
    "citizenship": "the_citizenship_act_1955.pdf",
    "passports act": "passports_act_1967.pdf",
    "passports act, 1967": "passports_act_1967.pdf",
    "passport act": "passports_act_1967.pdf",
    "passport": "passports_act_1967.pdf",
    "emigration act": "THE EMIGRATION ACT, 1983.pdf",
    "emigration act, 1983": "THE EMIGRATION ACT, 1983.pdf",
    "emigration": "THE EMIGRATION ACT, 1983.pdf",
    "foreigners act": "The Immigration and Foreigners Act, 2025.pdf",
    "immigration and foreigners act": "The Immigration and Foreigners Act, 2025.pdf",
    "foreigners act, 1946": "The Immigration and Foreigners Act, 2025.pdf",
    "foreigners rules": "Immigration and Foreigners Rules 2025.pdf",
    "citizenship 2015": "The_Citizenship_(Amendment)_Act,_2015_20012026.pdf",
    "citizenship 2019": "The_Citizenship_(Amendment)_Act,_2019_20012026.pdf",
}

# Cache for page numbers: (pdf_filename, section_number) -> page_number
_PAGE_CACHE: dict[tuple[str, str], int] = {}


def get_project_root() -> Path:
    """Return project root path."""
    return Path(__file__).resolve().parents[2]


def ensure_static_pdfs_synced() -> None:
    """Sync all PDF files from Data_Set/ and uploads/ into static/pdfs/."""
    root = get_project_root()
    static_pdf_dir = root / "static" / "pdfs"
    static_pdf_dir.mkdir(parents=True, exist_ok=True)

    sources = [root / "Data_Set", root / "uploads"]
    copied_count = 0

    for source_dir in sources:
        if not source_dir.exists():
            continue
        for pdf_file in source_dir.glob("*.pdf"):
            dest_file = static_pdf_dir / pdf_file.name
            if not dest_file.exists() or dest_file.stat().st_mtime < pdf_file.stat().st_mtime:
                try:
                    shutil.copy2(pdf_file, dest_file)
                    copied_count += 1
                except Exception as exc:
                    logger.warning("Failed to sync PDF %s: %s", pdf_file.name, exc)

    if copied_count > 0:
        logger.info("Synced %d PDF files to static/pdfs/", copied_count)


def resolve_pdf_filename(
    act_name: str | None,
    pdf_name: str | None = None,
    section_title: str | None = None,
) -> str:
    """
    Resolve Act title, pdf_name metadata, or section_title to exact filename in static/pdfs.
    """
    root = get_project_root()
    static_pdf_dir = root / "static" / "pdfs"

    # 1. Direct pdf_name match if provided
    if pdf_name:
        fname = Path(pdf_name).name
        if (static_pdf_dir / fname).exists():
            return fname
        if (root / "Data_Set" / fname).exists():
            return fname

    combined_str = f"{act_name or ''} {section_title or ''}".lower()

    # 2. If 'rule' or 'rules' is mentioned, target Immigration and Foreigners Rules 2025.pdf
    if "rules" in combined_str or "rule " in combined_str or "rules 2025" in combined_str:
        return "Immigration and Foreigners Rules 2025.pdf"

    if not act_name:
        return "the_citizenship_act_1955.pdf"

    act_lower = act_name.lower().strip()

    # 3. Check ACT_TO_PDF_MAP
    for key, target_pdf in ACT_TO_PDF_MAP.items():
        if key in act_lower:
            return target_pdf

    # 4. Check for match in actual static/pdfs directory files
    if static_pdf_dir.exists():
        for pdf_path in static_pdf_dir.glob("*.pdf"):
            if any(word in pdf_path.name.lower() for word in act_lower.split() if len(word) > 3):
                return pdf_path.name

    # Fallback to default
    return "the_citizenship_act_1955.pdf"


def is_hindi_page(text: str) -> bool:
    """Return True if page is predominantly Hindi text (>80 Devanagari chars or >15% ratio)."""
    if not text:
        return False
    hindi_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    ratio = hindi_chars / len(text)
    return hindi_chars > 80 or ratio > 0.15


def get_section_page_number(
    pdf_filename: str,
    section_number: str | None,
    section_title: str | None = None,
) -> int:
    """
    Locate the exact page number (1-based) in PDF where English section body definition starts.
    Uses section number, section title keyword matching, and skips Hindi, TOC index, and Form pages.
    """
    if not section_number or str(section_number).strip().upper() in ("N/A", "GENERAL", "NONE", ""):
        return 1

    sec_raw = str(section_number).strip()
    sec_num_match = re.search(r"(\d+[A-Z]?)", sec_raw, re.I)
    sec_digits = sec_num_match.group(1) if sec_num_match else sec_raw

    clean_title = section_title or ""
    cache_key = (pdf_filename, sec_digits, clean_title[:30])
    if cache_key in _PAGE_CACHE:
        return _PAGE_CACHE[cache_key]

    root = get_project_root()
    pdf_path = root / "static" / "pdfs" / pdf_filename
    if not pdf_path.exists():
        pdf_path = root / "Data_Set" / pdf_filename

    if not pdf_path.exists():
        return 1

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)

        title_words = [
            w.lower()
            for w in re.findall(r"[A-Za-z]{4,}", clean_title)
            if w.lower() not in (
                "this", "that", "with", "from", "under", "have", "such",
                "shall", "section", "rules", "act", "other", "said", "penalty", "refusal",
            )
        ][:5]

        candidates: list[tuple[int, int]] = []

        for idx in range(num_pages):
            page_text = doc[idx].get_text()

            # Skip Hindi pages (>20 Devanagari characters)
            if is_hindi_page(page_text):
                continue

            text_upper = page_text.upper()

            # Check if true TOC page (has ARRANGEMENT OF SECTIONS or table of contents, without body text like (1) or (a))
            has_body_subsections = bool(re.search(r"\(\d+\)\s+[A-Z]|\([a-z]\)\s+[A-Z]", page_text))
            is_toc_header = any(
                h in text_upper
                for h in [
                    "ARRANGEMENT OF SECTIONS",
                    "ARRANGEMENT OF CLAUSES",
                    "TABLE OF CONTENTS",
                    "LIST OF ABBREVIATIONS",
                ]
            )
            is_toc = is_toc_header or (
                idx < 3
                and not has_body_subsections
                and len(re.findall(r"(?:\n|^)\s*\d+\.\s+[A-Za-z]", page_text)) >= 6
            )

            # Check if Form / Schedule table page header
            is_form_header = bool(
                re.search(r"(?:\n|^)\s*(?:FORM|SCHEDULE|APPENDIX)\s+[I|V|X|\d+]", text_upper)
            )

            # Section Heading match at start of line
            heading_match = bool(
                re.search(
                    r"(?:\n|^)\s*" + re.escape(sec_digits) + r"\.\s*(?:\n|\s+)[A-Za-z]",
                    page_text,
                )
            )
            exact_sec_word = bool(
                re.search(r"(?<!sub-)\bSection\s+" + re.escape(sec_digits) + r"\b", page_text, re.I)
            )

            title_score = sum(1 for w in title_words if w in page_text.lower()) if title_words else 0

            if heading_match or exact_sec_word or title_score >= 2:
                score = 0
                if heading_match:
                    score += 60  # Massive boost for true Section Heading line!
                if exact_sec_word:
                    score += 20
                if title_score > 0:
                    score += (title_score * 10)
                if not is_toc:
                    score += 30
                if not is_form_header:
                    score += 20

                candidates.append((score, idx + 1))

        doc.close()

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            winner_page = candidates[0][1]
            _PAGE_CACHE[cache_key] = winner_page
            return winner_page

    except Exception as exc:
        logger.warning("Error scanning PDF %s for section %s: %s", pdf_filename, sec_digits, exc)

    return 1


def build_pdf_viewer_url(
    act_name: str,
    section_number: str | None = None,
    section_title: str | None = None,
    page_number: int | None = None,
    pdf_name: str | None = None,
) -> str:
    """
    Build web URL targeting pdf_viewer.html in a new web tab.
    """
    ensure_static_pdfs_synced()
    resolved_pdf = resolve_pdf_filename(act_name, pdf_name, section_title)

    # Determine page number with title matching if page_number is missing or 1
    if not page_number or page_number <= 0:
        resolved_page = get_section_page_number(resolved_pdf, section_number, section_title)
    else:
        resolved_page = page_number

    sec_str = str(section_number).strip() if section_number else "1"
    if sec_str.upper() in ("N/A", "NONE"):
        sec_str = "1"

    params = {
        "file": resolved_pdf,
        "section": sec_str,
        "page": str(resolved_page),
        "act": act_name or "Statutory Act",
    }
    if section_title:
        params["title"] = section_title[:60]

    query_str = urllib.parse.urlencode(params)
    return f"/app/static/pdf_viewer.html?{query_str}"
