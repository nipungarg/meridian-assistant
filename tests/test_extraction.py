"""Tests for PDF extraction and the symbol-font (check/cross) normalization."""
from pathlib import Path

import pytest

from meridian.config import get_settings
from meridian.ingestion.pdf_extract import extract_pdf, normalize_coverage_cell

FILES = get_settings().files_dir


@pytest.mark.parametrize("raw,expected", [
    ("3", "yes"),            # Wingdings check renders as "3"
    ("\u2713", "yes"),       # actual check mark
    ("7", "no"),             # Wingdings cross renders as "7"
    ("\u2717", "no"),        # actual cross
    ("Sub-contracted", "sub-contracted"),
    ("Pending (Q2)", "pending"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_normalize_coverage_cell(raw, expected):
    assert normalize_coverage_cell(raw) == expected


def test_service_area_header_parsed():
    doc = extract_pdf(FILES / "01_service_area_north.pdf")
    assert "North Region" in doc.doc_title
    assert doc.doc_type == "SERVICE AREA"
    assert doc.version == "v2.1"
    assert doc.updated_date == "2025-11-01"


def test_boilerplate_stripped():
    doc = extract_pdf(FILES / "03_hvac_pricing.pdf")
    # the repeating "INTERNAL USE ONLY ... Page N" footer must be gone
    assert "INTERNAL USE ONLY" not in doc.cleaned_text
    assert "$89" in doc.cleaned_text  # diagnostic fee survives


def test_coverage_table_extracted_with_symbols():
    doc = extract_pdf(FILES / "01_service_area_north.pdf")
    table = next(t for t in doc.tables if t and "HVAC" in " ".join(t[0]))
    loudoun = next(r for r in table if r[0] == "Loudoun")
    assert normalize_coverage_cell(loudoun[2]) == "yes"             # HVAC
    assert normalize_coverage_cell(loudoun[3]) == "sub-contracted"  # Plumbing
    assert normalize_coverage_cell(loudoun[4]) == "no"              # Electrical
