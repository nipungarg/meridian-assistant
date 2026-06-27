"""PDF extraction built on pdfplumber.

Two outputs per document:
  * ``cleaned_text`` - page text with the repeating company/version/page boilerplate
    stripped and consecutive duplicate lines collapsed (table cells remain inline).
  * ``tables`` - structured rows from ``page.extract_tables()`` for callers that need
    clean cells (e.g. the service-area coverage grid with its symbol-font check marks).

Symbol handling: the coverage grids use a symbol font where a check renders as the glyph
``3`` (U+2713) and a cross as ``7`` (U+2717). ``normalize_coverage_cell`` maps those to
``yes`` / ``no`` and recognises ``Sub-contracted`` / ``Pending`` text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

COMPANY_LINE = "Meridian Home Services"
# NOTE: only ASCII in this file; non-ASCII chars are written as \u escapes on purpose.
_VERSION_RE = re.compile(
    r"^Version\s+(?P<version>v[\d.]+)\s+\u00b7\s+Updated\s+(?P<updated>\d{4}-\d{2}-\d{2}).*$"
)
_PAGE_MARKER_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$")
_INTERNAL_RE = re.compile(r"INTERNAL USE ONLY", re.IGNORECASE)

# Tokens that mean "covered" / "not covered" inside the symbol-font coverage columns.
_CHECK_TOKENS = {"3", "\u2713", "\u2714"}
_CROSS_TOKENS = {"7", "\u2717", "\u2718", "x", "X"}


@dataclass
class ExtractedDoc:
    path: Path
    filename: str
    doc_title: str
    doc_type: str
    version: str
    updated_date: str
    cleaned_text: str
    tables: list[list[list[str]]] = field(default_factory=list)

    @property
    def base_metadata(self) -> dict:
        return {
            "source_file": self.filename,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "version": self.version,
            "updated_date": self.updated_date,
        }


def normalize_coverage_cell(value: str | None) -> str:
    """Map a single coverage-grid cell to a normalized status string."""
    if value is None:
        return "unknown"
    text = value.strip()
    if not text:
        return "unknown"
    if text in _CHECK_TOKENS:
        return "yes"
    if text in _CROSS_TOKENS:
        return "no"
    low = text.lower()
    if "sub" in low:  # "Sub-contracted"
        return "sub-contracted"
    if "pending" in low:  # "Pending (Q2)"
        return "pending"
    if low in {"yes", "y"}:
        return "yes"
    if low in {"no", "n"}:
        return "no"
    return text  # leave anything unexpected verbatim (caller may log)


_CID_RE = re.compile(r"\(cid:\d+\)\s*")


def _clean_line(line: str) -> str:
    # pdfplumber renders bullets as (cid:127) and the arrow glyph as the "fi" ligature.
    return _CID_RE.sub("", line).replace(" fi ", " -> ")


def _parse_header(first_page_lines: list[str], filename: str) -> dict:
    title, doc_type, version, updated = "", "", "", ""
    company_idx: int | None = None
    for i, line in enumerate(first_page_lines):
        s = line.strip()
        if s.startswith(COMPANY_LINE):  # pdfplumber merges "<company> <TYPE>" on one line
            company_idx = i
            rest = s[len(COMPANY_LINE):].strip()
            if rest:
                doc_type = rest
            break
    if company_idx is not None:
        for line in first_page_lines[company_idx + 1:]:
            s = line.strip()
            if not s or _VERSION_RE.match(s):
                continue
            title = s  # the H1 title line follows the company/type line
            break
    for line in first_page_lines:
        m = _VERSION_RE.match(line.strip())
        if m:
            version, updated = m.group("version"), m.group("updated")
            break
    if not title:
        title = filename
    return {"title": title, "doc_type": doc_type, "version": version, "updated": updated}


def _strip_boilerplate(pages: list[str], doc_type: str, title: str) -> str:
    kept: list[str] = []
    for page in pages:
        for raw in page.splitlines():
            line = _clean_line(raw.rstrip())
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(COMPANY_LINE):  # "<company> <TYPE>" header line
                continue
            if doc_type and stripped == doc_type.strip():
                continue
            if _VERSION_RE.match(stripped) or _PAGE_MARKER_RE.match(stripped):
                continue
            if _INTERNAL_RE.search(stripped) and "Page" in stripped:
                continue
            kept.append(line)
    deduped: list[str] = []
    for line in kept:
        if not deduped or deduped[-1].strip() != line.strip():
            deduped.append(line)
    return "\n".join(deduped).strip()


def extract_pdf(path: str | Path) -> ExtractedDoc:
    path = Path(path)
    pages: list[str] = []
    tables: list[list[list[str]]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            for tbl in page.extract_tables() or []:
                clean = [[(cell or "").strip() for cell in row] for row in tbl]
                tables.append(clean)

    first_lines = pages[0].splitlines() if pages else []
    header = _parse_header(first_lines, path.name)
    cleaned = _strip_boilerplate(pages, header["doc_type"], header["title"])
    return ExtractedDoc(
        path=path,
        filename=path.name,
        doc_title=header["title"],
        doc_type=header["doc_type"],
        version=header["version"],
        updated_date=header["updated"],
        cleaned_text=cleaned,
        tables=tables,
    )


if __name__ == "__main__":  # debug dumper: python -m meridian.ingestion.pdf_extract <file...>
    import sys

    from meridian.config import get_settings

    targets = sys.argv[1:]
    if not targets:
        targets = sorted(str(p) for p in get_settings().files_dir.glob("*.pdf"))
    for t in targets:
        doc = extract_pdf(t)
        print("=" * 88)
        print(f"FILE: {doc.filename}")
        print(f"  title={doc.doc_title!r} type={doc.doc_type!r} "
              f"version={doc.version!r} updated={doc.updated_date!r} tables={len(doc.tables)}")
        print("-" * 40, "CLEANED TEXT", "-" * 40)
        print(doc.cleaned_text)
        for ti, tbl in enumerate(doc.tables):
            print("-" * 40, f"TABLE {ti}", "-" * 40)
            for row in tbl:
                print(row)
