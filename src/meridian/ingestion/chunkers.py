"""Per-document-type chunking.

The corpus is small and highly structured, so chunk boundaries are chosen so that a single
retrieved chunk fully answers a likely question (e.g. one chunk per FAQ Q&A, one chunk per
branch's hours, one chunk per county's coverage). Every chunk carries source metadata for
citations and a short breadcrumb prefix (``Title > Section``) to sharpen embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from meridian.ingestion.pdf_extract import ExtractedDoc
from meridian.knowledge.service_area import ServiceAreaIndex

BRANCH_NAMES = [
    "Falls Church", "Tysons", "Herndon", "Rockville", "Columbia", "College Park",
    "Annapolis", "Glen Burnie", "Bowie", "Laurel", "Owings Mills",
]

# Curated section headings per document (the pack is fixed; this keeps boundaries reliable).
KNOWN_HEADINGS: dict[str, list[str]] = {
    "03_hvac_pricing.pdf": ["Diagnostic Fee", "Repair Tiers", "Maintenance Plans",
                            "After-Hours Surcharge"],
    "04_plumbing_pricing.pdf": ["Service Call Fee", "Common Services", "Emergency Plumbing"],
    "05_electrical_pricing.pdf": ["Service Call Fee", "Common Services"],
    "06_warranty_terms.pdf": ["Labor Warranty", "Parts Warranty", "Maintenance Plan Members",
                              "Exclusions", "Claim Process"],
    "07_cancellation_policy.pdf": ["Standard Cancellation Fees", "Rescheduling",
                                   "Meridian-Initiated Cancellations", "Fee Disputes"],
    "11_faq_emergencies.pdf": ["What Counts as an Emergency", "How to Request Emergency Service",
                               "Emergency Response Time SLA", "After-Hours Surcharges"],
    "12_booking_api_spec.pdf": ["Authentication", "POST /bookings", "GET /bookings",
                                "PATCH /bookings"],
}

_DASHES = ("\u2014", "\u2013")  # em dash, en dash


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _mk(doc: ExtractedDoc, section: str, body: str, idx: int) -> Chunk:
    body = body.strip()
    breadcrumb = f"{doc.doc_title} > {section}" if section else doc.doc_title
    meta = doc.base_metadata | {
        "section": section or doc.doc_title,
        "chunk_id": f"{doc.filename}#{idx}",
        "breadcrumb": breadcrumb,
    }
    return Chunk(text=f"[{breadcrumb}]\n{body}", metadata=meta)


def _split_on_headings(text: str, headings: list[str]) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] = ("", [])
    for line in lines:
        stripped = line.strip()
        matched = next((h for h in headings if stripped.startswith(h)), None)
        if matched:
            if current[1]:
                sections.append(current)
            current = (stripped, [])
        else:
            current[1].append(line)
    if current[1] or current[0]:
        sections.append(current)
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip() or h]


def _faq_chunks(doc: ExtractedDoc) -> list[Chunk]:
    lines = [ln for ln in doc.cleaned_text.splitlines() if ln.strip()]
    chunks: list[Chunk] = []
    q: str | None = None
    body: list[str] = []
    idx = 0
    intro_consumed = False
    for ln in lines:
        if ln.strip().endswith("?"):
            if q is not None:
                chunks.append(_mk(doc, q, f"{q}\n{' '.join(body)}", idx)); idx += 1
            q = ln.strip()
            body = []
            intro_consumed = True
        else:
            if q is None and not intro_consumed:
                continue
            body.append(ln.strip())
    if q is not None:
        chunks.append(_mk(doc, q, f"{q}\n{' '.join(body)}", idx))
    return chunks


def _branch_hours_chunks(doc: ExtractedDoc) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    rows: list[list[str]] = []
    for tbl in doc.tables:
        for row in tbl:
            if row and any((row[0] or "").startswith(b) for b in BRANCH_NAMES):
                rows.append([(c or "").strip() for c in row])
    if not rows:
        for line in doc.cleaned_text.splitlines():
            b = next((b for b in BRANCH_NAMES if line.strip().startswith(b)), None)
            if b:
                rest = line.strip()[len(b):].strip()
                rows.append([b] + rest.split("  "))
    for row in rows:
        branch = row[0]
        rest = " | ".join(c for c in row[1:] if c)
        text = (f"{branch} branch hours and region: {rest}. "
                f"Columns are Region, Mon-Fri, Saturday, Sunday.")
        chunks.append(_mk(doc, f"{branch} hours", text, idx)); idx += 1
    for section in ["Contact Center Hours", "24/7 Emergency Line"]:
        body = _grab_section(doc.cleaned_text, section)
        if body:
            chunks.append(_mk(doc, section, f"{section}\n{body}", idx)); idx += 1
    return chunks


def _grab_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        s = line.strip()
        if s.startswith(heading):
            capturing = True
            continue
        if capturing:
            is_heading = (
                s and len(s.split()) <= 6 and s[:1].isupper()
                and not s.endswith((".", ":", "?")) and "am" not in s and "$" not in s
                and not any(d in s for d in _DASHES) and not s[0].isdigit()
            )
            if is_heading:
                break
            if s:
                out.append(s)
    return "\n".join(out).strip()


def _service_area_chunks(doc: ExtractedDoc) -> list[Chunk]:
    index = ServiceAreaIndex()
    index._ingest_doc(doc)
    chunks: list[Chunk] = []
    for i, c in enumerate(index.render_chunks()):
        chunks.append(_mk(doc, c["section"], c["text"], i))
    return chunks


def _heading_chunks(doc: ExtractedDoc, headings: list[str]) -> list[Chunk]:
    sections = _split_on_headings(doc.cleaned_text, headings)
    chunks: list[Chunk] = []
    for i, (h, body) in enumerate(sections):
        if not body and not h:
            continue
        chunks.append(_mk(doc, h, f"{h}\n{body}" if h else body, i))
    return chunks


def _whole_doc_chunk(doc: ExtractedDoc) -> list[Chunk]:
    return [_mk(doc, "", doc.cleaned_text, 0)]


def chunk_document(doc: ExtractedDoc) -> list[Chunk]:
    name = doc.filename
    if name in ("01_service_area_north.pdf", "02_service_area_central.pdf"):
        return _service_area_chunks(doc)
    if name == "08_branch_hours.pdf":
        return _branch_hours_chunks(doc)
    if name in ("09_faq_booking.pdf", "10_faq_payments.pdf"):
        return _faq_chunks(doc)
    if name in KNOWN_HEADINGS:
        return _heading_chunks(doc, KNOWN_HEADINGS[name])
    return _whole_doc_chunk(doc)
