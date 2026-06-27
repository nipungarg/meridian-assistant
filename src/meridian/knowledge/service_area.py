"""Deterministic ZIP -> service coverage lookup parsed from the service-area PDFs.

ZIP eligibility must be exact (ranges like ``22030-22039``, sub-contracted/pending states,
and the symbol-font check marks), so this is parsed into a structured index used both by the
agent's ``check_service_area`` tool and the RAG chunker - rather than relying on vector search
over the table text.

Only North (file 01) and Central (file 02) regions have coverage docs. South ZIPs are
therefore *unverifiable from the knowledge pack* and resolve to ``unknown_zip`` -> handoff.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from meridian.config import get_settings
from meridian.ingestion.pdf_extract import ExtractedDoc, extract_pdf, normalize_coverage_cell

SERVICE_TYPES = ("hvac", "plumbing", "electrical")
_SERVICE_FILES = ("01_service_area_north.pdf", "02_service_area_central.pdf")
_DASH_RE = re.compile(r"[\u2013\u2014\-]")
_ZIP_RE = re.compile(r"\b\d{5}\b")
_REGION_RE = re.compile(r"Service Area\s*[\u2014\-]\s*(\w+)\s+Region", re.IGNORECASE)
_BULLET = "\u2022"
_NOTE_KEYWORDS = ("escalate", "out-of-area", "out of area", "sub-contract", "travel surcharge",
                  "not yet", "refer", "co-ordination", "coordination", "spot-approval",
                  "spot approval", "licensed")


@dataclass
class CountyCoverage:
    region: str
    county: str
    zip_spec: str
    explicit_zips: set[int] = field(default_factory=set)
    ranges: list[tuple[int, int]] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)

    def covers_zip(self, zip_code: str) -> bool:
        if not zip_code.isdigit():
            return False
        z = int(zip_code)
        if z in self.explicit_zips:
            return True
        return any(lo <= z <= hi for lo, hi in self.ranges)


@dataclass
class EligibilityResult:
    zip_code: str
    service_type: str | None
    in_coverage_table: bool
    status: str  # covered | not_covered | sub-contracted | pending | unknown_zip | mixed
    county: str | None = None
    region: str | None = None
    coverage_by_service: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    recommended_branch: str | None = None
    source_file: str | None = None

    @property
    def can_book(self) -> bool:
        return self.status in {"covered", "sub-contracted"}

    def summary(self) -> str:
        if not self.in_coverage_table:
            return (f"ZIP {self.zip_code} is not in any covered service area on file "
                    f"(North/Central). It requires Branch Manager spot-approval.")
        svc = self.service_type or "service"
        label = {
            "covered": f"{svc} is available",
            "not_covered": f"{svc} is NOT available",
            "sub-contracted": f"{svc} is sub-contracted (no same-day)",
            "pending": f"{svc} is pending / not yet active",
        }.get(self.status, self.status)
        return f"ZIP {self.zip_code} ({self.county}, {self.region} region): {label}."


def _parse_zip_spec(spec: str) -> tuple[set[int], list[tuple[int, int]]]:
    explicit: set[int] = set()
    ranges: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        nums = _ZIP_RE.findall(part)
        if _DASH_RE.search(part) and len(nums) >= 2:
            ranges.append((int(nums[0]), int(nums[1])))
        elif nums:
            for n in nums:
                explicit.add(int(n))
    return explicit, ranges


def _is_coverage_header(row: list[str]) -> bool:
    joined = " ".join((c or "").lower() for c in row)
    return "hvac" in joined and "plumbing" in joined and "electrical" in joined


def _status_from_label(label: str) -> str:
    label = (label or "").lower()
    if label in {"yes", "covered"}:
        return "covered"
    if "sub" in label:
        return "sub-contracted"
    if "pending" in label:
        return "pending"
    return "not_covered"


class ServiceAreaIndex:
    def __init__(self) -> None:
        self.counties: list[CountyCoverage] = []
        self.policy_notes: list[str] = []
        self.branch_assignments: list[str] = []
        self._docs: list[ExtractedDoc] = []

    @classmethod
    def from_files(cls, files_dir: str | Path | None = None) -> "ServiceAreaIndex":
        files_dir = Path(files_dir or get_settings().files_dir)
        index = cls()
        for fname in _SERVICE_FILES:
            path = files_dir / fname
            if path.exists():
                index._ingest_doc(extract_pdf(path))
        return index

    def _ingest_doc(self, doc: ExtractedDoc) -> None:
        self._docs.append(doc)
        m = _REGION_RE.search(doc.doc_title) or _REGION_RE.search(doc.cleaned_text)
        region = m.group(1).title() if m else doc.doc_title
        coverage_table = next((t for t in doc.tables if t and _is_coverage_header(t[0])), None)
        if coverage_table is not None:
            self._parse_coverage_table(coverage_table, region)
        else:
            self._parse_coverage_text(doc.cleaned_text, region)
        self._collect_notes(doc.cleaned_text)

    def _parse_coverage_table(self, table: list[list[str]], region: str) -> None:
        for row in table[1:]:
            cells = [(c or "").strip() for c in row]
            if len(cells) < 5 or not cells[0] or not _ZIP_RE.search(cells[1]):
                continue
            explicit, ranges = _parse_zip_spec(cells[1])
            coverage = {
                "hvac": normalize_coverage_cell(cells[2]),
                "plumbing": normalize_coverage_cell(cells[3]),
                "electrical": normalize_coverage_cell(cells[4]),
            }
            self.counties.append(
                CountyCoverage(region, cells[0], cells[1], explicit, ranges, coverage))

    def _parse_coverage_text(self, text: str, region: str) -> None:
        for line in text.splitlines():
            if not _ZIP_RE.search(line):
                continue
            m = re.match(
                r"^([A-Za-z'\.\s]+?)\s+(\d{5}.*?)\s+(\S+)\s+(\S+(?:\s\(\w+\))?)\s+(\S+)$", line)
            if not m:
                continue
            county, zip_spec, c1, c2, c3 = m.groups()
            explicit, ranges = _parse_zip_spec(zip_spec)
            coverage = {"hvac": normalize_coverage_cell(c1),
                        "plumbing": normalize_coverage_cell(c2),
                        "electrical": normalize_coverage_cell(c3)}
            self.counties.append(
                CountyCoverage(region, county.strip(), zip_spec, explicit, ranges, coverage))

    def _collect_notes(self, text: str) -> None:
        county_names = {c.county.lower() for c in self.counties}
        for line in text.splitlines():
            s = line.strip().lstrip(_BULLET).strip()
            if not s:
                continue
            low = s.lower()
            # Skip raw coverage-table rows ("Loudoun 20147, ... 3 7") but keep county notes
            # ("Prince George's County: electrical not yet licensed. Refer to EcoPower.").
            if any(low.startswith(cn) for cn in county_names) and _ZIP_RE.search(s):
                continue
            if any(k in low for k in _NOTE_KEYWORDS) and s not in self.policy_notes:
                self.policy_notes.append(s)

    def check(self, zip_code: str, service_type: str | None = None) -> EligibilityResult:
        zip_code = (zip_code or "").strip()
        match = next((c for c in self.counties if c.covers_zip(zip_code)), None)
        if match is None:
            return EligibilityResult(
                zip_code=zip_code, service_type=service_type, in_coverage_table=False,
                status="unknown_zip",
                notes=[n for n in self.policy_notes
                       if "escalate" in n.lower() or "spot" in n.lower()])
        coverage_by_service = dict(match.coverage)
        if service_type:
            status = _status_from_label(match.coverage.get(service_type, "unknown"))
        else:
            status = "covered" if all(
                _status_from_label(v) == "covered" for v in match.coverage.values()) else "mixed"
        return EligibilityResult(
            zip_code=zip_code, service_type=service_type, in_coverage_table=True, status=status,
            county=match.county, region=match.region, coverage_by_service=coverage_by_service,
            notes=self._notes_for(match, service_type), recommended_branch=self._branch_for(match),
            source_file=(_SERVICE_FILES[0] if match.region.lower() == "north" else _SERVICE_FILES[1]),
        )

    def _notes_for(self, county: CountyCoverage, service_type: str | None) -> list[str]:
        notes = [n for n in self.policy_notes if county.county.lower() in n.lower()]
        if (service_type and county.coverage.get(service_type) == "sub-contracted"
                and not any("same-day" in n.lower() for n in notes)):
            notes.append("Sub-contracted work: same-day service is not available.")
        return notes

    def _branch_for(self, county: CountyCoverage) -> str | None:
        # North assignments are stated in the doc; Central are inferred from branch locations.
        cl = county.county.lower()
        return {
            "fairfax": "Falls Church (overflow: Tysons)",
            "arlington": "Falls Church (overflow: Tysons)",
            "alexandria": "Falls Church (overflow: Tysons)",
            "loudoun": "Herndon",
            "montgomery": "Rockville",
            "howard": "Columbia",
            "prince george's": "College Park",
        }.get(cl)

    def render_chunks(self) -> list[dict]:
        chunks: list[dict] = []
        label = {"yes": "available", "no": "NOT available", "covered": "available",
                 "sub-contracted": "sub-contracted (no same-day service)",
                 "pending": "pending / not yet active"}
        for c in self.counties:
            cov = ", ".join(
                f"{svc.upper()}: {label.get(c.coverage.get(svc, ''), c.coverage.get(svc, 'unknown'))}"
                for svc in SERVICE_TYPES)
            text = (f"Service Area - {c.region} Region. {c.county} County covers ZIP codes "
                    f"{c.zip_spec}. Coverage by service - {cov}.")
            chunks.append({"text": text, "section": f"{c.region} / {c.county} coverage"})
        if self.policy_notes:
            chunks.append({"text": "Service-area policy notes: " + " ".join(self.policy_notes),
                           "section": "Service-area policy"})
        return chunks


_INDEX: ServiceAreaIndex | None = None


def get_service_area_index() -> ServiceAreaIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = ServiceAreaIndex.from_files()
    return _INDEX
