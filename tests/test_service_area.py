"""Tests for deterministic ZIP -> service eligibility."""
import pytest

from meridian.knowledge.service_area import ServiceAreaIndex


@pytest.fixture(scope="module")
def index():
    return ServiceAreaIndex.from_files()


@pytest.mark.parametrize("zip_code,service,status,can_book", [
    ("22030", "hvac", "covered", True),          # Fairfax range 22030-22039
    ("22044", "plumbing", "covered", True),       # Fairfax second range
    ("20147", "plumbing", "sub-contracted", True),  # Loudoun sub-contracted
    ("20147", "electrical", "not_covered", False),  # Loudoun electrical = cross
    ("22301", "electrical", "pending", False),      # Alexandria electrical Pending (Q2)
    ("20706", "electrical", "not_covered", False),  # Prince George's not licensed
    ("21042", "hvac", "covered", True),             # Howard (Central)
    ("20110", "electrical", "unknown_zip", False),  # Manassas - out of area
    ("99999", "hvac", "unknown_zip", False),        # nonsense ZIP
])
def test_eligibility(index, zip_code, service, status, can_book):
    r = index.check(zip_code, service)
    assert r.status == status
    assert r.can_book is can_book


def test_out_of_area_has_no_county(index):
    r = index.check("20110", "hvac")
    assert not r.in_coverage_table
    assert r.county is None
    assert "spot-approval" in " ".join(r.notes).lower()


def test_pg_electrical_refers_ecopower(index):
    r = index.check("20706", "electrical")
    assert any("ecopower" in n.lower() for n in r.notes)


def test_branch_assignment(index):
    assert "Falls Church" in (index.check("22030", "hvac").recommended_branch or "")
    assert index.check("20814", "hvac").recommended_branch == "Rockville"
