"""Tests for the mock Booking API (via TestClient) and the cancellation-fee logic."""
import datetime as dt

import pytest
from fastapi.testclient import TestClient

from meridian.api import mock_booking_api as api

HDR = {"Authorization": "Bearer test-token"}


@pytest.fixture()
def client():
    api.seed_bookings()
    return TestClient(api.app)


def _future(days: int) -> str:
    return (dt.date.today() + dt.timedelta(days=days)).isoformat()


def _create_payload(**over):
    base = {
        "customer_id": None,
        "customer_info": {"name": "Test User", "phone": "703-555-0000"},
        "service_type": "electrical", "job_type": "diagnostic",
        "zip_code": "20814", "preferred_date": _future(10),
        "preferred_window": "morning", "channel": "web_chat",
    }
    base.update(over)
    return base


def test_create_covered_zip(client):
    r = client.post("/v1/bookings", json=_create_payload(), headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "confirmed"
    assert body["booking_id"].startswith("BK-")
    assert body["confirmation_sent"] is True


def test_create_out_of_area(client):
    r = client.post("/v1/bookings", json=_create_payload(zip_code="20110"), headers=HDR)
    assert r.status_code == 200
    assert r.json()["status"] == "out_of_area"


def test_create_requires_auth(client):
    r = client.post("/v1/bookings", json=_create_payload())
    assert r.status_code == 401


def test_create_rejects_past_date(client):
    r = client.post("/v1/bookings", json=_create_payload(preferred_date=_future(-3)), headers=HDR)
    assert r.status_code == 400


def test_create_rejects_beyond_60_days(client):
    r = client.post("/v1/bookings", json=_create_payload(preferred_date=_future(90)), headers=HDR)
    assert r.status_code == 400


def test_get_seeded_booking(client):
    r = client.get("/v1/bookings/BK-00483921", headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert body["service_type"] == "plumbing"
    assert body["tech_name"] == "Marcus Webb"


def test_get_unknown_booking(client):
    assert client.get("/v1/bookings/BK-99999999", headers=HDR).status_code == 404


def test_reschedule_far_is_free(client):
    r = client.patch("/v1/bookings/BK-00391042", headers=HDR,
                     json={"action": "reschedule", "new_date": _future(20), "new_window": "afternoon"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rescheduled"
    assert body["fee_applied"] == 0


# ---- pure fee logic ----
def _record(hours_ahead: float, customer_id: str):
    appt = api._now() + dt.timedelta(hours=hours_ahead)
    return {
        "booking_id": "BK-TEST", "customer_id": customer_id,
        "appointment_window": {"date": appt.date().isoformat(),
                               "start_time": appt.strftime("%H:%M"), "end_time": "00:00"},
    }


def test_fee_more_than_24h():
    assert api._cancel_fee(_record(48, "C1")) == (0.0, False)


def test_fee_2_to_24h():
    assert api._cancel_fee(_record(5, "C2")) == (35.0, False)


def test_fee_under_2h_with_first_time_waiver():
    api._WAIVER_USED.discard("C3")
    fee, waiver = api._cancel_fee(_record(1, "C3"))
    assert (fee, waiver) == (0.0, True)          # first time -> waived
    fee2, waiver2 = api._cancel_fee(_record(1, "C3"))
    assert (fee2, waiver2) == (75.0, False)      # waiver already consumed
