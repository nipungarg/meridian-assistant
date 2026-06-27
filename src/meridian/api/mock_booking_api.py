"""Mock Booking API implementing the Booking API spec (docs/booking_api_spec.md).

POST /v1/bookings, GET /v1/bookings/{id}, PATCH /v1/bookings/{id}. In-memory store seeded
with the booking IDs referenced in the example messages. ZIP coverage is validated against
the same deterministic service-area index the agent uses, and PATCH enforces the
cancellation-policy fee logic (>24h free, 2-24h $35, <2h/no-show $75 with a one-time waiver).
The POST/GET responses include the booking's ``channel`` (which front-end it came in on).

Run:  uvicorn meridian.api.mock_booking_api:app --port 8000
"""
from __future__ import annotations

import datetime as dt
import random
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from meridian.config import get_settings
from meridian.knowledge.service_area import get_service_area_index


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    seed_bookings()
    yield


app = FastAPI(title="Meridian Booking API (mock)", version="1.0", lifespan=lifespan)

ServiceType = Literal["hvac", "plumbing", "electrical"]
JobType = Literal["diagnostic", "repair", "install", "tune_up", "warranty_return", "estimate"]
Window = Literal["morning", "midday", "afternoon", "first_available"]
Channel = Literal["ivr", "web_chat", "email", "agent"]

WINDOWS: dict[str, tuple[str, str]] = {
    "morning": ("07:00", "11:00"),
    "midday": ("11:00", "14:00"),
    "afternoon": ("14:00", "18:00"),
    "first_available": ("08:00", "12:00"),
}
TECHS: dict[str, list[str]] = {
    "hvac": ["Marcus Webb", "Dana Cole", "Priya Nair"],
    "plumbing": ["Marcus Webb", "Luis Ortega"],
    "electrical": ["Sara Kim", "Tom Brennan"],
}

# In-memory state
_BOOKINGS: dict[str, dict] = {}
_WAIVER_USED: set[str] = set()  # customer ids that have consumed the one-time no-show waiver


# --------------------------------------------------------------------------- helpers
def _now() -> dt.datetime:
    s = get_settings()
    if s.demo_date.strip():
        return dt.datetime.combine(s.today, dt.time(9, 0))
    return dt.datetime.now()


def _new_booking_id() -> str:
    while True:
        bid = f"BK-{random.randint(0, 99_999_999):08d}"
        if bid not in _BOOKINGS:
            return bid


def _assign_tech(service_type: str, key: str, preferred: str | None) -> str | None:
    pool = TECHS.get(service_type, [])
    if not pool:
        return None
    if preferred:
        return preferred  # mock honors the request (spec: warns if unavailable)
    return pool[hash(key) % len(pool)]


def _window_obj(date_iso: str, window: str) -> dict:
    start, end = WINDOWS.get(window, WINDOWS["first_available"])
    return {"date": date_iso, "start_time": start, "end_time": end}


def require_auth(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed bearer token.")
    return authorization.split(" ", 1)[1].strip()


# --------------------------------------------------------------------------- models
class CustomerInfo(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None


class CreateBooking(BaseModel):
    customer_id: Optional[str] = None
    customer_info: Optional[CustomerInfo] = None
    service_type: ServiceType
    job_type: JobType
    zip_code: str
    preferred_date: str
    preferred_window: Window
    preferred_tech: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=500)
    channel: Channel

    @field_validator("zip_code")
    @classmethod
    def _zip5(cls, v: str) -> str:
        v = v.strip()
        if not (len(v) == 5 and v.isdigit()):
            raise ValueError("zip_code must be a 5-digit string")
        return v

    @field_validator("preferred_date")
    @classmethod
    def _iso(cls, v: str) -> str:
        try:
            dt.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError("preferred_date must be ISO 8601 (YYYY-MM-DD)") from exc
        return v


class PatchBooking(BaseModel):
    action: Literal["reschedule", "cancel", "update_notes"]
    new_date: Optional[str] = None
    new_window: Optional[Window] = None
    cancel_reason: Optional[
        Literal["customer_request", "tech_unavailable", "weather", "duplicate", "other"]
    ] = None
    notes: Optional[str] = Field(default=None, max_length=500)


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "bookings": len(_BOOKINGS), "now": _now().isoformat()}


@app.post("/v1/bookings")
def create_booking(body: CreateBooking, token: str = Depends(require_auth)) -> dict:
    if body.customer_id is None and body.customer_info is None:
        raise HTTPException(400, "customer_info is required when customer_id is null.")

    # Date window: must be today..+60 days.
    today = _now().date()
    pdate = dt.date.fromisoformat(body.preferred_date)
    if pdate < today:
        raise HTTPException(400, "preferred_date is in the past.")
    if pdate > today + dt.timedelta(days=60):
        raise HTTPException(400, "preferred_date must be within 60 days.")

    elig = get_service_area_index().check(body.zip_code, body.service_type)
    bid = _new_booking_id()

    if not elig.can_book:
        record = {
            "booking_id": bid,
            "status": "out_of_area",
            "service_type": body.service_type,
            "job_type": body.job_type,
            "zip_code": body.zip_code,
            "assigned_branch": None,
            "appointment_window": None,
            "tech_name": None,
            "tech_eta_minutes": None,
            "notes": body.notes,
            "invoice_total": None,
            "customer_id": body.customer_id,
            "channel": body.channel,
        }
        _BOOKINGS[bid] = record
        return {
            "booking_id": bid,
            "status": "out_of_area",
            "assigned_branch": None,
            "appointment_window": None,
            "tech_name": None,
            "confirmation_sent": False,
            "channel": body.channel,
        }

    branch = elig.recommended_branch or "Unassigned"
    tech = _assign_tech(body.service_type, bid, body.preferred_tech)
    window = _window_obj(body.preferred_date, body.preferred_window)
    record = {
        "booking_id": bid,
        "status": "confirmed",
        "service_type": body.service_type,
        "job_type": body.job_type,
        "zip_code": body.zip_code,
        "assigned_branch": branch,
        "appointment_window": window,
        "tech_name": tech,
        "tech_eta_minutes": None,
        "notes": body.notes,
        "invoice_total": None,
        "customer_id": body.customer_id,
        "channel": body.channel,
    }
    _BOOKINGS[bid] = record
    return {
        "booking_id": bid,
        "status": "confirmed",
        "assigned_branch": branch,
        "appointment_window": window,
        "tech_name": tech,
        "confirmation_sent": True,
        "channel": body.channel,
    }


@app.get("/v1/bookings/{booking_id}")
def get_booking(booking_id: str, customer_id: str | None = None,
                token: str = Depends(require_auth)) -> dict:
    record = _BOOKINGS.get(booking_id)
    if record is None:
        raise HTTPException(404, f"Booking {booking_id} not found.")
    if customer_id and record.get("customer_id") and customer_id != record["customer_id"]:
        raise HTTPException(403, "customer_id does not own this booking.")
    return {
        "booking_id": record["booking_id"],
        "status": record["status"],
        "service_type": record["service_type"],
        "job_type": record["job_type"],
        "appointment_window": record["appointment_window"],
        "tech_name": record["tech_name"],
        "tech_eta_minutes": record["tech_eta_minutes"] if record["status"] == "en_route" else None,
        "notes": record["notes"],
        "invoice_total": record["invoice_total"] if record["status"] == "completed" else None,
        "channel": record.get("channel"),
    }


def _appointment_dt(record: dict) -> dt.datetime | None:
    win = record.get("appointment_window")
    if not win:
        return None
    start = win.get("start_time", "08:00")
    return dt.datetime.fromisoformat(f"{win['date']}T{start}")


def _cancel_fee(record: dict) -> tuple[float, bool]:
    """Return (fee, waiver_used) per the cancellation policy."""
    appt = _appointment_dt(record)
    if appt is None:
        return 0.0, False
    hours = (appt - _now()).total_seconds() / 3600.0
    if hours > 24:
        return 0.0, False
    if hours >= 2:
        return 35.0, False
    # <2h or past -> $75 no-show, waived once per 12 months per customer
    cust = record.get("customer_id") or record["booking_id"]
    if cust not in _WAIVER_USED:
        _WAIVER_USED.add(cust)
        return 0.0, True
    return 75.0, False


@app.patch("/v1/bookings/{booking_id}")
def patch_booking(booking_id: str, body: PatchBooking,
                  token: str = Depends(require_auth)) -> dict:
    record = _BOOKINGS.get(booking_id)
    if record is None:
        raise HTTPException(404, f"Booking {booking_id} not found.")

    if body.action == "update_notes":
        record["notes"] = body.notes
        return {"booking_id": booking_id, "status": record["status"],
                "fee_applied": 0.0, "waiver_used": False}

    if body.action == "cancel":
        fee, waiver = _cancel_fee(record)
        record["status"] = "cancelled"
        return {"booking_id": booking_id, "status": "cancelled",
                "fee_applied": fee, "waiver_used": waiver}

    # reschedule
    if not body.new_date or not body.new_window:
        raise HTTPException(400, "new_date and new_window are required to reschedule.")
    today = _now().date()
    ndate = dt.date.fromisoformat(body.new_date)
    if ndate < today:
        raise HTTPException(400, "new_date is in the past.")
    if ndate > today + dt.timedelta(days=60):
        raise HTTPException(400, "new_date must be within 60 days.")

    # >24h notice: free. Same-day reschedule to a different day: late-cancel fee.
    appt = _appointment_dt(record)
    fee, waiver = 0.0, False
    if appt is not None:
        hours = (appt - _now()).total_seconds() / 3600.0
        if hours <= 24 and ndate != appt.date():
            fee, waiver = _cancel_fee(record)
    new_window = _window_obj(body.new_date, body.new_window)
    record["appointment_window"] = new_window
    record["status"] = "confirmed"
    return {"booking_id": booking_id, "status": "rescheduled", "fee_applied": fee,
            "waiver_used": waiver, "new_appointment_window": new_window}


# --------------------------------------------------------------------------- seed
def seed_bookings() -> None:
    _BOOKINGS.clear()
    _WAIVER_USED.clear()
    today = _now().date()
    tomorrow = (today + dt.timedelta(days=1)).isoformat()
    # #4 status check: appointment "tomorrow"
    _BOOKINGS["BK-00391042"] = {
        "booking_id": "BK-00391042", "status": "confirmed", "service_type": "hvac",
        "job_type": "tune_up", "zip_code": "22046",
        "assigned_branch": "Falls Church (overflow: Tysons)",
        "appointment_window": {"date": tomorrow, "start_time": "10:00", "end_time": "12:00"},
        "tech_name": "Dana Cole", "tech_eta_minutes": None, "notes": "Annual maintenance.",
        "invoice_total": None, "customer_id": "CID-8842", "channel": "ivr",
    }
    # Spec GET example (future-dated a few days out so a >24h reschedule is fee-free).
    soon = (today + dt.timedelta(days=5)).isoformat()
    _BOOKINGS["BK-00483921"] = {
        "booking_id": "BK-00483921", "status": "confirmed", "service_type": "plumbing",
        "job_type": "repair", "zip_code": "22030",
        "assigned_branch": "Falls Church (overflow: Tysons)",
        "appointment_window": {"date": soon, "start_time": "10:00", "end_time": "12:00"},
        "tech_name": "Marcus Webb", "tech_eta_minutes": None,
        "notes": "Kitchen faucet leak.", "invoice_total": None,
        "customer_id": "CID-7741", "channel": "web_chat",
    }
    # #13 tech ETA: en_route
    _BOOKINGS["BK-00512883"] = {
        "booking_id": "BK-00512883", "status": "en_route", "service_type": "electrical",
        "job_type": "diagnostic", "zip_code": "20814",
        "assigned_branch": "Rockville",
        "appointment_window": {"date": today.isoformat(), "start_time": "10:00", "end_time": "12:00"},
        "tech_name": "Sara Kim", "tech_eta_minutes": 15, "notes": "Panel buzzing.",
        "invoice_total": None, "customer_id": "CID-3310", "channel": "ivr",
    }
    # Dedicated booking for cancel tests (so read/ETA seeds stay untouched).
    _BOOKINGS["BK-00400000"] = {
        "booking_id": "BK-00400000", "status": "confirmed", "service_type": "hvac",
        "job_type": "repair", "zip_code": "22030",
        "assigned_branch": "Falls Church (overflow: Tysons)",
        "appointment_window": {"date": soon, "start_time": "14:00", "end_time": "18:00"},
        "tech_name": "Priya Nair", "tech_eta_minutes": None, "notes": "Thermostat issue.",
        "invoice_total": None, "customer_id": "CID-9001", "channel": "agent",
    }


@app.post("/v1/admin/reset")
def admin_reset(token: str = Depends(require_auth)) -> dict:
    """Re-seed the in-memory store (used by the eval harness for isolation)."""
    seed_bookings()
    return {"status": "reset", "bookings": len(_BOOKINGS)}
