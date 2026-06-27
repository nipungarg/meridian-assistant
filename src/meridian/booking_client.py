"""Thin httpx client for the (mock) Booking API.

All agent tool calls go through here, so swapping the mock for the real internal API is a
one-line base-URL change. Errors are returned as structured dicts (never raised into the
graph) so the agent can fall back to a human handoff when the API is unreachable.
"""
from __future__ import annotations

from typing import Any

import httpx

from meridian.config import get_settings


class BookingAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().booking_api_token}"}


def _base() -> str:
    return get_settings().booking_api_base_url.rstrip("/")


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{_base()}{path}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.request(method, url, headers=_headers(), **kwargs)
    except httpx.HTTPError as exc:
        raise BookingAPIError(f"Booking API unreachable: {exc}") from exc
    if resp.status_code >= 400:
        detail: Any
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise BookingAPIError(str(detail), status_code=resp.status_code, payload=detail)
    return resp.json()


def create_booking(payload: dict) -> dict:
    return _request("POST", "/bookings", json=payload)


def get_booking(booking_id: str, customer_id: str | None = None) -> dict:
    params = {"customer_id": customer_id} if customer_id else None
    return _request("GET", f"/bookings/{booking_id}", params=params)


def reschedule_booking(booking_id: str, new_date: str, new_window: str) -> dict:
    return _request("PATCH", f"/bookings/{booking_id}",
                    json={"action": "reschedule", "new_date": new_date, "new_window": new_window})


def cancel_booking(booking_id: str, cancel_reason: str = "customer_request") -> dict:
    return _request("PATCH", f"/bookings/{booking_id}",
                    json={"action": "cancel", "cancel_reason": cancel_reason})


def reset() -> dict:
    """Re-seed the mock API (no-op-safe; used by the eval harness)."""
    return _request("POST", "/admin/reset")


def health() -> dict:
    base = _base()
    root = base[:-3] if base.endswith("/v1") else base
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{root}/health").json()
    except httpx.HTTPError as exc:
        raise BookingAPIError(f"Booking API unreachable: {exc}") from exc
