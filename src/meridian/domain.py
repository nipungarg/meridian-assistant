"""Shared domain vocabulary.

The customer/booking enums (service types, job types, time windows, contact channels,
cancellation reasons) are used by the router, the booking tools, the mock Booking API,
the CLI, and the Streamlit UI. Defining each ``Literal`` once here - and deriving the
runtime tuples from it via ``get_args`` - keeps a single source of truth with no drift.
"""
from __future__ import annotations

from typing import Literal, get_args

ServiceType = Literal["hvac", "plumbing", "electrical"]
JobType = Literal["diagnostic", "repair", "install", "tune_up", "warranty_return", "estimate"]
Window = Literal["morning", "midday", "afternoon", "first_available"]
Channel = Literal["ivr", "web_chat", "email", "agent"]
CancelReason = Literal["customer_request", "tech_unavailable", "weather", "duplicate", "other"]

SERVICE_TYPES: tuple[str, ...] = get_args(ServiceType)
CHANNELS: tuple[str, ...] = get_args(Channel)
