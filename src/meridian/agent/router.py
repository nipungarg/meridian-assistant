"""Structured triage router + deterministic safety overlay.

A single structured LLM call classifies intent, flags sensitive categories, and extracts
booking entities (resolving relative dates against TODAY). A deterministic keyword scan runs
in parallel for emergencies so a safety handoff never depends solely on the model.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from meridian.agent.llm import get_chat_llm
from meridian.agent.prompts import ROUTER_SYSTEM
from meridian.config import get_settings

Intent = Literal[
    "faq_policy", "service_area", "booking_create", "booking_reschedule",
    "booking_cancel", "booking_status", "smalltalk", "out_of_scope",
]
HandoffReason = Literal["emergency", "complaint", "fee_dispute", "commercial"]


class RouterDecision(BaseModel):
    intent: Intent
    handoff_reason: Optional[HandoffReason] = None
    zip_code: Optional[str] = None
    service_type: Optional[Literal["hvac", "plumbing", "electrical"]] = None
    job_type: Optional[Literal[
        "diagnostic", "repair", "install", "tune_up", "warranty_return", "estimate"]] = None
    preferred_date: Optional[str] = None
    preferred_window: Optional[Literal["morning", "midday", "afternoon", "first_available"]] = None
    preferred_tech: Optional[str] = None
    booking_id: Optional[str] = None
    cancel_reason: Optional[Literal[
        "customer_request", "tech_unavailable", "weather", "duplicate", "other"]] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    customer_address: Optional[str] = None
    search_query: str = Field(default="")
    rationale: str = Field(default="")


# High-precision ACTIVE-incident patterns only. Informational questions that merely contain
# words like "flood" or "emergency" must NOT trip this (the LLM router handles context).
_EMERGENCY_PATTERNS = [
    r"water (is )?(pouring|gushing|flooding|spraying|everywhere|coming up)",
    r"\bflooding\b", r"(house|basement|kitchen|bathroom|garage) is flood",
    r"burst pipe", r"pipe (just )?burst", r"pipe is burst",
    r"\bsparking\b", r"\bsparks\b", r"shooting sparks",
    r"burning smell", r"smell(s)? (like )?(something )?burning", r"\bsmoke\b",
    r"sewage (backup|backing up|is)", r"raw sewage",
    r"gas leak", r"smell(s)? (of )?gas", r"smell gas",
    r"electrical fire", r"(getting|got) shocked", r"electric shock",
    r"no (heat|cooling).{0,30}(freezing|frozen|below \d|baby|infant|elderly|sick)",
]
_URGENCY = [r"right now", r"immediately", r"\basap\b"]
_PROBLEM = [r"leak", r"no power", r"no heat", r"no cooling", r"not working", r"\bbroken\b",
            r"won'?t (turn|start)", r"stopped working", r"flooded"]
_FEE_DISPUTE_PATTERNS = [r"waive", r"no-?show fee", r"charged.*(fee|\$)", r"refund the (fee|charge)"]
_COMPLAINT_PATTERNS = [r"speak (to|with) (a )?manager", r"\bcomplaint\b", r"left a (huge )?mess",
                       r"unacceptable", r"terrible (job|service)"]
_COMMERCIAL_PATTERNS = [r"net-?30", r"volume discount", r"\d+\s+units", r"commercial account",
                        r"multiple (properties|units)", r"net 30 invoicing"]


def _matches(text: str, patterns: list[str]) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def deterministic_safety(text: str) -> Optional[str]:
    """Keyword overlay for sensitive categories (emergency is safety-critical, high-precision)."""
    if _matches(text, _EMERGENCY_PATTERNS):
        return "emergency"
    if _matches(text, _URGENCY) and _matches(text, _PROBLEM):
        return "emergency"
    if _matches(text, _FEE_DISPUTE_PATTERNS):
        return "fee_dispute"
    if _matches(text, _COMPLAINT_PATTERNS):
        return "complaint"
    if _matches(text, _COMMERCIAL_PATTERNS):
        return "commercial"
    return None


def _history_text(messages: list, limit: int = 6) -> str:
    out = []
    for m in messages[-limit:]:
        if isinstance(m, HumanMessage):
            out.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            out.append(f"Assistant: {m.content}")
    return "\n".join(out)


def run_router(messages: list, channel: str, active_flow: str | None, slots: dict) -> RouterDecision:
    s = get_settings()
    latest = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    context = (
        f"TODAY: {s.today.isoformat()}\n"
        f"CHANNEL: {channel}\n"
        f"ACTIVE FLOW: {active_flow or 'none'}\n"
        f"ALREADY-COLLECTED SLOTS: {slots or '{}'}\n\n"
        f"RECENT CONVERSATION:\n{_history_text(messages)}\n\n"
        f"LATEST CUSTOMER MESSAGE:\n{latest}"
    )
    llm = get_chat_llm(0.0).with_structured_output(RouterDecision)
    decision: RouterDecision = llm.invoke(
        [("system", ROUTER_SYSTEM), ("human", context)]
    )
    if not decision.search_query:
        decision.search_query = latest

    # Deterministic safety overlay wins for emergencies.
    safety = deterministic_safety(latest)
    if safety == "emergency":
        decision.handoff_reason = "emergency"
    elif safety and decision.handoff_reason is None:
        decision.handoff_reason = safety  # type: ignore[assignment]
    return decision


_YES = [r"^\s*y(es|eah|ep|up)?\b", r"\bconfirm", r"\bcorrect\b", r"sounds good", r"go ahead",
        r"please do", r"\bok(ay)?\b", r"\bsure\b", r"that works", r"do it", r"book it"]
_NO = [r"^\s*n(o|ope|ah)?\b", r"don'?t", r"do not", r"\bcancel that\b", r"\bchange\b",
       r"\bwrong\b", r"not right", r"\bwait\b"]


def parse_confirmation(text: str) -> str:
    low = text.strip().lower()
    if _matches(low, _NO):
        return "no"
    if _matches(low, _YES):
        return "yes"
    # LLM fallback for ambiguous phrasing
    try:
        llm = get_chat_llm(0.0)
        raw = llm.invoke(
            "Does this reply CONFIRM an action, DECLINE it, or is it UNCLEAR? "
            f"Reply with one word: yes, no, or unclear.\nReply: {text!r}"
        ).content.strip().lower()
        if "yes" in raw:
            return "yes"
        if raw.startswith("no") or "decline" in raw:
            return "no"
    except Exception:
        pass
    return "unclear"
