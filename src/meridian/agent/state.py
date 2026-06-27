"""Shared LangGraph state.

Conversation ``messages`` accumulate (add_messages reducer); booking flow fields
(``slots``, ``pending_action``, ``awaiting_confirmation``, ``active_flow``) persist across
turns via the checkpointer; per-turn outputs (``answer``, ``citations``, ``trace`` ...) are
reset by the triage node at the start of every turn.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    channel: str

    # routing (per turn)
    route: str
    intent: str
    handoff_reason: Optional[str]
    is_emergency: bool
    rationale: str
    search_query: str
    confirmation: str

    # booking flow (persisted across turns)
    active_flow: Optional[str]            # booking_create | booking_reschedule | booking_cancel
    slots: dict
    pending_action: Optional[dict]        # {kind, payload, summary}
    awaiting_confirmation: bool

    # outputs (per turn)
    answer: str
    citations: list                       # human-readable citation strings
    sources_detail: list                  # structured citation dicts for the UI
    tool_result: Optional[dict]
    handoff: Optional[dict]
    confidence: float
    retrieval_score: float
    trace: list[str]


PER_TURN_RESET: dict[str, Any] = {
    "route": "",
    "intent": "",
    "handoff_reason": None,
    "is_emergency": False,
    "rationale": "",
    "search_query": "",
    "confirmation": "",
    "answer": "",
    "citations": [],
    "sources_detail": [],
    "tool_result": None,
    "handoff": None,
    "confidence": 0.0,
    "retrieval_score": 0.0,
    "trace": [],
}
