"""LangGraph wiring for the Meridian assistant.

Flow per turn:  triage -> {answer | service_area | booking | confirm | status | smalltalk |
handoff} -> END. State persists across turns via a checkpointer, which is how slot-filling
and the confirm-before-commit gate work without LangGraph's interrupt() primitive (chosen for
multi-surface robustness across the Streamlit UI, CLI, and eval harness).

Guardrails enforced here:
  * Sensitive intents (emergency/complaint/fee_dispute/commercial) never reach the answerer.
  * ZIP eligibility, the 60-day window, and cancellation fees use deterministic logic / the API.
  * No write (POST/PATCH) happens until the customer explicitly confirms.
  * RAG answers only emit when retrieval is confident AND the model says it is answerable.
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from meridian import booking_client
from meridian.agent.handoff import build_handoff, customer_message
from meridian.agent.prompts import CONFIRM_HINT
from meridian.agent.router import parse_confirmation, run_router
from meridian.agent.state import PER_TURN_RESET, AgentState
from meridian.agent.tools import (
    WINDOW_LABEL,
    build_create_payload,
    humanize_slot_request,
    missing_create_slots,
    rag_answer,
)
from meridian.config import get_settings
from meridian.knowledge.service_area import get_service_area_index

ENTITY_KEYS = [
    "zip_code", "service_type", "job_type", "preferred_date", "preferred_window",
    "preferred_tech", "booking_id", "cancel_reason", "customer_name", "customer_phone",
    "customer_email", "customer_address",
]
_MODIFY_RE = re.compile(r"\b(actually|instead|rather|make it|change|different|switch)\b", re.I)


def _latest_human(messages: list) -> str:
    return next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def _ai(updates: dict, text: str) -> dict:
    updates["answer"] = text
    updates["messages"] = [AIMessage(content=text)]
    return updates


# --------------------------------------------------------------------------- triage
def triage(state: AgentState) -> dict:
    updates: dict = dict(PER_TURN_RESET)
    messages = state["messages"]
    latest = _latest_human(messages)
    trace: list[str] = []

    if state.get("awaiting_confirmation") and state.get("pending_action"):
        conf = parse_confirmation(latest)
        if conf == "yes":
            updates.update(route="confirm", confirmation="yes",
                           trace=["Awaiting confirmation -> 'yes'."])
            return updates
        if conf != "yes" and _MODIFY_RE.search(latest):
            # Treat as an edit: drop the pending action and re-route through the router.
            updates.update(awaiting_confirmation=False, pending_action=None)
            trace.append("Confirmation reply looks like an edit; reprocessing.")
        elif conf == "no":
            updates.update(route="confirm", confirmation="no",
                           trace=["Awaiting confirmation -> 'no'."])
            return updates
        else:
            updates.update(route="confirm", confirmation="unclear",
                           trace=["Awaiting confirmation -> unclear; will re-ask."])
            return updates

    decision = run_router(messages, state.get("channel", "web_chat"),
                          state.get("active_flow"), state.get("slots") or {})
    slots = dict(state.get("slots") or {})
    for k in ENTITY_KEYS:
        v = getattr(decision, k, None)
        if v:
            slots[k] = v
    updates.update(
        slots=slots,
        intent=decision.intent,
        handoff_reason=decision.handoff_reason,
        is_emergency=(decision.handoff_reason == "emergency"),
        rationale=decision.rationale,
        search_query=decision.search_query or latest,
    )
    trace.append(f"Router: intent={decision.intent}, handoff_reason={decision.handoff_reason}.")

    if decision.handoff_reason in {"emergency", "complaint", "fee_dispute", "commercial"}:
        updates["route"] = "handoff"
    elif decision.intent == "service_area":
        updates["route"] = "service_area"
    elif decision.intent in {"booking_create", "booking_reschedule", "booking_cancel"}:
        updates["route"] = "booking"
        updates["active_flow"] = decision.intent
    elif decision.intent == "booking_status":
        updates["route"] = "status"
    elif decision.intent == "smalltalk":
        updates["route"] = "smalltalk"
    else:  # faq_policy | out_of_scope -> let retrieval confidence decide
        updates["route"] = "answer"
    updates["trace"] = trace
    return updates


# --------------------------------------------------------------------------- answer (RAG)
def _pricing_coverage_caveat(state: AgentState, sources_detail: list[dict]) -> tuple[str, str | None]:
    """Deterministic coverage heads-up for pricing answers tied to a ZIP already in context.

    Pricing in the knowledge pack is ZIP-independent, so the RAG answerer happily quotes a
    price for any ZIP and only mentions coverage if a service-area chunk happens to be
    retrieved (unreliable / order-dependent). When the conversation already carries a ZIP
    that is out-of-area or not-covered, run the authoritative ZIP check and lead the pricing
    answer with a caveat instead of leaving it to chance.

    Returns ``(caveat_text, trace_msg)``; ``caveat_text`` is empty when nothing applies.
    """
    slots = state.get("slots") or {}
    zip_code = slots.get("zip_code")
    if not zip_code:
        return "", None
    # Only attach to pricing answers - not hours, warranty, payments, etc.
    is_pricing = any("pricing" in (s.get("source_file") or "").lower() for s in sources_detail)
    if not is_pricing:
        return "", None

    service = slots.get("service_type")
    elig = get_service_area_index().check(zip_code, service)
    trace_msg = (f"Pricing coverage check(zip={zip_code}, service={service}) -> "
                 f"{elig.status} (in_table={elig.in_coverage_table}).")
    if not elig.in_coverage_table:
        return (f"Quick heads-up: ZIP {zip_code} isn't in our standard service area on file, so a "
                f"visit there would need Branch Manager spot-approval. If we're able to serve you, "
                f"here's the ballpark:", trace_msg)
    if service and elig.status == "not_covered":
        county = f" ({elig.county})" if elig.county else ""
        return (f"Quick heads-up: we don't currently offer {service} service in {zip_code}{county}, "
                f"so this would need Branch Manager approval. If we're able to serve you, here's the "
                f"ballpark:", trace_msg)
    return "", trace_msg


def answer_node(state: AgentState) -> dict:
    updates: dict = {}
    trace = list(state.get("trace") or [])
    query = state.get("search_query") or _latest_human(state["messages"])
    res = rag_answer(query)
    trace.append(f"RAG: answerable={res['answerable']} confidence={res['confidence']} "
                 f"score={res['score']} reranker={res['used_reranker']} "
                 f"candidates={res['n_candidates']}")
    updates["trace"] = trace
    updates["retrieval_score"] = res["score"]
    updates["confidence"] = res["confidence"]
    if res["answerable"]:
        updates["citations"] = res["citations"]
        updates["sources_detail"] = res["sources_detail"]
        caveat, cov_trace = _pricing_coverage_caveat(state, res["sources_detail"])
        if cov_trace:
            trace.append(cov_trace)
        answer = f"{caveat} {res['answer']}".strip() if caveat else res["answer"]
        return _ai(updates, answer)
    # Low confidence or unsupported -> safe handoff.
    pkg = build_handoff("low_confidence", f"Could not confidently answer: {query}", state)
    updates["handoff"] = pkg
    return _ai(updates, customer_message("low_confidence"))


# --------------------------------------------------------------------------- service area
def _service_area_citation(elig) -> list[dict]:
    title = "Service Area - North Region" if (elig.region or "").lower() == "north" \
        else "Service Area - Central Region"
    return [{
        "index": 1, "doc_title": title,
        "section": f"{elig.region} / {elig.county} coverage" if elig.county else "Coverage",
        "source_file": elig.source_file or "", "version": "", "score": 1.0,
    }]


def service_area_node(state: AgentState) -> dict:
    updates: dict = {}
    trace = list(state.get("trace") or [])
    slots = state.get("slots") or {}
    zip_code = slots.get("zip_code")
    service = slots.get("service_type")

    if not zip_code:
        trace.append("Service-area check: ZIP missing -> asking.")
        updates["trace"] = trace
        return _ai(updates, "Happy to check! What's the 5-digit ZIP code of the service address?")

    elig = get_service_area_index().check(zip_code, service)
    trace.append(f"check_service_area(zip={zip_code}, service={service}) -> "
                 f"{elig.status} ({elig.county or 'n/a'}).")
    updates["trace"] = trace

    if not elig.in_coverage_table:
        pkg = build_handoff("out_of_area", elig.summary(), state,
                            recommended_route="Branch Manager (spot-approval)")
        updates["handoff"] = pkg
        msg = (f"ZIP {zip_code} isn't in our standard service area on file, so it needs Branch "
               f"Manager spot-approval. I can route your request to them - would you like that?")
        return _ai(updates, msg)

    updates["sources_detail"] = _service_area_citation(elig)
    updates["citations"] = [f"[1] {updates['sources_detail'][0]['doc_title']} "
                            f"({elig.source_file})"]
    note = (" " + " ".join(elig.notes)) if elig.notes else ""
    if service:
        if elig.status == "covered":
            msg = f"Yes - we service {zip_code} ({elig.county}, {elig.region} region) for {service}.{note}"
        elif elig.status == "sub-contracted":
            msg = (f"We do cover {service} in {zip_code} ({elig.county}), but it's handled via a "
                   f"sub-contractor and same-day service isn't available.{note}")
        elif elig.status == "pending":
            msg = f"{service.title()} service in {zip_code} ({elig.county}) is pending and not yet active.{note}"
        else:  # not_covered
            msg = f"I'm sorry - we don't currently offer {service} in {zip_code} ({elig.county}).{note}"
    else:
        cov = ", ".join(f"{k}: {v}" for k, v in elig.coverage_by_service.items())
        msg = f"For {zip_code} ({elig.county}, {elig.region} region), coverage is - {cov}.{note}"
    return _ai(updates, msg)


# --------------------------------------------------------------------------- booking
def _create_summary(slots: dict, elig_note: str) -> str:
    who = slots.get("customer_name") or "you"
    win = WINDOW_LABEL.get(slots.get("preferred_window", ""), slots.get("preferred_window", ""))
    tech = f", with {slots['preferred_tech']}" if slots.get("preferred_tech") else ""
    return (f"Please confirm this booking for {who}: {slots.get('service_type')} "
            f"{slots.get('job_type')} at ZIP {slots.get('zip_code')} on "
            f"{slots.get('preferred_date')} in the {win} window{tech}.{elig_note} {CONFIRM_HINT}")


def booking_node(state: AgentState) -> dict:
    updates: dict = {}
    trace = list(state.get("trace") or [])
    slots = dict(state.get("slots") or {})
    flow = state.get("active_flow") or "booking_create"
    channel = state.get("channel", "web_chat")

    if flow == "booking_create":
        zip_code, service = slots.get("zip_code"), slots.get("service_type")
        elig_note = ""
        if zip_code:
            elig = get_service_area_index().check(zip_code, service)
            trace.append(f"Pre-book eligibility(zip={zip_code}, service={service}) -> {elig.status}.")
            if not elig.in_coverage_table:
                updates["trace"] = trace
                pkg = build_handoff("out_of_area", elig.summary(), state)
                updates["handoff"] = pkg
                updates["active_flow"] = None
                return _ai(updates, f"ZIP {zip_code} isn't in our service area on file, so a new "
                                    f"booking there needs Branch Manager spot-approval. I'll route "
                                    f"this to them with your details.")
            if service and elig.status == "not_covered":
                updates["trace"] = trace
                updates["active_flow"] = None
                note = (" " + " ".join(elig.notes)) if elig.notes else ""
                return _ai(updates, f"I'm sorry - we don't offer {service} in {zip_code} "
                                    f"({elig.county}), so I can't book that.{note}")
            if elig.notes:
                elig_note = " Note:" + " " + " ".join(elig.notes)

        missing, issues = missing_create_slots(slots)
        if missing or issues:
            trace.append(f"Booking create: missing={missing} issues={len(issues)}.")
            updates["trace"] = trace
            updates["active_flow"] = "booking_create"
            return _ai(updates, humanize_slot_request(missing, issues))

        payload = build_create_payload(slots, channel)
        summary = _create_summary(slots, elig_note)
        trace.append("Booking create: slots complete -> awaiting confirmation.")
        updates.update(trace=trace, pending_action={"kind": "create", "payload": payload,
                                                     "summary": summary},
                       awaiting_confirmation=True, active_flow="booking_create")
        return _ai(updates, summary)

    if flow == "booking_reschedule":
        bid = slots.get("booking_id")
        new_date = slots.get("preferred_date")
        new_window = slots.get("preferred_window")
        need = []
        if not bid:
            need.append("your booking ID (looks like BK-XXXXXXXX)")
        if not new_date:
            need.append("the new date")
        if not new_window:
            need.append("the new time window (morning, midday, or afternoon)")
        if need:
            updates["trace"] = trace + [f"Reschedule: missing {need}."]
            updates["active_flow"] = "booking_reschedule"
            return _ai(updates, "To reschedule I need " + ", ".join(need) + ".")
        summary = (f"Please confirm: reschedule {bid} to {new_date} in the "
                   f"{WINDOW_LABEL.get(new_window, new_window)} window. A fee may apply per our "
                   f"cancellation policy if it's within 24 hours - I'll tell you the exact amount. "
                   f"{CONFIRM_HINT}")
        updates.update(trace=trace + ["Reschedule: ready -> awaiting confirmation."],
                       pending_action={"kind": "reschedule",
                                       "payload": {"booking_id": bid, "new_date": new_date,
                                                   "new_window": new_window}, "summary": summary},
                       awaiting_confirmation=True, active_flow="booking_reschedule")
        return _ai(updates, summary)

    # booking_cancel
    bid = slots.get("booking_id")
    if not bid:
        updates["trace"] = trace + ["Cancel: missing booking_id."]
        updates["active_flow"] = "booking_cancel"
        return _ai(updates, "Sure - what's the booking ID you'd like to cancel (BK-XXXXXXXX)?")
    reason = slots.get("cancel_reason") or "customer_request"
    summary = (f"Please confirm you want to cancel {bid}. A cancellation fee may apply per our "
               f"policy depending on timing - I'll tell you the exact amount. {CONFIRM_HINT}")
    updates.update(trace=trace + ["Cancel: ready -> awaiting confirmation."],
                   pending_action={"kind": "cancel",
                                   "payload": {"booking_id": bid, "cancel_reason": reason},
                                   "summary": summary},
                   awaiting_confirmation=True, active_flow="booking_cancel")
    return _ai(updates, summary)


# --------------------------------------------------------------------------- confirm + execute
def _fee_phrase(resp: dict) -> str:
    fee = resp.get("fee_applied", 0) or 0
    if resp.get("waiver_used"):
        return " No fee was charged (your one-time no-show waiver was applied)."
    if fee and fee > 0:
        return f" A ${fee:.0f} fee applies per our cancellation policy."
    return " No fee applies."


def confirm_node(state: AgentState) -> dict:
    updates: dict = {}
    trace = list(state.get("trace") or [])
    conf = state.get("confirmation", "unclear")
    pending = state.get("pending_action")

    if not pending:
        return _ai(updates, "Sorry, I lost track of that request. Could you tell me again what you'd like to do?")

    if conf == "no":
        updates.update(awaiting_confirmation=False, pending_action=None, active_flow=None,
                       trace=trace + ["Customer declined; cleared pending action."])
        return _ai(updates, "No problem - I won't proceed with that. Is there anything else I can help with?")
    if conf == "unclear":
        updates["trace"] = trace + ["Confirmation unclear; re-asking."]
        return _ai(updates, f"{pending['summary']}")

    kind = pending["kind"]
    payload = pending["payload"]
    try:
        if kind == "create":
            resp = booking_client.create_booking(payload)
            trace.append(f"POST /bookings -> {resp.get('status')} {resp.get('booking_id')}")
            if resp.get("status") == "out_of_area":
                updates.update(awaiting_confirmation=False, pending_action=None, active_flow=None,
                               trace=trace, tool_result=resp,
                               handoff=build_handoff("out_of_area",
                                                     f"ZIP {payload.get('zip_code')} out of area",
                                                     state))
                return _ai(updates, "It turns out that ZIP is outside our service area, so I'll "
                                    "route this to a Branch Manager for spot-approval.")
            win = resp.get("appointment_window") or {}
            tech = f" Your technician is {resp['tech_name']}." if resp.get("tech_name") else ""
            branch = f" Assigned branch: {resp['assigned_branch']}." if resp.get("assigned_branch") else ""
            msg = (f"You're booked! Confirmation {resp['booking_id']} - {payload['service_type']} "
                   f"{payload['job_type']} on {win.get('date')} from {win.get('start_time')} to "
                   f"{win.get('end_time')}.{tech}{branch} A confirmation has been sent.")
        elif kind == "reschedule":
            resp = booking_client.reschedule_booking(**payload)
            trace.append(f"PATCH /bookings (reschedule) -> {resp.get('status')}")
            win = resp.get("new_appointment_window") or {}
            msg = (f"Done - {resp['booking_id']} is rescheduled to {win.get('date')} "
                   f"({win.get('start_time')}-{win.get('end_time')}).{_fee_phrase(resp)}")
        else:  # cancel
            resp = booking_client.cancel_booking(**payload)
            trace.append(f"PATCH /bookings (cancel) -> {resp.get('status')}")
            msg = f"Your booking {resp['booking_id']} is cancelled.{_fee_phrase(resp)}"
    except booking_client.BookingAPIError as exc:
        trace.append(f"Booking API error: {exc.message}")
        updates.update(awaiting_confirmation=False, pending_action=None, trace=trace,
                       handoff=build_handoff("api_error", f"{kind} failed: {exc.message}", state))
        return _ai(updates, customer_message("api_error"))

    updates.update(awaiting_confirmation=False, pending_action=None, active_flow=None,
                   slots={}, tool_result=resp, trace=trace)
    return _ai(updates, msg)


# --------------------------------------------------------------------------- status
def status_node(state: AgentState) -> dict:
    updates: dict = {}
    trace = list(state.get("trace") or [])
    slots = state.get("slots") or {}
    bid = slots.get("booking_id")
    if not bid:
        updates["trace"] = trace + ["Status: booking_id missing -> asking."]
        return _ai(updates, "Sure - what's your booking ID? It looks like BK-XXXXXXXX.")
    try:
        data = booking_client.get_booking(bid, customer_id=slots.get("customer_id"))
        trace.append(f"GET /bookings/{bid} -> {data.get('status')}")
    except booking_client.BookingAPIError as exc:
        trace.append(f"GET /bookings/{bid} error: {exc.message}")
        updates["trace"] = trace
        if exc.status_code == 404:
            return _ai(updates, f"I couldn't find a booking with ID {bid}. Could you double-check it?")
        updates["handoff"] = build_handoff("api_error", f"status lookup failed: {exc.message}", state)
        return _ai(updates, customer_message("api_error"))

    updates["trace"] = trace
    updates["tool_result"] = data
    win = data.get("appointment_window") or {}
    when = f"{win.get('date')} from {win.get('start_time')} to {win.get('end_time')}" if win else "TBD"
    tech = data.get("tech_name")
    status = data.get("status")
    if status == "en_route":
        eta = data.get("tech_eta_minutes")
        msg = (f"Your technician {tech or ''} is on the way"
               + (f" - about {eta} minutes out" if eta is not None else "")
               + f". Your window is {when}.")
    elif status == "completed":
        inv = data.get("invoice_total")
        msg = f"Booking {bid} is completed" + (f"; invoice total ${inv:.0f}." if inv else ".")
    elif status in {"cancelled", "no_show"}:
        msg = f"Booking {bid} is marked {status}."
    else:
        who = f" with {tech}" if tech else ""
        msg = f"Booking {bid} is {status}. Your appointment is {when}{who}."
    return _ai(updates, msg)


# --------------------------------------------------------------------------- smalltalk + handoff
def smalltalk_node(state: AgentState) -> dict:
    return _ai({"trace": list(state.get("trace") or [])},
               "Hi! I'm the Meridian Home Services assistant. I can answer questions about hours, "
               "pricing, policies, and service areas, check appointment status, and book or "
               "reschedule visits. How can I help?")


def handoff_node(state: AgentState) -> dict:
    updates: dict = {}
    reason = state.get("handoff_reason") or "out_of_scope"
    summary = state.get("rationale") or _latest_human(state["messages"])
    pkg = build_handoff(reason, summary, state)
    updates["handoff"] = pkg
    updates["trace"] = list(state.get("trace") or []) + [f"Handoff -> {pkg['recommended_route']}."]
    return _ai(updates, customer_message(reason))


# --------------------------------------------------------------------------- assembly
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("triage", triage)
    g.add_node("answer", answer_node)
    g.add_node("service_area", service_area_node)
    g.add_node("booking", booking_node)
    g.add_node("confirm", confirm_node)
    g.add_node("status", status_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("handoff", handoff_node)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", lambda s: s["route"], {
        "answer": "answer", "service_area": "service_area", "booking": "booking",
        "confirm": "confirm", "status": "status", "smalltalk": "smalltalk", "handoff": "handoff",
    })
    for node in ["answer", "service_area", "booking", "confirm", "status", "smalltalk", "handoff"]:
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


class Assistant:
    """Conversational wrapper; one ``thread_id`` == one conversation (persisted state)."""

    def __init__(self) -> None:
        self.graph = build_graph()

    def chat(self, text: str, thread_id: str = "default", channel: str = "web_chat") -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)], "channel": channel}, config=config
        )
        return result
