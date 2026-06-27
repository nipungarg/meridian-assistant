"""Build a structured human-handoff package with full context.

The customer sees a short, safe message; the human agent receives a structured package
(reason, recommended route, collected fields, transcript, and what the assistant attempted).
Routes mirror the resolution paths in the example messages (file 13).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

ROUTE_FOR_REASON = {
    "emergency": "24/7 Emergency Line (1-800-555-0190)",
    "fee_dispute": "Branch Manager",
    "complaint": "Branch Manager",
    "commercial": "Sales / Commercial Accounts",
    "out_of_area": "Branch Manager (spot-approval)",
    "low_confidence": "Contact-center human agent",
    "out_of_scope": "Contact-center human agent",
    "api_error": "Contact-center human agent",
    "missing_info": "Contact-center human agent",
}

CUSTOMER_MESSAGE = {
    "emergency": ("This sounds like an emergency. Please call our 24/7 emergency line right now "
                  "at 1-800-555-0190 so we can dispatch help immediately - please don't use online "
                  "booking for emergencies."),
    "fee_dispute": ("I'm sorry for the trouble. Fee adjustments are handled by a Branch Manager, "
                    "so I'm routing you to them now with all the details."),
    "complaint": ("I'm really sorry about this experience. I'm escalating this to a Branch Manager "
                  "who can help - I won't make any commitments on their behalf."),
    "commercial": ("Commercial and multi-unit accounts are handled by our Sales team. I'll pass "
                   "your details along so they can reach out to you."),
    "out_of_area": ("That ZIP isn't in our standard service area, so it needs Branch Manager "
                    "spot-approval. I'm routing your request to them with the details."),
    "low_confidence": ("I want to make sure you get an accurate answer, so I'm connecting you with "
                       "a member of our team who can help."),
    "out_of_scope": ("That's a bit outside what I can handle, so I'm connecting you with a member "
                     "of our team who can help."),
    "api_error": ("I'm having trouble reaching our booking system right now, so I'm connecting you "
                  "with a team member who can complete this for you."),
    "missing_info": ("Let me connect you with a team member who can help finish this."),
}


def _transcript(messages: list, limit: int = 12) -> list[dict]:
    out = []
    for m in messages[-limit:]:
        if isinstance(m, HumanMessage):
            out.append({"role": "customer", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
    return out


def build_handoff(reason: str, summary: str, state: dict,
                  recommended_route: str | None = None) -> dict:
    slots = dict(state.get("slots") or {})
    collected = {k: v for k, v in slots.items() if v}
    return {
        "reason": reason,
        "recommended_route": recommended_route or ROUTE_FOR_REASON.get(reason, "Human agent"),
        "summary": summary,
        "channel": state.get("channel", "web_chat"),
        "collected_info": collected,
        "attempted": list(state.get("trace") or []),
        "transcript": _transcript(state.get("messages") or []),
    }


def customer_message(reason: str) -> str:
    return CUSTOMER_MESSAGE.get(reason, CUSTOMER_MESSAGE["out_of_scope"])
