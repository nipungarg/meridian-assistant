"""Tool functions used by the graph nodes: grounded RAG answering and booking slot logic.

These are plain functions (the graph orchestrates them deterministically rather than via a
free-form ReAct loop) and could be wrapped as LangChain ``@tool`` objects for a tool-calling
agent without changing their signatures.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from functools import lru_cache

from pydantic import BaseModel, Field

from meridian.agent.llm import get_chat_llm
from meridian.agent.prompts import GROUNDED_SYSTEM
from meridian.config import get_settings
from meridian.retrieval.citations import build_citations, format_context, format_sources
from meridian.retrieval.retriever import get_retriever

WINDOW_LABEL = {
    "morning": "morning (7-11am)", "midday": "midday (11am-2pm)",
    "afternoon": "afternoon (2-6pm)", "first_available": "first available",
}
REQUIRED_CREATE = ["service_type", "job_type", "zip_code", "preferred_date", "preferred_window"]
SLOT_PROMPTS = {
    "service_type": "which service you need (HVAC, plumbing, or electrical)",
    "job_type": "the type of job (e.g. diagnostic, repair, install, tune-up, or estimate)",
    "zip_code": "the 5-digit ZIP code of the service address",
    "preferred_date": "your preferred date",
    "preferred_window": "a time window (morning, midday, or afternoon)",
    "customer_name": "your name",
    "customer_phone": "a contact phone number",
}


class GroundedAnswer(BaseModel):
    answerable: bool
    answer: str = ""
    used_sources: list[int] = Field(default_factory=list)


# Cross-references in the knowledge pack: the plumbing pricing doc and the emergencies FAQ
# both defer to the HVAC after-hours surcharge ("same as HVAC policy" / "see the HVAC and
# Plumbing pricing documents...") instead of restating the figures. Resolving the reference
# at answer time lets emergency/after-hours answers state the actual amounts instead of
# pointing the customer at another document - without touching the vector index.
_HVAC_SURCHARGE_REFERENCES = (
    "same as hvac policy",
    "see the hvac and plumbing pricing documents",
)


@lru_cache(maxsize=1)
def _after_hours_surcharge_text() -> str:
    """Read the after-hours surcharge figures once from the HVAC pricing doc (single source)."""
    from pathlib import Path

    from meridian.ingestion.chunkers import chunk_document
    from meridian.ingestion.pdf_extract import extract_pdf

    path = Path(get_settings().files_dir) / "03_hvac_pricing.pdf"
    if not path.exists():
        return ""
    for ch in chunk_document(extract_pdf(path)):
        if "after-hours" in (ch.metadata.get("section") or "").lower():
            lines = ch.text.splitlines()[1:]   # drop the "[breadcrumb]" prefix line
            if lines and lines[0].strip().lower().startswith("after-hours"):
                lines = lines[1:]              # drop the repeated heading line
            return " ".join(ln.strip() for ln in lines if ln.strip())
    return ""


def _resolve_surcharge_cross_reference(context: str) -> str:
    """Append the HVAC after-hours figures when the context defers to them but omits them."""
    low = context.lower()
    if not any(ref in low for ref in _HVAC_SURCHARGE_REFERENCES):
        return context
    if "+$75" in context and "+$125" in context:
        return context  # the actual amounts are already in the retrieved sources
    surcharge = _after_hours_surcharge_text()
    if not surcharge:
        return context
    return (f"{context}\n\n[After-Hours Surcharge - resolved from the referenced HVAC pricing "
            f"policy]\n{surcharge}")


def rag_answer(query: str) -> dict:
    """Retrieve, check confidence, then answer using ONLY the retrieved sources."""
    retriever = get_retriever()
    result = retriever.retrieve(query)
    base = {
        "score": round(result.top_score, 3),
        "confidence": round(result.confidence, 3),  # embedding-similarity gate signal
        "used_reranker": result.used_reranker,
        "n_candidates": len(result.chunks),
    }
    if not result.chunks or not result.is_confident():
        return {**base, "answerable": False, "reason": "low_confidence",
                "answer": "", "citations": [], "sources_detail": []}

    context = _resolve_surcharge_cross_reference(format_context(result.chunks))
    llm = get_chat_llm(0.0).with_structured_output(GroundedAnswer)
    ga: GroundedAnswer = llm.invoke([
        ("system", GROUNDED_SYSTEM),
        ("human", f"SOURCES:\n{context}\n\nQUESTION: {query}"),
    ])
    if not ga.answerable or not ga.answer.strip():
        return {**base, "answerable": False, "reason": "not_in_context",
                "answer": "", "citations": [], "sources_detail": []}

    used = [i for i in ga.used_sources if 1 <= i <= len(result.chunks)]
    used_chunks = [result.chunks[i - 1] for i in used] or result.chunks
    cites = build_citations(used_chunks)
    return {
        **base,
        "answerable": True,
        "answer": ga.answer.strip(),
        "citations": format_sources(used_chunks),
        "sources_detail": [asdict(c) for c in cites],
    }


# --------------------------------------------------------------------------- booking slots
def _valid_iso_date(value: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _date_issue(value: str) -> str | None:
    d = _valid_iso_date(value)
    if d is None:
        return "I couldn't read that date - could you give it as a specific day (e.g. next Monday)?"
    today = get_settings().today
    if d < today:
        return "That date is in the past. What upcoming date works?"
    if d > today + dt.timedelta(days=60):
        return "We can only book up to 60 days ahead. Could you pick a sooner date?"
    return None


def missing_create_slots(slots: dict) -> tuple[list[str], list[str]]:
    """Return (missing_slot_keys, issue_messages) for a new booking."""
    missing = [k for k in REQUIRED_CREATE if not slots.get(k)]
    has_identity = bool(slots.get("customer_id")) or (
        slots.get("customer_name") and slots.get("customer_phone"))
    if not has_identity:
        if not slots.get("customer_name"):
            missing.append("customer_name")
        if not slots.get("customer_phone"):
            missing.append("customer_phone")
    issues: list[str] = []
    if slots.get("preferred_date"):
        issue = _date_issue(slots["preferred_date"])
        if issue:
            issues.append(issue)
    return missing, issues


def build_create_payload(slots: dict, channel: str) -> dict:
    payload = {
        "customer_id": slots.get("customer_id"),
        "service_type": slots.get("service_type"),
        "job_type": slots.get("job_type"),
        "zip_code": slots.get("zip_code"),
        "preferred_date": slots.get("preferred_date"),
        "preferred_window": slots.get("preferred_window"),
        "channel": channel if channel in {"ivr", "web_chat", "email", "agent"} else "web_chat",
    }
    if slots.get("preferred_tech"):
        payload["preferred_tech"] = slots["preferred_tech"]
    if slots.get("notes"):
        payload["notes"] = slots["notes"]
    if not slots.get("customer_id"):
        payload["customer_info"] = {
            "name": slots.get("customer_name"),
            "phone": slots.get("customer_phone"),
            "email": slots.get("customer_email"),
            "address": slots.get("customer_address"),
        }
    return payload


def humanize_slot_request(missing: list[str], issues: list[str]) -> str:
    parts = [SLOT_PROMPTS.get(k, k) for k in dict.fromkeys(missing)]
    msg = ""
    if parts:
        if len(parts) == 1:
            msg = f"To book that, I just need {parts[0]}."
        else:
            msg = "To book that, I need " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
    if issues:
        msg = (msg + " " + " ".join(issues)).strip()
    return msg or "Could you share a bit more detail so I can book that?"
