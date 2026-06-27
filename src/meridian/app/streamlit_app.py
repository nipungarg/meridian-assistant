"""Streamlit chat UI for the Meridian assistant.

Shows the grounded answer, the citations/sources it used, the agent's per-turn trace
(routing + tool calls), any booking API result, and the human-handoff card. Run with:

    streamlit run src/meridian/app/streamlit_app.py

NOTE: emoji are written as \\u / \\U escapes on purpose to keep this file pure ASCII.
"""
from __future__ import annotations

import logging
import uuid

import streamlit as st

from meridian import booking_client
from meridian.agent.graph import Assistant
from meridian.config import get_settings

# Streamlit's hot-reload watcher walks every imported module after each run. The
# `transformers` package (pulled in by the sentence-transformers reranker) lazily
# imports optional vision models on attribute access, several of which need
# `torchvision` - a dependency this text-only app intentionally does not install.
# The watcher trips those imports and floods the console with harmless
# "ModuleNotFoundError: No module named 'torchvision'" tracebacks. Quiet just that
# one noisy logger; all other Streamlit logging is left untouched.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

ICON_APP = "\U0001F6E0"        # hammer and wrench
ICON_OK = "\u2705"             # white check mark
ICON_ALERT = "\U0001F6A8"      # police car light
ICON_WAIT = "\u23F3"           # hourglass
ICON_HANDOFF = "\U0001F91D"    # handshake

st.set_page_config(page_title="Meridian Home Services Assistant", page_icon=ICON_APP,
                   layout="centered")


def _escape_dollars(text: str) -> str:
    """Stop Streamlit's markdown from treating currency as LaTeX.

    ``st.markdown`` (and ``st.warning`` etc.) render ``$...$`` as KaTeX math, so an
    answer like "$75 service fee ... over $200" otherwise shows up in a serif math
    italic font. The assistant never emits LaTeX, so escaping every "$" is safe.
    """
    return str(text).replace("$", "\\$")


SAMPLE_PROMPTS = [
    "What time does the Herndon branch open on Saturday?",
    "Roughly how much to replace a 40-gallon water heater?",
    "I'm at ZIP 20147 - do you service plumbing there?",
    "Book an electrical panel inspection in Rockville 20814 next Tuesday morning.",
    "What's the status of booking BK-00391042?",
    "Water is pouring out from under my sink right now!",
]


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


def _init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def _render_meta(meta: dict) -> None:
    if meta.get("awaiting"):
        st.info("Awaiting your confirmation before this action commits.", icon=ICON_WAIT)
    if meta.get("citations"):
        with st.expander(f"Sources ({len(meta['citations'])})", expanded=False):
            for c in meta["citations"]:
                st.markdown(f"- {_escape_dollars(c)}")
    if meta.get("tool_result"):
        with st.expander("Booking API result", expanded=False):
            st.json(meta["tool_result"])
    if meta.get("handoff"):
        h = meta["handoff"]
        st.warning(f"Human handoff -> **{h['recommended_route']}**  \n_{_escape_dollars(h['reason'])}_",
                   icon=ICON_HANDOFF)
        with st.expander("Handoff package (for the human agent)", expanded=False):
            st.json({k: h[k] for k in ("reason", "recommended_route", "summary",
                                       "channel", "collected_info", "attempted")})
    if meta.get("trace"):
        with st.expander("Agent trace", expanded=False):
            for step in meta["trace"]:
                st.markdown(f"- `{step}`")


def _process(text: str, channel: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "meta": {}})
    assistant = get_assistant()
    with st.spinner("Thinking..."):
        state = assistant.chat(text, thread_id=st.session_state.thread_id, channel=channel)
    meta = {
        "citations": state.get("citations") or [],
        "trace": state.get("trace") or [],
        "tool_result": state.get("tool_result"),
        "handoff": state.get("handoff"),
        "awaiting": state.get("awaiting_confirmation"),
    }
    st.session_state.messages.append(
        {"role": "assistant", "content": state.get("answer", ""), "meta": meta})


def main() -> None:
    _init_state()
    settings = get_settings()

    with st.sidebar:
        st.header("Meridian Assistant")
        st.caption("Grounded RAG + agentic booking demo")
        channel = st.selectbox("Channel", ["web_chat", "ivr", "email", "agent"], index=0)
        st.divider()
        try:
            h = booking_client.health()
            st.success(f"Booking API: up ({h.get('bookings')} bookings)", icon=ICON_OK)
        except Exception:
            st.error("Booking API: unreachable. Start it with `make api`.", icon=ICON_ALERT)
        st.caption(f"Model: `{settings.openai_chat_model}`  \nReranker: `{settings.reranker}`  \n"
                   f"Today: `{settings.today.isoformat()}`")
        st.divider()
        st.caption("Try a sample:")
        for i, p in enumerate(SAMPLE_PROMPTS):
            if st.button(p, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_prompt = p
        if st.button("Reset conversation", type="primary", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex[:8]
            st.session_state.messages = []
            st.rerun()

    st.title("Meridian Home Services")
    st.caption("Ask about hours, pricing, policies, or service areas - or book, reschedule, "
               "and check a visit. Everything is grounded in the knowledge pack with citations.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(_escape_dollars(msg["content"]))
            if msg["role"] == "assistant":
                _render_meta(msg.get("meta", {}))

    prompt = st.chat_input("Type your message...")
    if st.session_state.pending_prompt and not prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    if prompt:
        _process(prompt, channel)
        st.rerun()


main()
