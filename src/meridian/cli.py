"""Simple terminal chat REPL for the Meridian assistant.

Run:  python -m meridian.cli            (start the mock API first; see README)
      python -m meridian.cli --debug    (print the agent's per-turn trace)
"""
from __future__ import annotations

import argparse
import uuid

from meridian.agent.graph import Assistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Meridian assistant.")
    parser.add_argument("--channel", default="web_chat",
                        choices=["ivr", "web_chat", "email", "agent"])
    parser.add_argument("--debug", action="store_true", help="show the agent's per-turn trace")
    args = parser.parse_args()

    assistant = Assistant()
    thread = uuid.uuid4().hex[:8]
    print("Meridian Home Services assistant. Type 'exit' to quit, 'reset' for a new conversation.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        if text.lower() == "reset":
            thread = uuid.uuid4().hex[:8]
            print("(started a new conversation)")
            continue
        state = assistant.chat(text, thread_id=thread, channel=args.channel)
        print(f"\nAssistant: {state.get('answer', '')}")
        if state.get("citations"):
            print("  Sources: " + " ; ".join(state["citations"]))
        if state.get("handoff"):
            print(f"  [HANDOFF -> {state['handoff']['recommended_route']}]")
        if args.debug:
            for step in state.get("trace", []):
                print("   . " + step)


if __name__ == "__main__":
    main()
