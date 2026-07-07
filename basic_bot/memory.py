"""Conversation memory — orchestration layer.

Assembles the model's context window from stored messages and summaries.
All storage access goes through the MessageStore protocol. This module
has no opinion about whether data lives in Firestore, SQLite, or
anything else.

Messages from storage carry seq and metadata fields. This module strips
them down to role/content for the Claude API context window, but prefixes
each message with its sequence number so the model can reference specific
messages by number. The richer data is available to callers that need it
(history endpoint, UI).
"""

from basic_bot.config import WINDOW_FLOOR
from basic_bot.store import MessageStore


def get_messages(
    store: MessageStore, user_id: str, limit: int = WINDOW_FLOOR,
) -> list[dict]:
    """Get recent messages for a user, in chronological order.

    Returns full message dicts including seq and metadata.
    """
    return store.get_messages(user_id, limit)


def load_window(
    store: MessageStore,
    user_id: str,
    current_message: str,
) -> tuple[list[dict[str, str]], str, dict]:
    """Assemble the model's context: verbatim window + summary + position.

    Returns (messages, summary, position) where messages is the API-ready
    list stripped to role/content (everything after the summary boundary,
    plus the incoming turn). Each message is prefixed with its sequence
    number so the model can reference messages by number.
    """
    state = store.get_state(user_id)
    after = state["summarized_through"]
    window = store.get_messages_after(user_id, after)

    messages = [
        {"role": m["role"], "content": f"<!-- seq:{m['seq']} -->{m['content']}"}
        for m in window
    ]

    current_seq = state["next_seq"]
    messages.append({"role": "user", "content": f"(seq:{current_seq}) {current_message}"})

    latest = state["next_seq"] - 1
    position = {
        "total": latest,
        "summarized_through": after,
        "window_start": window[0]["seq"] if window else None,
        "window_end": window[-1]["seq"] if window else None,
    }

    return messages, state["summary"], position