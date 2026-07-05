"""Conversation memory — orchestration layer.

Assembles the model's context window from stored messages and summaries.
All storage access goes through the MessageStore protocol. This module
has no opinion about whether data lives in Firestore, SQLite, or
anything else.
"""

from basic_bot.config import WINDOW_FLOOR
from basic_bot.store import MessageStore


def get_messages(
    store: MessageStore, user_id: str, limit: int = WINDOW_FLOOR,
) -> list[dict[str, str]]:
    """Get recent messages for a user, in chronological order."""
    return store.get_messages(user_id, limit)


def load_window(
    store: MessageStore,
    user_id: str,
    current_message: str,
) -> tuple[list[dict[str, str]], str, dict]:
    """Assemble the model's context: verbatim window + summary + position.

    Returns (messages, summary, position) where messages is the API-ready
    list (everything after the summary boundary, plus the incoming turn).
    """
    state = store.get_state(user_id)
    after = state["summarized_through"]
    window = store.get_messages_after(user_id, after)

    messages = [{"role": m["role"], "content": m["content"]} for m in window]
    messages.append({"role": "user", "content": current_message})

    latest = state["next_seq"] - 1
    position = {
        "total": latest,
        "summarized_through": after,
        "window_start": window[0]["seq"] if window else None,
        "window_end": window[-1]["seq"] if window else None,
    }

    return messages, state["summary"], position