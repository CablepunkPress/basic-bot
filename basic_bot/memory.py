"""Conversation memory — orchestration layer.

Assembles the model's context window from stored messages and summaries.
All storage access goes through the MessageStore protocol. This module
has no opinion about whether data lives in Firestore, SQLite, or
anything else.

Messages from storage carry seq, created_at, and metadata. This module
strips them down to role/content for the context window, but prefixes
each message with an annotation of its seq and created_at (see
basic_bot.annotations), so the model can refer to messages by number
and knows when each was written. The richer data is available to
callers that need it (history endpoint, UI).
"""

import basic_bot.config as config
from basic_bot.annotations import annotate
from basic_bot.annotations import strip as strip_annotations
from basic_bot.store import MessageStore


def get_messages(
    store: MessageStore, user_id: str, limit: int | None = None,
) -> list[dict]:
    """Get recent messages for a user, in chronological order.

    Returns full message dicts including seq and metadata. The default
    limit is read from config when called, so overrides apply.
    """
    if limit is None:
        limit = config.WINDOW_FLOOR
    return store.get_messages(user_id, limit)


def load_history(
    store: MessageStore,
    user_id: str,
) -> tuple[list[dict[str, str]], str, dict]:
    """The stored conversation as the model sees it, before a new message.

    Returns (messages, summary, state): everything after the summary
    boundary, stripped to role/content and each prefixed with its
    annotation, exactly as load_window begins the next turn's prompt.
    Warming reads this, so the cache it prepares matches that prompt
    token for token.
    """
    state = store.get_state(user_id)
    window = store.get_messages_after(user_id, state["summarized_through"])
    messages = [
        {
            "role": m["role"],
            "content": (
                annotate(m["seq"], m.get("created_at"))
                + strip_annotations(m["content"]).strip()
            ),
        }
        for m in window
    ]
    return messages, state["summary"], state


def load_window(
    store: MessageStore,
    user_id: str,
    current_message: str,
    sent_at: str,
) -> tuple[list[dict[str, str]], str, dict]:
    """Assemble the model's context: verbatim window + summary + position.

    sent_at is when the current message arrived, a UTC ISO timestamp.
    It's the time saved with the message, so the annotation the model
    sees now matches the one it will see on every later turn.

    Returns (messages, summary, position) where messages is the API-ready
    list stripped to role/content (everything after the summary boundary,
    plus the incoming turn), each prefixed with its annotation.
    """
    state = store.get_state(user_id)
    after = state["summarized_through"]
    window = store.get_messages_after(user_id, after)

    messages = [
        {
            "role": m["role"],
            "content": (
                annotate(m["seq"], m.get("created_at"))
                + strip_annotations(m["content"]).strip()
            ),
        }
        for m in window
    ]

    current_seq = state["next_seq"]
    messages.append({
        "role": "user",
        "content": annotate(current_seq, sent_at) + current_message,
    })

    latest = state["next_seq"] - 1
    position = {
        "total": latest,
        "summarized_through": after,
        "window_start": window[0]["seq"] if window else None,
        "window_end": window[-1]["seq"] if window else None,
    }

    return messages, state["summary"], position
