"""What the model sees of the conversation.

The prompt holds the summary and every message after it. This module
reads those messages from the store and turns each into what the model
is shown. A user's message is shown as it was written. So is a reply,
but a reply that used tools comes after its calls, each followed by a
short note standing in for the result:

    the call           search_web(query="Sierra AI founders")
    the note           This result was shown in full during that turn, ...
    the reply          Sierra was founded by ...

A call and its note are messages in the form every chat API uses for
tool calls, and the model's own chat template shows them the way it
shows calls during a turn. So the model can see which of its replies
came from a lookup, and what a real call looks like. A result is shown
whole only during the turn that called for it. Afterward the note says
it has been removed since, so the model calls the tool again rather
than recalling what the result said.

A message's number and time aren't shown. The model reads them through
its memory tools.

Warming and the turn it prepares for build their messages here, the
same way, so the chat server's cache matches the turn's prompt.
"""

import basic_bot.config as config
from basic_bot.annotations import sites_of
from basic_bot.annotations import strip as strip_annotations
from basic_bot.providers.protocol import ToolCall
from basic_bot.store import MessageStore

KEPT = (
    "This result was shown in full during that turn, and the reply after "
    "it was written from it. It has been removed since. To see it again, "
    "call the tool again."
)
NOTES = {
    "nothing found": "Result from an earlier turn: nothing was found.",
    "error": "Result from an earlier turn: the tool failed.",
    "refused": "Not run: the turn had reached its limit of tool calls.",
}


def get_messages(
    store: MessageStore, user_id: str, limit: int | None = None,
) -> list[dict]:
    """Return the user's most recent messages, oldest first, with all their fields.

    This is for the page's history, not the prompt. The limit is read
    from config when called, so an agent's config.toml applies.
    """
    if limit is None:
        limit = config.WINDOW_FLOOR
    return store.get_messages(user_id, limit)


def tools_of(message: dict) -> list[dict] | None:
    """Return a stored message's record of tool calls.

    It's None for a user's message and for a reply saved before tools
    were recorded, and an empty list for a reply that used no tools.
    """
    return (message.get("metadata") or {}).get("tools")


def _note(record: dict) -> str:
    """Return the note that stands in for one call's result."""
    note = NOTES.get(record.get("outcome", "ok"))
    if note:
        return note
    sites = sites_of(record)
    return KEPT + (f" Sources: {', '.join(sites)}." if sites else "")


def _as_seen(message: dict) -> list[dict]:
    """Return one stored message as the model sees it.

    A reply that used tools becomes each call and its note, in the
    order the calls were made, then the reply. Each call's id comes
    from the reply's number, so it's the same every time it's built.
    """
    seen: list[dict] = []
    if message["role"] == "assistant":
        for i, record in enumerate(tools_of(message) or [], start=1):
            call_id = f"call_{message['seq']}_{i}"
            call = ToolCall(
                name=record.get("tool", "unknown"),
                input=record.get("input") or {},
                id=call_id,
            )
            seen.append({"role": "assistant", "content": "", "tool_calls": [call]})
            seen.append({
                "role": "tool_result",
                "results": [{"tool_call_id": call_id, "content": _note(record)}],
            })
    seen.append({
        "role": message["role"],
        "content": strip_annotations(message["content"]).strip(),
    })
    return seen


def _shown(window: list[dict]) -> list[dict]:
    """Return the stored messages as the model sees them, in order."""
    return [part for message in window for part in _as_seen(message)]


def load_history(
    store: MessageStore,
    user_id: str,
) -> tuple[list[dict], str, dict]:
    """Return the conversation as the model sees it before a new message.

    Returns the messages after the summary, the summary, and the
    conversation's state. Warming uses this, and the messages are the
    ones load_window starts the next turn's prompt with.
    """
    state = store.get_state(user_id)
    window = store.get_messages_after(user_id, state["summarized_through"])
    return _shown(window), state["summary"], state


def load_window(
    store: MessageStore,
    user_id: str,
    current_message: str,
) -> tuple[list[dict], str, dict]:
    """Return the conversation as the model sees it, with the new message.

    Returns the messages after the summary followed by the new one, the
    summary, and where the conversation stands: how many messages there
    are in all, how many are summarized, how many come after the
    summary, and the numbers of the first and last of those.
    """
    state = store.get_state(user_id)
    after = state["summarized_through"]
    window = store.get_messages_after(user_id, after)

    messages = _shown(window)
    messages.append({"role": "user", "content": current_message})

    position = {
        "total": state["next_seq"] - 1,
        "summarized_through": after,
        "count": len(window),
        "window_start": window[0]["seq"] if window else None,
        "window_end": window[-1]["seq"] if window else None,
    }

    return messages, state["summary"], position
