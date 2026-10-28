"""Universal tool: recall specific messages by sequence number or date.

Always available to every bot (belt tool). Provides deterministic
lookup of past messages — by exact sequence number, sequence range,
or date range. Complements search_archive's semantic search with
precise, addressable retrieval.

What a result can hold is limited, since it goes into the chat
model's prompt. A message longer than config.ARCHIVE_MESSAGE_CHARS is
shown as its beginning and a note saying how long it is; the message
stays whole in the store. A result shows messages in order until
config.ARCHIVE_RESULT_CHARS, then says where to pick up. search_archive
shows its messages through fit_messages() here too, so both tools
show a message the same way.
"""

import json

import basic_bot.config as config
from basic_bot.annotations import to_seconds

TOOL = {
    "name": "recall_message",
    "description": (
        "Look up specific past messages by their sequence number or by date. "
        "Use this when the user references a specific message number like #222, "
        "or asks about a conversation from a specific date or time period. "
        "For topic-based searches, use search_archive instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "seq": {
                "type": "integer",
                "description": "A single message sequence number to look up. Returns the message plus a few surrounding messages for context.",
            },
            "start_seq": {
                "type": "integer",
                "description": "Start of a sequence range (inclusive). Use with end_seq.",
            },
            "end_seq": {
                "type": "integer",
                "description": "End of a sequence range (inclusive). Use with start_seq.",
            },
            "after_date": {
                "type": "string",
                "description": "Start date in ISO format (e.g., '2026-06-01'). Use with before_date.",
            },
            "before_date": {
                "type": "string",
                "description": "End date in ISO format (e.g., '2026-06-07'). Use with after_date.",
            },
        },
    },
}

# Context window around a single seq lookup
SEQ_CONTEXT = 2

# Maximum range span to prevent unbounded queries
MAX_SEQ_RANGE = 50
MAX_DATE_RESULTS = 50

# Ends a message cut for length. It tells the model that it isn't
# seeing the whole message, so it doesn't quote it as complete, and
# that the message itself was abnormal, so it doesn't go looking for
# the rest: the tools never show more of it.
CUT_NOTE = (
    "\n[Cut for length: this message is {total:,} characters long, far "
    "longer than a message should be; only its first {shown:,} are "
    "shown here.]"
)


def _cut(content: str) -> tuple[str, int]:
    """A message's content as the tools show it, and how many of its
    characters that is.

    Within config.ARCHIVE_MESSAGE_CHARS, the content is whole.
    Otherwise it's cut at the last whitespace within the limit, so a
    word isn't split, and the note is added.
    """
    limit = config.ARCHIVE_MESSAGE_CHARS
    if len(content) <= limit:
        return content, len(content)
    head = content[:limit]
    space = max(head.rfind(" "), head.rfind("\n"))
    if space > 0:
        head = head[:space]
    head = head.rstrip()
    return head + CUT_NOTE.format(total=len(content), shown=len(head)), len(head)


def _format_message(msg: dict) -> tuple[dict, int]:
    """Format a message for tool output; also returns the characters
    of it shown."""
    content, shown = _cut(msg["content"])
    formatted = {
        "seq": msg["seq"],
        "role": msg["role"],
        "content": content,
    }
    if msg.get("metadata"):
        formatted["metadata"] = msg["metadata"]
    if msg.get("created_at"):
        formatted["created_at"] = to_seconds(msg["created_at"])
    return formatted, shown


def fit_messages(messages: list[dict], room: int) -> tuple[list[dict], int]:
    """Format messages in order while what they show fits in room.

    Returns the formatted messages and the characters they show. The
    first message is always included, so a lookup never comes back
    empty.
    """
    formatted: list[dict] = []
    used = 0
    for msg in messages:
        item, shown = _format_message(msg)
        if formatted and used + shown > room:
            break
        formatted.append(item)
        used += shown
    return formatted, used


def _stopped_note(shown: list[dict], messages: list[dict]) -> str | None:
    """Where to pick up, if the result stopped before the last message."""
    if len(shown) == len(messages):
        return None
    last = shown[-1]["seq"]
    following = messages[len(shown)]["seq"]
    return (
        f"Stopped after #{last} to keep this result short. "
        f"{len(messages) - len(shown)} more message(s) matched; "
        f"ask for #{following} onward to read them."
    )


def _notes(*notes: str | None) -> str | None:
    """The notes that apply, as one, or None."""
    present = [n for n in notes if n]
    return " ".join(present) if present else None


def handler(context: dict, **tool_input) -> str:
    """Execute recall_message with seq or date lookup."""
    store = context["store"]
    user_id = context["user_id"]

    seq = tool_input.get("seq")
    start_seq = tool_input.get("start_seq")
    end_seq = tool_input.get("end_seq")
    after_date = tool_input.get("after_date")
    before_date = tool_input.get("before_date")

    room = config.ARCHIVE_RESULT_CHARS

    # Single seq: look up with surrounding context
    if seq is not None:
        start = max(1, seq - SEQ_CONTEXT)
        end = seq + SEQ_CONTEXT
        messages = store.get_messages_in_range(user_id, start, end)

        if not messages:
            return json.dumps({"error": f"No message found at #{seq}."})

        shown, _ = fit_messages(messages, room)
        return json.dumps({
            "target_seq": seq,
            "messages": shown,
            "note": _stopped_note(shown, messages),
        })

    # Seq range
    if start_seq is not None and end_seq is not None:
        if end_seq - start_seq > MAX_SEQ_RANGE:
            return json.dumps({
                "error": f"Range too large. Maximum span is {MAX_SEQ_RANGE} messages.",
            })

        messages = store.get_messages_in_range(user_id, start_seq, end_seq)

        if not messages:
            return json.dumps({"error": f"No messages found in range #{start_seq}–#{end_seq}."})

        shown, _ = fit_messages(messages, room)
        return json.dumps({
            "range": f"#{start_seq}–#{end_seq}",
            "count": len(shown),
            "messages": shown,
            "note": _notes(
                f"Results capped at {MAX_SEQ_RANGE}. There may be more messages outside this range."
                if len(messages) >= MAX_SEQ_RANGE else None,
                _stopped_note(shown, messages),
            ),
        })

    # Date range
    if after_date is not None and before_date is not None:
        messages = store.get_messages_by_date(
            user_id, after_date, before_date, limit=MAX_DATE_RESULTS,
        )

        if not messages:
            return json.dumps({
                "error": f"No messages found between {after_date} and {before_date}.",
            })

        shown, _ = fit_messages(messages, room)
        return json.dumps({
            "date_range": f"{after_date} to {before_date}",
            "count": len(shown),
            "messages": shown,
            "note": _notes(
                f"Results capped at {MAX_DATE_RESULTS}. There may be more messages in this date range."
                if len(messages) >= MAX_DATE_RESULTS else None,
                _stopped_note(shown, messages),
            ),
        })

    return json.dumps({
        "error": "Provide either seq, start_seq+end_seq, or after_date+before_date.",
    })
