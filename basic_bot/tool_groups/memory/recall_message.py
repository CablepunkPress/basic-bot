"""Look up past messages by number, by a range of numbers, or by dates."""

import json

import basic_bot.config as config
from _archive import fit_messages

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
