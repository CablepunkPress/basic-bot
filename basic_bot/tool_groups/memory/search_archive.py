"""Search the archive of past conversations by meaning."""

import json

import basic_bot.config as config
from _archive import fit_messages
from basic_bot.rag import search_memory

TOOL = {
    "name": "search_archive",
    "description": (
        "Search your archive of past conversations with this user, by topic. "
        "Use this when the user asks about something the two of you discussed "
        "that isn't in your recent messages or summary. The archive holds only "
        "your conversations, never facts about the world, so don't search it "
        "to check a name, date, or fact: answer those from what you know, or "
        "with a reference tool if you have one. "
        "Returns verbatim excerpts from past turns with their sequence numbers. "
        "For looking up a specific message number or date, use recall_message instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A few keywords describing what to search for in past conversations",
            },
        },
        "required": ["query"],
    },
}


def handler(context: dict, **tool_input) -> str:
    """Execute search_archive and return JSON results."""
    store = context["store"]
    user_id = context["user_id"]
    query = tool_input.get("query", "")
    results = search_memory(store, user_id, query, context["embedder"])

    if not results:
        return json.dumps({"results": [], "note": "No matching past conversations found."})

    room = config.ARCHIVE_RESULT_CHARS
    formatted = []
    left_out = []
    for result in results:
        seq_range = f"#{result['seq_start']}–#{result['seq_end']}"
        messages = store.get_messages_in_range(
            user_id, result["seq_start"], result["seq_end"],
        )
        if not messages:
            continue
        if left_out:
            left_out.append(seq_range)
            continue
        # The best match is always shown. After it, a turn is shown
        # whole or not at all, and once one is left out, so is every
        # match after it, so the result keeps the ranking's order.
        shown, used = fit_messages(messages, room)
        if formatted and (len(shown) < len(messages) or used > room):
            left_out.append(seq_range)
            continue
        formatted.append({
            "seq_range": seq_range,
            "seq_start": result["seq_start"],
            "seq_end": result["seq_end"],
            "messages": shown,
        })
        room -= used

    note = None
    if left_out:
        note = (
            f"{len(left_out)} more matching turn(s) left out to keep this "
            f"result short: {', '.join(left_out)}. Use recall_message to "
            f"read one."
        )
    return json.dumps({"results": formatted, "note": note})
