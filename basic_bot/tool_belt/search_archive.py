"""Universal tool: search long-term conversation memory via RAG.

Always available to every bot (belt tool). Lets the agent retrieve
verbatim past exchanges that have left the sliding window and been
folded into the summary. Results include sequence number ranges for
cross-referencing with recall_message.
"""

import json

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
    query = tool_input.get("query", "")
    results = search_memory(
        context["store"],
        context["user_id"],
        query,
        context["embedder"],
    )

    if not results:
        return json.dumps({"results": [], "note": "No matching past conversations found."})

    formatted = []
    for result in results:
        formatted.append({
            "seq_range": f"#{result['seq_start']}–#{result['seq_end']}",
            "seq_start": result["seq_start"],
            "seq_end": result["seq_end"],
            "content": result["content"],
        })

    return json.dumps({"results": formatted})
