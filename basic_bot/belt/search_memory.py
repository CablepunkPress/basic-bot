"""Universal tool: search long-term conversation memory via RAG.

Always available to every bot (belt tool). Lets the agent retrieve
verbatim past exchanges that have left the sliding window and been
folded into the summary.
"""

import json

TOOL = {
    "name": "search_memory",
    "description": (
        "Search your long-term memory for specific past conversations. "
        "Use this when the user asks about something you discussed before "
        "that isn't visible in your current conversation window or summary. "
        "Returns verbatim excerpts from past turns."
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
    """Execute search_memory and return JSON results."""
    from basic_bot.rag import search_memory

    query = tool_input.get("query", "")
    results = search_memory(
        context["user_id"],
        query,
        collection=context["collection"],
    )

    if not results:
        return json.dumps({"results": [], "note": "No matching past conversations found."})
    return json.dumps({"results": results})