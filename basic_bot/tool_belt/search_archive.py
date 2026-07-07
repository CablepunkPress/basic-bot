"""Universal tool: search long-term conversation memory via RAG.

Always available to every bot (belt tool). Lets the agent retrieve
verbatim past exchanges that have left the sliding window and been
folded into the summary. Results include sequence number ranges for
cross-referencing with recall_message.
"""

import json

TOOL = {
    "name": "search_archive",
    "description": (
        "Search the archive for specific past conversations by topic. "
        "Use this when the user asks about something you discussed before "
        "that isn't visible in your current conversation window or summary. "
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
    from basic_bot.rag import search_memory

    query = tool_input.get("query", "")
    results = search_memory(
        context["store"],
        context["user_id"],
        query,
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