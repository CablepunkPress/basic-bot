"""Show archived messages the same way in both memory tools."""

import basic_bot.config as config
from basic_bot.annotations import render_tools, to_seconds

CUT_NOTE = (
    "\n[Cut for length: this message is {total:,} characters long, far "
    "longer than a message should be; only its first {shown:,} are "
    "shown here.]"
)


def _cut(content: str) -> tuple[str, int]:
    """Return a message's content as the tools show it, and how many characters of it are shown."""
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
    """Format a message for a tool's result, and count the characters shown."""
    content, shown = _cut(msg["content"])
    formatted = {
        "seq": msg["seq"],
        "role": msg["role"],
        "content": content,
    }
    metadata = dict(msg.get("metadata") or {})
    tools = metadata.pop("tools", None)
    if metadata:
        formatted["metadata"] = metadata
    if tools is not None:
        formatted["tools"] = render_tools(tools)
    if msg.get("created_at"):
        formatted["created_at"] = to_seconds(msg["created_at"])
    return formatted, shown


def fit_messages(messages: list[dict], room: int) -> tuple[list[dict], int]:
    """Format messages in order while they fit in room. The first message is always included."""
    formatted: list[dict] = []
    used = 0
    for msg in messages:
        item, shown = _format_message(msg)
        if formatted and used + shown > room:
            break
        formatted.append(item)
        used += shown
    return formatted, used
