"""How a message's record is shown, and cleaning notes out of text.

The conversation goes into the prompt as plain messages. A message's
number, time, and tools reach the model only through the memory tools,
as fields of their results:

    "seq": 108, "created_at": "2026-10-11T21:50:13+00:00",
    "tools": "search_web(query=\"Sierra AI founders\") → sierra.ai, techcrunch.com"

seq is the message's sequence number, its position in the whole
conversation, starting at 1. created_at is when it was written, in UTC,
ISO 8601, to the second. The summary model sees each reply's tools the
same way.

Each call shows its inputs, each with its name, and when the tool gave
them, the sites its answer came from. The names keep inputs from being
mistaken for one another: a model may send them in any order, and
read_wikipedia("Cast", title="Brazil") read as if "Cast" were the
title (October 31, #494). A call that failed, found nothing, or was
refused says so. "none" means the reply was written without a tool. A
reply saved before tools were recorded has no tools at all, because
nobody knows what it used. The full record, with whole addresses,
stays in the database; the model sees only enough to know where a
reply's facts came from.

Messages once began with annotations, such as
<!-- seq:107 created_at:2026-10-11T21:50:13+00:00 tools:none -->, and
some stored text and model replies still hold them. strip removes
them, and strip_comments removes any <!-- … --> note from a reply,
except inside fenced code.
"""

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

# Any annotation, with or without created_at or tools
PATTERN = re.compile(r"<!--\s*seq:\d+[^>]*-->")

COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
FENCE = re.compile(r"(```.*?```)", re.DOTALL)

# The most of any one input shown, in characters
INPUT_CHARS = 80

# The most sites shown for one call
SITES = 4

# What a call that didn't simply succeed shows after its input
OUTCOME_NOTES = {
    "error": " (error)",
    "nothing found": " (nothing found)",
    "refused": " (refused)",
}


def to_seconds(timestamp: str) -> str:
    """A stored timestamp in UTC, to the second.

    "2026-10-11T21:50:13.450045+00:00" becomes
    "2026-10-11T21:50:13+00:00": the same moment, without fractions
    of a second the model can't use.
    """
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _clean(value) -> str:
    """One input as it's shown: one line, no angle brackets, not too long."""
    text = value if isinstance(value, str) else str(value)
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip().replace('"', "'")
    if len(text) > INPUT_CHARS:
        text = text[:INPUT_CHARS - 1].rstrip() + "…"
    return text


def _site(url: str) -> str:
    """A page's site: "https://www.example.com/a/b" becomes "example.com"."""
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def _render_call(record: dict) -> str:
    """One call: search_web(query="…") → example.com, other.org"""
    inputs = record.get("input") or {}
    args = [f'{_clean(key)}="{_clean(value)}"' for key, value in inputs.items()]
    text = f"{_clean(record.get('tool', '?'))}({', '.join(args)})"

    text += OUTCOME_NOTES.get(record.get("outcome", "ok"), "")

    sites = []
    for source in record.get("sources") or []:
        site = _site(source.get("url", "")) if isinstance(source, dict) else ""
        if site and site not in sites:
            sites.append(site)
    if sites:
        text += " → " + ", ".join(sites[:SITES])
    return text


def render_tools(records: list[dict]) -> str:
    """A reply's tool calls, as the memory tools and the summary model show them.

    An empty list is "none": the reply used no tools.
    """
    if not records:
        return "none"
    return "; ".join(_render_call(r) for r in records)


def strip(text: str) -> str:
    """Remove every annotation from text."""
    return PATTERN.sub("", text)


def strip_comments(text: str) -> str:
    """Remove every comment from a reply, except inside fenced code."""
    parts = FENCE.split(text)
    for i in range(0, len(parts), 2):
        cleaned = COMMENT.sub("", parts[i])
        parts[i] = re.sub(r"\n{3,}", "\n\n", cleaned)
    return "".join(parts)
