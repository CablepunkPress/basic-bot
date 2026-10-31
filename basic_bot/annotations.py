"""Message annotations: what the model sees before each message.

    <!-- seq:107 created_at:2026-10-11T21:50:13+00:00 -->

seq is the message's sequence number, its position in the whole
conversation, starting at 1. created_at is when it was written, in UTC,
ISO 8601, to the second. Both are the database's own column names, so
an annotation, recall_message's seq parameter, and its created_at
results all use one vocabulary.

A reply also says which tools were used to write it, from the tools
column saved with it:

    <!-- seq:108 created_at:… tools:search_web("Sierra AI founders") → sierra.ai, techcrunch.com -->
    <!-- seq:110 created_at:… tools:none -->

Each call shows its input and, when the tool gave them, the sites its
answer came from. A call that failed, found nothing, or was refused
says so. "tools:none" means the reply was written without a tool. A
reply saved before tools were recorded has no tools part at all,
because nobody knows what it used. The full record, with whole
addresses, stays in the database; the model sees only enough to know
where a reply's facts came from.

Annotations are added each time the prompt is built and never stored.
A message's annotation never changes, so the history stays identical
from turn to turn and the prompt cache keeps it. The newest message is
annotated with the time it arrived, which tells the model the current
time.

The pattern also recognizes the older form, <!-- seq:12 -->, so text
saved before created_at was added, and replies that echo any form, are
cleaned the same way. Nothing inside an annotation may contain ">",
which would end the pattern early, so inputs are cleaned of it. This
is the only definition: memory, the summary, and the providers all
use it.
"""

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

# Any annotation, with or without created_at or tools
PATTERN = re.compile(r"<!--\s*seq:\d+[^>]*-->")

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
    """One call: search_web("query") → example.com, other.org"""
    inputs = record.get("input") or {}
    args = []
    for i, (key, value) in enumerate(inputs.items()):
        shown = f'"{_clean(value)}"'
        args.append(shown if i == 0 else f"{_clean(key)}={shown}")
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
    """A reply's tool calls, as the model and the summary model see them.

    An empty list is "none": the reply used no tools.
    """
    if not records:
        return "none"
    return "; ".join(_render_call(r) for r in records)


def annotate(seq: int, created_at: str | None, tools: list[dict] | None = None) -> str:
    """The annotation for one message.

    tools is the reply's tool record. None, for a user's message or a
    reply saved before tools were recorded, leaves the tools part out.
    """
    parts = [f"seq:{seq}"]
    if created_at:
        parts.append(f"created_at:{to_seconds(created_at)}")
    if tools is not None:
        parts.append(f"tools:{render_tools(tools)}")
    return f"<!-- {' '.join(parts)} -->"


def strip(text: str) -> str:
    """Remove every annotation from text."""
    return PATTERN.sub("", text)
