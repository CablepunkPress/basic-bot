"""Message annotations: what the model sees before each message.

    <!-- seq:107 created_at:2026-10-11T21:50:13+00:00 -->

seq is the message's sequence number, its position in the whole
conversation, starting at 1. created_at is when it was written, in UTC,
ISO 8601, to the second. Both are the database's own column names, so
an annotation, recall_message's seq parameter, and its created_at
results all use one vocabulary.

Annotations are added each time the prompt is built and never stored.
A message's annotation never changes, so the history stays identical
from turn to turn and the prompt cache keeps it. The newest message is
annotated with the time it arrived, which tells the model the current
time.

The pattern also recognizes the older form, <!-- seq:12 -->, so text
saved before created_at was added, and replies that echo either form,
are cleaned the same way. This is the only definition: memory and the
providers all use it.
"""

import re
from datetime import datetime, timezone

# Any annotation, with or without created_at
PATTERN = re.compile(r"<!--\s*seq:\d+[^>]*-->")


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


def annotate(seq: int, created_at: str | None) -> str:
    """The annotation for one message."""
    if created_at:
        return f"<!-- seq:{seq} created_at:{to_seconds(created_at)} -->"
    return f"<!-- seq:{seq} -->"


def strip(text: str) -> str:
    """Remove every annotation from text."""
    return PATTERN.sub("", text)
