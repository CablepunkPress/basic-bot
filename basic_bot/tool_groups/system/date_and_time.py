"""Read the current date and time from the user's computer."""

import json
from datetime import datetime, timezone

TOOL = {
    "name": "date_and_time",
    "description": (
        "Get the current date and time from the user's computer, in its own "
        "time zone, with the day of the week, and in UTC. You have no clock "
        "of your own, and neither your training nor the conversation tells "
        "you today's date. Use this whenever you need the date, the day, or "
        "the time: when the user asks, or before you work out how long ago "
        "or how soon something is."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def handler(context: dict, **tool_input) -> str:
    """Return the computer's local date and time, and UTC."""
    now = datetime.now().astimezone()
    return json.dumps({
        "date": f"{now:%A, %B} {now.day}, {now.year}",
        "time": f"{now.hour % 12 or 12}:{now:%M %p}",
        "time_zone": now.tzname(),
        "local": now.isoformat(timespec="seconds"),
        "utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
    })
