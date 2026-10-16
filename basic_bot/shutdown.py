"""Whether the agent is shutting down.

Shutdown begins when a front end closes its session, or when the
session receives a signal to stop. From then on, the rest of the
engine behaves accordingly: no server is started, and a request cut
off because its server was stopped is reported as stopped, not
failed.

It lives in its own module because several parts of the engine need
it, whichever host is in use: the server lifecycle, the chat loop,
and the session.

The flag is set once and never cleared. A threading.Event, so any
thread can read it safely: a reply or a fold may be running on one
thread while shutdown begins on another.
"""

import threading

_shutdown = threading.Event()


def begin_shutdown() -> None:
    """Mark that shutdown has begun. Safe to call more than once."""
    _shutdown.set()


def shutting_down() -> bool:
    """Whether shutdown has begun."""
    return _shutdown.is_set()
