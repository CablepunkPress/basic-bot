"""Send what a running agent is doing to the front ends listening."""

import queue
import threading


class Events:
    """Pass each event to every listener, each through its own queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue] = []

    def listen(self) -> queue.Queue:
        """Start listening, and return the queue the events arrive in."""
        listener: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.append(listener)
        return listener

    def stop_listening(self, listener: queue.Queue) -> None:
        """Stop sending events to a listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def publish(self, kind: str, text: str) -> None:
        """Send an event to every listener."""
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.put({"kind": kind, "text": text})
