"""Keep a llama-server from outliving the engine that started it.

    python watchdog.py -- <llama-server command ...>

server.py starts this in place of llama-server, and it starts
llama-server. To the engine, it looks exactly like llama-server:

- Output: llama-server writes to the same log; this writes nothing.
- Exit: when llama-server exits, so does this, with the same exit
  code, so a server that dies during startup is reported at once.
- Stopping: a stop request (SIGTERM) is passed on to llama-server,
  which is given GRACE seconds before being killed. GRACE is shorter
  than server.py's own wait, so a stop always finishes here first.

What it adds: the engine holds the other end of this program's stdin
and never writes to it. When the engine ends, however it ends,
including a force quit or a crash, the operating system closes that
pipe. This program sees the end of input and stops llama-server.

It uses only the standard library and imports nothing from the
engine, so it reads no config and can't affect anything by loading.
It runs as a file, not as part of the package, for the same reason.
"""

import signal
import subprocess
import sys
import threading

# Seconds llama-server gets to exit after a stop request before it's
# killed. Must stay below server.py's wait, so this side finishes first.
GRACE = 5

# How often to check whether llama-server has exited on its own.
POLL_INTERVAL = 0.2


def _exit_code(returncode: int) -> int:
    """Mirror llama-server's exit. A process ended by a signal has a
    negative returncode; the shell convention for it is 128 + signal."""
    return returncode if returncode >= 0 else 128 - returncode


def main() -> None:
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.exit("usage: watchdog.py -- <command ...>")

    stop = threading.Event()

    # A stop request from the engine. The handler only sets the event;
    # the main loop does the stopping, outside the handler.
    def request_stop(signum, frame):
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request_stop)

    # llama-server inherits stdout and stderr, which are the log file.
    # Its stdin is empty: the pipe from the engine is for this program
    # alone.
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    # The engine never writes, so read() returns only when the pipe
    # closes, which happens when the engine ends, however it ends.
    def watch_engine():
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        stop.set()

    threading.Thread(target=watch_engine, daemon=True).start()

    # Wait for llama-server to exit on its own, or for a stop request
    while child.poll() is None and not stop.wait(POLL_INTERVAL):
        pass

    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=GRACE)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    sys.exit(_exit_code(child.returncode))


if __name__ == "__main__":
    main()
