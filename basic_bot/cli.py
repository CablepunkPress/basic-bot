"""Terminal chat for a Bountiful agent.

Run from the agent's directory, with the agent's Python:

    .venv/bin/python -m basic_bot.cli

Everything here goes through basic_bot.session. If this file ever
needs something the session doesn't offer, that's engine work living
in a front end, and it belongs in the session instead.

Engine logs go to ~/.{agent-id}/{agent-id}.log, so they don't interrupt
the conversation.
"""

import sys
from pathlib import Path

HELP = """\
Commands:
  /models              list models for the current host
  /model N             switch to model N in that list
  /host local | api    switch host
  /reasoning on | off  turn Deep Reasoning on or off
  /effort LEVEL        set effort, e.g. /effort high
  /history             show recent messages
  /help                show this list
  /quit                leave (Ctrl-D works too)
Anything else is sent as a message."""

HOST_NAMES = {"local": "local", "api": "API"}


def _describe(controls) -> str:
    """What's running and how it's set, from the session's controls."""
    host = HOST_NAMES.get(controls.host, controls.host)
    lines = [f"Running {controls.model.display_name} ({host})"]
    if controls.reasoning_locked:
        lines.append("Deep Reasoning: always on")
    elif controls.reasoning_shown:
        lines.append(f"Deep Reasoning: {'on' if controls.reasoning_on else 'off'}")
    if controls.effort_levels:
        lines.append(
            f"Effort: {controls.effort} "
            f"(choices: {', '.join(controls.effort_levels)})"
        )
    return "\n".join(lines)


def _badge(reply) -> str:
    """The same details the web UI shows under each reply."""
    parts = [reply.display_name]
    if reply.effort:
        parts.append(reply.effort.capitalize())
    if reply.thinking:
        parts.append("Deep Reasoning")
    if reply.fallback:
        parts.append("fallback")
    return "[" + " · ".join(parts) + "]"


def _command(session, line: str) -> bool:
    """Handle one slash command. Returns False when it's time to quit."""
    name, _, arg = line[1:].partition(" ")
    name, arg = name.lower(), arg.strip().lower()

    if name in ("quit", "exit"):
        return False

    if name == "help":
        print(HELP)

    elif name == "models":
        controls = session.controls()
        for number, model in enumerate(controls.models, start=1):
            mark = "*" if model.id == controls.model.id else " "
            print(f" {mark} {number}. {model.display_name}")

    elif name == "model":
        models = session.controls().models
        if not arg.isdigit() or not 1 <= int(arg) <= len(models):
            print(f"Use /model N, with N from 1 to {len(models)}. See /models.")
            return True
        model = models[int(arg) - 1]
        if model.host == "local":
            print(f"Loading {model.display_name}...", flush=True)
        session.select_model(model.id)
        print(_describe(session.controls()))

    elif name == "host":
        if arg not in HOST_NAMES:
            print("Use /host local or /host api.")
            return True
        print(f"Switching to {HOST_NAMES[arg]}...", flush=True)
        session.select_host(arg)
        print(_describe(session.controls()))

    elif name == "reasoning":
        if arg not in ("on", "off"):
            print("Use /reasoning on or /reasoning off.")
            return True
        session.set_reasoning(arg == "on")
        print(_describe(session.controls()))

    elif name == "effort":
        if not arg:
            print("Use /effort LEVEL, e.g. /effort high.")
            return True
        session.set_effort(arg)
        print(_describe(session.controls()))

    elif name == "history":
        for message in session.history():
            who = "you" if message["role"] == "user" else session.name
            text = " ".join(message["content"].split())
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  #{message['seq']} {who}: {text}")

    else:
        print(f"Unknown command /{name}. Type /help for the list.")

    return True


def _converse(session) -> None:
    from basic_bot.session import SessionError

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        try:
            if line.startswith("/"):
                if not _command(session, line):
                    return
                continue

            print("Thinking...", end="", flush=True)
            try:
                reply = session.send(line)
            finally:
                print("\r" + " " * 11 + "\r", end="", flush=True)
            print(f"{session.name}> {reply.text}")
            print(_badge(reply))

        except SessionError as e:
            print(e)
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> None:
    agent_path = Path.cwd()
    if not (agent_path / "dashboard.json").exists():
        sys.exit(
            "Run this from an agent's directory, the folder with dashboard.json."
        )

    from basic_bot.session import Session, SessionError

    try:
        session = Session.open(agent_path)
    except SessionError as e:
        sys.exit(f"ERROR: {e}")

    with session:
        print(f"{session.name}  (engine log: {session.log_path})")
        print("Starting...", flush=True)
        try:
            session.start()
        except SessionError as e:
            sys.exit(f"ERROR: {e}")
        print(_describe(session.controls()))
        print("Type /help for commands.")
        _converse(session)
        print("Stopping...", flush=True)


if __name__ == "__main__":
    main()
