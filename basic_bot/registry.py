"""The agents on this machine.

An agent lives in the home folder, in a folder named by its id, and
nowhere else: the id, the folder's name, and the database's name are
one word. The registry records only ids, in agents.json, since the
folder always follows from the id. Registering an id makes it
something 'bountiful ID' can start; forgetting it touches nothing
but this list.
"""

import json
import os
import tempfile
from pathlib import Path

from basic_bot.agent import AGENT_FILE, AgentError, load_agent
from basic_bot.paths import AGENTS_REGISTRY


class RegistryError(ValueError):
    """An agent can't be registered or found. The message is user-facing."""


def folder(agent_id: str) -> Path:
    """The folder of the agent with this id: always ~/ID."""
    return Path.home() / agent_id


def _read() -> list[str]:
    try:
        data = json.loads(AGENTS_REGISTRY.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        raise RegistryError(f"{AGENTS_REGISTRY} couldn't be read: {e}") from e
    agents = data.get("agents") if isinstance(data, dict) else None
    if not isinstance(agents, list):
        raise RegistryError(f"{AGENTS_REGISTRY} isn't a list of ids.")
    return [a for a in agents if isinstance(a, str)]


def _write(agent_ids: list[str]) -> None:
    """Write the list whole, so a crash never leaves it half written."""
    AGENTS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=AGENTS_REGISTRY.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"agents": sorted(agent_ids)}, indent=2) + "\n")
    os.replace(temp, AGENTS_REGISTRY)


def ids() -> list[str]:
    """Every registered id, sorted."""
    return sorted(_read())


def registered(agent_id: str) -> bool:
    return agent_id in _read()


def register(agent_id: str) -> None:
    """Register the agent in ~/ID, once it's checked to be that agent.

    The folder must exist, hold agent.json, and that file's id must be
    this id. The database isn't checked: the session finds it, or
    creates it, when the agent starts.
    """
    found = _read()
    if agent_id in found:
        raise RegistryError(f"'{agent_id}' is already registered.")

    place = folder(agent_id)
    shown = f"~/{agent_id}"
    if not place.is_dir():
        raise RegistryError(f"{shown} doesn't exist.")
    if not (place / AGENT_FILE).is_file():
        raise RegistryError(f"{shown} has no {AGENT_FILE}.")
    try:
        file_id = load_agent(place)["id"]
    except AgentError as e:
        raise RegistryError(str(e)) from e
    if file_id != agent_id:
        raise RegistryError(f"{shown}/{AGENT_FILE} has the id '{file_id}'.")

    _write([*found, agent_id])


def forget(agent_id: str) -> None:
    """Unregister an agent. Its folder is never touched."""
    found = _read()
    if agent_id not in found:
        raise RegistryError(f"'{agent_id}' isn't registered.")
    found.remove(agent_id)
    _write(found)
