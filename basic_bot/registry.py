"""The agents on this machine."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from basic_bot.agent import AgentError, load_agent
from basic_bot.paths import AGENTS_REGISTRY


class RegistryError(ValueError):
    """An agent can't be recorded or found. The message is user-facing."""


@dataclass(frozen=True)
class Entry:
    """One recorded agent."""
    agent_id: str
    folder: Path

    @property
    def present(self) -> bool:
        """True if the folder is still there and still an agent."""
        return (self.folder / "agent.json").is_file()


def _read() -> dict:
    try:
        data = json.loads(AGENTS_REGISTRY.read_text())
    except FileNotFoundError:
        return {"agents": {}}
    except (OSError, ValueError) as e:
        raise RegistryError(f"{AGENTS_REGISTRY} couldn't be read: {e}") from e
    agents = data.get("agents") if isinstance(data, dict) else None
    return {"agents": dict(agents) if isinstance(agents, dict) else {}}


def _write(data: dict) -> None:
    """Write the list whole, so a crash never leaves it half written."""
    AGENTS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=AGENTS_REGISTRY.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    os.replace(temp, AGENTS_REGISTRY)


def entries() -> list[Entry]:
    """Every recorded agent, by id."""
    return [
        Entry(agent_id, Path(folder))
        for agent_id, folder in sorted(_read()["agents"].items())
    ]


def find(agent_id: str) -> Entry | None:
    """The recorded agent with this id, or None."""
    for entry in entries():
        if entry.agent_id == agent_id:
            return entry
    return None


def id_taken(agent_id: str) -> Path | None:
    """The folder of another agent already using this id, or None.

    A recorded folder that's gone doesn't count; recording the id
    again replaces it.
    """
    entry = find(agent_id)
    if entry and entry.present:
        return entry.folder
    return None


def record(folder: str | Path) -> Entry:
    """Record an agent folder, reading its id from its agent.json."""
    folder = Path(folder).expanduser().resolve()
    try:
        agent_id = load_agent(folder)["id"]
    except AgentError as e:
        raise RegistryError(str(e)) from e

    data = _read()
    existing = data["agents"].get(agent_id)
    if existing and Path(existing) != folder and (Path(existing) / "agent.json").is_file():
        raise RegistryError(
            f"An agent with the id '{agent_id}' is already recorded at "
            f"{existing}. Two agents can't share an id. If {folder} is a "
            f"copy, keep one; if it moved, run 'bountiful forget "
            f"{agent_id}' and import it again."
        )

    data["agents"][agent_id] = str(folder)
    _write(data)
    return Entry(agent_id, folder)


def forget(agent_id: str) -> Path:
    """Remove an agent from the list. Its folder is never touched.

    Returns the folder that was recorded.
    """
    data = _read()
    if agent_id not in data["agents"]:
        raise RegistryError(f"No agent with the id '{agent_id}' is recorded.")
    folder = Path(data["agents"].pop(agent_id))
    _write(data)
    return folder
