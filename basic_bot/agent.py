"""An agent's agent.json: its identity, read with defaults.

agent.json is the agent's license: issued once, when the agent is
created, in the folder created for it. Creating an agent writes only
id, name, and description, and nothing else ever writes them.
Everything else is read here with a default, so a field added later
works for agents created before it, without anyone editing their
files.

A front end that keeps its own fields in agent.json, such as an icon
or colors, reads them with its own defaults.
"""

import json
from pathlib import Path

AGENT_FILE = "agent.json"


class AgentError(ValueError):
    """agent.json is missing or can't be used. The message is user-facing."""


def load_agent(agent_path: str | Path) -> dict:
    """The agent's agent.json, with defaults for missing fields."""
    path = Path(agent_path) / AGENT_FILE
    try:
        agent = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise AgentError(
            f"{path} is missing, so {Path(agent_path)} isn't an agent's folder."
        ) from e
    except (OSError, ValueError) as e:
        raise AgentError(f"{path} couldn't be read: {e}") from e

    agent_id = agent.get("id") if isinstance(agent, dict) else None
    if not isinstance(agent_id, str) or not agent_id:
        raise AgentError(f"{path} has no id.")

    agent.setdefault("name", agent_id)
    agent.setdefault("description", "A Bountiful agent")
    return agent
