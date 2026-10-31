"""An agent: its folder, its identity, and creating one.

An agent is a folder in the home folder, at ~/ID, holding everything
that is its own:

    agent.json      its identity, and the tool groups it has added
    persona.md      from the template, then the user's
    context/        Markdown the agent keeps in mind
    data/           its database, logs, and backups
    .gitignore      keeps data/ out of git, if the user chooses git

and, only if the user makes them, config.toml (overrides of the
engine's settings) and tools/ (the user's own tool groups).

agent.json is the agent's license. Creating an agent writes its id,
name, and description; the id never changes, and the name may be
changed by hand, since only display reads it. Its "tools" section
records the tool groups the agent has added beyond the default ones,
each with its settings, and is the one part written afterward, by
'bountiful add' and 'bountiful remove' through write_tools(). Every
field is read here with a default, so a field added later works for
agents created before it, without anyone editing their files. A front
end that keeps its own fields in agent.json, such as an icon or
colors, reads them with its own defaults.

The other files come from templates in basic_bot/templates/agent/.
They are copied once, when the agent is created, and then belong to
the user; later versions never overwrite them.

Where an agent's folder goes is the registry's rule (~/ID). This
module only creates the folder it is given, and never inside one that
already exists.
"""

import json
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

AGENT_FILE = "agent.json"
TOOLS = "tools"
DATA_DIR = "data"
TEMPLATES = Path(__file__).parent / "templates" / "agent"

# Template files whose real names would be hidden, which packaging
# would skip, are stored without the dot and renamed when copied.
_RENAMED = {"gitignore": ".gitignore"}

# An id becomes a folder name, a keyring entry, and a log prefix, so it
# must be simple. It's lowercase, which also keeps it the same folder
# on a Mac, whose disk ignores case, and on Linux, whose disk doesn't.
VALID_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
RESERVED_IDS = {
    "bountiful": "it is the name of the application",
    "my-agent": "it is the old template's placeholder name",
}

# Letters that aren't an accent on a plain letter, so dropping accents
# alone would lose them
_LETTERS = str.maketrans({
    "ł": "l", "ø": "o", "đ": "d", "ð": "d",
    "æ": "ae", "œ": "oe", "þ": "th",
})


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


def tool_groups(agent: dict) -> dict[str, dict]:
    """The tool groups an agent has added, each with its settings.

    Takes agent.json as load_agent() returns it. Anything malformed is
    left out rather than stopping the agent.
    """
    groups = agent.get(TOOLS)
    if not isinstance(groups, dict):
        return {}
    return {
        name: settings if isinstance(settings, dict) else {}
        for name, settings in groups.items()
        if isinstance(name, str)
    }


def write_tools(agent_path: str | Path, groups: dict[str, dict]) -> None:
    """Write agent.json's tool groups, leaving every other field as it is.

    The file is written whole through a temporary file, so a crash
    never leaves it half written. With no groups, the section is
    removed rather than left empty.
    """
    path = Path(agent_path) / AGENT_FILE
    try:
        agent = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise AgentError(f"{path} couldn't be read: {e}") from e
    if not isinstance(agent, dict):
        raise AgentError(f"{path} has no id.")

    if groups:
        agent[TOOLS] = {name: groups[name] for name in sorted(groups)}
    else:
        agent.pop(TOOLS, None)

    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(agent, indent=4) + "\n")
    os.replace(temp, path)


def id_from(name: str) -> str:
    """Turn a name like "Leeroy Jones" or "O'Brien" into an id like
    'leeroy-jones' or 'o-brien'.

    Accents are dropped, so Zoë becomes zoe, and anything else that
    isn't a letter or number becomes a single hyphen. A name with no
    Latin letters or numbers, such as 小明, gives an empty id.
    """
    text = name.casefold().translate(_LETTERS)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def id_problem(agent_id: str) -> str | None:
    """Why an id can't be used, or None if it's fine.

    Whether another agent already uses it is the registry's question,
    not this one's.
    """
    if not agent_id:
        return "The agent's id can't be empty."
    if agent_id in RESERVED_IDS:
        return f"'{agent_id}' can't be used because {RESERVED_IDS[agent_id]}."
    if not VALID_ID.fullmatch(agent_id):
        return (
            f"'{agent_id}' can't be used. Use lowercase letters, numbers, "
            f"hyphens, and underscores, starting with a letter or number."
        )
    return None


def create_agent(
    folder: str | Path, agent_id: str, name: str,
    description: str = "A Bountiful agent",
) -> Path:
    """Create a new agent's folder, with its agent.json and templates.

    The folder must not exist yet: an agent is never created inside a
    folder that's already there, such as ~/Music for an agent named
    music. Returns the new folder.
    """
    problem = id_problem(agent_id)
    if problem:
        raise AgentError(problem)

    folder = Path(folder).expanduser()
    if folder.exists():
        raise AgentError(
            f"{folder} already exists. An agent needs a new folder of its "
            f"own; choose another name or place."
        )

    folder.mkdir(parents=True)
    try:
        identity = {"id": agent_id, "name": name, "description": description}
        (folder / AGENT_FILE).write_text(json.dumps(identity, indent=4) + "\n")

        for template in sorted(TEMPLATES.rglob("*")):
            if template.is_dir() or template.name.startswith("."):
                continue
            relative = template.relative_to(TEMPLATES)
            target = folder / relative.with_name(
                _RENAMED.get(relative.name, relative.name)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, target)

        (folder / DATA_DIR).mkdir()
    except BaseException:
        # Leave nothing half-made behind
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder
