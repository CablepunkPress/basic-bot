"""Load an agent's tools.

A tool is a name, a schema, and a handler. Tools come in groups. A
group is a folder that holds:

- tool.json, which describes the group
- one .py file per tool, with TOOL and handler(context, **input)
- shared code, in files whose names start with an underscore
- settings.toml, the group's default settings, if it has any
- instructions.md, which joins the system prompt, if it has one

Every agent has the memory group, which comes with basic-bot. Other
groups come from installed packages, through the basic_bot.tool_groups
entry point, or from a folder of custom groups. An agent gets the
built-in groups, which the session is given when it opens, and the
plugin groups it adds in agent.json. A custom group replaces an
installed group with the same name.
"""

import importlib.metadata
import importlib.util
import json
import logging
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import basic_bot.config as config
from basic_bot.agent import AgentError, load_agent, tool_groups

logger = logging.getLogger(__name__)

MEMORY_GROUP = Path(__file__).parent / "tool_groups" / "memory"
ENTRY_POINT = "basic_bot.tool_groups"
MANIFEST = "tool.json"
SETTINGS = "settings.toml"
INSTRUCTIONS = "instructions.md"

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True)
class Group:
    """A tool group, as its tool.json describes it for people."""
    name: str
    source: str
    description: str
    sends: str
    keys: list[dict]
    folder: Path
    tools: list[str]


def read_group(folder: Path, source: str) -> Group:
    """Read a group's tool.json and the names of its tools."""
    try:
        data = json.loads((folder / MANIFEST).read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return Group(
        name=folder.name,
        source=source,
        description=str(data.get("description", "")),
        sends=str(data.get("sends", "")),
        keys=[k for k in data.get("keys", []) if isinstance(k, dict) and k.get("name")],
        folder=folder,
        tools=sorted(f.stem for f in folder.glob("*.py") if not f.name.startswith("_")),
    )


def offered_groups() -> dict[str, tuple[Path, str]]:
    """Return every tool group offered by an installed package, with the package's name."""
    groups: dict[str, tuple[Path, str]] = {}
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT):
        source = entry_point.dist.name if entry_point.dist else entry_point.name
        try:
            root = Path(entry_point.load())
        except Exception:
            logger.exception("Couldn't load the tool groups of '%s'", source)
            continue
        if not root.is_dir():
            logger.warning("'%s' names no folder of tool groups: %s", source, root)
            continue
        for folder in sorted(root.iterdir()):
            if not (folder / MANIFEST).exists():
                continue
            if folder.name in groups:
                logger.warning(
                    "Tool group '%s' is offered twice; keeping the one from %s",
                    folder.name, groups[folder.name][1],
                )
                continue
            groups[folder.name] = (folder, source)
    return groups


def installed_groups() -> dict[str, Path]:
    """Return every tool group offered by an installed package, by name."""
    return {name: folder for name, (folder, _) in offered_groups().items()}


def custom_groups(folder: Path | None) -> dict[str, Path]:
    """Return the custom tool groups in a folder, by name."""
    groups: dict[str, Path] = {}
    if folder is None or not folder.is_dir():
        return groups
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if not (entry / MANIFEST).exists():
            logger.warning("Skipping %s: it has no %s", entry, MANIFEST)
            continue
        groups[entry.name] = entry
    return groups


def default_settings(folder: Path) -> dict:
    """Return a group's default settings from its settings.toml."""
    path = folder / SETTINGS
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.error("Couldn't read %s: %s", path, e)
        return {}


def _fill(value: object, settings: dict, group: str) -> object:
    """Fill {{ name }} placeholders in a schema's text from settings.

    Builds new dicts and lists rather than changing the old ones, so
    the module's own TOOL is never altered.
    """
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in settings:
                logger.warning(
                    "Tool group '%s' names {{ %s }}, which its settings don't have",
                    group, name,
                )
                return ""
            return str(settings[name])
        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, dict):
        return {k: _fill(v, settings, group) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, settings, group) for v in value]
    return value


def _fill_schema(schema: dict, settings: dict, group: str) -> dict:
    """A tool's schema with its placeholders filled in."""
    return {key: _fill(value, settings, group) for key, value in schema.items()}


def _load_module_from_file(module_name: str, file_path: Path):
    """Load a single .py file as a module under a synthetic name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_group(group_dir: Path, settings: dict) -> dict:
    """Load all tools from one group folder, with the agent's settings.

    The folder is temporarily added to sys.path so tool files can
    import sibling _ modules directly (`from _auth import ...`). Shared
    modules are removed from sys.modules afterward so same-named shared
    modules in different groups never collide.
    """
    registry: dict = {}
    group = group_dir.name
    shared_names = [p.stem for p in group_dir.glob("_*.py")]

    sys.path.insert(0, str(group_dir))
    try:
        for py_file in sorted(group_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            synthetic = f"agent_tools_{group}_{py_file.stem}"
            try:
                module = _load_module_from_file(synthetic, py_file)
            except Exception:
                logger.exception("Failed to load tool file %s", py_file)
                continue
            if hasattr(module, "TOOL") and hasattr(module, "handler"):
                schema = _fill_schema(module.TOOL, settings, group)
                registry[schema["name"]] = {
                    "schema": schema,
                    "handler": module.handler,
                    "group": group,
                    "settings": settings,
                }
    finally:
        sys.path.remove(str(group_dir))
        for name in shared_names:
            sys.modules.pop(name, None)

    return registry


def _wanted(agent_path: Path, built_in: tuple[str, ...]) -> tuple[dict, set[str]]:
    """Return the groups to load with their settings, and the plugins the agent added."""
    wanted: dict[str, dict | None] = {name: None for name in built_in}
    if not config.PLUGINS_ENABLED:
        logger.info("Plugin tools are turned off")
        return wanted, set()
    try:
        added = tool_groups(load_agent(agent_path))
    except AgentError as e:
        logger.error("Couldn't read the agent's tool groups: %s", e)
        added = {}
    wanted.update(added)
    return wanted, set(added)


def load_groups(
    agent_path: Path, built_in: tuple[str, ...], custom_dir: Path | None,
) -> tuple[dict, list[str]]:
    """Load the built-in groups and the plugin groups this agent added."""
    wanted, added = _wanted(agent_path, built_in)
    installed = installed_groups() if wanted else {}
    custom = custom_groups(custom_dir)
    for group in custom:
        if group not in wanted:
            logger.info("Skipping the custom group '%s': not added", group)

    registry: dict = {}
    instructions: list[str] = []
    loaded: dict[str, list[str]] = {"built-in": [], "plugin": []}
    for group, settings in wanted.items():
        if group in custom:
            folder = custom[group]
            if group in installed:
                logger.info("Using the custom group '%s' in place of the installed one", group)
        elif group in installed:
            folder = installed[group]
        elif group in added:
            logger.warning("agent.json adds '%s', which isn't installed", group)
            continue
        else:
            logger.warning("The tool group '%s' isn't installed", group)
            continue

        tools = _load_group(folder, default_settings(folder) if settings is None else settings)
        for name in tools:
            if name in registry:
                logger.warning(
                    "Tool '%s' from group '%s' replaces the one from '%s'",
                    name, group, registry[name]["group"],
                )
        registry.update(tools)
        loaded["plugin" if group in added else "built-in"].extend(tools)
        text = _read_instructions(folder)
        if text and tools:
            instructions.append(text)

    logger.info("Built-in tools loaded: %s", loaded["built-in"] or "none")
    logger.info("Plugin tools loaded: %s", loaded["plugin"] or "none")
    return registry, instructions


def _read_instructions(folder: Path) -> str:
    """A group's instructions.md, or "" if it has none or it can't be read."""
    path = folder / INSTRUCTIONS
    if not path.exists():
        return ""
    try:
        return path.read_text().strip()
    except OSError as e:
        logger.error("Couldn't read %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_tools(
    agent_path: Path,
    built_in: tuple[str, ...] = (),
    custom_dir: Path | None = None,
) -> tuple[dict, list[str]]:
    """Assemble an agent's tools, and its tool groups' instructions."""
    agent_path = Path(agent_path)
    registry = _load_group(MEMORY_GROUP, {})
    logger.info("Memory tools loaded: %s", list(registry) or "none")
    others, instructions = load_groups(agent_path, built_in, custom_dir)
    for name in others:
        if name in registry:
            logger.warning(
                "Tool '%s' from group '%s' replaces the memory tool",
                name, others[name]["group"],
            )
    registry.update(others)

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry, instructions
