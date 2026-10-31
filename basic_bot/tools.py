"""Tool registry assembly: the tools jack.

The engine's tools jack has one shape: a tool is a name, a schema, and
a handler. Tools come in three kinds, and the engine knows none of the
plugins or custom groups by name.

Default tools (the tool belt) ship with the engine, in
basic_bot.tool_belt, and every agent has them.

Plugin tools (the tool box) come from installed packages. A package
offers its groups by declaring a folder under the basic_bot.tool_box
entry point; the engine reads them in place and never copies them.

Custom tools (the tool shed) are the user's own groups, folders in the
agent's tools/ folder. A custom group with the same name as a plugin
takes its place for that agent, and the log says so.

Plugin and custom groups load only when the agent has added them: the
"tools" section of its agent.json names each one, with its settings.
A custom folder that isn't named there is skipped, and a name with no
group behind it is logged.

A group is a folder with a tool.json manifest. Each tool is a .py file
defining a TOOL dict (the schema) and a handler(context, **input)
function. Files starting with _ are shared modules (e.g., _auth.py),
importable by sibling tools in the same group with plain imports:
`from _auth import auth_headers`.

Each tool's handler receives the agent's id and its group's settings
in its context, as context["agent_id"] and context["settings"], so one
group serves every agent with that agent's own settings and keys. A
description can name a setting as {{ name }}, filled in when the tools
load.

A group may also carry an instructions.md: guidance for using its
tools, such as when to check a fact, that goes into the system prompt
of every agent that has the group, and of no other. The system prompt
changes only at startup, so it costs nothing in cache.

The factory calls build_tools() once at startup.
"""

import importlib
import importlib.metadata
import importlib.util
import logging
import pkgutil
import re
import sys
from pathlib import Path

import basic_bot.config as config
from basic_bot.agent import AgentError, load_agent, tool_groups

logger = logging.getLogger(__name__)

BELT_PACKAGE = "basic_bot.tool_belt"
TOOL_BOX_ENTRY_POINT = "basic_bot.tool_box"
MANIFEST = "tool.json"
CUSTOM_DIR = "tools"
INSTRUCTIONS = "instructions.md"

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


# ---------------------------------------------------------------------------
# Tool Belt: package-based discovery (engine-internal)
# ---------------------------------------------------------------------------

def _discover_package(package_name: str) -> dict:
    """Scan a Python package for tool modules. Skips _ prefixed names."""
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        logger.debug("Tool package '%s' not found — skipping", package_name)
        return {}

    registry: dict = {}
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname.startswith("_"):
            continue
        if ispkg:
            registry.update(_discover_package(f"{package_name}.{modname}"))
        else:
            module = importlib.import_module(f"{package_name}.{modname}")
            if hasattr(module, "TOOL") and hasattr(module, "handler"):
                registry[module.TOOL["name"]] = {
                    "schema": module.TOOL,
                    "handler": module.handler,
                    "group": "default",
                    "settings": {},
                }
    return registry


def load_belt() -> dict:
    """Load all belt tools from the engine package."""
    tools = _discover_package(BELT_PACKAGE)
    logger.info("Default tools loaded: %s", list(tools.keys()) or "none")
    return tools


# ---------------------------------------------------------------------------
# Plugin and custom groups
# ---------------------------------------------------------------------------

def plugin_groups() -> dict[str, Path]:
    """Every plugin tool group installed, by name.

    Each package that plugs into the tools jack declares a folder of
    groups under the basic_bot.tool_box entry point. If two packages
    offer a group of the same name, the first found is kept.
    """
    groups: dict[str, Path] = {}
    for entry_point in importlib.metadata.entry_points(group=TOOL_BOX_ENTRY_POINT):
        try:
            root = Path(entry_point.load())
        except Exception:
            logger.exception("Couldn't load tool box '%s'", entry_point.name)
            continue
        if not root.is_dir():
            logger.warning("Tool box '%s' names no folder: %s", entry_point.name, root)
            continue
        for folder in sorted(root.iterdir()):
            if not (folder / MANIFEST).exists():
                continue
            if folder.name in groups:
                logger.warning(
                    "Tool group '%s' is offered twice; keeping %s",
                    folder.name, groups[folder.name],
                )
                continue
            groups[folder.name] = folder
    return groups


def custom_groups(agent_path: Path) -> dict[str, Path]:
    """The agent's custom tool groups: folders in its tools/ with a manifest."""
    tools_dir = agent_path / CUSTOM_DIR
    groups: dict[str, Path] = {}
    if not tools_dir.is_dir():
        return groups
    for entry in sorted(tools_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if not (entry / MANIFEST).exists():
            logger.warning(
                "Skipping tools/%s/ — no %s manifest", entry.name, MANIFEST,
            )
            continue
        groups[entry.name] = entry
    return groups


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


def load_added(agent_path: Path) -> tuple[dict, dict, list[str]]:
    """Load the plugin and custom groups the agent has added.

    Returns the plugin tools, the custom tools, and the instructions of
    the groups that carry them, in the groups' order. All are empty if
    TOOL_BOX_ENABLED is off.
    """
    if not config.TOOL_BOX_ENABLED:
        logger.info("Plugin and custom tools disabled")
        return {}, {}, []

    try:
        added = tool_groups(load_agent(agent_path))
    except AgentError as e:
        logger.error("Couldn't read the agent's tool groups: %s", e)
        added = {}

    plugins = plugin_groups() if added else {}
    custom = custom_groups(agent_path)
    for group in custom:
        if group not in added:
            logger.info("Skipping %s/%s/: not added", CUSTOM_DIR, group)

    loaded = {"plugin": {}, "custom": {}}
    instructions: list[str] = []
    for group, settings in sorted(added.items()):
        if group in custom:
            kind, folder = "custom", custom[group]
            if group in plugins:
                logger.info(
                    "Using the custom %s/%s/ in place of the plugin", CUSTOM_DIR, group,
                )
        elif group in plugins:
            kind, folder = "plugin", plugins[group]
        else:
            logger.warning("agent.json adds '%s', which isn't installed", group)
            continue

        tools = _load_group(folder, settings)
        for name in tools:
            for other in loaded.values():
                if name in other:
                    logger.warning(
                        "Tool '%s' from group '%s' replaces the one from '%s'",
                        name, group, other[name]["group"],
                    )
                    del other[name]
                    break
        loaded[kind].update(tools)
        text = _read_instructions(folder)
        if text and tools:
            instructions.append(text)

    logger.info("Plugin tools loaded: %s", list(loaded["plugin"]) or "none")
    logger.info("Custom tools loaded: %s", list(loaded["custom"]) or "none")
    return loaded["plugin"], loaded["custom"], instructions


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

def build_tools(agent_path: Path) -> tuple[dict, list[str]]:
    """Assemble an agent's tools, and its tool groups' instructions.

    Default tools load first; plugin and custom tools overlay them, so
    one with the same name as a default tool replaces it. The
    instructions go into the system prompt.
    """
    agent_path = Path(agent_path)
    registry = load_belt()
    plugins, custom, instructions = load_added(agent_path)
    for added in (plugins, custom):
        for name in added:
            if name in registry:
                logger.warning(
                    "Tool '%s' from group '%s' replaces the default one",
                    name, added[name]["group"],
                )
        registry.update(added)

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry, instructions
