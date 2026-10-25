"""Tool registry assembly: the tools jack.

The engine's tools jack has one shape: a tool is a name, a schema, and
a handler. Three kinds of plug fit it, and the engine knows none of
them by name.

Belt tools ship with the engine (basic_bot.tool_belt) and every agent
has them.

Tool box groups are curated add-ons from an installed package. A
package offers its groups by declaring a folder under the
basic_bot.tool_box entry point; the engine reads them in place and
never copies them. An agent uses a curated group when its tools/
folder holds a settings file named for it, tools/<group>.toml, which
also holds the agent's settings for that group. Deleting the file
turns the group off.

The agent's own groups are folders in its tools/ folder. An agent's
own group with the same name as a curated group wins, and the log says
so, the way box tools already override belt tools. Its settings, if
any, are in tools/<group>.toml too.

A group is a folder with a tool.json manifest. Each tool is a .py file
defining a TOOL dict (the schema) and a handler(context, **input)
function. Files starting with _ are shared modules (e.g., _auth.py),
importable by sibling tools in the same group with plain imports:
`from _auth import auth_headers`.

Each tool's handler receives the agent's id and its group's settings
in its context, as context["agent_id"] and context["settings"], so one
group serves every agent with that agent's own settings and keys. A
description can name a setting as {{ name }}, filled in when the tools
load, so the model sees, for example, the GitHub account it works in.

A group may also carry an instructions.md: guidance for using its
tools, such as when to check a fact, that goes into the system prompt
of every agent that has the group on, and of no other. It plugs into
the prompt the way the tools plug into the registry. The system prompt
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
import tomllib
from pathlib import Path

import basic_bot.config as config

logger = logging.getLogger(__name__)

BELT_PACKAGE = "basic_bot.tool_belt"
TOOL_BOX_ENTRY_POINT = "basic_bot.tool_box"
MANIFEST = "tool.json"
SETTINGS_SUFFIX = ".toml"
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
                    "group": "belt",
                    "settings": {},
                }
    return registry


def load_belt() -> dict:
    """Load all belt tools from the engine package."""
    tools = _discover_package(BELT_PACKAGE)
    logger.info("Belt tools loaded: %s", list(tools.keys()) or "none")
    return tools


# ---------------------------------------------------------------------------
# Tool Box: curated groups from packages, and the agent's own groups
# ---------------------------------------------------------------------------

def curated_groups() -> dict[str, Path]:
    """Every curated tool group installed, by name.

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


def own_groups(tools_dir: Path) -> dict[str, Path]:
    """The agent's own tool groups: folders in tools/ with a manifest."""
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


def read_settings(tools_dir: Path, group: str) -> dict:
    """An agent's settings for one group, from tools/<group>.toml.

    An absent file is empty settings. A file that can't be read is
    logged and treated as empty, so one bad file doesn't stop the agent.
    """
    path = tools_dir / f"{group}{SETTINGS_SUFFIX}"
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.error("Couldn't read tools/%s: %s", path.name, e)
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


def load_box(tools_dir: Path) -> tuple[dict, list[str]]:
    """Load the agent's tool box: curated groups it uses, and its own.

    A curated group is used when tools/<group>.toml exists. Every
    group folder in tools/ with a manifest is the agent's own. Returns
    the tools, and the instructions of the groups that carry them, in
    the groups' order. Both are empty if the tool box is disabled.
    """
    if not config.TOOL_BOX_ENABLED:
        logger.info("Tool box disabled")
        return {}, []

    own = own_groups(tools_dir)
    curated = curated_groups()

    chosen: dict[str, Path] = {}
    if tools_dir.is_dir():
        for settings_file in sorted(tools_dir.glob(f"*{SETTINGS_SUFFIX}")):
            group = settings_file.stem
            if group in own:
                continue  # settings for the agent's own group
            if group in curated:
                chosen[group] = curated[group]
            else:
                logger.warning(
                    "tools/%s names no installed tool group", settings_file.name,
                )
    for group, folder in own.items():
        if group in curated and (tools_dir / f"{group}{SETTINGS_SUFFIX}").exists():
            logger.info(
                "Using the agent's own tools/%s/ in place of the curated group",
                group,
            )
        chosen[group] = folder

    registry: dict = {}
    instructions: list[str] = []
    for group, folder in sorted(chosen.items()):
        tools = _load_group(folder, read_settings(tools_dir, group))
        for name in tools:
            if name in registry:
                logger.warning(
                    "Tool '%s' from group '%s' replaces the one from '%s'",
                    name, group, registry[name]["group"],
                )
        registry.update(tools)
        text = _read_instructions(folder)
        if text and tools:
            instructions.append(text)

    logger.info("Box tools loaded: %s", list(registry.keys()) or "none")
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

def build_tools(agent_path: Path) -> tuple[dict, list[str]]:
    """Assemble an agent's tools, and its tool groups' instructions.

    Belt tools load first. Box tools overlay — a box tool with the same
    name as a belt tool replaces it. The instructions go into the
    system prompt.
    """
    registry = {}
    registry.update(load_belt())
    box, instructions = load_box(agent_path / "tools")
    registry.update(box)

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry, instructions


def build_registry(agent_path: Path) -> dict:
    """The agent's tools alone, for callers that don't need instructions."""
    return build_tools(agent_path)[0]
