"""Tool registry assembly — discovers and combines belt and box tools.

Belt tools ship with the engine (basic_bot.tool_belt) and are always
loaded via package discovery.

Box tools load from the agent's tools/ directory on the filesystem.
Each subdirectory of tools/ is a tool group. Files starting with _
are shared modules (e.g., _auth.py) importable by sibling tools in
the same group via plain imports: `from _auth import auth_headers`.
Loose .py files directly in tools/ are also loaded.

The factory calls build_registry() once at startup.
"""

import importlib
import importlib.util
import logging
import pkgutil
import sys
from pathlib import Path

from basic_bot.config import TOOL_BOX_ENABLED

logger = logging.getLogger(__name__)

BELT_PACKAGE = "basic_bot.tool_belt"


# ---------------------------------------------------------------------------
# Tool Belt — package-based discovery (engine-internal)
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
                }
    return registry


def load_belt() -> dict:
    """Load all belt tools from the engine package."""
    tools = _discover_package(BELT_PACKAGE)
    logger.info("Belt tools loaded: %s", list(tools.keys()) or "none")
    return tools


# ---------------------------------------------------------------------------
# Tool Box — filesystem-based discovery (agent tools/ directory)
# ---------------------------------------------------------------------------

def _load_module_from_file(module_name: str, file_path: Path):
    """Load a single .py file as a module under a synthetic name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_group(group_dir: Path) -> dict:
    """Load all tools from one directory.

    The directory is temporarily added to sys.path so tool files can
    import sibling _ modules directly (`from _auth import ...`). Shared
    modules are removed from sys.modules afterward so same-named shared
    modules in different groups never collide.
    """
    registry: dict = {}
    shared_names = [p.stem for p in group_dir.glob("_*.py")]

    sys.path.insert(0, str(group_dir))
    try:
        for py_file in sorted(group_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            synthetic = f"agent_tools_{group_dir.name}_{py_file.stem}"
            try:
                module = _load_module_from_file(synthetic, py_file)
            except Exception:
                logger.exception("Failed to load tool file %s", py_file)
                continue
            if hasattr(module, "TOOL") and hasattr(module, "handler"):
                registry[module.TOOL["name"]] = {
                    "schema": module.TOOL,
                    "handler": module.handler,
                }
    finally:
        sys.path.remove(str(group_dir))
        for name in shared_names:
            sys.modules.pop(name, None)

    return registry


def load_box(tools_dir: Path) -> dict:
    """Load box tools from an agent's tools/ directory.

    Loose .py files in tools/ load first, then each subdirectory as a
    group. Returns an empty dict if disabled or the directory is absent.
    """
    if not TOOL_BOX_ENABLED:
        logger.info("Tool box disabled")
        return {}

    if not tools_dir.is_dir():
        logger.debug("No tools directory at %s — skipping", tools_dir)
        return {}

    registry: dict = {}
    registry.update(_load_group(tools_dir))

    for entry in sorted(tools_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith(("_", ".")) \
                and entry.name != "__pycache__":
            registry.update(_load_group(entry))

    logger.info("Box tools loaded: %s", list(registry.keys()) or "none")
    return registry


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_registry(agent_path: Path) -> dict:
    """Assemble the complete tool registry for an agent.

    Belt tools load first. Box tools overlay — a box tool with the same
    name as a belt tool replaces it.
    """
    registry = {}
    registry.update(load_belt())
    registry.update(load_box(agent_path / "tools"))

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry
