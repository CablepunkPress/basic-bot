"""Tool registry assembly — discovers and combines belt and box tools.

Belt tools ship with the engine (basic_bot.tool_belt) and are always
loaded. Box tools come from the downstream bot's tool_box package and
can be disabled with TOOL_BOX_ENABLED=false.

The factory calls build_registry() once at startup. The resulting dict
is stored on the BotRuntime and passed through the chat loop.
"""

import importlib
import logging
import pkgutil

from basic_bot.config import TOOL_BOX_ENABLED

logger = logging.getLogger(__name__)

BELT_PACKAGE = "basic_bot.tool_belt"


def _discover(package_name: str) -> dict:
    """Scan a package for tool modules, recursing into subpackages.

    Each module exposing a TOOL dict and a handler callable is registered.
    Modules and packages starting with _ are skipped.
    """
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
            registry.update(_discover(f"{package_name}.{modname}"))
        else:
            module = importlib.import_module(f"{package_name}.{modname}")
            if hasattr(module, "TOOL") and hasattr(module, "handler"):
                name = module.TOOL["name"]
                registry[name] = {
                    "schema": module.TOOL,
                    "handler": module.handler,
                }
    return registry


def load_belt() -> dict:
    """Load all belt tools from the engine package."""
    tools = _discover(BELT_PACKAGE)
    logger.info("Belt tools loaded: %s", list(tools.keys()) or "none")
    return tools


def load_box(package_name: str) -> dict:
    """Load box tools from a downstream bot's tool_box package.

    Returns an empty dict if TOOL_BOX_ENABLED is false or the package
    has no tool_box.
    """
    if not TOOL_BOX_ENABLED:
        logger.info("Tool box disabled")
        return {}

    tools = _discover(f"{package_name}.tool_box")
    logger.info("Box tools loaded: %s", list(tools.keys()) or "none")
    return tools


def build_registry(package_name: str) -> dict:
    """Assemble the complete tool registry from all sources.

    Belt tools load first. Box tools overlay — a box tool with the same
    name as a belt tool replaces it.
    """
    registry = {}
    registry.update(load_belt())
    registry.update(load_box(package_name))

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry
