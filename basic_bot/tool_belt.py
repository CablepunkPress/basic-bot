"""Tool belt — engine-shipped tools loaded from the basic_bot package.

Belt tools (search_archive, recall_message) are always active and inherited
by every downstream bot. These are not user-configurable.
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

BELT_PACKAGE = "basic_bot.tool_belt"


def _discover(package_name: str) -> dict:
    """Scan a package for tool modules. Skips _ prefixed names."""
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


def load() -> dict:
    """Load all belt tools from the engine package."""
    tools = _discover(BELT_PACKAGE)
    logger.info("Belt tools loaded: %s", list(tools.keys()) or "none")
    return tools
