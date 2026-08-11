"""Tool box — agent-specific tools loaded from the downstream bot's package.

Currently uses package-based discovery (importing {package_name}.tool_box).
This will be replaced with filesystem-based loading from agent_path/tools/
when the factory refactor lands.
"""

import importlib
import logging
import pkgutil

from basic_bot.config import TOOL_BOX_ENABLED

logger = logging.getLogger(__name__)


def _discover(package_name: str) -> dict:
    """Scan a package for tool modules, recursing into subpackages.

    Skips modules and packages starting with _.
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


def load(package_name: str) -> dict:
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
