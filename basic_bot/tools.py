"""Tool registry assembly — combines belt and box tools into one registry.

The factory calls build_registry() once at startup. The resulting dict
is stored on the BotRuntime and passed through the chat loop.
"""

import logging

from basic_bot import tool_belt, tool_box

logger = logging.getLogger(__name__)


def build_registry(package_name: str) -> dict:
    """Assemble the complete tool registry from all sources.

    Belt tools load first. Box tools overlay — a box tool with the same
    name as a belt tool replaces it (allows downstream override, though
    overriding belt tools is not recommended).
    """
    registry = {}
    registry.update(tool_belt.load())
    registry.update(tool_box.load(package_name))

    logger.info(
        "Tool registry: %d tools [%s]",
        len(registry),
        ", ".join(registry.keys()) or "none",
    )
    return registry
