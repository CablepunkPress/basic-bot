import importlib
import json
import logging
import pkgutil
from typing import cast

import anthropic
from anthropic.types import MessageParam, ToolUseBlock

from basic_bot.config import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    TOOL_BOX_ENABLED,
)
from basic_bot.memory import load_window
from basic_bot.models import MODELS, build_api_kwargs, extract_reply
from basic_bot.runtime import BotRuntime
from basic_bot.store import MessageStore

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def _discover_tools(package_name: str) -> dict:
    """Scan a Python package for tool modules, recursing into subpackages.

    Each module that exposes a TOOL dict (the schema) and a handler callable
    is registered. Modules and packages starting with _ are skipped.
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
            sub_tools = _discover_tools(f"{package_name}.{modname}")
            registry.update(sub_tools)
        else:
            module = importlib.import_module(f"{package_name}.{modname}")
            if hasattr(module, "TOOL") and hasattr(module, "handler"):
                name = module.TOOL["name"]
                registry[name] = {
                    "schema": module.TOOL,
                    "handler": module.handler,
                }
    return registry


def build_tool_registry(package_name: str) -> dict:
    """Build the combined tool registry for a bot.

    Tool belt tools (basic_bot.tool_belt) are always loaded. Plugin tools from the
    downstream bot's own tool_box package are auto-discovered and loaded by default.
    For the downstream bot, plugin tools can be disabled with env var TOOL_BOX_ENABLED=false.
    """
    registry = _discover_tools("basic_bot.tool_belt")

    if TOOL_BOX_ENABLED:
        plugin_tools = _discover_tools(f"{package_name}.tool_box")
        registry.update(plugin_tools)

    return registry


def _execute_tool(
    registry: dict, name: str, context: dict, tool_input: dict,
) -> str:
    """Dispatch a tool call and return the result as a JSON string."""
    entry = registry.get(name)
    if not entry:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        result = entry["handler"](context, **tool_input)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        logger.exception("Tool '%s' failed", name)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_memory_section(
    summary: str, position: dict | None, tool_registry: dict,
) -> str:
    """Describe the conversation's memory state for the model."""
    if not position or not position.get("total"):
        return ""

    total = position["total"]
    through = position["summarized_through"]
    w_start = position["window_start"]
    w_end = position["window_end"]

    parts = ["# MEMORY"]
    if summary and through:
        parts.append(
            f"Here is a running summary of earlier messages (1 through {through}):\n"
            f"{summary}\n\n"
            f"You can see messages {w_start} through {w_end} verbatim below. "
            f"This conversation has {total} messages so far, not counting the "
            "user's current message."
        )
    else:
        parts.append(
            f"You can see all {total} messages of this conversation verbatim "
            f"(messages {w_start} through {w_end}), not counting the user's "
            "current message."
        )

    if through and tool_registry:
        parts.append(
            "If you need to recall exact words or specific details from before "
            "the summary, use your search_archive tool."
        )

    return "\n\n".join(parts)


def build_system_prompt(
    runtime: BotRuntime,
    model_id: str,
    effort: str | None = None,
    thinking: bool = False,
    summary: str = "",
    position: dict | None = None,
) -> str:
    """Build the system prompt with persona, model awareness, and memory state."""
    model_config = MODELS.get(model_id, MODELS[DEFAULT_MODEL])
    display_name = model_config["display_name"]

    config_lines = [f"You are currently running on Claude {display_name}."]

    if model_config["effort_levels"] and effort:
        config_lines.append(f"Effort level is set to {effort}.")
    elif not model_config["effort_levels"]:
        config_lines.append("This model does not use effort levels.")

    config_lines.append(
        "Deep Reasoning is enabled." if thinking else "Deep Reasoning is disabled."
    )

    parts = [
        runtime.persona,
        "# MODEL\n\n" + "\n".join(config_lines),
    ]

    memory_section = _build_memory_section(summary, position, runtime.tool_registry)
    if memory_section:
        parts.append(memory_section)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat orchestration
# ---------------------------------------------------------------------------

async def chat_with_claude(
    runtime: BotRuntime,
    user_id: str,
    user_message: str,
    model_id: str = DEFAULT_MODEL,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Chat with Claude using conversation memory and tools."""

    logger.info("Processing chat for user %s", user_id)

    window, summary, position = load_window(
        runtime.store, user_id, user_message,
    )
    logger.info(
        "Context: %d prior messages, summary=%s, boundary=seq %s",
        len(window) - 1, bool(summary), position["summarized_through"],
    )

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": m["role"], "content": m["content"]})
        for m in window
    ]

    if model_id not in MODELS:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = DEFAULT_MODEL

    system_prompt = build_system_prompt(
        runtime, model_id, effort, thinking, summary, position,
    )

    tool_schemas = [
        entry["schema"] for entry in runtime.tool_registry.values()
    ]
    context = {"user_id": user_id, "store": runtime.store}

    model_config = MODELS[model_id]
    logger.info(
        "Sending to %s (effort=%s, thinking=%s) — %d messages, %d tools",
        model_config["display_name"], effort, thinking,
        len(messages), len(tool_schemas),
    )

    try:
        kwargs = build_api_kwargs(model_id, system_prompt, messages, effort, thinking)
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        while True:
            response = client.messages.create(**kwargs)

            tool_use_blocks = [
                b for b in response.content if isinstance(b, ToolUseBlock)
            ]

            if not tool_use_blocks:
                break

            assistant_content = [block.model_dump() for block in response.content]
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in tool_use_blocks:
                logger.info("Executing tool: %s", block.name)
                result = _execute_tool(
                    runtime.tool_registry, block.name, context, block.input,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages

        actual_model = response.model
        if actual_model != model_id:
            logger.warning(
                "Model mismatch: requested %s, got %s", model_id, actual_model,
            )

        actual_config = MODELS.get(actual_model, model_config)
        reply = extract_reply(response)

        logger.info(
            "Response from %s (%d chars)",
            actual_config["display_name"], len(reply),
        )

        return {
            "reply": reply,
            "model_used": actual_model,
            "display_name": actual_config["display_name"],
            "effort": effort,
            "thinking": thinking,
            "fallback": False,
            "model_mismatch": actual_model != model_id,
        }

    except Exception:
        logger.exception(
            "Error with %s, falling back to %s",
            model_config["display_name"],
            MODELS[FALLBACK_MODEL]["display_name"],
        )

        if model_id == FALLBACK_MODEL and not effort and not thinking:
            raise

        try:
            fallback_prompt = build_system_prompt(
                runtime, FALLBACK_MODEL, summary=summary, position=position,
            )
            fallback_kwargs = build_api_kwargs(
                FALLBACK_MODEL, fallback_prompt, messages,
            )
            response = client.messages.create(**fallback_kwargs)

            actual_model = response.model
            actual_config = MODELS.get(actual_model, MODELS[FALLBACK_MODEL])
            reply = extract_reply(response)

            logger.info(
                "Fallback to %s succeeded (%d chars)",
                actual_config["display_name"], len(reply),
            )

            return {
                "reply": reply,
                "model_used": actual_model,
                "display_name": actual_config["display_name"],
                "effort": None,
                "thinking": False,
                "fallback": True,
                "model_mismatch": actual_model != FALLBACK_MODEL,
            }

        except Exception:
            logger.exception("Fallback also failed")
            raise