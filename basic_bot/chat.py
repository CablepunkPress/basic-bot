"""Chat orchestration.

Provider-agnostic conversation loop: loads memory, builds the system
prompt, calls the provider, handles tool execution, and returns the
result. The provider handles all API translation internally.
"""

import json
import logging

from basic_bot.providers.protocol import ChatResponse, InferenceProvider, ModelInfo
from basic_bot.memory import load_window
from basic_bot.runtime import BotRuntime

logger = logging.getLogger(__name__)


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
    parts.append(
        "Messages below contain <!-- seq:N --> annotations for internal tracking. "
        "Your reply is plain prose beginning with your first content word."
    )
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
    model_info: ModelInfo,
    effort: str | None = None,
    thinking: bool = False,
    summary: str = "",
    position: dict | None = None,
) -> str:
    """Build the system prompt with persona, model awareness, and memory state."""
    config_lines = [
        f"You are currently running on {model_info.family} {model_info.display_name}."
    ]

    if model_info.effort_levels and effort:
        config_lines.append(f"Effort level is set to {effort}.")
    elif not model_info.effort_levels:
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

async def chat_with_model(
    runtime: BotRuntime,
    user_id: str,
    user_message: str,
    model_id: str | None = None,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Chat using conversation memory and tools.

    Returns a result dict with the reply text and metadata reflecting
    what the API actually used (not what was requested, where detectable).
    """
    provider: InferenceProvider = runtime.provider

    if model_id is None:
        model_id = provider.get_default_model()

    models = provider.get_models()
    if model_id not in models:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = provider.get_default_model()

    model_info = models[model_id]
    fallback_id = provider.get_fallback_model()
    fallback_info = models[fallback_id]

    logger.info("Processing chat for user %s", user_id)

    window, summary, position = load_window(
        runtime.store, user_id, user_message,
    )
    logger.info(
        "Context: %d prior messages, summary=%s, boundary=seq %s",
        len(window) - 1, bool(summary), position["summarized_through"],
    )

    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in window
    ]

    system_prompt = build_system_prompt(
        runtime, model_info, effort, thinking, summary, position,
    )

    tool_schemas = [
        entry["schema"] for entry in runtime.tool_registry.values()
    ]
    context = {"user_id": user_id, "store": runtime.store}

    logger.info(
        "Sending to %s (effort=%s, thinking=%s) — %d messages, %d tools",
        model_info.display_name, effort, thinking,
        len(messages), len(tool_schemas),
    )

    try:
        response = _chat_loop(
            provider, messages, system_prompt,
            tool_schemas, context, runtime.tool_registry,
            model_id, effort, thinking,
        )

        if response.model_used != model_id:
            logger.warning(
                "Model mismatch: requested %s, got %s",
                model_id, response.model_used,
            )

        used_info = models.get(response.model_used, model_info)

        logger.info(
            "Response from %s (%d chars, thinking=%s)",
            used_info.display_name, len(response.text), response.thinking,
        )

        return {
            "reply": response.text,
            "model_used": response.model_used,
            "display_name": used_info.display_name,
            "effort": effort,
            "thinking": response.thinking,
            "fallback": False,
            "model_mismatch": response.model_used != model_id,
        }

    except Exception:
        logger.exception(
            "Error with %s, falling back to %s",
            model_info.display_name, fallback_info.display_name,
        )

        if model_id == fallback_id and not effort and not thinking:
            raise

        try:
            fallback_prompt = build_system_prompt(
                runtime, fallback_info, summary=summary, position=position,
            )

            response = _chat_loop(
                provider, messages, fallback_prompt,
                tool_schemas, context, runtime.tool_registry,
                fallback_id, None, False,
            )

            used_info = models.get(response.model_used, fallback_info)

            logger.info(
                "Fallback to %s succeeded (%d chars, thinking=%s)",
                used_info.display_name, len(response.text), response.thinking,
            )

            return {
                "reply": response.text,
                "model_used": response.model_used,
                "display_name": used_info.display_name,
                "effort": None,
                "thinking": response.thinking,
                "fallback": True,
                "model_mismatch": response.model_used != fallback_id,
            }

        except Exception:
            logger.exception("Fallback also failed")
            raise


def _chat_loop(
    provider: InferenceProvider,
    messages: list[dict],
    system: str,
    tool_schemas: list[dict],
    context: dict,
    tool_registry: dict,
    model_id: str,
    effort: str | None,
    thinking: bool,
) -> ChatResponse:
    """Call the provider in a loop, executing tools until no more are requested."""
    while True:
        response = provider.chat(
            messages=messages,
            system=system,
            tools=tool_schemas or None,
            model_id=model_id,
            effort=effort,
            thinking=thinking,
        )

        if not response.tool_calls:
            return response

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response.text,
            "tool_calls": response.tool_calls,
        })

        # Execute tools and append results
        results = []
        for tc in response.tool_calls:
            logger.info("Executing tool: %s", tc.name)
            result = _execute_tool(tool_registry, tc.name, context, tc.input)
            results.append({"tool_call_id": tc.id, "content": result})

        messages.append({"role": "tool_result", "results": results})
