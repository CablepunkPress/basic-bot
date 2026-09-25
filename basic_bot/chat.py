"""Chat orchestration.

Provider-agnostic conversation loop: loads memory, builds the prompt,
calls the provider, handles tool execution, and returns the result.
The provider handles all API translation internally.

The prompt is ordered from most stable to most volatile so the
inference server's prompt cache can reuse as much as possible. The
system prompt (persona, instructions, context, tools, summary)
changes only at startup or at a fold. The transcript only grows.
Per-turn state rides in a note at the end of the newest message.
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
# System prompt — stable between startup and fold
# ---------------------------------------------------------------------------

def _build_tools_section(tool_registry: dict) -> str:
    """List the available tools by name. Fixed for the session."""
    if not tool_registry:
        return ""
    names = ", ".join(sorted(tool_registry.keys()))
    return f"# TOOLS\n\nYou have {len(tool_registry)} tools: {names}."


def _build_memory_section(summary: str, summarized_through: int) -> str:
    """Present the rolling summary. Changes only at a fold."""
    if summary and summarized_through:
        return (
            "# MEMORY\n\n"
            f"Running summary of messages 1 through {summarized_through}:\n\n"
            f"{summary}"
        )
    return "# MEMORY\n\nNo earlier messages have been summarized yet."


def build_system_prompt(
    runtime: BotRuntime,
    summary: str = "",
    summarized_through: int = 0,
) -> str:
    """Build the system prompt: stable prefix, tools, and memory."""
    parts = [runtime.persona]

    tools_section = _build_tools_section(runtime.tool_registry)
    if tools_section:
        parts.append(tools_section)

    parts.append(_build_memory_section(summary, summarized_through))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Turn note — per-turn state, appended to the newest user message
# ---------------------------------------------------------------------------

def _model_name(model_info: ModelInfo) -> str:
    """Full model name without repeating the family.

    Claude display names omit the family ("Haiku 4.5"), local display
    names include it ("Qwen3.5 9B Q5_K_M"). Prepend only when missing.
    """
    if model_info.display_name.startswith(model_info.family):
        return model_info.display_name
    return f"{model_info.family} {model_info.display_name}"


def _build_turn_note(
    model_info: ModelInfo, effort: str | None, thinking: bool,
) -> str:
    """State the current model and settings for this turn only."""
    lines = [f"You are running on {_model_name(model_info)}."]

    if model_info.effort_levels and effort:
        lines.append(f"Effort is set to {effort}.")
    elif not model_info.effort_levels:
        lines.append("This model does not use effort levels.")

    lines.append(
        "Deep Reasoning is enabled." if thinking else "Deep Reasoning is disabled."
    )

    return "<!-- turn note, written by the system: " + " ".join(lines) + " -->"


def _prepare_messages(window: list[dict], turn_note: str) -> list[dict]:
    """Copy the window for the API and attach the turn note.

    The note goes on the newest user message in the prompt only. The
    store keeps the user's original words. A fresh copy is built per
    attempt so a fallback never inherits a failed attempt's tool calls.
    """
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in window
    ]
    messages[-1]["content"] += "\n\n" + turn_note
    return messages


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
    provider: InferenceProvider = runtime.chat_provider

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
    summarized_through = position["summarized_through"]
    logger.info(
        "Context: %d prior messages, summary=%s, boundary=seq %s",
        len(window) - 1, bool(summary), summarized_through,
    )

    # One system prompt for every attempt — it does not depend on the model
    system_prompt = build_system_prompt(runtime, summary, summarized_through)

    tool_schemas = [
        entry["schema"] for entry in runtime.tool_registry.values()
    ]
    context = {"user_id": user_id, "store": runtime.store, "embedder": runtime.embedder}

    messages = _prepare_messages(
        window, _build_turn_note(model_info, effort, thinking),
    )

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
            fallback_messages = _prepare_messages(
                window, _build_turn_note(fallback_info, None, False),
            )

            response = _chat_loop(
                provider, fallback_messages, system_prompt,
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
