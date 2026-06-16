import logging
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import MessageParam

from basic_bot.config import (
    CONVERSATION_COLLECTION,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    MESSAGE_LIMIT,
    SUMMARY_INTERVAL,
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL,
)
from basic_bot.memory import get_messages_after, get_state, load_window, save_summary
from basic_bot.models import MODELS, build_api_kwargs, extract_reply

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

_PERSONA_PATH = Path(__file__).parent / "instructions" / "persona.md"


def _build_memory_section(summary: str, position: dict | None) -> str:
    """Describe the conversation's memory state for the model: what it can see
    verbatim, what's been summarized, and how many messages exist."""
    if not position or not position.get("total"):
        return ""

    total = position["total"]
    through = position["summarized_through"]
    w_start = position["window_start"]
    w_end = position["window_end"]

    if summary and through:
        return (
            f"Here is a running summary of earlier messages (1 through {through}):\n"
            f"{summary}\n"
            f"You can see messages {w_start} through {w_end} verbatim below. "
            f"This conversation has {total} messages so far, not counting the "
            "user's current message."
        )
    return (
        f"You can see all {total} messages of this conversation verbatim "
        f"(messages {w_start} through {w_end}), not counting the user's "
        "current message."
    )


def build_system_prompt(
    model_id: str,
    effort: str | None = None,
    thinking: bool = False,
    summary: str = "",
    position: dict | None = None,
) -> str:
    """Build the system prompt with persona, model awareness, and memory state."""
    model_config = MODELS.get(model_id, MODELS[DEFAULT_MODEL])
    display_name = model_config["display_name"]
    persona = _PERSONA_PATH.read_text().strip()

    config_lines = [f"You are currently running on Claude {display_name}."]

    if model_config["effort_levels"] and effort:
        config_lines.append(f"Effort level is set to {effort}.")
    elif not model_config["effort_levels"]:
        config_lines.append("This model does not use effort levels.")

    config_lines.append(
        "Deep Reasoning is enabled." if thinking else "Deep Reasoning is disabled."
    )

    parts = [persona, "\n".join(config_lines)]
    memory_section = _build_memory_section(summary, position)
    if memory_section:
        parts.append(memory_section)
    return "\n".join(parts)


def summarize_batch(existing_summary: str, messages: list[dict[str, str]]) -> str:
    """Fold a batch of aged-out messages into the rolling summary using Haiku."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    system = (
        "You maintain a running summary of a conversation between a user and an AI assistant. "
        "Integrate the new messages into the existing summary, if one is provided. "
        "Preserve durable facts: names, preferences, decisions, and ongoing topics. "
        "Compress older detail rather than dropping it entirely. "
        "Write in plain prose, third person. "
        f"Keep the result under roughly {SUMMARY_MAX_TOKENS} tokens."
    )

    if existing_summary:
        user_content = f"EXISTING SUMMARY:\n{existing_summary}\n\nNEW MESSAGES:\n{transcript}"
    else:
        user_content = f"NEW MESSAGES:\n{transcript}"

    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return extract_reply(response)


def maybe_summarize(user_id: str, collection: str = CONVERSATION_COLLECTION) -> bool:
    """Fold the oldest aged-out batch into the summary, if the tail has grown
    past the threshold. Called after a turn is saved. Returns True if it folded.

    Self-contained error handling: a summarization failure leaves the summary
    and boundary untouched (so the next turn retries) and never disturbs the
    user-facing turn that already completed.
    """
    state = get_state(user_id, collection)
    boundary = state["summarized_through"]
    latest = state["next_seq"] - 1

    if latest - boundary < MESSAGE_LIMIT + SUMMARY_INTERVAL:
        return False

    try:
        tail = get_messages_after(user_id, boundary, collection)
        chunk = tail[:SUMMARY_INTERVAL]
        batch = [{"role": m["role"], "content": m["content"]} for m in chunk]
        new_through = chunk[-1]["seq"]

        new_summary = summarize_batch(state["summary"], batch)
        save_summary(user_id, new_summary, new_through, collection)

        logger.info(
            "Summarized user %s through seq %s (%d messages folded, summary %d chars)",
            user_id, new_through, len(batch), len(new_summary)
        )
        return True
    except Exception:
        logger.exception(
            "Summarization failed for user %s — summary left unchanged", user_id
        )
        return False


async def chat_with_claude(
    user_id: str,
    user_message: str,
    model_id: str = DEFAULT_MODEL,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Chat with Claude using conversation memory. Returns reply and metadata."""

    logger.info("Processing chat request for user: %s", user_id)

    window, summary, position = load_window(user_id, user_message)
    logger.info(
        "Loaded context: %d prior messages, summary=%s, boundary=seq %s",
        len(window) - 1, bool(summary), position["summarized_through"]
    )

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": m["role"], "content": m["content"]})
        for m in window
    ]

    if model_id not in MODELS:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = DEFAULT_MODEL

    model_config = MODELS[model_id]
    system_prompt = build_system_prompt(model_id, effort, thinking, summary, position)

    logger.info(
        "Sending request to %s (effort=%s, thinking=%s) with %d messages",
        model_config["display_name"], effort, thinking, len(messages)
    )

    try:
        kwargs = build_api_kwargs(model_id, system_prompt, messages, effort, thinking)
        response = client.messages.create(**kwargs)

        actual_model = response.model
        if actual_model != model_id:
            logger.warning(
                "Model mismatch: requested %s, got %s", model_id, actual_model
            )

        actual_config = MODELS.get(actual_model, model_config)
        reply = extract_reply(response)

        logger.info(
            "Received response from %s (effort=%s, thinking=%s, %d chars)",
            actual_config["display_name"], effort, thinking, len(reply)
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
            "Error with %s, attempting fallback to %s",
            model_config["display_name"], MODELS[FALLBACK_MODEL]["display_name"]
        )

        if model_id == FALLBACK_MODEL and not effort and not thinking:
            raise

        try:
            fallback_prompt = build_system_prompt(
                FALLBACK_MODEL, summary=summary, position=position
            )
            fallback_kwargs = build_api_kwargs(
                FALLBACK_MODEL, fallback_prompt, messages
            )
            response = client.messages.create(**fallback_kwargs)

            actual_model = response.model
            actual_config = MODELS.get(actual_model, MODELS[FALLBACK_MODEL])
            reply = extract_reply(response)

            logger.info(
                "Fallback to %s succeeded (%d chars)",
                actual_config["display_name"], len(reply)
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