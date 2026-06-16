import logging
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import MessageParam

from basic_bot.config import DEFAULT_MODEL, FALLBACK_MODEL, MESSAGE_LIMIT, SUMMARY_MODEL, SUMMARY_MAX_TOKENS
from basic_bot.memory import get_messages
from basic_bot.models import MODELS, build_api_kwargs, extract_reply

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

_PERSONA_PATH = Path(__file__).parent / "instructions" / "persona.md"


def build_system_prompt(
    model_id: str,
    effort: str | None = None,
    thinking: bool = False,
) -> str:
    """Build the system prompt with persona text and full model awareness."""
    model_config = MODELS.get(model_id, MODELS[DEFAULT_MODEL])
    display_name = model_config["display_name"]
    persona = _PERSONA_PATH.read_text().strip()

    config_lines = [
        f"You are currently running on Claude {display_name}.",
        f"Your current sliding window is {MESSAGE_LIMIT} messages.",
    ]

    if model_config["effort_levels"] and effort:
        config_lines.append(f"Effort level is set to {effort}.")
    elif not model_config["effort_levels"]:
        config_lines.append("This model does not use effort levels.")

    if thinking:
        config_lines.append("Deep Reasoning is enabled.")
    else:
        config_lines.append("Deep Reasoning is disabled.")

    return f"{persona}\n" + "\n".join(config_lines)


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


async def chat_with_claude(
    user_id: str,
    user_message: str,
    model_id: str = DEFAULT_MODEL,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Chat with Claude using conversation memory. Returns reply and metadata."""

    logger.info("Processing chat request for user: %s", user_id)

    memory = get_messages(user_id)
    logger.info("Retrieved %d messages from history", len(memory))

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": msg["role"], "content": msg["content"]})
        for msg in memory
    ]
    messages.append({"role": "user", "content": user_message})

    if model_id not in MODELS:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = DEFAULT_MODEL

    model_config = MODELS[model_id]
    system_prompt = build_system_prompt(model_id, effort, thinking)

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
                "Model mismatch: requested %s, got %s",
                model_id, actual_model
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
            fallback_prompt = build_system_prompt(FALLBACK_MODEL)
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