import logging
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import MessageParam

from basic_bot.config import DEFAULT_MODEL, FALLBACK_MODEL
from basic_bot.memory import get_messages
from basic_bot.models import MODELS, build_api_kwargs, extract_reply

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

_PERSONA_PATH = Path(__file__).parent / "instructions" / "persona.md"


def build_system_prompt(model_id: str) -> str:
    """Build the system prompt with persona text and model awareness."""
    display_name = MODELS.get(model_id, MODELS[DEFAULT_MODEL])["display_name"]
    persona = _PERSONA_PATH.read_text().strip()
    return f"{persona}\nYou are currently running on {display_name}."


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
    system_prompt = build_system_prompt(model_id)

    logger.info(
        "Sending request to %s (effort=%s, thinking=%s) with %d messages",
        model_config["display_name"], effort, thinking, len(messages)
    )

    try:
        kwargs = build_api_kwargs(model_id, system_prompt, messages, effort, thinking)
        response = client.messages.create(**kwargs)
        reply = extract_reply(response)

        logger.info("Received response from %s (%d chars)",
                     model_config["display_name"], len(reply))

        return {
            "reply": reply,
            "model_used": model_id,
            "display_name": model_config["display_name"],
            "effort": effort,
            "thinking": thinking,
            "fallback": False,
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
            reply = extract_reply(response)

            logger.info(
                "Fallback to %s succeeded (%d chars)",
                MODELS[FALLBACK_MODEL]["display_name"], len(reply)
            )

            return {
                "reply": reply,
                "model_used": FALLBACK_MODEL,
                "display_name": MODELS[FALLBACK_MODEL]["display_name"],
                "effort": None,
                "thinking": False,
                "fallback": True,
            }

        except Exception:
            logger.exception("Fallback also failed")
            raise