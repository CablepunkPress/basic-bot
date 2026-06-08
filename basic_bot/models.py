import logging

from anthropic.types import TextBlock

from basic_bot.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EXTENDED_BUDGET_TOKENS,
    THINKING_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

MODELS = {
    "claude-haiku-4-5-20251001": {
        "display_name": "Haiku 4.5",
        "effort_levels": None,
        "thinking_type": "extended",
    },
    "claude-sonnet-4-6": {
        "display_name": "Sonnet 4.6",
        "effort_levels": ["low", "medium", "high", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-6": {
        "display_name": "Opus 4.6",
        "effort_levels": ["low", "medium", "high", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-7": {
        "display_name": "Opus 4.7",
        "effort_levels": ["low", "medium", "high", "xhigh", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-8": {
        "display_name": "Opus 4.8",
        "effort_levels": ["low", "medium", "high", "xhigh", "max"],
        "thinking_type": "adaptive",
    },
}


def build_api_kwargs(
    model_id: str,
    system_prompt: str,
    messages: list,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Build kwargs for client.messages.create based on model capabilities."""
    model_config = MODELS[model_id]

    kwargs: dict = {
        "model": model_id,
        "system": system_prompt,
        "messages": messages,
    }

    if effort and model_config["effort_levels"]:
        if effort in model_config["effort_levels"]:
            kwargs["output_config"] = {"effort": effort}
        else:
            logger.warning(
                "Effort '%s' not supported by %s, ignoring",
                effort, model_config["display_name"]
            )

    if thinking and model_config["thinking_type"] == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["max_tokens"] = THINKING_MAX_TOKENS
    elif thinking and model_config["thinking_type"] == "extended":
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": EXTENDED_BUDGET_TOKENS,
        }
        kwargs["max_tokens"] = THINKING_MAX_TOKENS
    else:
        kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

    return kwargs


def extract_reply(response) -> str:
    """Extract the text reply from a Claude response, skipping any thinking blocks."""
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text
    raise ValueError("No text block found in response")