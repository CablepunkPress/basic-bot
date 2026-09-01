"""Anthropic Claude provider.

Translates between the engine's internal message format and the
Anthropic Messages API. Handles model capabilities, thinking modes,
effort levels, and response parsing.
"""

import logging
import re

import anthropic
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from basic_bot.providers.protocol import ChatResponse, ModelInfo, ToolCall

logger = logging.getLogger(__name__)

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')

# Token limits
DEFAULT_MAX_TOKENS = 8192
THINKING_MAX_TOKENS = 16384
EXTENDED_BUDGET_TOKENS = 10000


MODELS: dict[str, ModelInfo] = {
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Haiku 4.5",
        provider="Anthropic",
        family="Claude",
        host="api",
        effort_levels=None,
        thinking_type="extended",
        rank=1,
    ),
    "claude-sonnet-4-6": ModelInfo(
        id="claude-sonnet-4-6",
        display_name="Sonnet 4.6",
        provider="Anthropic",
        family="Claude",
        host="api",
        effort_levels=["low", "medium", "high", "max"],
        thinking_type="adaptive",
        rank=2,
    ),
    "claude-opus-4-6": ModelInfo(
        id="claude-opus-4-6",
        display_name="Opus 4.6",
        provider="Anthropic",
        family="Claude",
        host="api",
        effort_levels=["low", "medium", "high", "max"],
        thinking_type="adaptive",
        rank=3,
    ),
    "claude-opus-4-7": ModelInfo(
        id="claude-opus-4-7",
        display_name="Opus 4.7",
        provider="Anthropic",
        family="Claude",
        host="api",
        effort_levels=["low", "medium", "high", "xhigh", "max"],
        thinking_type="adaptive",
        rank=4,
    ),
    "claude-opus-4-8": ModelInfo(
        id="claude-opus-4-8",
        display_name="Opus 4.8",
        provider="Anthropic",
        family="Claude",
        host="api",
        effort_levels=["low", "medium", "high", "xhigh", "max"],
        thinking_type="adaptive",
        rank=5,
    ),
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"


class ClaudeProvider:
    """Anthropic Claude inference provider."""

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic()

    def get_models(self) -> dict[str, ModelInfo]:
        return MODELS

    def get_default_model(self) -> str:
        return DEFAULT_MODEL

    def get_fallback_model(self) -> str:
        return FALLBACK_MODEL

    def chat(
        self,
        *,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        model_id: str,
        effort: str | None = None,
        thinking: bool = False,
        sampling: dict | None = None,
    ) -> ChatResponse:
        """Send messages to Claude and return a ChatResponse."""
        if model_id not in MODELS:
            logger.warning("Unknown model '%s', using default", model_id)
            model_id = DEFAULT_MODEL

        model_info = MODELS[model_id]
        api_messages = self._encode_messages(messages)
        kwargs = self._build_kwargs(
            model_id, model_info, system, api_messages, effort, thinking,
        )

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        return self._decode_response(response)

    def _encode_messages(self, messages: list[dict]) -> list[dict]:
        """Translate engine-format messages to Anthropic format."""
        encoded = []
        for msg in messages:
            if msg["role"] == "tool_result":
                # Engine tool results → Anthropic user message with tool_result blocks
                encoded.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r["tool_call_id"],
                            "content": r["content"],
                        }
                        for r in msg["results"]
                    ],
                })
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                # Engine assistant + tool_calls → Anthropic content blocks
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                encoded.append({"role": "assistant", "content": content})
            else:
                encoded.append({"role": msg["role"], "content": msg["content"]})
        return encoded

    def _build_kwargs(
        self,
        model_id: str,
        model_info: ModelInfo,
        system: str,
        messages: list[dict],
        effort: str | None,
        thinking: bool,
    ) -> dict:
        """Build kwargs for client.messages.create."""
        kwargs: dict = {
            "model": model_id,
            "system": system,
            "messages": messages,
        }

        if effort and model_info.effort_levels:
            if effort in model_info.effort_levels:
                kwargs["output_config"] = {"effort": effort}
            else:
                logger.warning(
                    "Effort '%s' not supported by %s, ignoring",
                    effort, model_info.display_name,
                )

        if thinking and model_info.thinking_type == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["max_tokens"] = THINKING_MAX_TOKENS
        elif thinking and model_info.thinking_type == "extended":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": EXTENDED_BUDGET_TOKENS,
            }
            kwargs["max_tokens"] = THINKING_MAX_TOKENS
        else:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        return kwargs

    def _decode_response(self, response) -> ChatResponse:
        """Translate an Anthropic response to ChatResponse."""
        text = ""
        tool_calls = []

        for block in response.content:
            if isinstance(block, TextBlock):
                text = _SEQ_ANNOTATION.sub('', block.text).strip()
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(ToolCall(
                    name=block.name,
                    input=block.input,
                    id=block.id,
                ))

        thinking = any(
            isinstance(b, ThinkingBlock) for b in response.content
        )

        return ChatResponse(
            text=text,
            model_used=response.model,
            thinking=thinking,
            tool_calls=tool_calls,
        )
