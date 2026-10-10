"""Anthropic Claude provider.

Translates between the engine's internal message format and the
Anthropic Messages API. Handles model capabilities, thinking modes,
effort levels, and response parsing.

The engine and UI see each model's reasoning and effort controls
through ModelInfo. How thinking is requested from the API, adaptive
or extended, is private to this connector.
"""

import logging
import re

import anthropic
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from basic_bot.providers.protocol import (
    REASONING_OPTIONAL,
    ChatResponse,
    ModelInfo,
    ToolCall,
)

logger = logging.getLogger(__name__)

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')

# Token limits
DEFAULT_MAX_TOKENS = 8192
THINKING_MAX_TOKENS = 16384
EXTENDED_BUDGET_TOKENS = 10000

# How each model's thinking is requested from the API.
#   adaptive  the model decides how much to think
#   extended  a fixed thinking budget
_THINKING_MODE = {
    "claude-haiku-4-5-20251001": "extended",
    "claude-sonnet-4-6": "adaptive",
    "claude-opus-4-6": "adaptive",
    "claude-opus-4-7": "adaptive",
    "claude-opus-4-8": "adaptive",
}

# Effort defaults to "high", the API's own default when none is sent.
MODELS: dict[str, ModelInfo] = {
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Haiku 4.5",
        publisher="Anthropic",
        family="Claude",
        host="api",
        rank=1,
        reasoning=REASONING_OPTIONAL,
        reasoning_default=False,
    ),
    "claude-sonnet-4-6": ModelInfo(
        id="claude-sonnet-4-6",
        display_name="Sonnet 4.6",
        publisher="Anthropic",
        family="Claude",
        host="api",
        rank=2,
        reasoning=REASONING_OPTIONAL,
        reasoning_default=False,
        effort_levels=["low", "medium", "high", "max"],
        effort_default="high",
    ),
    "claude-opus-4-6": ModelInfo(
        id="claude-opus-4-6",
        display_name="Opus 4.6",
        publisher="Anthropic",
        family="Claude",
        host="api",
        rank=3,
        reasoning=REASONING_OPTIONAL,
        reasoning_default=False,
        effort_levels=["low", "medium", "high", "max"],
        effort_default="high",
    ),
    "claude-opus-4-7": ModelInfo(
        id="claude-opus-4-7",
        display_name="Opus 4.7",
        publisher="Anthropic",
        family="Claude",
        host="api",
        rank=4,
        reasoning=REASONING_OPTIONAL,
        reasoning_default=False,
        effort_levels=["low", "medium", "high", "xhigh", "max"],
        effort_default="high",
    ),
    "claude-opus-4-8": ModelInfo(
        id="claude-opus-4-8",
        display_name="Opus 4.8",
        publisher="Anthropic",
        family="Claude",
        host="api",
        rank=5,
        reasoning=REASONING_OPTIONAL,
        reasoning_default=False,
        effort_levels=["low", "medium", "high", "xhigh", "max"],
        effort_default="high",
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
        thinking: bool | None = None,
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
        thinking: bool | None,
    ) -> dict:
        """Build kwargs for client.messages.create."""
        reasoning_on, level = model_info.resolve(thinking, effort)

        kwargs: dict = {
            "model": model_id,
            "system": system,
            "messages": messages,
        }

        if level:
            kwargs["output_config"] = {"effort": level}

        mode = _THINKING_MODE.get(model_id)
        if reasoning_on and mode == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["max_tokens"] = THINKING_MAX_TOKENS
        elif reasoning_on and mode == "extended":
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
