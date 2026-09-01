"""Inference provider protocol and shared data types.

Every provider (Claude, Gemini, local) implements InferenceProvider
and returns results using these types. The engine never imports
provider-specific SDKs — translation happens at the boundary.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelInfo:
    """A model available from a provider."""
    id: str                                # "claude-haiku-4-5-20251001"
    display_name: str                      # "Haiku 4.5"
    provider: str                          # "Anthropic"
    family: str                            # "Claude"
    host: str                              # "api" or "local"
    rank: int = 0                          # display order: lower = smaller/cheaper
    effort_levels: list[str] | None = None # ["low", "medium", "high", "max"]
    thinking_type: str | None = None       # "adaptive", "extended", None


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""
    name: str
    input: dict
    id: str


@dataclass
class ChatResponse:
    """What a provider returns from a single chat round-trip."""
    text: str
    model_used: str
    thinking: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)


class InferenceProvider(Protocol):
    """Contract for inference providers."""

    def get_models(self) -> dict[str, ModelInfo]:
        """Return available models keyed by model ID."""
        ...

    def get_default_model(self) -> str:
        """Return the default model ID for this provider."""
        ...

    def get_fallback_model(self) -> str:
        """Return the fallback model ID for this provider."""
        ...

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
        """Send messages and return a response.

        Messages are in engine format:
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "...", "tool_calls": [...]}
            {"role": "tool_result", "results": [{"tool_call_id": "...", "content": "..."}]}

        Tools are JSON Schema definitions (name, description, input_schema).
        The provider translates to its own API format internally.
        """
        ...
