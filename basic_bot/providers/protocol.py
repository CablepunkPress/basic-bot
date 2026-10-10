"""Inference provider protocol and shared data types.

Every provider (Claude, Gemini, local) implements InferenceProvider
and returns results using these types. The engine never imports
provider-specific SDKs — translation happens at the boundary.

A provider is the backend that answers requests. A model's publisher
is the organization that made it. The two differ for local models:
Muse is published by Meta, but provided by llama-server on this
machine.

Reasoning and effort are the engine's own vocabulary. ModelInfo says
what the user can control for each model. How each control reaches
the model is the provider's private business.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

# Reasoning modes
REASONING_NONE = "none"            # the model does not reason
REASONING_OPTIONAL = "optional"    # the user can switch reasoning on and off
REASONING_ALWAYS = "always"        # the model always reasons


@dataclass
class ModelInfo:
    """A model available from a provider, as the user sees it."""
    id: str                                   # "claude-haiku-4-5-20251001"
    display_name: str                         # "Haiku 4.5"
    publisher: str                            # "Anthropic", "Meta" — who made it not who runs it
    family: str                               # "Claude", "Muse" — groups models in the UI
    host: str                                 # "api" or "local"
    rank: int = 0                             # display order: lower = smaller/cheaper
    reasoning: str = REASONING_NONE           # "none", "optional", or "always"
    reasoning_default: bool = False           # on or off when nothing is chosen
    effort_levels: list[str] | None = None    # lowest to highest
    effort_default: str | None = None         # level used when nothing is chosen
    effort_needs_reasoning: bool = False      # effort does nothing with reasoning off

    def resolve(
        self, thinking: bool | None, effort: str | None,
    ) -> tuple[bool, str | None]:
        """Turn requested settings into the settings that will be used.

        None means "use this model's default." Requests the model can't
        honor are corrected: reasoning can't be switched off on an
        "always" model, an unknown effort level becomes the default,
        and effort is dropped where it would have no effect.

        Returns (reasoning_on, effort_level).
        """
        if self.reasoning == REASONING_ALWAYS:
            reasoning_on = True
        elif self.reasoning == REASONING_OPTIONAL:
            reasoning_on = self.reasoning_default if thinking is None else thinking
        else:
            reasoning_on = False

        level = None
        if self.effort_levels:
            if effort is None:
                level = self.effort_default
            elif effort in self.effort_levels:
                level = effort
            else:
                logger.warning(
                    "Effort '%s' not supported by %s — using %s",
                    effort, self.display_name, self.effort_default,
                )
                level = self.effort_default

            if self.effort_needs_reasoning and not reasoning_on:
                level = None

        return reasoning_on, level


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
        thinking: bool | None = None,
        sampling: dict | None = None,
    ) -> ChatResponse:
        """Send messages and return a response.

        Messages are in engine format:
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "...", "tool_calls": [...]}
            {"role": "tool_result", "results": [{"tool_call_id": "...", "content": "..."}]}

        Tools are JSON Schema definitions (name, description, input_schema).
        The provider translates to its own API format internally.

        thinking and effort of None mean "use the model's default."
        Providers resolve them with ModelInfo.resolve().
        """
        ...
