"""Chat provider registry.

Composite InferenceProvider that holds local and API providers in
one catalog. Routes chat() to the correct provider based on model_id.

select() makes a model the running model right away: it starts or
stops the local chat server as needed. chat() calls select() too, so
a front end that never selects explicitly still gets the right model.

Engine-core code sees a single InferenceProvider. The registry is
invisible to chat.py, app.py, and everything downstream.
"""

import logging
from typing import Callable

from basic_bot.providers.protocol import ChatResponse, ModelInfo

logger = logging.getLogger(__name__)


class ChatProviderRegistry:
    """Composite provider routing to local or API backends.

    Holds a LocalProvider per available local chat model and an
    optional ClaudeProvider. Merges their catalogs into one
    get_models(). Routes chat() by looking up which provider owns
    the requested model_id.

    Nothing is running when the registry is built. The launcher
    calls select() with the default model to start it.
    """

    def __init__(
        self,
        *,
        local_providers: dict,
        api_provider=None,
        default_model: str,
        start_local: Callable[[str], None],
        stop_local: Callable[[], None],
    ):
        self._local_providers = local_providers
        self._api_provider = api_provider
        self._default_model = default_model
        self._start_local = start_local
        self._stop_local = stop_local

        # What is actually running right now
        self._active_type: str | None = None
        self._active_local_model: str | None = None

        # Ownership: model_id → (provider, type)
        self._providers: dict = {}
        self._model_types: dict[str, str] = {}

        for model_id, provider in local_providers.items():
            self._providers[model_id] = provider
            self._model_types[model_id] = "local"

        if api_provider:
            for model_id in api_provider.get_models():
                self._providers[model_id] = api_provider
                self._model_types[model_id] = "api"

    # --- Protocol: model catalog ---

    def get_models(self) -> dict[str, ModelInfo]:
        merged: dict[str, ModelInfo] = {}
        for provider in self._local_providers.values():
            merged.update(provider.get_models())
        if self._api_provider:
            merged.update(self._api_provider.get_models())
        return merged

    def get_default_model(self) -> str:
        return self._default_model

    def get_fallback_model(self) -> str:
        return self._default_model

    # --- Public: active state ---

    @property
    def active_local_model(self) -> str | None:
        """The local model currently loaded, or None if on API."""
        return self._active_local_model

    # --- Selection ---

    def select(self, model_id: str) -> str:
        """Make model_id the running model, right away.

        Stops the local chat server when moving to the API, and
        starts it, or swaps models, when moving to a local model.
        Unknown models fall back to the default. Returns the model
        actually selected.

        If the local server fails to start, the registry records
        that nothing is running and the error propagates.
        """
        if model_id not in self._providers:
            logger.warning("Unknown model '%s', using default", model_id)
            model_id = self._default_model

        target_type = self._model_types[model_id]

        if target_type == "api":
            if self._active_type == "local":
                logger.info("Switching to API — stopping local chat server")
                self._stop_local()
                self._active_local_model = None
            self._active_type = "api"
            return model_id

        # target_type == "local"
        if (
            self._active_type == "local"
            and self._active_local_model == model_id
        ):
            return model_id

        if self._active_type == "local":
            logger.info(
                "Switching local model: %s → %s",
                self._active_local_model, model_id,
            )
            self._stop_local()
        else:
            logger.info("Starting local chat server for %s", model_id)

        # Nothing is running until the start succeeds
        self._active_type = None
        self._active_local_model = None

        self._start_local(model_id)
        self._active_type = "local"
        self._active_local_model = model_id
        return model_id

    # --- Protocol: chat ---

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
        model_id = self.select(model_id)
        provider = self._providers[model_id]

        return provider.chat(
            messages=messages,
            system=system,
            tools=tools,
            model_id=model_id,
            effort=effort,
            thinking=thinking,
            sampling=sampling,
        )
