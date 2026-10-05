"""Chat provider registry.

Composite InferenceProvider that holds local and API providers in one
catalog, grouped into hosts: "local" models run on this machine, "api"
models run on a provider's servers. Routes chat() to the provider that
owns the requested model.

The registry is the single owner of the local chat server. select()
makes a model the running one right away. suspend() and resume() let
work that needs the chat server's memory, such as embedding or a fold,
stop it and bring it back. Suspends nest: only the outermost resume
restarts the server.

Engine-core code sees a single InferenceProvider.
"""

import logging
from typing import Callable

from basic_bot.providers.protocol import ChatResponse, InferenceProvider, ModelInfo

logger = logging.getLogger(__name__)

HOST_LOCAL = "local"
HOST_API = "api"


class NoModelsError(RuntimeError):
    """Neither a local model nor an API provider is available."""


class ChatProviderRegistry:
    """Composite provider routing to local or API backends."""

    def __init__(
        self,
        *,
        local_providers: dict[str, InferenceProvider],
        api_provider: InferenceProvider | None = None,
        local_default: str | None,
        starting_host: str,
        start_local: Callable[[str], None],
        stop_local: Callable[[], None],
    ):
        if not local_providers and not api_provider:
            raise NoModelsError(
                "No chat models are downloaded and no API key is stored.\n"
                "Run 'python3 build.py' to download the local models."
            )

        self._local_providers = local_providers
        self._api_provider = api_provider
        self._start_local = start_local
        self._stop_local = stop_local

        # Ownership: model_id → provider, and model_id → host
        self._providers: dict[str, InferenceProvider] = {}
        self._hosts: dict[str, str] = {}
        for model_id, provider in local_providers.items():
            self._providers[model_id] = provider
            self._hosts[model_id] = HOST_LOCAL
        if api_provider:
            for model_id in api_provider.get_models():
                self._providers[model_id] = api_provider
                self._hosts[model_id] = HOST_API

        # Each host's default model
        self._defaults: dict[str, str] = {}
        if local_providers:
            if local_default is not None and local_default in local_providers:
                self._defaults[HOST_LOCAL] = local_default
            else:
                self._defaults[HOST_LOCAL] = next(iter(local_providers))
        if api_provider:
            self._defaults[HOST_API] = api_provider.get_default_model()

        if starting_host not in self._defaults:
            starting_host = HOST_LOCAL if HOST_LOCAL in self._defaults else HOST_API
        self._starting_host = starting_host

        # What the user has selected, and whether the local server is up.
        # These differ during a suspend, or after a failed start.
        self._active_host: str | None = None
        self._active_local_model: str | None = None
        self._server_up = False
        self._suspended = 0

    # --- Protocol: model catalog ---

    def get_models(self) -> dict[str, ModelInfo]:
        merged: dict[str, ModelInfo] = {}
        for provider in self._local_providers.values():
            merged.update(provider.get_models())
        if self._api_provider:
            merged.update(self._api_provider.get_models())
        return merged

    def get_default_model(self) -> str:
        """The default model of the current host, or the starting host."""
        return self._defaults[self._active_host or self._starting_host]

    def get_fallback_model(self) -> str:
        """Fallback stays within the current host.

        A local failure must never send the conversation to the API.
        """
        return self.get_default_model()

    # --- Hosts ---

    def hosts(self) -> list[str]:
        """The hosts with at least one model, local first."""
        return [h for h in (HOST_LOCAL, HOST_API) if h in self._defaults]

    def default_for(self, host: str) -> str:
        return self._defaults[host]

    @property
    def active_host(self) -> str | None:
        return self._active_host

    @property
    def active_local_model(self) -> str | None:
        """The local model selected, or None if on the API."""
        return self._active_local_model

    # --- Selection ---

    def select(self, model_id: str) -> str:
        """Make model_id the running model, right away.

        Moving to the API stops the local server. Moving to a local
        model starts it, or swaps models. Unknown models fall back to
        the default. Returns the model actually selected.

        If the server fails to start, the selection stands and the
        error propagates; the next select() or chat() tries again.
        """
        if model_id not in self._providers:
            logger.warning("Unknown model '%s', using default", model_id)
            model_id = self.get_default_model()

        if self._hosts[model_id] == HOST_API:
            if self._active_host == HOST_LOCAL:
                logger.info("Switching to API — stopping local chat server")
            self._stop_server()
            self._active_host = HOST_API
            self._active_local_model = None
            return model_id

        # Local: nothing to do if this model is already running, or
        # selected and waiting for a resume
        if (
            self._active_host == HOST_LOCAL
            and self._active_local_model == model_id
            and (self._server_up or self._suspended)
        ):
            return model_id

        self._stop_server()
        self._active_host = HOST_LOCAL
        self._active_local_model = model_id
        if not self._suspended:
            self._start_server()
        return model_id

    # --- Suspend and resume ---

    def suspend(self) -> None:
        """Stop the local chat server until the matching resume()."""
        self._suspended += 1
        if self._suspended == 1 and self._server_up:
            logger.info("Pausing local chat server")
            self._stop_server()

    def resume(self) -> None:
        """Restart the local chat server after the outermost suspend."""
        if self._suspended == 0:
            logger.warning("resume() without a matching suspend()")
            return
        self._suspended -= 1
        if (
            self._suspended == 0
            and self._active_host == HOST_LOCAL
            and not self._server_up
        ):
            self._start_server()

    def _start_server(self) -> None:
        model_id = self._active_local_model
        if model_id is None:
            # Callers select a local model first; this guards the invariant
            raise RuntimeError("No local model selected to start")
        logger.info("Starting local chat server for %s", model_id)
        self._start_local(model_id)
        self._server_up = True

    def _stop_server(self) -> None:
        if self._server_up:
            self._stop_local()
            self._server_up = False

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
