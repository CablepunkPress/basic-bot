"""Chat router.

Composite InferenceProvider that holds the chat providers in one
catalog, grouped into hosts. Today there is one host, "local": models
that run on this machine. The host list is what front ends read, so
a host added later appears in their controls without changes to
them. Routes chat() to the provider that owns the requested model.

The router is the single owner of the local chat server. select()
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


class NoModelsError(RuntimeError):
    """No chat model is available."""


class ChatRouter:
    """Composite provider routing to the provider that owns each model."""

    def __init__(
        self,
        *,
        local_providers: dict[str, InferenceProvider],
        local_default: str | None,
        start_local: Callable[[str], None],
        stop_local: Callable[[], None],
    ):
        if not local_providers:
            raise NoModelsError(
                "No chat models are downloaded.\n"
                "Run 'python3 build.py' to download them."
            )

        self._local_providers = local_providers
        self._start_local = start_local
        self._stop_local = stop_local

        # Ownership: model_id → provider, and model_id → host
        self._providers: dict[str, InferenceProvider] = {}
        self._hosts: dict[str, str] = {}
        for model_id, provider in local_providers.items():
            self._providers[model_id] = provider
            self._hosts[model_id] = HOST_LOCAL

        # Each host's default model
        if local_default is not None and local_default in local_providers:
            self._defaults = {HOST_LOCAL: local_default}
        else:
            self._defaults = {HOST_LOCAL: next(iter(local_providers))}

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
        return merged

    def get_default_model(self) -> str:
        """The default model of the current host, or of local before any selection."""
        return self._defaults[self._active_host or HOST_LOCAL]

    def get_fallback_model(self) -> str:
        """Fallback stays within the current host.

        A failure on one host must never send the conversation to another.
        """
        return self.get_default_model()

    # --- Hosts ---

    def hosts(self) -> list[str]:
        """The hosts with at least one model, local first."""
        return list(self._defaults)

    def default_for(self, host: str) -> str:
        return self._defaults[host]

    @property
    def active_host(self) -> str | None:
        return self._active_host

    @property
    def active_local_model(self) -> str | None:
        """The local model selected, or None before any selection."""
        return self._active_local_model

    # --- Selection ---

    def select(self, model_id: str) -> str:
        """Make model_id the running model, right away.

        Starts the local server, or swaps models. Unknown models fall
        back to the default. Returns the model actually selected.

        If the server fails to start, the selection stands and the
        error propagates; the next select() or chat() tries again.
        """
        if model_id not in self._providers:
            logger.warning("Unknown model '%s', using default", model_id)
            model_id = self.get_default_model()

        # Nothing to do if this model is already running, or selected
        # and waiting for a resume
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

    # --- Protocol: warming ---

    def warm(
        self,
        *,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        model_id: str,
        effort: str | None = None,
        thinking: bool | None = None,
    ) -> None:
        """Prepare the running model's cache for the next request.

        Only for the model already running: warming never starts,
        stops, or switches a server, and does nothing while suspended.
        """
        if (
            model_id != self._active_local_model
            or not self._server_up
            or self._suspended
        ):
            return
        self._providers[model_id].warm(
            messages=messages,
            system=system,
            tools=tools,
            model_id=model_id,
            effort=effort,
            thinking=thinking,
        )

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
