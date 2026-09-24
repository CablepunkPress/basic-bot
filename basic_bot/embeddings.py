"""Embedding providers.

LocalEmbedder talks to llama-server's /v1/embeddings endpoint.
ManagedEmbedder wraps it with on-demand server lifecycle — starts
the embedding server before each use, stops it after. Engine-core
code calls embed() without knowing a server was started and stopped.

The lifecycle callbacks are injected by the factory, so this module
has no knowledge of infrastructure or server management.
"""

import json
import logging
import urllib.request
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding providers.

    Any class with an embed() method matching this signature
    satisfies the protocol. Both LocalEmbedder and ManagedEmbedder
    implement it without inheriting from anything.
    """

    ctx_size: int

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]: ...


class LocalEmbedder:
    """Local embedding model via llama-server."""

    def __init__(self, base_url: str, ctx_size: int, query_prefix: str = "", passage_prefix: str = "", model_name: str = ""):
        self._endpoint = base_url.rstrip("/") + "/v1/embeddings"
        self.ctx_size = ctx_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        logger.info("Local embedder configured: %s → %s", model_name or "unknown", self._endpoint)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        if task == "query":
            inputs = [self._query_prefix + t for t in texts]
        else:
            inputs = [self._passage_prefix + t for t in texts]

        payload = json.dumps({"input": inputs}).encode("utf-8")

        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]


class ManagedEmbedder:
    """Wraps an embedder with on-demand server lifecycle.

    Starts the embedding server before each embed call, stops it
    after. Engine-core code sees the same embed() interface.
    The start/stop callbacks are injected by the factory.
    """

    def __init__(self, embedder: Embedder, start_fn, stop_fn):
        self._embedder = embedder
        self.ctx_size = embedder.ctx_size
        self._start = start_fn
        self._stop = stop_fn


    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        self._start()
        try:
            return self._embedder.embed(texts, task)
        finally:
            self._stop()
