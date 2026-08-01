"""Embedding provider abstraction.

Turns text into vectors. The rest of the system calls embed() and never
knows which provider produced the vectors. Swap providers with the
EMBEDDING_PROVIDER env var without touching any caller.

Providers translate a generic task ("document" | "query") into their own
mechanism: the Google GenAI SDK uses task_type in EmbedContentConfig;
EmbeddingGemma uses prompt prefixes prepended to the input text.
Provider-specific imports are deferred into each class so an environment
only installs the SDK for the provider it actually uses. The local
provider needs no SDK at all — stdlib urllib over HTTP.
"""

import json
import logging
import urllib.request
from typing import Protocol

from basic_bot.config import (
    EMBEDDING_LOCATION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_URL,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        ...


class LocalEmbedder:
    """Qwen3-Embedding via llama-server on localhost. Default for local installs.

    llama-server serves an OpenAI-compatible /v1/embeddings endpoint.
    Qwen3-Embedding uses an instruction prefix on queries only; documents
    are embedded with no prefix.
    """

    QUERY_INSTRUCT = (
        "Instruct: Given a search query, retrieve relevant passages "
        "from past conversation that answer the query\nQuery: "
    )

    def __init__(self, base_url: str):
        self._endpoint = base_url.rstrip("/") + "/v1/embeddings"
        logger.info("Local embedder ready: %s", self._endpoint)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        if task == "query":
            inputs = [self.QUERY_INSTRUCT + t for t in texts]
        else:
            inputs = texts

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


class VertexEmbedder:
    """Google GenAI embeddings via Vertex AI. Used by oravec.io (cloud)."""

    def __init__(self, model_name: str, location: str):
        self._model_name = model_name
        self._location = location
        self._client = None  # lazy — no network or credentials at import time

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                location=self._location,
            )
            logger.info("GenAI embedder ready: %s (%s)", self._model_name, self._location)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        self._ensure_client()
        assert self._client is not None
        from google.genai.types import EmbedContentConfig

        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=EmbedContentConfig(task_type=task_type),
        )
        if not result.embeddings:
            return []
        return [e.values for e in result.embeddings if e.values is not None]


def _build_embedder() -> EmbeddingProvider:
    if EMBEDDING_PROVIDER == "local":
        return LocalEmbedder(EMBEDDING_URL)
    if EMBEDDING_PROVIDER == "vertex":
        return VertexEmbedder(EMBEDDING_MODEL, EMBEDDING_LOCATION)
    raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER!r}")


_embedder: EmbeddingProvider | None = None


def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    """Embed a list of texts into vectors.

    task is "document" (storing turns) or "query" (searching). Returns one
    vector per input text, in order.
    """
    global _embedder
    if _embedder is None:
        _embedder = _build_embedder()
    return _embedder.embed(texts, task)
